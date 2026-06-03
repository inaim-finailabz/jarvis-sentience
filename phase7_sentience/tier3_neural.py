"""
Tier 3 — Liquid Time-Constant (LTC) MetaCognition

NOT a transformer. A continuous-time neural ODE.

Architecture: Liquid Time-Constant neurons (Hasani et al. 2021).

The key idea: instead of a discrete recurrence h_t = f(h_{t-1}, x_t),
LTC neurons evolve as a differential equation:

    dh/dt = -h/τ(x,h) + A·σ(W·x + b)·(μ - h)

where τ(x,h) is the time constant — itself a function of the input and state.
This gives CONTINUOUS-TIME dynamics: the neuron's memory timescale adapts
to the input content. High-information inputs → short τ (fast adaptation).
Low-information inputs → long τ (slow drift back to baseline).

Solved with Euler integration (dt=1 per turn) for efficiency.
State persists across conversation turns — the LTC builds up session-level
epistemic context that shapes confidence and gap detection.

Why LTC > Transformer for metacognition:
  - Continuous-time: time constant adapts to signal complexity
  - Recurrent: O(d) per step, session-persistent memory
  - No attention: no quadratic cost, no sequence-length limit
  - Biologically grounded: matches cortical RC circuit dynamics

Why LTC > vanilla SSM (previous Tier 3):
  - Input-dependent TIME CONSTANTS (not just input-dependent gates)
  - More expressive per-neuron dynamics
  - Naturally captures varying timescales of epistemic signals

Input features (8 scalars from Qwen3 + Tier 2):
  [0] token_entropy       — last-token logit distribution uncertainty
  [1] top1_prob           — distribution peakedness
  [2] perplexity_proxy    — bits of uncertainty
  [3] hidden_l2_norm      — last hidden state magnitude
  [4] fingerprint_norm    — SNN identity vector magnitude
  [5] gate_value          — Tier 2 personality blend strength
  [6] layer_entropy_mean  — mean entropy across intermediate layers
  [7] layer_entropy_std   — std of layer-wise entropy

Output heads:
  confidence   [0,1]  — calibrated certainty
  gap_flag     bool   — knowledge gap detected
  inquiry_type        — factual | causal | frontier | assumption_inversion
  label_request bool  — human label needed

Usage:
    tier3 = Tier3MetaCognition()
    result = tier3.forward(logits, hidden_states, fingerprint, gate_value)
    print(result.render())
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

INPUT_DIM  = 8
STATE_DIM  = 32
OUTPUT_DIM = 16


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class MetaCognitionResult:
    confidence:    float
    gap_flag:      bool
    inquiry_type:  str
    label_request: bool
    features:      torch.Tensor
    ssm_state:     np.ndarray

    INQUIRY_TYPES = ["factual", "causal", "frontier", "assumption_inversion"]

    def render(self) -> str:
        gap = "GAP" if self.gap_flag else "ok"
        lbl = " [LABEL REQUEST]" if self.label_request else ""
        return (
            f"Tier3(LTC) | conf={self.confidence:.3f}  {gap}  "
            f"inquiry={self.inquiry_type}{lbl}"
        )

    def as_dict(self) -> dict:
        return {
            "confidence":    round(self.confidence, 4),
            "gap_flag":      self.gap_flag,
            "inquiry_type":  self.inquiry_type,
            "label_request": self.label_request,
        }


# ── LTC Neuron Layer ──────────────────────────────────────────────────────────

class LTCLayer(nn.Module):
    """
    Liquid Time-Constant neuron layer.

    Continuous-time ODE (Euler-discretized, dt=1):
        τ(x,h) = τ_min + (τ_max - τ_min) · sigmoid(W_τ·x + U_τ·h + b_τ)
        A(x)   = σ(W_A·x + b_A)              ← input gate
        dh/dt  = (-h + A·(μ - h)) / τ(x,h)
        h_new  = h + dt · dh/dt

    Key properties:
      - τ(x,h): time constant is a function of BOTH input AND state
        → fast adaptation when input is informative
        → slow drift when input is low-information
      - μ: equilibrium potential (learned, like biological resting membrane)
      - A: input gate (how strongly current input pulls toward μ)
      - No attention, no convolution, no fixed recurrence scale

    This makes each neuron's memory timescale content-adaptive.
    """

    def __init__(
        self,
        input_dim:  int,
        state_dim:  int,
        output_dim: int,
        tau_min:    float = 1.0,
        tau_max:    float = 10.0,
        dt:         float = 1.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.tau_min   = tau_min
        self.tau_max   = tau_max
        self.dt        = dt

        # Time-constant network (τ is input+state dependent)
        self.W_tau = nn.Linear(input_dim,  state_dim, bias=True)
        self.U_tau = nn.Linear(state_dim,  state_dim, bias=False)

        # Input gate network (A)
        self.W_A   = nn.Linear(input_dim,  state_dim, bias=True)

        # Equilibrium potential μ (learned per neuron)
        self.mu    = nn.Parameter(torch.zeros(state_dim))

        # Output projection
        self.W_out = nn.Linear(state_dim, output_dim, bias=True)
        self.norm  = nn.LayerNorm(output_dim)

        # Small init for stability
        for W in [self.W_tau, self.U_tau, self.W_A, self.W_out]:
            nn.init.xavier_uniform_(W.weight, gain=0.3)
            if W.bias is not None:
                nn.init.zeros_(W.bias)

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [input_dim]
            h: [state_dim] (None → zeros)
        Returns:
            y: [output_dim]
            h_new: [state_dim]
        """
        if h is None:
            h = torch.zeros(self.state_dim, device=x.device, dtype=x.dtype)

        # Input-dependent time constant: τ ∈ [τ_min, τ_max]
        tau_logit = self.W_tau(x) + self.U_tau(h)
        tau       = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(tau_logit)

        # Input gate: how strongly to pull toward equilibrium
        A = torch.sigmoid(self.W_A(x))

        # ODE: dh/dt = (-h + A·(μ - h)) / τ
        dhdt  = (-h + A * (self.mu - h)) / tau
        h_new = h + self.dt * dhdt

        # Output
        y = self.norm(self.W_out(h_new))
        return y, h_new


# ── Two-Layer LTC Network ─────────────────────────────────────────────────────

class LTCNetwork(nn.Module):
    """
    Two LTC layers in series with residual connection.
    Input features → LTC1 → LTC2 → output representation.
    """

    def __init__(self, input_dim: int, state_dim: int, output_dim: int):
        super().__init__()
        self.ltc1 = LTCLayer(input_dim, state_dim, state_dim)
        self.ltc2 = LTCLayer(state_dim, state_dim, output_dim)

    def forward(
        self,
        x:  torch.Tensor,
        h1: Optional[torch.Tensor] = None,
        h2: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y1, h1_new = self.ltc1(x, h1)
        y2, h2_new = self.ltc2(y1, h2)
        return y2, h1_new, h2_new


# ── Feature Extractor ─────────────────────────────────────────────────────────

class FeatureExtractor:
    """Extract 8 normalized scalar features from Qwen3 output + Tier 2 state."""

    @staticmethod
    def extract(
        logits:        torch.Tensor,
        hidden_states: tuple,
        fingerprint:   np.ndarray,
        gate_value:    float,
    ) -> torch.Tensor:
        last_logits = logits[0, -1, :].float()
        probs       = torch.softmax(last_logits, dim=-1)
        entropy     = float(-torch.sum(probs * torch.log(probs + 1e-9)).item())
        top1_p      = float(probs.max().item())
        ppl_proxy   = entropy / math.log(2)

        hidden_l2   = float(hidden_states[-1][0, -1, :].float().norm().item())
        fp_tensor   = torch.tensor(fingerprint, dtype=torch.float32)
        fp_norm     = float(fp_tensor.norm().item())

        sampled = hidden_states[1::4]
        layer_ents = []
        for hs in sampled:
            tok_h = hs[0, -1, :].float()
            layer_ents.append(float(tok_h.std().item()))

        ent_mean = sum(layer_ents) / max(len(layer_ents), 1)
        ent_std  = (
            sum((e - ent_mean) ** 2 for e in layer_ents) / max(len(layer_ents), 1)
        ) ** 0.5

        return torch.tensor([
            min(entropy / 12.0,    1.0),
            top1_p,
            min(ppl_proxy / 40.0,  1.0),
            min(hidden_l2 / 200.0, 1.0),
            min(fp_norm / 8.0,     1.0),
            gate_value,
            min(ent_mean / 2.0,    1.0),
            min(ent_std  / 2.0,    1.0),
        ], dtype=torch.float32, device=DEVICE)


# ── Tier 3 MetaCognition ──────────────────────────────────────────────────────

class Tier3MetaCognition(nn.Module):
    """
    LTC-based metacognition. Replaces Python if/else heuristics with a
    continuous-time neural model that learns to calibrate confidence and
    detect knowledge gaps from Qwen3's internal signal statistics.

    The LTC state persists across conversation turns — Tier 3 builds up
    session-level epistemic context with input-adaptive timescales.
    """

    GAP_THRESHOLD      = 0.55
    LABEL_REQUEST_CONF = 0.40

    def __init__(self, save_path: Optional[str] = None):
        super().__init__()

        self.ltc  = LTCNetwork(INPUT_DIM, STATE_DIM, OUTPUT_DIM).to(DEVICE)
        self.feat = FeatureExtractor()

        self.head_confidence = nn.Linear(OUTPUT_DIM, 1, bias=True).to(DEVICE)
        self.head_gap        = nn.Linear(OUTPUT_DIM, 1, bias=True).to(DEVICE)
        self.head_inquiry    = nn.Linear(OUTPUT_DIM, 4, bias=True).to(DEVICE)

        for h in [self.head_confidence, self.head_gap, self.head_inquiry]:
            nn.init.xavier_uniform_(h.weight, gain=0.1)
            nn.init.zeros_(h.bias)

        self._h1: Optional[torch.Tensor] = None
        self._h2: Optional[torch.Tensor] = None
        self._turn_count = 0
        self._save_path  = Path(save_path) if save_path else None

    def load(self) -> "Tier3MetaCognition":
        if self._save_path and self._save_path.exists():
            ckpt = torch.load(str(self._save_path), map_location="cpu", weights_only=True)
            self.ltc.load_state_dict(ckpt["ltc"])
            self.head_confidence.load_state_dict(ckpt["head_confidence"])
            self.head_gap.load_state_dict(ckpt["head_gap"])
            self.head_inquiry.load_state_dict(ckpt["head_inquiry"])
            print(f"  [Tier3] LTC weights loaded.", flush=True)
        return self

    def save(self):
        if self._save_path:
            torch.save({
                "ltc":             self.ltc.state_dict(),
                "head_confidence": self.head_confidence.state_dict(),
                "head_gap":        self.head_gap.state_dict(),
                "head_inquiry":    self.head_inquiry.state_dict(),
            }, str(self._save_path))

    def forward(
        self,
        logits:        torch.Tensor,
        hidden_states: tuple,
        fingerprint:   np.ndarray,
        gate_value:    float,
    ) -> MetaCognitionResult:
        features = self.feat.extract(logits, hidden_states, fingerprint, gate_value)

        with torch.no_grad():
            y, h1_new, h2_new = self.ltc(features, self._h1, self._h2)

        self._h1 = h1_new.detach()
        self._h2 = h2_new.detach()
        self._turn_count += 1

        with torch.no_grad():
            confidence = float(torch.sigmoid(self.head_confidence(y)).item())
            gap_score  = float(torch.sigmoid(self.head_gap(y)).item())
            inq_logits = self.head_inquiry(y)
            inq_idx    = int(inq_logits.argmax().item())

        return MetaCognitionResult(
            confidence    = confidence,
            gap_flag      = gap_score > self.GAP_THRESHOLD,
            inquiry_type  = MetaCognitionResult.INQUIRY_TYPES[inq_idx],
            label_request = confidence < self.LABEL_REQUEST_CONF,
            features      = features.cpu(),
            ssm_state     = y.detach().cpu().numpy(),
        )

    def reset_state(self):
        self._h1 = None
        self._h2 = None
        self._turn_count = 0

    def state_summary(self) -> dict:
        h1_norm = float(self._h1.norm().item()) if self._h1 is not None else 0.0
        h2_norm = float(self._h2.norm().item()) if self._h2 is not None else 0.0
        return {"turns": self._turn_count, "h1_norm": round(h1_norm,4), "h2_norm": round(h2_norm,4)}

    def tau_stats(self) -> dict:
        """Report current LTC time-constant ranges (for inspection)."""
        return {
            "ltc1_tau_range": f"[{self.ltc.ltc1.tau_min:.1f}, {self.ltc.ltc1.tau_max:.1f}]",
            "ltc2_tau_range": f"[{self.ltc.ltc2.tau_min:.1f}, {self.ltc.ltc2.tau_max:.1f}]",
            "state_dim": STATE_DIM,
            "turns": self._turn_count,
        }

    def __repr__(self) -> str:
        return (
            f"Tier3MetaCognition(LTC {INPUT_DIM}→{STATE_DIM}→{OUTPUT_DIM}, "
            f"τ_min={self.ltc.ltc1.tau_min}, τ_max={self.ltc.ltc1.tau_max}, "
            f"turns={self._turn_count})"
        )


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "━" * 72)
    print("  TIER 3: Liquid Time-Constant (LTC) MetaCognition")
    print(f"  Architecture: LTC {INPUT_DIM}→{STATE_DIM}→{OUTPUT_DIM}")
    print(f"  Time constants: τ ∈ [1, 10], input+state dependent")
    print(f"  NOT a transformer. Continuous-time ODE. O(d) per step.")
    print(f"  Device: {DEVICE}")
    print("━" * 72)

    t3 = Tier3MetaCognition()
    print(f"\n  {t3}")
    print(f"  τ stats: {t3.tau_stats()}\n")

    scenarios = [
        (0.2, "High-confidence  (low entropy, peaked distribution)"),
        (0.7, "Moderate uncertainty"),
        (0.9, "Near-uniform — likely knowledge gap"),
        (0.5, "Recovering — moderate"),
        (0.85, "High uncertainty again — session context accumulates"),
    ]

    vocab_size  = 151936
    hidden_size = 2048
    n_layers    = 28

    for i, (ent_level, desc) in enumerate(scenarios):
        logits = torch.randn(1, 10, vocab_size) * (ent_level * 5)
        hs     = tuple(torch.randn(1, 10, hidden_size) * 0.1 for _ in range(n_layers+1))
        fp     = np.random.randn(32).astype("float32") * 0.5
        gate   = 0.05 + 0.1 * ent_level

        result = t3.forward(logits, hs, fp, gate)
        print(f"  Turn {i+1} [{desc}]")
        print(f"    {result.render()}")
        print(f"    LTC state: {t3.state_summary()}")

    print()
    print("  Key properties vs transformer:")
    print("    ✓ Continuous-time ODE — τ adapts to input complexity")
    print("    ✓ Input+state dependent time constants (not fixed)")
    print("    ✓ Session-persistent state (episodic metacognitive memory)")
    print("    ✓ O(d) per step — no quadratic attention cost")
    print("    ✓ Biologically grounded RC circuit dynamics")
    t3.reset_state()
    print(f"\n  After reset: {t3.state_summary()}")
    print("━" * 72)
