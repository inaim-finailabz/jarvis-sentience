"""
Tier 2 — Neural PersonalityConnectome (Echo State + Hypernetwork + SNN)

Architecture — three genuinely non-transformer components in series:

  ┌─────────────────────────────────────────────────────────────────┐
  │  Qwen3 last hidden [2048]                                       │
  │        ↓                                                        │
  │  [Echo State Network — Reservoir Computing]                     │
  │    Fixed sparse random recurrent reservoir (N=512 neurons)      │
  │    Spectral radius < 1 → echo state property (fading memory)   │
  │    Reservoir is FROZEN — never trained                          │
  │    Only linear readout W_out [512→12] is learned                │
  │    → 12-dim trait activations                                   │
  │        ↓                                                        │
  │  [SNN — Leaky Integrate-and-Fire + STDP]                        │
  │    Spiking dynamics: 12 → 64 LIF → 32 output neurons           │
  │    STDP local learning (no global error signal)                 │
  │    → [32] identity fingerprint, unique per agent                │
  │        ↓                                                        │
  │  [Hypernetwork Decoder]                                         │
  │    fingerprint [32] → generates two rank-r matrices dynamically │
  │    W_A [rank, 2048]  W_B [2048, rank]  (LoRA-style low rank)    │
  │    personality_residual = W_B(z) @ W_A(z) @ last_hidden        │
  │    Different agent fingerprint → different projection matrix    │
  │        ↓                                                        │
  │  personality_residual [2048]                                    │
  │  → added to Qwen3 last hidden before LM head                   │
  └─────────────────────────────────────────────────────────────────┘

Why NOT a transformer:
  ESN:         No attention. Fixed recurrent dynamics. Echo state property.
  SNN:         Spike-based. STDP local rule. Biologically plausible.
  Hypernetwork: Generates dynamic weight matrices. Not a fixed projection.

What changed from old architecture:
  Before: JSON traits → system prompt string injection
  Now:    Qwen3 hidden → ESN reservoir → SNN spikes → HyperNet → weight residual
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent))
from snn_personality import SNNPersonalityLayer, N_TRAITS, N_OUTPUT, T_DEFAULT

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

RESERVOIR_SIZE = 512
LORA_RANK      = 16


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class Tier2Result:
    personality_residual: torch.Tensor
    fingerprint:          np.ndarray
    trait_vector:         list[float]
    gate_value:           float
    snn_sparsity:         float
    snn_energy_proxy:     float

    def render(self) -> str:
        return (
            f"Tier2(ESN+SNN+HyperNet) | gate={self.gate_value:.4f}  "
            f"sparsity={self.snn_sparsity:.3f}  "
            f"energy_proxy={self.snn_energy_proxy:.4f}  "
            f"|residual|={self.personality_residual.norm().item():.4f}"
        )


# ── Echo State Network ────────────────────────────────────────────────────────

class EchoStateEncoder(nn.Module):
    """
    Reservoir computing encoder: Qwen3 hidden [D] → trait activations [n_traits].

    The reservoir (W_res) is a large fixed sparse recurrent network — NEVER trained.
    Random recurrent connections create rich nonlinear projections of the input.
    Only the linear readout W_out: [reservoir_size → n_traits] is learned.

    Echo State Property: spectral radius < 1 → reservoir response fades,
    ensuring stable dynamics while maintaining rich transient representations.

    Biological analog: cortical microcircuit as liquid-state machine — the cortex
    as a high-dimensional nonlinear transformer from which linear readout extracts
    behaviorally relevant features.
    """

    def __init__(
        self,
        input_dim:       int   = 2048,
        reservoir_size:  int   = RESERVOIR_SIZE,
        n_traits:        int   = N_TRAITS,
        spectral_radius: float = 0.9,
        sparsity:        float = 0.1,
        seed:            int   = 42,
    ):
        super().__init__()
        self.reservoir_size = reservoir_size

        rng = torch.Generator()
        rng.manual_seed(seed)

        # Input projection (fixed, not trained)
        W_in = torch.randn(reservoir_size, input_dim, generator=rng) * 0.1
        self.register_buffer("W_in", W_in)

        # Sparse recurrent reservoir (fixed, not trained)
        W_res = torch.randn(reservoir_size, reservoir_size, generator=rng)
        mask  = (torch.rand(reservoir_size, reservoir_size, generator=rng) < sparsity).float()
        W_res = W_res * mask
        with torch.no_grad():
            eig = torch.linalg.eigvals(W_res)
            sr  = eig.abs().max().item()
            if sr > 1e-6:
                W_res = W_res * (spectral_radius / sr)
        self.register_buffer("W_res", W_res)

        # Reservoir state (persistent, detached from graph)
        self.register_buffer("h_res", torch.zeros(reservoir_size))

        # Only trained component: linear readout
        self.readout = nn.Linear(reservoir_size, n_traits, bias=True)
        nn.init.xavier_uniform_(self.readout.weight, gain=0.3)
        nn.init.zeros_(self.readout.bias)

    def reset_state(self):
        self.h_res.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [input_dim] → [n_traits] in [0,1]"""
        x = x.float().to(DEVICE)
        new_h = torch.tanh(self.W_in @ x + self.W_res @ self.h_res)
        self.h_res = new_h.detach()
        return torch.sigmoid(self.readout(new_h))


# ── Hypernetwork Decoder ──────────────────────────────────────────────────────

class HypernetworkDecoder(nn.Module):
    """
    Hypernetwork: SNN fingerprint GENERATES the projection weight matrices.

    Given fingerprint z ∈ R^{N_OUTPUT}:
      W_A(z) = MLP_A(z) reshaped to [rank, hidden_size]
      W_B(z) = MLP_B(z) reshaped to [hidden_size, rank]
      personality_residual = W_B(z) @ (W_A(z) @ last_hidden)

    The projection matrix is unique per agent (z differs) and evolves over time
    as STDP updates the SNN weights that produce z.

    This is dynamic LoRA: the adapter matrices are generated fresh from the
    agent's live neural identity, not stored as static parameters.
    """

    def __init__(
        self,
        fingerprint_dim: int = N_OUTPUT,
        hidden_size:     int = 2048,
        rank:            int = LORA_RANK,
    ):
        super().__init__()
        self.rank        = rank
        self.hidden_size = hidden_size

        self.hyper_A = nn.Sequential(
            nn.Linear(fingerprint_dim, 128, bias=True),
            nn.GELU(),
            nn.Linear(128, rank * hidden_size, bias=False),
        )
        self.hyper_B = nn.Sequential(
            nn.Linear(fingerprint_dim, 128, bias=True),
            nn.GELU(),
            nn.Linear(128, hidden_size * rank, bias=False),
        )

        for net in [self.hyper_A, self.hyper_B]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.05)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, fingerprint: torch.Tensor, last_hidden: torch.Tensor) -> torch.Tensor:
        """fingerprint [N_OUTPUT], last_hidden [hidden_size] → residual [hidden_size]"""
        fp = fingerprint.float().to(DEVICE)
        h  = last_hidden.float().to(DEVICE)

        W_A = self.hyper_A(fp).view(self.rank, self.hidden_size)    # [rank, hidden]
        W_B = self.hyper_B(fp).view(self.hidden_size, self.rank)    # [hidden, rank]

        return W_B @ (W_A @ h)   # [hidden_size]


# ── Tier 2 Layer ──────────────────────────────────────────────────────────────

class Tier2PersonalityLayer(nn.Module):
    """
    Three non-transformer components in series:
      ESN (reservoir computing) → SNN (spikes + STDP) → Hypernetwork (dynamic LoRA)
    """

    TRAIT_NAMES = [
        "openness", "conscientiousness", "extraversion", "agreeableness",
        "neuroticism", "skepticism", "curiosity", "pragmatism",
        "empathy", "creativity", "discipline", "resilience",
    ]

    def __init__(self, agent_name: str, hidden_size: int = 2048, snn_dir: str = "."):
        super().__init__()
        self.agent_name  = agent_name
        self.hidden_size = hidden_size

        agent_seed = abs(hash(agent_name)) % (2 ** 20)
        self.esn   = EchoStateEncoder(
            input_dim=hidden_size, reservoir_size=RESERVOIR_SIZE,
            n_traits=N_TRAITS, seed=agent_seed,
        ).to(DEVICE)

        self.snn         = SNNPersonalityLayer(agent_name, snn_dir=snn_dir)
        self.hypernetwork = HypernetworkDecoder(
            fingerprint_dim=N_OUTPUT, hidden_size=hidden_size, rank=LORA_RANK,
        ).to(DEVICE)

        self.gate = nn.Parameter(torch.tensor(-3.0, device=DEVICE))
        self._save_path = Path(snn_dir) / f"{agent_name.lower()}_tier2.pt"

    def load(self) -> "Tier2PersonalityLayer":
        if self._save_path.exists():
            ckpt = torch.load(str(self._save_path), map_location="cpu", weights_only=True)
            if "esn_readout" in ckpt:
                self.esn.readout.load_state_dict(ckpt["esn_readout"])
            if "hypernetwork" in ckpt:
                self.hypernetwork.load_state_dict(ckpt["hypernetwork"])
            if "gate" in ckpt:
                self.gate.data = ckpt["gate"].to(DEVICE)
            print(f"  [Tier2] {self.agent_name}: ESN+HyperNet weights loaded.", flush=True)
        return self

    def save(self):
        torch.save({
            "esn_readout":  self.esn.readout.state_dict(),
            "hypernetwork": self.hypernetwork.state_dict(),
            "gate":         self.gate.data.cpu(),
        }, str(self._save_path))

    def forward(self, last_hidden: torch.Tensor) -> Tier2Result:
        last_hidden = last_hidden.to(DEVICE)

        # 1. ESN: reservoir dynamics → traits
        with torch.no_grad():
            trait_tensor = self.esn(last_hidden)
        trait_vector = trait_tensor.cpu().tolist()

        # 2. SNN: spiking dynamics → fingerprint
        snn_result = self.snn.forward(trait_vector, T=T_DEFAULT)
        fp_tensor  = torch.tensor(
            snn_result.fingerprint, dtype=torch.float32, device=DEVICE
        )

        # 3. Hypernetwork: fingerprint → dynamic LoRA matrices → residual
        with torch.no_grad():
            raw_residual = self.hypernetwork(fp_tensor, last_hidden)

        gate_val             = float(torch.sigmoid(self.gate).item())
        personality_residual = gate_val * raw_residual

        return Tier2Result(
            personality_residual = personality_residual,
            fingerprint          = snn_result.fingerprint,
            trait_vector         = trait_vector,
            gate_value           = gate_val,
            snn_sparsity         = snn_result.sparsity,
            snn_energy_proxy     = snn_result.energy_proxy,
        )

    def stdp_update(self):
        return self.snn.stdp_update()

    def traits_as_system_prompt(self, trait_vector: list[float]) -> str:
        dominant = sorted(
            zip(self.TRAIT_NAMES, trait_vector), key=lambda kv: kv[1], reverse=True
        )[:4]
        desc = ", ".join(f"{n} ({v:.2f})" for n, v in dominant)
        return (
            f"You are {self.agent_name}. "
            f"Your dominant personality traits (ESN-derived): {desc}. "
            f"Let these shape your reasoning and responses."
        )

    def __repr__(self) -> str:
        return (
            f"Tier2({self.agent_name!r}: "
            f"ESN(res={RESERVOIR_SIZE})→SNN(12→64→32)→HyperNet(rank={LORA_RANK}), "
            f"gate={float(torch.sigmoid(self.gate).item()):.4f})"
        )


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    tmpdir = tempfile.mkdtemp()

    print("\n" + "━" * 72)
    print("  TIER 2: Echo State Network + SNN + Hypernetwork")
    print(f"  ESN reservoir={RESERVOIR_SIZE}  SNN 12→64→32  HyperNet rank={LORA_RANK}")
    print(f"  Device: {DEVICE}")
    print("━" * 72)

    hidden_size = 2048
    fake_hidden = torch.randn(hidden_size, device=DEVICE) * 0.1
    agents = ["Jarvis", "Nora", "Zed"]
    results = {}

    for name in agents:
        t2 = Tier2PersonalityLayer(name, hidden_size=hidden_size, snn_dir=tmpdir)
        r  = t2(fake_hidden)
        results[name] = (t2, r)
        print(f"\n  [{name}]  {r.render()}")
        print(f"  traits[:4]: {[round(x,3) for x in r.trait_vector[:4]]}")

    print("\n  Fingerprint cosine distances:")
    names = list(results.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            fa = results[names[i]][1].fingerprint
            fb = results[names[j]][1].fingerprint
            cos = float(np.dot(fa,fb) / (np.linalg.norm(fa)*np.linalg.norm(fb)+1e-9))
            print(f"  {names[i]} ↔ {names[j]}: dist={1-cos:.4f}")

    print("\n  Architecture (genuinely non-transformer):")
    print(f"  ESN: {RESERVOIR_SIZE} reservoir neurons (FROZEN random recurrent)")
    print(f"       {RESERVOIR_SIZE * N_TRAITS} trainable readout weights only")
    print(f"  SNN: STDP local learning, spiking, no backprop through Qwen3")
    print(f"  HyperNet: fingerprint → dynamic {LORA_RANK}×{hidden_size} matrices (evolving with STDP)")
    print("━" * 72)
