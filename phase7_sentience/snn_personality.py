"""
SNN Personality Layer — Phase 4 Neuromorphic Substrate (M4.1–M4.4)

Implements the Tier 2 PersonalityConnectome as a Spiking Neural Network
with Spike-Timing Dependent Plasticity (STDP), running on PyTorch / MPS.

Biological motivation:
  Biological neurons communicate via discrete spikes, not continuous values.
  Uniqueness emerges from the specific pattern of synaptic weights shaped
  by the agent's history. STDP ("neurons that fire together, wire together")
  is the local learning rule — no global error signal, no backpropagation.

Architecture:
  Input   (12 neurons)  — one per personality trait; fires at rate ∝ trait value
  Hidden  (64 LIF neurons) — leaky integrate-and-fire; sparse, asynchronous
  Output  (32 neurons)  — agent's "neural fingerprint"; read out as identity vector

Neuron model — Leaky Integrate-and-Fire (LIF):
  V(t+1) = τ · V(t) + W · spike_in(t)     ← membrane potential update
  spike(t) = 1  if V(t) ≥ V_th            ← fire if threshold crossed
  V(t)    = V_reset  after spike           ← reset after firing

STDP learning rule (Bi & Poo 1998):
  If pre fires BEFORE post (Δt > 0):  Δw = A+ · exp(−Δt / τ+)   [potentiation]
  If pre fires AFTER  post (Δt < 0):  Δw = −A− · exp( Δt / τ−)  [depression]

  Net effect: synapses that predict spikes get stronger; others weaken.
  This is fully local — each synapse only needs its own pre/post spike times.

Milestones:
  M4.1  Personality layer implemented as SNN with STDP replacing EWC
  M4.2  Identity probe: SNN activations classify agent identity > 90% accuracy
  M4.3  STDP updates within a single forward pass (no batch, no optimiser)
  M4.4  Energy proxy: sparse spike count vs equivalent dense activation

Usage:
    from snn_personality import SNNPersonalityLayer
    snn = SNNPersonalityLayer("Nora", snn_dir="./agents")
    spikes, fingerprint = snn.forward(trait_vector, T=20)
    snn.stdp_update()
    identity_vec = snn.fingerprint()
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Device ────────────────────────────────────────────────────────────────────

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

# ── Network dimensions ────────────────────────────────────────────────────────

N_TRAITS  = 12    # input: one neuron per personality trait
N_HIDDEN  = 64    # hidden LIF layer
N_OUTPUT  = 32    # output fingerprint layer
T_DEFAULT = 20    # simulation timesteps per forward pass

# ── LIF parameters ───────────────────────────────────────────────────────────

TAU_MEM   = 0.9   # membrane time constant (decay per step)
V_THRESH  = 0.5   # firing threshold (calibrated for W=[0,0.02], τ=0.9, T=20)
V_RESET   = 0.0   # reset potential after spike
V_REST    = 0.0   # resting potential

# ── STDP parameters ──────────────────────────────────────────────────────────

STDP_A_PLUS  = 0.01   # potentiation amplitude
STDP_A_MINUS = 0.012  # depression amplitude (slightly larger → weight decay)
STDP_TAU_P   = 20.0   # potentiation time constant (timesteps)
STDP_TAU_M   = 20.0   # depression time constant
STDP_W_MAX   = 2.0    # weight ceiling
STDP_W_MIN   = 0.0    # weight floor (no negative weights in this layer)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SNNForwardResult:
    spike_hidden:  torch.Tensor   # [T, N_HIDDEN] — spike trains
    spike_output:  torch.Tensor   # [T, N_OUTPUT]
    v_hidden:      torch.Tensor   # [T, N_HIDDEN] — membrane potentials
    fingerprint:   np.ndarray     # [N_OUTPUT] — mean firing rate = identity vector
    sparsity:      float          # fraction of neurons that fired
    energy_proxy:  float          # spike count / (T * N_HIDDEN) — lower = more efficient

    def render(self) -> str:
        fired_h = int(self.spike_hidden.sum().item())
        fired_o = int(self.spike_output.sum().item())
        return (f"SNN forward: hidden={fired_h} spikes / {T_DEFAULT*N_HIDDEN} max  "
                f"output={fired_o} spikes  sparsity={self.sparsity:.3f}  "
                f"energy_proxy={self.energy_proxy:.4f}")


@dataclass
class STDPUpdate:
    dW_ih: torch.Tensor    # [N_HIDDEN, N_TRAITS] weight delta
    dW_ho: torch.Tensor    # [N_OUTPUT, N_HIDDEN]
    n_potentiated: int
    n_depressed:   int

    def render(self) -> str:
        return (f"STDP: potentiated={self.n_potentiated}  "
                f"depressed={self.n_depressed}  "
                f"||dW_ih||={self.dW_ih.norm().item():.4f}  "
                f"||dW_ho||={self.dW_ho.norm().item():.4f}")


# ── LIF Layer ─────────────────────────────────────────────────────────────────

class LIFLayer(nn.Module):
    """
    Leaky Integrate-and-Fire neuron layer.
    Stateful: membrane potential V persists across timesteps.
    """

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        # Sparse init: mean weight ~0.01 so typical input per step
        # (trait_rate × n_in × mean_w) ≈ 0.07 < V_th*(1-τ)=0.1 → ~20% sparsity
        self.W   = nn.Parameter(
            torch.empty(n_out, n_in).uniform_(0.0, 0.02), requires_grad=False
        )
        self.n_in  = n_in
        self.n_out = n_out
        self.reset_state()

    def reset_state(self):
        self.V      = torch.full((self.n_out,), V_REST, device=DEVICE)
        self.spikes: list[torch.Tensor] = []   # spike history [T × n_out]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        One timestep: integrate input, fire if threshold crossed, reset.
        x: [n_in] — binary spike input or continuous rate
        Returns: [n_out] — binary spike output
        """
        self.V  = TAU_MEM * self.V + self.W @ x
        spike   = (self.V >= V_THRESH).float()
        self.V  = torch.where(spike.bool(), torch.tensor(V_RESET, device=DEVICE), self.V)
        self.spikes.append(spike.clone())
        return spike


# ── STDP Rule ─────────────────────────────────────────────────────────────────

def compute_stdp(
    pre_spikes:  torch.Tensor,   # [T, n_pre]
    post_spikes: torch.Tensor,   # [T, n_post]
) -> tuple[torch.Tensor, int, int]:
    """
    Compute STDP weight update from pre/post spike trains.

    For each pre-post pair:
      Δw = Σ_t  A+ · Σ_{t'>t} spike_post(t') · exp(−(t'−t)/τ+)   (potentiation)
           −A− · Σ_{t'<t} spike_post(t') · exp(−(t−t')/τ−)       (depression)

    Vectorised over all pairs simultaneously — O(T² × n_pre × n_post) but
    small T (20 steps) makes this fast.

    Returns (dW [n_post, n_pre], n_potentiated, n_depressed)
    """
    T, n_pre  = pre_spikes.shape
    _,  n_post = post_spikes.shape

    dW           = torch.zeros(n_post, n_pre, device=DEVICE)
    n_pot, n_dep = 0, 0

    # Pre-compute exponential decay kernels
    times       = torch.arange(T, dtype=torch.float32, device=DEVICE)
    diff        = times.unsqueeze(0) - times.unsqueeze(1)    # [T, T] Δt matrix

    # Potentiation kernel: exp(−Δt / τ+) for Δt > 0 (post after pre)
    pot_kernel  = torch.where(diff > 0,
                              torch.exp(-diff / STDP_TAU_P),
                              torch.zeros_like(diff))

    # Depression kernel: exp(Δt / τ−) for Δt < 0 (post before pre)
    dep_kernel  = torch.where(diff < 0,
                              torch.exp(diff / STDP_TAU_M),
                              torch.zeros_like(diff))

    # Vectorised over neuron pairs
    # pot[pre_t, post_t'] · pre_spike(pre_t) · post_spike(post_t')
    # Shape: pre=[T, n_pre] post=[T, n_post] kernel=[T, T]
    pre_T   = pre_spikes.float()    # [T, n_pre]
    post_T  = post_spikes.float()   # [T, n_post]

    # Potentiation: dW_pot[j, i] = Σ_t Σ_t' pre[t,i] · pot_kernel[t,t'] · post[t',j]
    #             = (pot_kernel.T @ pre_T).T dotted with post somehow
    # Efficient: (pre_T.T @ pot_kernel @ post_T) gives [n_pre, n_post] — transpose
    pot  = (pre_T.T @ pot_kernel @ post_T).T   # [n_post, n_pre]
    dep  = (pre_T.T @ dep_kernel @ post_T).T   # [n_post, n_pre]

    dW_pot  = STDP_A_PLUS  * pot
    dW_dep  = STDP_A_MINUS * dep
    dW     += dW_pot - dW_dep

    n_pot  = int((dW_pot > 1e-6).sum().item())
    n_dep  = int((dW_dep > 1e-6).sum().item())

    return dW, n_pot, n_dep


# ── SNNPersonalityLayer ────────────────────────────────────────────────────────

class SNNPersonalityLayer:
    """
    A two-layer SNN (input → hidden → output) that encodes personality traits
    as spiking dynamics and learns via STDP.

    Replaces the EWC-protected trait vector as the identity mechanism:
    instead of a 12D float vector, the agent's identity is its unique
    synaptic weight matrix and the spike patterns it produces.

    Uniqueness: different initial weights + different STDP history → different
    spike fingerprints, even for the same trait vector input.
    """

    def __init__(self, agent_name: str, snn_dir: str = "."):
        self.name    = agent_name
        self.snn_dir = Path(snn_dir)
        self.snn_dir.mkdir(parents=True, exist_ok=True)
        self.snn_file = self.snn_dir / f"{agent_name.lower()}_snn.pt"

        self._build_network()
        self._stdp_history: list[STDPUpdate] = []
        self._forward_history: list[SNNForwardResult] = []
        self._n_updates = 0

        if self.snn_file.exists():
            self._load()

    def _build_network(self):
        seed = abs(hash(self.name)) % (2**31)
        torch.manual_seed(seed)
        self.layer_ih = LIFLayer(N_TRAITS,  N_HIDDEN).to(DEVICE)
        self.layer_ho = LIFLayer(N_HIDDEN,  N_OUTPUT).to(DEVICE)

    # ── Rate encoding ─────────────────────────────────────────────────────────

    @staticmethod
    def encode_traits(trait_vector: list[float], T: int = T_DEFAULT) -> torch.Tensor:
        """
        Rate coding: each trait value → Bernoulli spike train over T timesteps.
        Higher trait value → higher firing probability per timestep.
        Returns: [T, N_TRAITS] binary spike tensor.
        """
        rates  = torch.tensor(trait_vector, dtype=torch.float32, device=DEVICE)
        rates  = rates.clamp(0.05, 0.95)
        spikes = torch.bernoulli(rates.unsqueeze(0).expand(T, -1))
        return spikes    # [T, N_TRAITS]

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        trait_vector: list[float],
        T:            int = T_DEFAULT,
    ) -> SNNForwardResult:
        """
        Run T timesteps of SNN dynamics given the trait vector.
        Returns spike trains, membrane potentials, and the identity fingerprint.
        """
        self.layer_ih.reset_state()
        self.layer_ho.reset_state()

        input_spikes          = self.encode_traits(trait_vector, T)   # [T, N_TRAITS]
        self._last_input_spikes = input_spikes   # cache for STDP
        v_hidden_log          = []

        for t in range(T):
            s_h = self.layer_ih(input_spikes[t])   # [N_HIDDEN]
            _   = self.layer_ho(s_h)               # [N_OUTPUT]
            v_hidden_log.append(self.layer_ih.V.clone())

        spike_h = torch.stack(self.layer_ih.spikes)   # [T, N_HIDDEN]
        spike_o = torch.stack(self.layer_ho.spikes)   # [T, N_OUTPUT]
        v_h     = torch.stack(v_hidden_log)           # [T, N_HIDDEN]

        # Fingerprint = mean firing rate per output neuron
        fingerprint  = spike_o.mean(dim=0).cpu().numpy()  # [N_OUTPUT]
        sparsity     = float(spike_h.mean().item())
        energy_proxy = float(spike_h.sum().item()) / (T * N_HIDDEN)

        result = SNNForwardResult(
            spike_hidden  = spike_h,
            spike_output  = spike_o,
            v_hidden      = v_h,
            fingerprint   = fingerprint,
            sparsity      = sparsity,
            energy_proxy  = energy_proxy,
        )
        self._forward_history.append(result)
        return result

    # ── STDP update ───────────────────────────────────────────────────────────

    def stdp_update(self) -> STDPUpdate | None:
        """
        Apply STDP to both synapse layers from the most recent forward pass.
        Fully local — each synapse updates from its own pre/post spike times.
        No global error signal, no backpropagation.
        Returns STDPUpdate with weight deltas applied.
        """
        if not self._forward_history:
            return None

        result = self._forward_history[-1]

        # Layer ih: input → hidden
        input_sp = self.layer_ih.spikes   # already stored as list
        # Reconstruct input spike tensor from last forward
        # (stored in layer; we need to re-run or cache — cache approach here)
        # For simplicity: use hidden as proxy for input correlation
        spike_h  = result.spike_hidden   # [T, N_HIDDEN]
        spike_o  = result.spike_output   # [T, N_OUTPUT]

        # STDP for hidden → output synapses
        dW_ho, pot_ho, dep_ho = compute_stdp(spike_h, spike_o)
        with torch.no_grad():
            self.layer_ho.W.data += dW_ho
            self.layer_ho.W.data.clamp_(STDP_W_MIN, STDP_W_MAX)

        # STDP for input → hidden synapses
        # pre = input spike train, post = hidden spike train
        # Need the input spikes from last forward — reconstructed from rate encoding
        # We use the stored result's hidden spikes as proxy: STDP(hidden→hidden) shapes ih
        # Correct approach: cache input spikes during forward()
        if hasattr(self, '_last_input_spikes') and self._last_input_spikes is not None:
            dW_ih, pot_ih, dep_ih = compute_stdp(
                self._last_input_spikes,   # [T, N_TRAITS]  pre
                spike_h,                   # [T, N_HIDDEN]  post
            )
            # dW_ih shape: [N_HIDDEN, N_TRAITS] = self.layer_ih.W.shape ✓
            with torch.no_grad():
                self.layer_ih.W.data += dW_ih * 0.1
                self.layer_ih.W.data.clamp_(STDP_W_MIN, STDP_W_MAX)
        else:
            dW_ih = torch.zeros_like(self.layer_ih.W)
            pot_ih = dep_ih = 0

        upd = STDPUpdate(
            dW_ih         = dW_ih,
            dW_ho         = dW_ho,
            n_potentiated = pot_ho + pot_ih,
            n_depressed   = dep_ho + dep_ih,
        )
        self._stdp_history.append(upd)
        self._n_updates += 1
        self._save()
        return upd

    # ── Identity fingerprint ──────────────────────────────────────────────────

    def fingerprint(self, trait_vector: list[float] | None = None,
                    T: int = T_DEFAULT) -> np.ndarray:
        """
        Return the agent's neural identity fingerprint.
        If trait_vector provided, runs a fresh forward pass first.
        """
        if trait_vector is not None:
            result = self.forward(trait_vector, T=T)
            return result.fingerprint
        if self._forward_history:
            return self._forward_history[-1].fingerprint
        return np.zeros(N_OUTPUT)

    # ── Energy report (M4.4) ──────────────────────────────────────────────────

    def energy_report(self) -> dict:
        """
        Estimate energy efficiency proxy vs. equivalent dense activation.
        Spike count × estimated energy per spike vs. dense matmul.
        """
        if not self._forward_history:
            return {}
        last     = self._forward_history[-1]
        n_spikes = int(last.spike_hidden.sum().item() + last.spike_output.sum().item())
        n_dense  = T_DEFAULT * (N_HIDDEN + N_OUTPUT)   # equivalent dense ops
        ratio    = n_spikes / max(1, n_dense)
        return {
            "n_spikes":       n_spikes,
            "n_dense_equiv":  n_dense,
            "sparsity":       round(last.sparsity, 4),
            "energy_ratio":   round(ratio, 4),
            "m4_4_passed":    ratio < 0.5,   # less than 50% of dense ops
            "device":         str(DEVICE),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        torch.save({
            "W_ih":      self.layer_ih.W.data.cpu(),
            "W_ho":      self.layer_ho.W.data.cpu(),
            "n_updates": self._n_updates,
        }, str(self.snn_file))

    def _load(self):
        ckpt = torch.load(str(self.snn_file), map_location="cpu",
                          weights_only=True)
        self.layer_ih.W.data = ckpt["W_ih"].to(DEVICE)
        self.layer_ho.W.data = ckpt["W_ho"].to(DEVICE)
        self._n_updates       = ckpt.get("n_updates", 0)

    def __repr__(self) -> str:
        return (f"SNNPersonalityLayer({self.name!r}, "
                f"arch={N_TRAITS}→{N_HIDDEN}→{N_OUTPUT}, "
                f"device={DEVICE}, updates={self._n_updates})")


# ── Standalone demo / milestone check ─────────────────────────────────────────

if __name__ == "__main__":
    import sys, tempfile
    sys.path.insert(0, str(Path(__file__).parent))
    from sentient_agent import PERSONALITY_PRESETS

    print("\n" + "━" * 68)
    print(f"  SNN PERSONALITY LAYER — Phase 4 Demo")
    print(f"  Device: {DEVICE}  |  Architecture: {N_TRAITS}→{N_HIDDEN}→{N_OUTPUT}")
    print(f"  Neuron: LIF (τ={TAU_MEM}, V_th={V_THRESH})  |  "
          f"Learning: STDP (A+={STDP_A_PLUS}, A-={STDP_A_MINUS})")
    print("━" * 68)

    tmpdir = tempfile.mkdtemp()

    # ── M4.1: Three agents with different presets ──────────────────────────────
    print("\n  M4.1 — Three SNNs with different personality presets")
    agents = {}
    for preset in ["biologist", "physicist", "critic"]:
        traits   = list(PERSONALITY_PRESETS[preset].values())
        # Pad to 12 traits if needed
        while len(traits) < N_TRAITS:
            traits.append(0.5)
        traits = traits[:N_TRAITS]
        snn    = SNNPersonalityLayer(preset, snn_dir=tmpdir)
        result = snn.forward(traits, T=T_DEFAULT)
        agents[preset] = (snn, traits, result)
        print(f"  {preset:<12} {result.render()}")

    # ── M4.3: STDP update within single pass (no batch optimiser) ─────────────
    print("\n  M4.3 — STDP local updates (no global optimiser)")
    for preset, (snn, traits, result) in agents.items():
        upd = snn.stdp_update()
        if upd:
            print(f"  {preset:<12} {upd.render()}")

    # ── M4.2: Identity probe — fingerprint distinctiveness ─────────────────────
    print("\n  M4.2 — Fingerprint distinctiveness (cosine distance between agents)")
    presets = list(agents.keys())
    for i in range(len(presets)):
        for j in range(i + 1, len(presets)):
            fa = agents[presets[i]][2].fingerprint
            fb = agents[presets[j]][2].fingerprint
            dot = float(np.dot(fa, fb))
            na  = float(np.linalg.norm(fa))
            nb  = float(np.linalg.norm(fb))
            cos = dot / (na * nb + 1e-9)
            print(f"  {presets[i]:<12} ↔ {presets[j]:<12}  "
                  f"cosine_sim={cos:.4f}  distance={1-cos:.4f}")

    # ── M4.4: Energy efficiency proxy ─────────────────────────────────────────
    print("\n  M4.4 — Energy efficiency proxy (spike count vs dense equivalent)")
    for preset, (snn, _, _) in agents.items():
        er = snn.energy_report()
        status = "✓" if er.get("m4_4_passed") else "—"
        print(f"  {preset:<12} sparsity={er['sparsity']:.3f}  "
              f"energy_ratio={er['energy_ratio']:.3f}  {status}")

    # ── Run multiple passes to show STDP convergence ───────────────────────────
    print("\n  STDP convergence (biologist, 50 passes):")
    snn_b, traits_b, _ = agents["biologist"]
    fe_vals = []
    for i in range(50):
        r = snn_b.forward(traits_b, T=T_DEFAULT)
        u = snn_b.stdp_update()
        fe_vals.append(r.sparsity)

    # Plot sparsity convergence as ASCII
    mx = max(fe_vals) + 1e-9
    sampled = fe_vals[::5]
    for i, v in enumerate(sampled):
        bar = "█" * int(v / mx * 40)
        print(f"  Pass {i*5:>3} │ {bar:<40} sparsity={v:.4f}")

    print()
    final_sparsity = fe_vals[-1]
    print(f"  Initial sparsity: {fe_vals[0]:.4f}")
    print(f"  Final sparsity:   {final_sparsity:.4f}")
    print(f"  Converged: {'✓' if abs(fe_vals[-1] - fe_vals[-5]) < 0.02 else '—'}")
    print("━" * 68)
