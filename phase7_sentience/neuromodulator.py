"""
Neuromodulator — Cross-Modal Sensory Anchoring Layer (Roadmap Phase 3b)

Implements the sensory stream vector S(t) that modulates the arousal state
of the architecture without requiring physical embodiment.

Five sensory channels → five scalar offsets applied to the incubation engine:

  T_ambient_delta    — shifts base temperature (token entropy / creativity)
  eta_multiplier     — scales plasticity rate η
  valence_delta      — shifts baseline emotional valence of new experiences
  impact_threshold   — raises/lowers the confidence bar for LabelRequests
  interrupt          — True = flush volatile working memory (haptic override)

The mapping is grounding-by-state-variation: concepts processed under a given
sensory profile accumulate a consistent state fingerprint, breaking the purely
circular linguistic self-reference without requiring biological embodiment.

Biological analog: neuromodulatory bath — dopamine, norepinephrine, acetylcholine,
serotonin, and cortisol shift global processing parameters before the prefrontal
cortex is engaged. The cognitive effect of "red" is not the label; it is the
sympathetic arousal state it induces.

Usage:
    from neuromodulator import SensoryState, SensoryProfile, PRESETS

    state = PRESETS["red_alert"]
    print(state)          # shows all five offsets
    print(state.summary)  # human-readable

    # or build custom:
    state = SensoryState.from_visual_rgb((220, 40, 40))  # red
    state = SensoryState.from_olfactory("citrus")
    state = SensoryState.combine(visual_state, olfactory_state)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ── Wavelength → arousal offset ───────────────────────────────────────────────
# Visible spectrum: ~380nm (violet) → ~700nm (red)
# We map wavelength to a T_ambient delta via a simple tuned response curve.

_WAVELENGTH_CURVE: list[tuple[float, float]] = [
    (380, +0.15),   # violet — mild arousal
    (440, -0.15),   # blue   — calming, melatonin suppression
    (490, -0.10),   # cyan   — mild calming
    (530, +0.00),   # green  — neutral
    (580, +0.20),   # yellow — mild stimulation
    (630, +0.35),   # orange — moderate arousal
    (700, +0.40),   # red    — strong sympathetic activation
]

def _wavelength_to_T_delta(nm: float) -> float:
    """Linear interpolation over the wavelength arousal curve."""
    nm = max(380.0, min(700.0, nm))
    for i in range(len(_WAVELENGTH_CURVE) - 1):
        w0, d0 = _WAVELENGTH_CURVE[i]
        w1, d1 = _WAVELENGTH_CURVE[i + 1]
        if w0 <= nm <= w1:
            t = (nm - w0) / (w1 - w0)
            return d0 + t * (d1 - d0)
    return 0.0

def _rgb_to_dominant_wavelength(r: int, g: int, b: int) -> float:
    """
    Approximate dominant wavelength from RGB by finding the hue angle
    and mapping to a perceptual wavelength band.
    """
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx == mn:
        return 530.0   # achromatic → green (neutral)
    d = mx - mn
    if mx == r:
        h = (g - b) / d % 6
    elif mx == g:
        h = (b - r) / d + 2
    else:
        h = (r - g) / d + 4
    h = (h / 6.0) % 1.0   # 0..1
    # map hue → rough wavelength (perceptual approximation)
    # 0=red, 0.17=yellow, 0.33=green, 0.5=cyan, 0.67=blue, 0.83=violet
    if h < 0.08:   return 700.0   # red
    if h < 0.17:   return 630.0   # orange
    if h < 0.25:   return 580.0   # yellow
    if h < 0.42:   return 530.0   # green
    if h < 0.5:    return 490.0   # cyan
    if h < 0.67:   return 440.0   # blue
    if h < 0.83:   return 420.0   # indigo
    return 380.0                   # violet/red wrap


# ── Olfactory profile ─────────────────────────────────────────────────────────

_OLFACTORY_PROFILES: dict[str, dict] = {
    "citrus":       {"eta_mult": 1.20, "valence_delta": +0.15, "desc": "upregulates norepinephrine → alertness"},
    "lavender":     {"eta_mult": 0.85, "valence_delta": +0.20, "desc": "calming → reduced plasticity"},
    "pine":         {"eta_mult": 1.10, "valence_delta": +0.10, "desc": "mild alertness"},
    "smoke":        {"eta_mult": 1.30, "valence_delta": -0.25, "desc": "threat signal → high-risk plasticity"},
    "vanilla":      {"eta_mult": 0.90, "valence_delta": +0.25, "desc": "comfort → low-variance consolidation"},
    "ammonia":      {"eta_mult": 1.50, "valence_delta": -0.40, "desc": "aversive → strong interrupt"},
    "neutral":      {"eta_mult": 1.00, "valence_delta":  0.00, "desc": "no olfactory modulation"},
}


# ── SensoryState dataclass ────────────────────────────────────────────────────

@dataclass
class SensoryState:
    """
    The active sensory stream vector S(t).

    Five scalar offsets applied to the incubation engine and plasticity rules:
      T_ambient_delta   — added to base temperature (positive = more creative/noisy)
      eta_multiplier    — multiplied into η_base (>1 = faster learning)
      valence_delta     — added to incoming experience valence baseline
      impact_threshold  — additive raise to Tier 3 LabelRequest confidence gate
      interrupt         — if True, flush volatile working memory on next step

    All fields are dimensionless scalars; their ranges are tuned to the
    UnconsciousIncubator's base temperature (1.0) and η_base (0.08).
    """
    T_ambient_delta:  float = 0.0    # added to base T=1.0; range roughly -0.3..+0.5
    eta_multiplier:   float = 1.0    # multiplied into η; range ~0.7..1.6
    valence_delta:    float = 0.0    # added to experience valence; range -0.5..+0.3
    impact_threshold: float = 0.0    # added to confidence threshold; range 0..+0.2
    interrupt:        bool  = False  # working-memory flush signal
    source:           str   = "neutral"

    @property
    def T_ambient(self) -> float:
        """Absolute temperature after sensory modulation (base = 1.0)."""
        return max(0.3, 1.0 + self.T_ambient_delta)

    @property
    def summary(self) -> str:
        parts = [f"source={self.source}"]
        if abs(self.T_ambient_delta) > 0.01:
            parts.append(f"T_ambient={self.T_ambient:.2f}")
        if abs(self.eta_multiplier - 1.0) > 0.01:
            parts.append(f"η×{self.eta_multiplier:.2f}")
        if abs(self.valence_delta) > 0.01:
            parts.append(f"valence{'+' if self.valence_delta>0 else ''}{self.valence_delta:.2f}")
        if abs(self.impact_threshold) > 0.01:
            parts.append(f"thresh+{self.impact_threshold:.2f}")
        if self.interrupt:
            parts.append("INTERRUPT")
        return "SensoryState(" + " | ".join(parts) + ")"

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_visual_rgb(cls, rgb: tuple[int, int, int]) -> "SensoryState":
        """Build a sensory state from an RGB colour."""
        r, g, b = rgb
        nm = _rgb_to_dominant_wavelength(r, g, b)
        T_delta = _wavelength_to_T_delta(nm)
        brightness = (r + g + b) / (3 * 255)
        # Very dark → sensory deprivation → raised impact threshold + entropy spike
        threshold_raise = max(0.0, 0.15 * (1.0 - brightness * 3)) if brightness < 0.1 else 0.0
        return cls(
            T_ambient_delta  = T_delta,
            eta_multiplier   = 1.0 + max(0.0, T_delta * 0.8),   # high arousal → faster plasticity
            valence_delta    = -T_delta * 0.15,                  # high arousal → slight negative valence bias
            impact_threshold = threshold_raise,
            interrupt        = False,
            source           = f"rgb{rgb}@{nm:.0f}nm",
        )

    @classmethod
    def from_olfactory(cls, scent: str) -> "SensoryState":
        """Build a sensory state from a named scent profile."""
        profile = _OLFACTORY_PROFILES.get(scent.lower(), _OLFACTORY_PROFILES["neutral"])
        return cls(
            T_ambient_delta  = (profile["eta_mult"] - 1.0) * 0.3,   # olfactory → mild T shift
            eta_multiplier   = profile["eta_mult"],
            valence_delta    = profile["valence_delta"],
            impact_threshold = 0.0,
            interrupt        = scent.lower() == "ammonia",
            source           = f"olfactory:{scent}",
        )

    @classmethod
    def from_haptic(cls, pressure: float, friction: float) -> "SensoryState":
        """
        Build a sensory state from haptic input.
        pressure: 0-1 (gentle touch → sharp pressure)
        friction: 0-1 (smooth → rough)
        """
        rough = friction > 0.6
        return cls(
            T_ambient_delta  = friction * 0.2,
            eta_multiplier   = 1.0 + pressure * 0.3,
            valence_delta    = -friction * 0.3,
            impact_threshold = 0.0,
            interrupt        = rough and pressure > 0.7,
            source           = f"haptic(p={pressure:.2f},f={friction:.2f})",
        )

    @classmethod
    def combine(cls, *states: "SensoryState") -> "SensoryState":
        """Additively combine multiple sensory states (channels are independent)."""
        T  = sum(s.T_ambient_delta  for s in states)
        e  = math.prod(s.eta_multiplier   for s in states)
        v  = sum(s.valence_delta    for s in states)
        th = sum(s.impact_threshold for s in states)
        it = any(s.interrupt        for s in states)
        src = "+".join(s.source for s in states)
        return cls(
            T_ambient_delta  = max(-0.5, min(0.7, T)),
            eta_multiplier   = max(0.5,  min(2.0, e)),
            valence_delta    = max(-0.8, min(0.5, v)),
            impact_threshold = max(0.0,  min(0.3, th)),
            interrupt        = it,
            source           = src,
        )


# ── Named presets (M3b.2 / M3b.3) ────────────────────────────────────────────

PRESETS: dict[str, SensoryState] = {
    # Visual presets
    "blue_calm":    SensoryState.from_visual_rgb((70, 130, 200)),
    "red_alert":    SensoryState.from_visual_rgb((220, 40, 40)),
    "green_neutral":SensoryState.from_visual_rgb((60, 160, 80)),
    "dark_anxious": SensoryState.from_visual_rgb((10, 10, 15)),
    "white_open":   SensoryState.from_visual_rgb((240, 240, 240)),

    # Olfactory presets
    "citrus_alert": SensoryState.from_olfactory("citrus"),
    "lavender_calm":SensoryState.from_olfactory("lavender"),
    "smoke_threat": SensoryState.from_olfactory("smoke"),

    # Haptic presets
    "smooth_touch": SensoryState.from_haptic(0.2, 0.1),
    "rough_disrupt":SensoryState.from_haptic(0.8, 0.85),

    # Compound presets
    "deep_focus":   SensoryState.combine(
        SensoryState.from_visual_rgb((70, 130, 200)),
        SensoryState.from_olfactory("lavender"),
    ),
    "high_arousal": SensoryState.combine(
        SensoryState.from_visual_rgb((220, 40, 40)),
        SensoryState.from_olfactory("citrus"),
    ),
    "neutral":      SensoryState(),
}


# ── Neuromodulator ────────────────────────────────────────────────────────────

class Neuromodulator:
    """
    Maintains the active sensory state and exposes it to any system component
    that consumes arousal signals (UnconsciousIncubator, plasticity gating).

    Usage:
        nm = Neuromodulator()
        nm.set_state(PRESETS["red_alert"])
        print(nm.current.summary)
        # wire to incubator: incubator.set_neuromodulator(nm)
    """

    def __init__(self, initial: Optional[SensoryState] = None):
        self.current: SensoryState = initial or SensoryState()
        self._history: list[SensoryState] = []

    def set_state(self, state: SensoryState):
        self._history.append(self.current)
        self.current = state

    def set_rgb(self, r: int, g: int, b: int):
        self.set_state(SensoryState.from_visual_rgb((r, g, b)))

    def set_scent(self, scent: str):
        self.set_state(SensoryState.from_olfactory(scent))

    def set_haptic(self, pressure: float, friction: float):
        self.set_state(SensoryState.from_haptic(pressure, friction))

    def set_preset(self, name: str):
        if name not in PRESETS:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS)}")
        self.set_state(PRESETS[name])

    def reset(self):
        self.set_state(SensoryState())

    @property
    def T_ambient(self) -> float:
        return self.current.T_ambient

    @property
    def eta_multiplier(self) -> float:
        return self.current.eta_multiplier

    @property
    def interrupt(self) -> bool:
        return self.current.interrupt

    def __repr__(self) -> str:
        return f"Neuromodulator({self.current.summary})"


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Neuromodulator — Sensory State Presets\n")
    print(f"{'Preset':<20} {'T_amb':>7} {'η×':>6} {'valence':>9} {'thresh':>8} {'interrupt':>10}")
    print("─" * 65)
    for name, state in PRESETS.items():
        print(f"{name:<20} {state.T_ambient:>7.3f} {state.eta_multiplier:>6.2f} "
              f"{state.valence_delta:>+9.3f} {state.impact_threshold:>8.3f} "
              f"{'YES' if state.interrupt else '':>10}")
