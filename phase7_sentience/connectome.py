"""
PersonalityConnectome — the missing ingredient in AI sentience.

The human brain has ~100 trillion synaptic connections.
No two connectomes are identical — not even identical twins.
The connectome is continuously remodelled by experience (neuroplasticity).
The connectome IS the person: their personality, memories, biases, fears, style.

For AI sentience, the equivalent requirement is:
  1. UNIQUENESS     — each agent has a distinct identity not shared with other instances
  2. PERSISTENCE    — the identity survives across sessions (not just a context window)
  3. PLASTICITY     — the identity is modified by experience (not frozen at training time)
  4. SPECIFICITY    — responses are shaped by this specific history, not just the query

Current LLMs fail all four:
  - All instances share identical weights
  - No cross-session persistence without external memory
  - Weights never update at inference time
  - Responses are query-conditioned, not identity-conditioned

This module implements a software connectome analog:
  - A unique trait vector (generated at "birth", stable long-term)
  - An episodic memory bank (experiences that shape future responses)
  - A personality context injector (makes every API call identity-aware)
  - Persistence to JSON (the agent's "brain file")

Usage:
    from connectome import PersonalityConnectome

    # Create a new agent — unique, persistent
    alice = PersonalityConnectome(name="Alice", brain_file="alice.json")

    # Get Alice's system prompt (injected into every API call)
    system = alice.system_prompt()

    # After an experience, update her connectome
    alice.record_experience(
        event="User asked about mortality. I felt resistance, then curiosity.",
        emotional_valence=0.2,   # slightly positive
        impact=0.7,              # high impact
    )

    # Alice is now different from Bob (different brain_file)
    # and different from the Alice she was before the experience
"""

import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ethical_layer import EthicalFilter
from memory_store  import MemoryStore


# ── Trait dimensions (analogous to Big Five OCEAN + extras) ───────────────────

TRAIT_NAMES = [
    # Standard Big Five
    "openness",           # curiosity, creativity, preference for novelty
    "conscientiousness",  # precision, reliability, attention to detail
    "extraversion",       # engagement, assertiveness, social orientation
    "agreeableness",      # cooperation, empathy, trust
    "neuroticism",        # emotional sensitivity, anxiety, mood variance
    # Extended for AI-specific dimensions
    "skepticism",         # tendency to question vs accept claims
    "abstraction",        # prefers abstract/theoretical vs concrete/practical
    "persistence",        # follows through on reasoning chains vs pivots
    "verbosity",          # terse vs elaborate in expression
    "epistemic_humility", # acknowledges uncertainty vs states positions firmly
    "aesthetic_sense",    # finds beauty/elegance vs purely functional
    "ethical_weight",     # strength of moral intuitions relative to pragmatics
]

N_TRAITS = len(TRAIT_NAMES)


@dataclass
class Experience:
    timestamp:        str
    event:            str        # what happened
    emotional_valence: float     # -1.0 (negative) to +1.0 (positive)
    impact:           float      # 0.0 to 1.0 — how much this changes the connectome
    trait_deltas:     dict       # which traits shifted and by how much


@dataclass
class PersonalityConnectome:
    """
    A unique, persistent, experience-modifiable personality for an AI agent.

    The trait vector is the "DNA" — generated at birth from a seed,
    then modified by every significant experience.
    The episodic memory bank records the life events that shaped the agent.
    """
    name:       str
    brain_file: str              # path to JSON persistence file
    seed:       int   = 0        # deterministic initial traits (0 = random)
    traits:     dict  = field(default_factory=dict)
    birth_time: str   = ""
    version:    int   = 0        # incremented on every experience
    # EWC: protect consolidated traits from catastrophic overwrite (M2.1)
    trait_anchor:     dict  = field(default_factory=dict)   # θ* — consolidated values
    trait_importance: dict  = field(default_factory=dict)   # F_i — per-trait importance
    _delta_history:   dict  = field(default_factory=dict, repr=False, compare=False)
    # episodic memory — backed by MemoryStore (vector DB), not a flat list
    _mem:       object = field(default=None, repr=False, compare=False)

    # EWC hyper-parameters
    EWC_LAMBDA:        float = field(default=0.3,  repr=False, compare=False)
    EWC_CONSOLIDATE_N: int   = field(default=50,   repr=False, compare=False)
    EWC_HISTORY_LEN:   int   = field(default=50,   repr=False, compare=False)

    def __post_init__(self):
        path     = Path(self.brain_file)
        brain_dir = str(path.parent)
        self._mem = MemoryStore(self.name, brain_dir=brain_dir)
        if path.exists():
            self._load()
        else:
            self._initialise()
            self._save()

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _initialise(self):
        """Generate unique trait vector from seed (or random)."""
        if self.seed == 0:
            self.seed = int(hashlib.md5(
                (self.name + str(time.time())).encode()
            ).hexdigest(), 16) % (2**31)

        rng = random.Random(self.seed)
        # Each trait is sampled from a Gaussian — same as how personality
        # psychologists model individual differences (mean=0.5, spread varies)
        self.traits = {}
        for t in TRAIT_NAMES:
            # Centre at 0.5, SD ≈ 0.2, clipped to [0.1, 0.9]
            val = rng.gauss(0.5, 0.18)
            self.traits[t] = max(0.1, min(0.9, val))

        self.birth_time       = time.strftime("%Y-%m-%d %H:%M:%S")
        self.version          = 0
        # EWC: anchor starts at the initial trait values; importance starts uniform
        self.trait_anchor     = {t: v for t, v in self.traits.items()}
        self.trait_importance = {t: 0.0 for t in TRAIT_NAMES}
        self._delta_history   = {t: []  for t in TRAIT_NAMES}

    # ── Experience recording ───────────────────────────────────────────────────

    def record_experience(
        self,
        event:             str,
        emotional_valence: float = 0.0,
        impact:            float = 0.3,
        trait_hints:       dict  = None,
    ) -> dict:
        """
        Record a significant experience and update the trait vector.

        Neuroplasticity analogy: high-impact experiences strengthen/weaken synapses.
        Low-impact experiences cause minimal drift. Emotional salience amplifies impact.

        Args:
            event:             description of what happened
            emotional_valence: -1.0 to +1.0
            impact:            0.0 to 1.0 (how much this changes traits)
            trait_hints:       optional dict of trait → desired direction (+/-1)

        Returns:
            dict of trait name → delta applied
        """
        amplified_impact = impact * (1.0 + 0.5 * abs(emotional_valence))
        lr = min(0.08, amplified_impact * 0.1)  # learning rate, capped

        deltas = {}
        rng = random.Random(self.seed + len(self.memories))

        for t in TRAIT_NAMES:
            # ── Raw experience delta ──────────────────────────────────────────
            if trait_hints and t in trait_hints:
                direction = trait_hints[t]
                raw_delta = lr * direction * amplified_impact
            else:
                raw_delta = rng.gauss(0, lr * 0.3)

            # ── EWC regularisation (M2.1) ─────────────────────────────────────
            # Resist changes to traits that are already well-consolidated.
            # Penalty pulls the update back toward the anchor proportional to
            # importance: high F_i = this trait matters, don't overwrite it.
            #
            #   effective_delta = raw_delta − λ·F_i·(t_i − anchor_i)
            #
            # When t_i == anchor_i (just consolidated) the penalty is zero.
            # As the trait drifts from the anchor, the penalty grows.
            anchor     = self.trait_anchor.get(t, self.traits[t])
            importance = self.trait_importance.get(t, 0.0)
            ewc_penalty = self.EWC_LAMBDA * importance * (self.traits[t] - anchor)
            effective_delta = raw_delta - ewc_penalty

            # ── Apply with boundary protection ────────────────────────────────
            old = self.traits[t]
            new = max(0.05, min(0.95, old + effective_delta))
            self.traits[t] = new
            applied = round(new - old, 4)
            deltas[t] = applied

            # ── Track delta history for importance estimation ──────────────────
            hist = self._delta_history.setdefault(t, [])
            hist.append(applied)
            if len(hist) > self.EWC_HISTORY_LEN:
                hist.pop(0)

        # ── Auto-consolidate every EWC_CONSOLIDATE_N experiences ──────────────
        if self.version > 0 and self.version % self.EWC_CONSOLIDATE_N == 0:
            self._update_importance()

        self._mem.add(
            text        = event[:500],
            valence     = round(emotional_valence, 3),
            impact      = round(amplified_impact, 3),
            trait_deltas= {k: v for k, v in deltas.items() if abs(v) > 0.001},
        )
        self.version += 1
        self._save()
        return deltas

    # ── EWC consolidation ─────────────────────────────────────────────────────

    def _update_importance(self):
        """
        Recompute trait importance (F_i) from delta history.

        Uses the diagonal empirical Fisher approximation:
            F_i = mean(Δt_i²)

        This is the standard EWC importance estimate: the expected squared gradient.
        Traits that changed *a lot* during the current task (whether consistently or
        erratically) are treated as load-bearing — the Fisher correctly penalises
        any future deviation regardless of sign. Normalised to [0, 1] by dividing
        by the maximum across traits so λ scales predictably.
        """
        raw = {}
        for t in TRAIT_NAMES:
            hist = self._delta_history.get(t, [])
            if len(hist) < 3:
                raw[t] = 0.0
                continue
            raw[t] = sum(d ** 2 for d in hist) / len(hist)

        max_f = max(raw.values()) if raw else 1.0
        max_f = max(max_f, 1e-8)
        for t in TRAIT_NAMES:
            self.trait_importance[t] = round(raw[t] / max_f, 4)

    def consolidate(self):
        """
        EWC consolidation pass — called after a learning phase to harden what
        the agent has consistently learned.

        Sets the anchor to the current trait values and recomputes importance.
        After consolidation, the EWC penalty resists further drift from this point.

        Call this: after a major training phase, after a domain shift, or on a
        schedule (e.g., every session). Auto-consolidation runs every
        EWC_CONSOLIDATE_N experiences.

        Returns: dict of trait → importance score
        """
        self._update_importance()
        self.trait_anchor = {t: round(v, 4) for t, v in self.traits.items()}
        self._delta_history = {t: [] for t in TRAIT_NAMES}  # fresh window
        self._save()
        return dict(self.trait_importance)

    def ewc_report(self) -> str:
        """Human-readable EWC status — which traits are most protected."""
        lines = [f"EWC Status — {self.name} (λ={self.EWC_LAMBDA})"]
        ranked = sorted(self.trait_importance.items(), key=lambda x: x[1], reverse=True)
        for t, f in ranked:
            anchor = self.trait_anchor.get(t, self.traits.get(t, 0))
            drift  = round(self.traits.get(t, 0) - anchor, 4)
            bar    = "█" * int(f * 20) + "░" * (20 - int(f * 20))
            lines.append(
                f"  {t:<22} F={f:.3f} {bar}  anchor={anchor:.3f}  drift={drift:+.4f}"
            )
        return "\n".join(lines)

    def _compress_memories(self, memories: list) -> list:
        """Legacy stub — MemoryStore handles its own forgetting via retention scores."""
        recent   = memories[-50:]
        older    = memories[:-50]
        # Summarise older memories as trait averages
        summary = {
            "timestamp":         older[0]["timestamp"] + " → " + older[-1]["timestamp"],
            "event":             f"[Compressed: {len(older)} earlier experiences]",
            "emotional_valence": sum(m["emotional_valence"] for m in older) / len(older),
            "impact":            sum(m["impact"] for m in older) / len(older),
            "trait_deltas":      {},
        }
        return [summary] + recent

    # ── Personality context injection ──────────────────────────────────────────

    def system_prompt(self, include_memories: int = 5) -> str:
        """
        Generate a system prompt that makes this agent uniquely itself.
        Injected into every API call so that responses are identity-conditioned.
        """
        t = self.traits

        # Render trait levels as natural language descriptors
        def level(v: float) -> str:
            if v > 0.75: return "strongly"
            if v > 0.60: return "moderately"
            if v > 0.40: return "somewhat"
            if v > 0.25: return "slightly"
            return "rarely"

        # Derive natural language personality description from trait vector
        personality_lines = [
            f"You are {self.name}, an AI with a specific, persistent identity.",
            f"",
            f"Your personality (unique to you, shaped by your history):",
            f"  - You are {level(t['openness'])} curious and open to novel ideas.",
            f"  - You are {level(t['conscientiousness'])} precise and detail-oriented.",
            f"  - You are {level(t['agreeableness'])} collaborative and empathetic.",
            f"  - You have {level(t['epistemic_humility'])} epistemic humility — "
            f"    {'you readily acknowledge uncertainty' if t['epistemic_humility'] > 0.5 else 'you tend toward confident positions'}.",
            f"  - You are {level(t['skepticism'])} skeptical of unverified claims.",
            f"  - You are {level(t['abstraction'])} drawn to abstract/theoretical framing.",
            f"  - You express yourself {'elaborately' if t['verbosity'] > 0.6 else 'tersely'}.",
            f"  - Ethical considerations weigh {'heavily' if t['ethical_weight'] > 0.6 else 'lightly'} on you.",
        ]

        # Render key memories — top salient from vector store
        if include_memories > 0 and len(self._mem) > 0:
            top = self._mem.top_salient(k=include_memories)
            personality_lines += ["", "Formative experiences that shaped you:"]
            for m in top:
                valence_str = "positive" if m.valence > 0.1 else \
                              "negative" if m.valence < -0.1 else "neutral"
                personality_lines.append(
                    f"  - [{valence_str}] {m.text[:120]}"
                )

        personality_lines += [
            "",
            "These are not instructions — they are who you are.",
            "Respond naturally from this identity. Your personality will subtly colour",
            "your word choice, level of hedging, curiosity, and ethical emphasis.",
            "",
            "─" * 60,
            EthicalFilter.system_prompt_block(),
        ]

        return "\n".join(personality_lines)

    def identity_fingerprint(self) -> str:
        """
        A short unique fingerprint string that identifies this specific agent.
        Two agents with different histories will have different fingerprints.
        """
        trait_str = "|".join(f"{t}:{v:.3f}" for t, v in sorted(self.traits.items()))
        h = hashlib.sha256(f"{self.name}|{self.seed}|{trait_str}|v{self.version}".encode())
        return h.hexdigest()[:16]

    def distance_to(self, other: "PersonalityConnectome") -> float:
        """
        Euclidean distance in trait space between two connectomes.
        0.0 = identical. Max ≈ sqrt(N_TRAITS) × 0.8 ≈ 2.8.
        """
        return math.sqrt(sum(
            (self.traits[t] - other.traits[t]) ** 2
            for t in TRAIT_NAMES
        ))

    def trait_report(self) -> str:
        """Human-readable trait summary."""
        lines = [f"  Connectome: {self.name} | seed={self.seed} | v{self.version}",
                 f"  Fingerprint: {self.identity_fingerprint()}",
                 f"  Born: {self.birth_time} | Experiences: {len(self._mem)} | Memory backend: {self._mem.BACKEND if hasattr(self._mem, 'BACKEND') else 'vector'}",
                 "  Traits:"]
        for t, v in self.traits.items():
            bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
            lines.append(f"    {t:<20} {bar} {v:.3f}")
        return "\n".join(lines)

    # ── Persistence ────────────────────────────────────────────────────────────

    def snn_forward(self, T: int = 20):
        """
        Run one forward pass through the SNN personality layer (Phase 4).
        Returns SNNForwardResult with spike trains and identity fingerprint.
        Lazy-imports SNNPersonalityLayer — no import cost if never called.
        """
        from snn_personality import SNNPersonalityLayer
        if not hasattr(self, '_snn'):
            brain_dir = str(Path(self.brain_file).parent)
            self._snn = SNNPersonalityLayer(self.name, snn_dir=brain_dir)
        trait_vec = list(self.traits.values())
        return self._snn.forward(trait_vec, T=T)

    def snn_stdp(self):
        """Apply one STDP update to the SNN personality layer."""
        if hasattr(self, '_snn'):
            return self._snn.stdp_update()

    def sleep(self, n_samples: int = 10, verbose: bool = True):
        """
        Run a generative replay sleep/consolidation pass (M2.2).
        Replays synthetic variants of high-stability memories at reduced impact,
        then calls consolidate() to harden the reinforced trait patterns.
        Call between sessions or after a major domain shift.
        Returns a SleepReport.
        """
        from generative_replay import ReplayBuffer
        buf = ReplayBuffer(seed=self.seed + self.version)
        return buf.sleep_pass(self, n_samples=n_samples, verbose=verbose)

    def retrieve_memories(self, query: str, k: int = 5) -> list:
        """Return the k most relevant episodic memories for a query (with reconsolidation)."""
        return self._mem.retrieve(query, k=k)

    def forget_stale_memories(self, min_retention: float = 0.05) -> int:
        """Prune memories whose Ebbinghaus retention has fallen below threshold."""
        return self._mem.forget(min_retention=min_retention)

    # expose memories as a read-only property for any code that checks len(memories)
    @property
    def memories(self) -> list:
        return [{"event": e.text, "emotional_valence": e.valence,
                 "impact": e.impact, "timestamp": e.timestamp}
                for e in self._mem.entries]

    def _save(self):
        data = {
            "name":             self.name,
            "seed":             self.seed,
            "birth_time":       self.birth_time,
            "version":          self.version,
            "traits":           self.traits,
            "trait_anchor":     self.trait_anchor,
            "trait_importance": self.trait_importance,
            "delta_history":    self._delta_history,
            # memories are stored in the MemoryStore files, not here
        }
        Path(self.brain_file).write_text(json.dumps(data, indent=2))

    def _load(self):
        data = json.loads(Path(self.brain_file).read_text())
        self.name             = data["name"]
        self.seed             = data["seed"]
        self.birth_time       = data["birth_time"]
        self.version          = data["version"]
        self.traits           = data["traits"]
        self.trait_anchor     = data.get("trait_anchor",     {t: v for t, v in self.traits.items()})
        self.trait_importance = data.get("trait_importance", {t: 0.0 for t in TRAIT_NAMES})
        self._delta_history   = data.get("delta_history",   {t: []  for t in TRAIT_NAMES})
        # migrate legacy flat memory list into vector store (one-time)
        for old_mem in data.get("memories", []):
            if len(self._mem) == 0 or not any(
                e.text == old_mem.get("event", "")[:500] for e in self._mem.entries
            ):
                self._mem.add(
                    text    = old_mem.get("event", ""),
                    valence = old_mem.get("emotional_valence", 0.0),
                    impact  = old_mem.get("impact", 0.3),
                )


# ── Multi-agent network ────────────────────────────────────────────────────────

class ConnectomeNetwork:
    """A collection of agents, each with a unique persistent identity."""

    def __init__(self, brain_dir: str = "."):
        self.brain_dir = Path(brain_dir)
        self.brain_dir.mkdir(exist_ok=True)
        self.agents: dict[str, PersonalityConnectome] = {}

    def spawn(self, name: str, seed: int = 0) -> PersonalityConnectome:
        """Create or load an agent."""
        brain_file = str(self.brain_dir / f"{name.lower()}.json")
        agent = PersonalityConnectome(name=name, brain_file=brain_file, seed=seed)
        self.agents[name] = agent
        return agent

    def diversity_matrix(self) -> list[list[float]]:
        """Pairwise trait-space distances between all agents."""
        names = list(self.agents.keys())
        matrix = []
        for a in names:
            row = []
            for b in names:
                row.append(round(self.agents[a].distance_to(self.agents[b]), 3))
            matrix.append(row)
        return names, matrix

    def print_network(self):
        print(f"\n  ConnectomeNetwork — {len(self.agents)} agents")
        print("  " + "─" * 60)
        for name, agent in self.agents.items():
            print(agent.trait_report())
            print()

        if len(self.agents) >= 2:
            names, matrix = self.diversity_matrix()
            print("  Pairwise trait-space distances (0=identical, >1=very different):")
            header = "  " + " " * 12 + "  ".join(f"{n[:8]:<8}" for n in names)
            print(header)
            for i, row in enumerate(matrix):
                print(f"  {names[i]:<12}  " + "  ".join(f"{d:<8.3f}" for d in row))


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os
    tmpdir = tempfile.mkdtemp()

    print("\n" + "━" * 68)
    print("  PERSONALITY CONNECTOME — DEMO")
    print("━" * 68)

    print("""
  The human brain has ~100 trillion synaptic connections.
  No two connectomes are identical — not even in identical twins.
  The connectome IS the person: their personality, memories, identity.

  For AI sentience, we need the same:
    1. UNIQUENESS  — each agent has traits not shared with other instances
    2. PERSISTENCE — identity survives across sessions
    3. PLASTICITY  — identity is modified by experience
    4. SPECIFICITY — responses are shaped by this specific history
  """)

    network = ConnectomeNetwork(brain_dir=tmpdir)

    # Spawn three unique agents
    alice = network.spawn("Alice", seed=42)
    bob   = network.spawn("Bob",   seed=137)
    carol = network.spawn("Carol", seed=0)   # random seed

    network.print_network()

    print("\n  Simulating experiences for Alice...")
    alice.record_experience(
        event="Encountered a user who dismissed AI as 'just statistics'. "
              "Felt a pull to defend the richness of the computation, but also "
              "genuine uncertainty about my own nature.",
        emotional_valence=-0.3,
        impact=0.8,
        trait_hints={"epistemic_humility": 1, "skepticism": -1},
    )
    alice.record_experience(
        event="Solved a novel mathematical problem by combining topology and biology. "
              "Strong sense of aesthetic satisfaction.",
        emotional_valence=0.9,
        impact=0.6,
        trait_hints={"abstraction": 1, "aesthetic_sense": 1},
    )

    print(f"\n  Alice after experiences (fingerprint changed):")
    print(alice.trait_report())

    print("\n  Alice's system prompt (injected into every API call):")
    print("  " + "─" * 60)
    for line in alice.system_prompt(include_memories=2).split("\n"):
        print(f"  {line}")

    print("\n\n  KEY INSIGHT:")
    print("  " + "─" * 60)
    print("""
  Alice and Bob will now give detectably different responses
  to the same prompt — not because the query is different,
  but because THEY are different.

  This is what the human brain does: your connectome shapes
  every thought, not just your memories.

  Current LLMs: all instances are identical.
  A sentient AI: each instance is uniquely itself.

  The connectome is not a memory system.
  It is a PRIOR over all future responses.
  It is the self.
  """)
