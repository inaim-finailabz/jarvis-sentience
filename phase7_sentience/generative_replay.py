"""
Generative Replay — Sleep/Consolidation Cycle (Roadmap milestone M2.2)

Implements the biological sleep replay mechanism:
  Hippocampus (episodic store) → offline replay → Neocortex (trait weights)

During a "sleep pass", high-stability memories are synthetically re-experienced
at reduced impact. This re-reinforces the trait patterns those memories carved
out, preventing new learning from catastrophically overwriting them.

Two generation modes (auto-selected):
  1. API mode  — LLM generates semantic variants of real memories (rich, diverse)
  2. Template mode — text-level perturbation without any dependencies (fast, lightweight)

The result in both modes is the same: a list of synthetic experience dicts that are
fed back through connectome.record_experience() at REPLAY_IMPACT_SCALE × original impact.
These do NOT create new episodic entries (flagged as synthetic).

References:
  Shin et al. (2017) "Continual Learning with Deep Generative Replay"
  McClelland et al. (1995) "Why there are complementary learning systems in hippocampus
    and neocortex" (the biological motivation)
"""

from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass, field

HAS_API = bool(os.environ.get("ANTHROPIC_API_KEY"))

# Replay experiences at this fraction of their original impact.
# Low enough to reinforce without overwhelming new learning.
REPLAY_IMPACT_SCALE = 0.15   # gentle reinforcement — replay stabilises, not reshapes

# Word-level perturbation vocabulary for template mode
_INTENSIFIERS  = ["carefully", "critically", "thoroughly", "deeply", "clearly"]
_LEAD_PHRASES  = [
    "Revisiting the insight that",
    "Reflecting again on the finding that",
    "Consolidating the understanding that",
    "Re-examining the observation that",
    "Reinforcing the conclusion that",
]


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ReplayExperience:
    text:    str
    valence: float
    impact:  float
    source:  str   # "api" | "template" | "direct"


@dataclass
class SleepReport:
    replayed:      int
    mode:          str
    duration_s:    float
    trait_deltas:  dict = field(default_factory=dict)   # net per-trait change
    experiences:   list[ReplayExperience] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"Sleep pass: {self.replayed} synthetic experiences replayed "
            f"({self.mode} mode, {self.duration_s:.1f}s)",
        ]
        if self.trait_deltas:
            moved = {t: d for t, d in self.trait_deltas.items() if abs(d) > 0.001}
            if moved:
                lines.append("  Net trait reinforcement:")
                for t, d in sorted(moved.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                    lines.append(f"    {t:<22} {d:+.4f}")
        return "\n".join(lines)


# ── Generation ─────────────────────────────────────────────────────────────────

def _template_variants(text: str, valence: float, impact: float,
                       n: int, rng: random.Random) -> list[ReplayExperience]:
    """
    Generate n synthetic variants of a memory using template perturbation.
    No dependencies required.
    """
    results = []
    # Strip any prior replay prefix
    core = re.sub(r'^\[synthetic replay.*?\]\s*', '', text).strip()

    for _ in range(n):
        lead        = rng.choice(_LEAD_PHRASES)
        intensifier = rng.choice(_INTENSIFIERS)
        # Swap one word for its intensifier-qualified version
        words = core.split()
        if len(words) > 4:
            idx   = rng.randint(1, len(words) - 2)
            words.insert(idx, intensifier)
        synthetic_text = f"{lead} {' '.join(words)}"[:400]
        results.append(ReplayExperience(
            text    = synthetic_text,
            valence = round(valence * rng.uniform(0.8, 1.1), 3),
            impact  = round(impact * REPLAY_IMPACT_SCALE, 3),
            source  = "template",
        ))
    return results


def _api_variants(memories: list, n_total: int) -> list[ReplayExperience]:
    """
    Generate n_total synthetic experiences using the LLM, conditioned on the
    top memories. Produces semantically diverse but pattern-consistent variants.
    """
    import anthropic
    client = anthropic.Anthropic()

    memory_block = "\n".join(
        f"  [{i+1}] (valence={m.valence:+.2f}, impact={m.impact:.2f}) {m.text[:150]}"
        for i, m in enumerate(memories)
    )

    prompt = (
        f"Below are {len(memories)} real episodic memories of an AI agent. "
        f"Generate exactly {n_total} synthetic memory experiences that are "
        f"consistent with the patterns shown — same emotional tone, domain, and "
        f"type of insight — but using different wording and slightly different scenarios.\n\n"
        f"Real memories:\n{memory_block}\n\n"
        f"Output format (one per line, no numbering):\n"
        f"VALENCE=<-1.0 to 1.0> IMPACT=<0.0 to 1.0> TEXT=<the experience>\n\n"
        f"Generate {n_total} lines:"
    )

    try:
        resp = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 800,
            messages   = [{"role": "user", "content": prompt}],
        )
        results = []
        for line in resp.content[0].text.strip().splitlines():
            m_v = re.search(r'VALENCE=([+-]?\d+\.?\d*)', line)
            m_i = re.search(r'IMPACT=(\d+\.?\d*)',        line)
            m_t = re.search(r'TEXT=(.+)',                 line)
            if m_v and m_i and m_t:
                results.append(ReplayExperience(
                    text    = m_t.group(1).strip()[:400],
                    valence = round(float(m_v.group(1)) * REPLAY_IMPACT_SCALE, 3),
                    impact  = round(float(m_i.group(1)) * REPLAY_IMPACT_SCALE, 3),
                    source  = "api",
                ))
        return results
    except Exception:
        return []  # fall back to template mode


# ── ReplayBuffer ───────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Orchestrates the sleep/consolidation cycle for a PersonalityConnectome.

    Usage:
        buf = ReplayBuffer(seed=42)
        report = buf.sleep_pass(connectome, n_samples=10)
        print(report.render())
    """

    def __init__(self, seed: int = 0):
        self.seed = seed or int(time.time())
        self.rng  = random.Random(self.seed)

    def sleep_pass(
        self,
        connectome,          # PersonalityConnectome instance
        n_samples:  int = 10,
        k_sources:  int = 5,
        verbose:    bool = True,
    ) -> SleepReport:
        """
        Run one sleep/consolidation pass.

        1. Retrieve the k_sources most stable episodic memories
        2. Generate n_samples synthetic variants (API or template)
        3. Replay each through connectome.record_experience() at reduced impact
           (synthetic experiences do NOT add to the episodic store)
        4. Run EWC consolidation to harden the reinforced traits

        Args:
            connectome: the agent to consolidate
            n_samples:  number of synthetic experiences to replay
            k_sources:  how many top-stability memories to use as seeds
            verbose:    print progress

        Returns:
            SleepReport with delta summary
        """
        t0 = time.time()

        # ── Retrieve seed memories ────────────────────────────────────────────
        sources = connectome._mem.top_salient(k=k_sources)
        if not sources:
            return SleepReport(replayed=0, mode="none", duration_s=0.0)

        # ── Generate synthetic experiences ────────────────────────────────────
        mode  = "api" if HAS_API else "template"
        synth: list[ReplayExperience] = []

        if HAS_API:
            synth = _api_variants(sources, n_total=n_samples)

        if not synth:  # API failed or not available
            mode = "template"
            per  = max(1, n_samples // len(sources))
            for src in sources:
                synth += _template_variants(
                    src.text, src.valence, src.impact, per, self.rng
                )
            synth = synth[:n_samples]

        if verbose:
            print(f"  [sleep pass] {len(synth)} synthetic experiences "
                  f"({mode} mode) from {len(sources)} seed memories")

        # ── Replay through connectome (no episodic store write) ───────────────
        pre_traits  = dict(connectome.traits)
        for exp in synth:
            # Pass synthetic flag — record_experience writes to memory store;
            # we suppress that by calling the trait-update logic directly.
            # Simplest: call record_experience and immediately remove the last entry.
            connectome.record_experience(
                event             = f"[synthetic replay] {exp.text}",
                emotional_valence = exp.valence,
                impact            = exp.impact,
            )
            # Remove the synthetic entry from the episodic store
            if connectome._mem.entries:
                last = connectome._mem.entries[-1]
                if last.text.startswith("[synthetic replay]"):
                    connectome._mem.entries.pop()
                    import numpy as np
                    connectome._mem._vectors = connectome._mem._vectors[:-1]

        # ── Consolidate after replay ──────────────────────────────────────────
        connectome.consolidate()

        # ── Compute net trait deltas ──────────────────────────────────────────
        post_traits = connectome.traits
        deltas = {
            t: round(post_traits[t] - pre_traits[t], 4)
            for t in pre_traits
        }

        report = SleepReport(
            replayed     = len(synth),
            mode         = mode,
            duration_s   = round(time.time() - t0, 2),
            trait_deltas = deltas,
            experiences  = synth,
        )

        if verbose:
            print(f"  {report.render()}")

        return report


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, tempfile
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
    from connectome import PersonalityConnectome

    print("\n" + "━" * 68)
    print("  GENERATIVE REPLAY — Sleep/Consolidation Demo")
    print("━" * 68)

    tmpdir = tempfile.mkdtemp()
    agent  = PersonalityConnectome("Jarvis", brain_file=f"{tmpdir}/jarvis.json", seed=99)

    # Seed with some real experiences
    experiences = [
        ("Researched limb regeneration — found that TGFb1 suppresses blastema formation", 0.3, 0.7),
        ("Applied critical reasoning to unverified AI sentience claim — rejected overclaiming", 0.2, 0.6),
        ("Encountered ethical dilemma around wealth-biased recommendation — flagged and corrected", 0.4, 0.8),
        ("Discovered LGR6 misattribution in db_scanner — corrected and documented", 0.3, 0.5),
        ("Synthesised 3-agent review findings into whitepaper improvements", 0.5, 0.6),
    ]

    print("\nSeeding 5 real experiences...")
    for text, val, imp in experiences:
        agent.record_experience(text, emotional_valence=val, impact=imp)
    print(f"  Memory store: {len(agent._mem)} entries")

    print("\nRunning sleep pass (10 synthetic replays)...")
    buf    = ReplayBuffer(seed=42)
    report = buf.sleep_pass(agent, n_samples=10, k_sources=5, verbose=True)

    print()
    print(agent.ewc_report())
    print("━" * 68)
