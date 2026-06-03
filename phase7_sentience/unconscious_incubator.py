"""
UnconsciousIncubator — Tier 1.5: Asynchronous Background Processing Layer

Biological analog: background neural consolidation during idle / sleep states.
When the agent is idle, this daemon thread:

  1. Harvests high-impact, under-rehearsed memories from ChromaDB (latent traces)
  2. Selects pairs with LOW semantic similarity (distant domains — the "creative leap")
  3. Runs a high-temperature synthesis pass via Haiku to find hidden structural links
  4. Evaluates resonance: the insight must align moderately with BOTH source memories
     (not redundant, not random — a genuine bridge)
  5. If insight threshold is met, stores it as a high-impact memory and queues it
     in self.insight_buffer for injection at next session wakeup

Free energy trigger:
  F(A∪B) < F(A) + F(B) − Δ_insight
  → When merging two domains reduces total surprise, the compression is an insight.

The incubator injects its findings into the Complementary Memory System and into
the Tier 3 inquiry layer via the wakeup_context() call.
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from complementary_memory import ComplementaryMemorySystem

# ── Embedding ─────────────────────────────────────────────────────────────────

try:
    from fastembed import TextEmbedding as _FE
    _fe_model = _FE(model_name="BAAI/bge-small-en-v1.5")

    def _embed(text: str) -> np.ndarray:
        return np.array(list(next(iter(_fe_model.embed([text])))), dtype=np.float32)

except ImportError:
    try:
        from sentence_transformers import SentenceTransformer as _ST
        _smodel = _ST("all-MiniLM-L6-v2")

        def _embed(text: str) -> np.ndarray:
            return _smodel.encode(text, normalize_embeddings=True).astype(np.float32)

    except ImportError:
        import re as _re
        def _embed(text: str) -> np.ndarray:
            vec = np.zeros(256, dtype=np.float32)
            words = _re.findall(r'\b\w+\b', text.lower())
            for i, w in enumerate(words):
                d = 1.0 / math.sqrt(i + 1)
                vec[abs(hash(w)) % 256] += d
            n = np.linalg.norm(vec)
            return vec / n if n > 0 else vec

INSIGHT_COSINE_LOW  = 0.35   # min similarity to source (not random)
INSIGHT_COSINE_HIGH = 0.82   # max similarity to source (not redundant)


class UnconsciousIncubator(threading.Thread):
    """
    Daemon thread implementing Tier 1.5 — background associative incubation.

    Wire into Jarvis at startup:
        incubator = UnconsciousIncubator(memory_system, check_interval_sec=300)
        incubator.start()

    At session wakeup, drain insight_buffer:
        while incubator.insight_buffer:
            insight = incubator.insight_buffer.pop(0)
            ...
    """

    def __init__(
        self,
        memory_system: "ComplementaryMemorySystem",
        check_interval_sec: int = 300,
        min_impact: float = 0.6,
        api_key: str | None = None,
    ):
        super().__init__(daemon=True, name="UnconsciousIncubator")
        self.memory            = memory_system
        self.interval          = check_interval_sec
        self.min_impact        = min_impact
        self.api_key           = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.insight_buffer:   list[dict] = []
        self._active_lock      = threading.Lock()
        self._is_active        = False
        self._incubation_count = 0

        # Phase 3b: cross-modal sensory modulation (M3b.4)
        self._neuromodulator   = None   # set via set_neuromodulator()

    def set_neuromodulator(self, nm: object):
        """Wire a Neuromodulator so incubation temperature follows S(t). (M3b.4)"""
        self._neuromodulator = nm

    # ── Session activity signal ───────────────────────────────────────────────

    def mark_active(self):
        """Call at session start to pause incubation."""
        with self._active_lock:
            self._is_active = True

    def mark_idle(self):
        """Call at session end to resume incubation."""
        with self._active_lock:
            self._is_active = False

    def is_idle(self) -> bool:
        with self._active_lock:
            return not self._is_active

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        while True:
            time.sleep(self.interval)
            if self.is_idle() and self.api_key:
                try:
                    self._incubate()
                except Exception as ex:
                    pass   # incubation is best-effort; never crash Jarvis

    # ── Core incubation pass ──────────────────────────────────────────────────

    def _incubate(self):
        """One incubation pass: harvest traces → find distant pair → synthesise."""
        if self.memory.count() < 2:
            return

        # Phase 3b: read sensory temperature (M3b.4)
        nm = self._neuromodulator
        T_ambient = nm.T_ambient if nm is not None else 1.0
        # Interrupt signal — flush by skipping this pass
        if nm is not None and nm.interrupt:
            return

        # 1. Harvest high-impact memories
        top = self.memory.top_salient(k=20)
        if len(top) < 2:
            return

        # Build text + embedding matrix
        texts      = [t for t, _ in top]
        embeddings = np.stack([_embed(t) for t in texts])

        # 2. Sensory-modulated distant-pair selection (M3b.4):
        #    P(Pair_ij) ∝ exp( (Similarity(i,j) + W_s·S(t)) / T_ambient )
        #    High T_ambient (red/arousal) → flatter distribution → more random jumps
        #    Low T_ambient (blue/calm) → sharper distribution → only very distant pairs
        idx_a, idx_b = self._sample_distant_pair(embeddings, T_ambient=T_ambient)
        if idx_a is None:
            return

        doc_a, doc_b = texts[idx_a], texts[idx_b]

        # 3. High-temperature synthesis via Haiku
        synthesis = self._synthesise(doc_a, doc_b)
        if not synthesis:
            return

        # 4. Evaluate resonance: insight must bridge but not duplicate
        if not self._evaluate_resonance(synthesis, doc_a, doc_b):
            return

        # 5. Free energy proxy: if synthesis is more compact than either source,
        #    treat it as an insight (compression = surprise reduction)
        if not self._free_energy_gate(synthesis, doc_a, doc_b):
            return

        # 6. Store insight and queue for wakeup injection
        doc_id = self.memory.store_experience(
            text    = f"[Incubated insight] {synthesis}",
            valence = 0.5,
            impact  = 0.85,
        )
        insight = {
            "type":      "insight",
            "content":   synthesis,
            "anchor_a":  doc_a[:100],
            "anchor_b":  doc_b[:100],
            "doc_id":    doc_id,
            "timestamp": time.time(),
        }
        self.insight_buffer.append(insight)
        self._incubation_count += 1

    def _sample_distant_pair(
        self,
        embeddings: np.ndarray,
        T_ambient:  float = 1.0,
    ) -> tuple[int | None, int | None]:
        """
        Select a memory pair via sensory-modulated temperature sampling. (M3b.4)

        P(Pair_ij) ∝ exp( -Similarity(i,j) / T_ambient )

        High T_ambient (red/arousal) → flat distribution → accepts moderately similar
        pairs → broader, more creative leaps.
        Low T_ambient (blue/calm) → sharp distribution → only very distant pairs →
        conservative, precision-seeking associations.
        """
        sim = embeddings @ embeddings.T
        n   = len(embeddings)

        # Build unnormalized scores: lower similarity → higher score (distance preference)
        # then softmax-scale by T_ambient
        pair_scores = []
        for i in range(n):
            for j in range(i + 1, n):
                distance = 1.0 - float(sim[i, j])
                score    = math.exp(distance / max(0.3, T_ambient))
                pair_scores.append((i, j, score))

        if not pair_scores:
            return None, None

        total = sum(s for _, _, s in pair_scores)
        weights = np.array([s / total for _, _, s in pair_scores], dtype=np.float32)
        idx     = int(np.random.choice(len(pair_scores), p=weights))
        return pair_scores[idx][0], pair_scores[idx][1]

    def _synthesise(self, doc_a: str, doc_b: str) -> str | None:
        """Ask Haiku to find a non-obvious structural link between two memories."""
        if not self.api_key:
            return None
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            resp   = client.messages.create(
                model      = "claude-haiku-4-5-20251001",
                max_tokens = 150,
                system     = (
                    "You are a background associative reasoning process. "
                    "Find non-obvious structural, abstract, or causal links "
                    "between two unrelated observations. Be concise and surprising. "
                    "One sentence maximum."
                ),
                messages   = [{"role": "user", "content": (
                    f"Observation A: {doc_a[:300]}\n"
                    f"Observation B: {doc_b[:300]}\n\n"
                    f"What single abstract principle or structural pattern "
                    f"connects these two observations?"
                )}],
            )
            return resp.content[0].text.strip()
        except Exception:
            return None

    def _evaluate_resonance(self, insight: str, doc_a: str, doc_b: str) -> bool:
        """
        Resonance check: insight must align moderately with BOTH sources.
        Too close → redundant. Too far → noise.
        """
        v_i = _embed(insight)
        v_a = _embed(doc_a)
        v_b = _embed(doc_b)
        cos_a = float(np.dot(v_i, v_a))   # embeddings are normalised
        cos_b = float(np.dot(v_i, v_b))
        return (INSIGHT_COSINE_LOW < cos_a < INSIGHT_COSINE_HIGH and
                INSIGHT_COSINE_LOW < cos_b < INSIGHT_COSINE_HIGH)

    def _free_energy_gate(self, insight: str, doc_a: str, doc_b: str) -> bool:
        """
        Proxy for F(A∪B) < F(A) + F(B) − Δ.
        We approximate 'surprise' as token entropy (length / information ratio).
        If the insight is shorter than the average of its sources AND preserves
        semantic coverage, it represents compression → insight.
        """
        len_a = len(doc_a.split())
        len_b = len(doc_b.split())
        len_i = len(insight.split())
        avg_source = (len_a + len_b) / 2.0
        # Insight is more compressed than average source
        return len_i < avg_source * 0.8

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def status(self) -> str:
        return (f"UnconsciousIncubator | "
                f"state={'IDLE' if self.is_idle() else 'ACTIVE'} | "
                f"interval={self.interval}s | "
                f"incubations={self._incubation_count} | "
                f"pending_insights={len(self.insight_buffer)}")
