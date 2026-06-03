"""
ComplementaryMemorySystem — ChromaDB-backed episodic memory (Roadmap M1.1–M1.3)

Implements the complementary learning systems (CLS) architecture:
  - Fast episodic write with content-addressable retrieval (ChromaDB)
  - Rehearsal-stabilized power-law forgetting (R2.3)
  - Reconsolidation on retrieval — memory is re-stabilized each access (R2.4)
  - "Waking up" session continuity bridge — top-k salience memories injected
    into the system prompt at session start (R2.5)

Storage: local ChromaDB (persistent, no server required)
Embedding: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── Embedding backend ─────────────────────────────────────────────────────────
# Preference: fastembed (ONNX, ~50MB) → sentence-transformers → hash fallback

try:
    from fastembed import TextEmbedding as _FE
    _fe_model = _FE(model_name="BAAI/bge-small-en-v1.5")
    EMBED_DIM = 384
    BACKEND   = "fastembed"

    def _embed(text: str) -> list[float]:
        return list(next(iter(_fe_model.embed([text]))))

except ImportError:
    try:
        from sentence_transformers import SentenceTransformer
        _smodel = SentenceTransformer("all-MiniLM-L6-v2")
        EMBED_DIM = 384
        BACKEND   = "sentence-transformers"

        def _embed(text: str) -> list[float]:
            return _smodel.encode(text, normalize_embeddings=True).tolist()

    except ImportError:
        EMBED_DIM = 256
        BACKEND   = "feature-hashing"

        def _embed(text: str) -> list[float]:
            import re
            vec   = np.zeros(EMBED_DIM, dtype=np.float32)
            words = re.findall(r'\b\w+\b', text.lower())
            for i, w in enumerate(words):
                decay = 1.0 / math.sqrt(i + 1)
                vec[abs(hash(w)) % EMBED_DIM] += decay
                if i + 1 < len(words):
                    bg = w + "_" + words[i + 1]
                    vec[abs(hash(bg)) % EMBED_DIM] += decay * 0.5
            norm = np.linalg.norm(vec)
            return (vec / norm if norm > 0 else vec).tolist()


# ── Noise-entropy proxy ───────────────────────────────────────────────────────

def _text_entropy(text: str) -> float:
    """
    Inverse type-token ratio: measures repetitiveness/verbosity of a string.

    High-temperature synthesis (arousal state) produces verbose, repetitive
    token sequences — these score high here. The KV-cache projection formula
    (R2.5) divides salience weight by this value, suppressing noisy insights
    while preferring compact, information-dense ones.

    Score = total_tokens / unique_tokens  (floor 0.5, so weight denominator ≥ 0.5).
    """
    words = text.lower().split()
    if not words:
        return 0.5
    return max(0.5, len(words) / max(1, len(set(words))))


# ── ChromaDB backend ──────────────────────────────────────────────────────────

try:
    import chromadb
    _CHROMA_OK = True
except ImportError:
    _CHROMA_OK = False


class ComplementaryMemorySystem:
    """
    ChromaDB-backed episodic memory with rehearsal-stabilized decay.

    Roadmap milestones covered:
      M1.1 — ChromaDB collection replacing JSON append log
      M1.2 — Rehearsal-stabilized S_k update (R2.3) + reconsolidation (R2.4)
      M1.3 — wakeup_context() generates session-start memory injection

    Usage:
        cms = ComplementaryMemorySystem("jarvis", brain_dir="./jarvis_brain")
        cms.store_experience(text="Ran sentience suite — score 7.8/10",
                             valence=0.6, impact=0.7)
        ctx = cms.wakeup_context()  # inject into system prompt at session start
    """

    def __init__(
        self,
        agent_id:  str,
        brain_dir: str = ".",
        alpha:     float = 0.05,   # consolidation coefficient
        gamma:     float = 0.01,   # base decay dampener
    ):
        self.agent_id  = agent_id.lower()
        self.brain_dir = Path(brain_dir)
        self.brain_dir.mkdir(parents=True, exist_ok=True)
        self.alpha = alpha
        self.gamma = gamma

        if _CHROMA_OK:
            chroma_path = str(self.brain_dir / "chroma_episodic")
            self._client = chromadb.PersistentClient(path=chroma_path)
            self._col = self._client.get_or_create_collection(
                name="episodic_memory",
                metadata={"hnsw:space": "cosine"},
            )
            self._backend = "chromadb"
        else:
            # Fallback: in-memory list (no persistence)
            self._col     = None
            self._entries = []
            self._backend = "in-memory"

    # ── Write ─────────────────────────────────────────────────────────────────

    def store_experience(
        self,
        text:    str,
        valence: float = 0.0,   # -1 to +1
        impact:  float = 0.3,   # 0 to 1
    ) -> str:
        """Encode and store an experience. Returns the document ID."""
        embedding        = _embed(text)
        doc_id           = f"mem_{self.agent_id}_{int(time.time() * 1000)}"
        initial_stability = float(impact * (1.0 + abs(valence)))
        now              = time.time()

        if self._backend == "chromadb":
            self._col.add(
                ids        =[doc_id],
                embeddings =[embedding],
                documents  =[text],
                metadatas  =[{
                    "agent_id":  self.agent_id,
                    "timestamp": now,
                    "valence":   float(valence),
                    "impact":    float(impact),
                    "stability": initial_stability,
                    "rehearsals": 1,
                }],
            )
        else:
            self._entries.append({
                "id":        doc_id,
                "agent_id":  self.agent_id,
                "text":      text,
                "embedding": embedding,
                "timestamp": now,
                "valence":   float(valence),
                "impact":    float(impact),
                "stability": initial_stability,
                "rehearsals": 1,
            })
        return doc_id

    # ── Retrieve + reconsolidate ──────────────────────────────────────────────

    def retrieve_and_reconsolidate(
        self,
        query_text: str,
        top_k:      int = 5,
    ) -> list[tuple[str, float]]:
        """
        Retrieve the top-k most semantically similar memories and apply
        the R2.3 rehearsal-stabilized stability update to each retrieved entry.

        Returns list of (text, stability_score) tuples, highest-stability first.
        """
        query_vec = _embed(query_text)
        now       = time.time()
        results   = []

        if self._backend == "chromadb":
            n = self._col.count()
            if n == 0:
                return []
            resp = self._col.query(
                query_embeddings=[query_vec],
                n_results=min(top_k, n),
                where={"agent_id": self.agent_id},
            )
            if not resp["ids"] or not resp["ids"][0]:
                return []

            for idx in range(len(resp["ids"][0])):
                doc_id = resp["ids"][0][idx]
                meta   = resp["metadatas"][0][idx]
                text   = resp["documents"][0][idx]

                # R2.3 dynamic stability update on retrieval
                delta_t = now - meta["timestamp"]
                s_new   = meta["stability"] * (
                    1.0 + self.alpha * abs(meta["valence"]) * meta["impact"]
                    * math.exp(-self.gamma * delta_t)
                )
                meta.update({
                    "stability":  float(s_new),
                    "rehearsals": meta["rehearsals"] + 1,
                    "timestamp":  now,
                })
                self._col.update(ids=[doc_id], metadatas=[meta])
                results.append((text, s_new))

        else:
            # In-memory fallback: cosine similarity
            q = np.array(query_vec, dtype=np.float32)
            scored = []
            for e in self._entries:
                if e["agent_id"] != self.agent_id:
                    continue
                v  = np.array(e["embedding"], dtype=np.float32)
                cs = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-8))
                scored.append((e, cs))
            scored.sort(key=lambda x: -x[1])
            for e, _ in scored[:top_k]:
                delta_t = now - e["timestamp"]
                s_new   = e["stability"] * (
                    1.0 + self.alpha * abs(e["valence"]) * e["impact"]
                    * math.exp(-self.gamma * delta_t)
                )
                e.update({"stability": float(s_new),
                          "rehearsals": e["rehearsals"] + 1,
                          "timestamp": now})
                results.append((e["text"], s_new))

        return results

    def top_salient(self, k: int = 5) -> list[tuple[str, float]]:
        """Return the k highest-stability memories (session-start salience ranking)."""
        if self._backend == "chromadb":
            n = self._col.count()
            if n == 0:
                return []
            all_docs = self._col.get(
                where={"agent_id": self.agent_id},
                include=["documents", "metadatas"],
            )
            pairs = list(zip(all_docs["documents"], all_docs["metadatas"]))
            pairs.sort(key=lambda x: -x[1].get("stability", 0))
            return [(text, meta.get("stability", 0.0)) for text, meta in pairs[:k]]
        else:
            entries = [e for e in self._entries if e["agent_id"] == self.agent_id]
            entries.sort(key=lambda e: -e["stability"])
            return [(e["text"], e["stability"]) for e in entries[:k]]

    def prune_stale(self, min_retention: float = 0.05) -> int:
        """Delete entries whose Ebbinghaus retention has fallen below threshold."""
        now     = time.time()
        pruned  = 0
        if self._backend == "chromadb":
            all_docs = self._col.get(
                where={"agent_id": self.agent_id},
                include=["metadatas"],
            )
            for doc_id, meta in zip(all_docs["ids"], all_docs["metadatas"]):
                hours   = max(0.0, (now - meta["timestamp"]) / 3600.0)
                s       = max(0.01, meta["stability"])
                retention = (1.0 + hours / s) ** -0.5
                if retention < min_retention:
                    self._col.delete(ids=[doc_id])
                    pruned += 1
        else:
            keep = []
            for e in self._entries:
                hours     = max(0.0, (now - e["timestamp"]) / 3600.0)
                s         = max(0.01, e["stability"])
                retention = (1.0 + hours / s) ** -0.5
                if retention >= min_retention:
                    keep.append(e)
                else:
                    pruned += 1
            self._entries = keep
        return pruned

    def count(self) -> int:
        if self._backend == "chromadb":
            return self._col.count()
        return len([e for e in self._entries if e["agent_id"] == self.agent_id])

    # ── Session continuity bridge (R2.5 / M1.3) ──────────────────────────────

    def wakeup_context(
        self,
        top_k: int = 5,
        insight_buffer: Optional[list] = None,
    ) -> str:
        """
        Generate the session-start autobiographical context block.

        Retrieves top-k highest-salience memories plus entropy-filtered incubator
        insights. Insights from the UnconsciousIncubator are weighted by
            w = (impact × |valence|) / max(0.5, H_noise)
        where H_noise = total_tokens / unique_tokens is a repetitiveness proxy.
        High-temperature synthesis produces verbose, repetitive strings (high
        H_noise) and is down-weighted relative to compressed, dense insights.
        This implements the KV-cache entropy gate described in R2.5.

        Returns empty string if no memories or insights exist yet.
        """
        memories = self.top_salient(k=top_k)
        if not memories and not insight_buffer:
            return ""

        lines = ["[Waking up — context from prior sessions]"]

        for text, stability in memories:
            lines.append(f"  • (stability={stability:.2f}) {text[:115].strip()}")

        if insight_buffer:
            weighted: list[tuple[float, str]] = []
            for ins in insight_buffer:
                content = ins.get("content", "")
                if not content:
                    continue
                impact  = float(ins.get("impact",  0.85))
                valence = float(ins.get("valence", 0.5))
                H = _text_entropy(content)
                w = (impact * max(0.01, abs(valence))) / H
                weighted.append((w, content))
            weighted.sort(key=lambda x: -x[0])
            if weighted:
                lines.append("  [Incubated insights — entropy-filtered]")
                for _, content in weighted[:3]:
                    lines.append(f"  ◆ {content[:115].strip()}")

        return "\n".join(lines)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def status(self) -> str:
        n = self.count()
        return (f"ComplementaryMemorySystem [{self.agent_id}] | "
                f"backend={self._backend} | entries={n} | "
                f"α={self.alpha} γ={self.gamma}")
