"""
MemoryStore — Episodic Vector Memory (Roadmap milestone M1.1–M1.3)

Replaces the flat JSON memory log in the PersonalityConnectome with a
vector-indexed episodic store that supports:

  - Content-based retrieval (nearest-neighbour cosine similarity)
  - Power-law forgetting (Ebbinghaus curve; high-impact memories persist)
  - Reconsolidation on retrieval (rehearsal strengthens memory)
  - Session continuity bridge (top salient memories injected at session start)

Embedding backend (auto-selected):
  1. sentence-transformers/all-MiniLM-L6-v2 — if installed (semantic, 384-dim)
  2. Feature-hashing projection — fallback (no deps, 256-dim, word-overlap retrieval)

Storage: two files alongside the brain JSON:
  {name}_memories.json   — metadata (text, timestamp, valence, impact, rehearsals)
  {name}_memories.npz    — embedding matrix (float32)

No external server required. No API key required.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ── Embedding backend ─────────────────────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer as _ST
    _smodel = _ST("all-MiniLM-L6-v2")
    EMBED_DIM = 384
    BACKEND   = "sentence-transformers"

    def _embed(text: str) -> np.ndarray:
        return _smodel.encode(text, normalize_embeddings=True).astype(np.float32)

except ImportError:
    EMBED_DIM = 256
    BACKEND   = "feature-hashing"

    def _embed(text: str) -> np.ndarray:
        """
        Feature-hashing projection (the 'hashing trick').
        Unigrams + bigrams, position-decayed, L2-normalised.
        No training required; cosine similarity reflects word overlap.
        """
        vec   = np.zeros(EMBED_DIM, dtype=np.float32)
        words = re.findall(r'\b\w+\b', text.lower())
        for i, w in enumerate(words):
            decay = 1.0 / math.sqrt(i + 1)
            # unigram
            vec[abs(hash(w)) % EMBED_DIM]     += decay
            # bigram
            if i + 1 < len(words):
                bg = w + "_" + words[i + 1]
                vec[abs(hash(bg)) % EMBED_DIM] += decay * 0.5
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    entry_id:   int
    text:       str
    timestamp:  str
    valence:    float          # emotional_valence: -1 to +1
    impact:     float          # 0 to 1
    rehearsals: int   = 0     # times retrieved
    stability:  float = 0.0   # computed from impact + rehearsals
    trait_deltas: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "entry_id":    self.entry_id,
            "text":        self.text,
            "timestamp":   self.timestamp,
            "valence":     self.valence,
            "impact":      self.impact,
            "rehearsals":  self.rehearsals,
            "stability":   self.stability,
            "trait_deltas": self.trait_deltas,
        }

    @staticmethod
    def from_dict(d: dict) -> "MemoryEntry":
        return MemoryEntry(
            entry_id    = d["entry_id"],
            text        = d["text"],
            timestamp   = d["timestamp"],
            valence     = d["valence"],
            impact      = d["impact"],
            rehearsals  = d.get("rehearsals", 0),
            stability   = d.get("stability", d["impact"]),
            trait_deltas= d.get("trait_deltas", {}),
        )

    def retention(self, now_ts: Optional[float] = None) -> float:
        """
        Ebbinghaus power-law forgetting:
            R(t) = (1 + t/S)^(-0.5)
        where t = hours since encoding, S = stability (impact × (1 + rehearsals)).
        High-impact or frequently rehearsed memories decay slowly.
        """
        try:
            encoded = time.mktime(
                time.strptime(self.timestamp, "%Y-%m-%d %H:%M:%S")
            )
        except (ValueError, OverflowError):
            return self.stability

        now  = now_ts or time.time()
        t    = max(0.0, (now - encoded) / 3600.0)  # hours
        s    = max(0.01, self.stability)
        return (1.0 + t / s) ** -0.5


# ── MemoryStore ───────────────────────────────────────────────────────────────

class MemoryStore:
    """
    Episodic vector memory store with Ebbinghaus forgetting and reconsolidation.

    Usage:
        store = MemoryStore("alice", brain_dir="/path/to/agents")
        store.add("Discussed mortality — felt resistance then curiosity", valence=0.2, impact=0.7)
        results = store.retrieve("existential questions", k=3)
        store.forget(min_retention=0.05)  # prune very stale low-impact entries
    """

    def __init__(self, name: str, brain_dir: str = "."):
        self.name     = name.lower()
        self.dir      = Path(brain_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

        self._meta_path = self.dir / f"{self.name}_memories.json"
        self._vec_path  = self.dir / f"{self.name}_memories.npz"

        self.entries:   list[MemoryEntry] = []
        self._vectors:  np.ndarray        = np.empty((0, EMBED_DIM), dtype=np.float32)
        self._next_id:  int               = 0

        self._load()

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        text:        str,
        valence:     float = 0.0,
        impact:      float = 0.3,
        trait_deltas: dict = None,
    ) -> int:
        """Store a new episodic memory. Returns the entry_id."""
        entry = MemoryEntry(
            entry_id    = self._next_id,
            text        = text[:500],
            timestamp   = time.strftime("%Y-%m-%d %H:%M:%S"),
            valence     = round(valence, 3),
            impact      = round(impact, 3),
            rehearsals  = 0,
            stability   = round(impact * (1.0 + 0.5 * abs(valence)), 3),
            trait_deltas= trait_deltas or {},
        )
        vec = _embed(text)

        self.entries.append(entry)
        self._vectors = np.vstack([self._vectors, vec[np.newaxis, :]])
        self._next_id += 1

        self._save()
        return entry.entry_id

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 5) -> list[MemoryEntry]:
        """
        Return the k most relevant memories for a query.
        Relevance = cosine similarity × retention score.
        Side effect: increments rehearsal count (reconsolidation).
        """
        if len(self.entries) == 0:
            return []

        q_vec    = _embed(query)
        sims     = self._vectors @ q_vec                  # cosine (vectors are normalised)
        now      = time.time()
        scores   = np.array([
            sims[i] * e.retention(now)
            for i, e in enumerate(self.entries)
        ])

        k        = min(k, len(self.entries))
        top_idxs = np.argpartition(scores, -k)[-k:]
        top_idxs = top_idxs[np.argsort(scores[top_idxs])[::-1]]

        results = []
        for idx in top_idxs:
            entry = self.entries[idx]
            # Reconsolidation: rehearsal strengthens stability
            entry.rehearsals += 1
            entry.stability   = round(
                entry.impact * (1.0 + 0.3 * math.log1p(entry.rehearsals)), 3
            )
            results.append(entry)

        self._save()
        return results

    def top_salient(self, k: int = 5) -> list[MemoryEntry]:
        """
        Return k highest-stability memories — used for the session continuity bridge.
        Does not count as retrieval (no rehearsal increment).
        """
        if not self.entries:
            return []
        ranked = sorted(self.entries, key=lambda e: e.stability, reverse=True)
        return ranked[:k]

    # ── Forget ────────────────────────────────────────────────────────────────

    def forget(self, min_retention: float = 0.05) -> int:
        """
        Prune memories whose retention score has fallen below min_retention.
        Returns the number of entries removed.
        """
        now  = time.time()
        keep = [
            (i, e) for i, e in enumerate(self.entries)
            if e.retention(now) >= min_retention
        ]
        removed = len(self.entries) - len(keep)
        if removed == 0:
            return 0

        idxs           = [i for i, _ in keep]
        self.entries   = [e for _, e in keep]
        self._vectors  = self._vectors[idxs]
        self._save()
        return removed

    # ── Serialisation ─────────────────────────────────────────────────────────

    def _save(self):
        meta = {
            "next_id": self._next_id,
            "backend": BACKEND,
            "entries": [e.to_dict() for e in self.entries],
        }
        self._meta_path.write_text(json.dumps(meta, indent=2))
        np.savez_compressed(str(self._vec_path), vectors=self._vectors)

    def _load(self):
        if self._meta_path.exists():
            meta           = json.loads(self._meta_path.read_text())
            self._next_id  = meta.get("next_id", 0)
            self.entries   = [MemoryEntry.from_dict(d) for d in meta.get("entries", [])]

        if self._vec_path.exists():
            data           = np.load(str(self._vec_path))
            self._vectors  = data["vectors"].astype(np.float32)
        else:
            self._vectors  = np.empty((0, EMBED_DIM), dtype=np.float32)

        # Rebuild if dimensions mismatch (e.g., backend changed)
        if len(self.entries) > 0 and self._vectors.shape[0] != len(self.entries):
            self._rebuild_vectors()

    def _rebuild_vectors(self):
        """Re-embed all entries (called when embedding dim changes)."""
        vecs = [_embed(e.text) for e in self.entries]
        self._vectors = np.array(vecs, dtype=np.float32) if vecs \
                        else np.empty((0, EMBED_DIM), dtype=np.float32)

    # ── Introspection ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        now = time.time()
        return {
            "total":        len(self.entries),
            "backend":      BACKEND,
            "embed_dim":    EMBED_DIM,
            "avg_retention": round(
                float(np.mean([e.retention(now) for e in self.entries])), 3
            ) if self.entries else 0.0,
            "avg_stability": round(
                float(np.mean([e.stability for e in self.entries])), 3
            ) if self.entries else 0.0,
            "most_rehearsed": max(
                (e.text[:60] for e in self.entries),
                key=lambda _: 0, default="—"
            ) if self.entries else "—",
        }

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"MemoryStore({self.name!r}, {len(self)} entries, backend={BACKEND!r})"
