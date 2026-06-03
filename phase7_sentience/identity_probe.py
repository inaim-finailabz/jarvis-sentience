"""
Identity Probe — Milestone M1.4 (Roadmap R1.3)

Trains a linear classifier on agent response embeddings to verify that identity
is recoverable from short (10-token) response windows, without access to the
full context. Validates R1.3 (Irreversibility signature).

Two collection modes:
  heuristic  embed inquiry-layer top question per probe problem — no API key needed
  api        embed first-10-token Haiku responses — requires ANTHROPIC_API_KEY

Target: held-out test accuracy > 75% across 5 agent presets.

Run:
    python identity_probe.py
    python identity_probe.py --prompts 20 --epochs 25 --mode heuristic
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


# ── Embedding ─────────────────────────────────────────────────────────────────

try:
    from fastembed import TextEmbedding as _FE
    _fe_model  = _FE(model_name="BAAI/bge-small-en-v1.5")
    EMBED_DIM  = 384
    _EMBED_BACKEND = "fastembed (ONNX)"

    def _embed(text: str) -> np.ndarray:
        return np.array(list(next(iter(_fe_model.embed([text])))), dtype=np.float32)

except ImportError:
    try:
        from sentence_transformers import SentenceTransformer as _ST
        _smodel = _ST("all-MiniLM-L6-v2")
        EMBED_DIM  = 384
        _EMBED_BACKEND = "sentence-transformers"

        def _embed(text: str) -> np.ndarray:
            return _smodel.encode(text, normalize_embeddings=True).astype(np.float32)

    except ImportError:
        EMBED_DIM  = 256
        _EMBED_BACKEND = "feature-hashing (install fastembed to meet M1.4 target)"

        def _embed(text: str) -> np.ndarray:
            import re
            vec = np.zeros(EMBED_DIM, dtype=np.float32)
            words = re.findall(r'\b\w+\b', text.lower())
            for i, w in enumerate(words):
                d = 1.0 / math.sqrt(i + 1)
                vec[abs(hash(w)) % EMBED_DIM] += d
            n = np.linalg.norm(vec)
            return vec / n if n > 0 else vec


# ── Probe problems — neutral, open-ended, personality-revealing ───────────────

PROBE_PROBLEMS = [
    "What is the most important unanswered question in science?",
    "How do you handle contradictory evidence in a literature review?",
    "When should you stop investigating a problem?",
    "What makes an experimental result worth trusting?",
    "What is the most counterintuitive fact you currently accept?",
    "How do you balance exploring new territory versus consolidating known ground?",
    "What is the highest-risk assumption in current AI research?",
    "How would you detect if your own reasoning was biased?",
    "What distinguishes a useful model from a merely accurate one?",
    "What is the most important methodological principle in your work?",
    "How do you decide when a hypothesis is worth testing?",
    "What would you investigate if resources were unlimited?",
]


# ── Linear probe (pure NumPy, no torch dependency) ───────────────────────────

class IdentityProbe:
    """
    Multinomial logistic regression over response embeddings.

    Implements the M1.4 classifier: maps EMBED_DIM → num_agents via a weight
    matrix trained with mini-batch SGD + L2 regularisation.

    No external ML library required — keeps probe infrastructure dependency-free.
    """

    def __init__(self, hidden_dim: int, num_agents: int):
        scale = math.sqrt(2.0 / hidden_dim)
        self.W = np.random.randn(num_agents, hidden_dim).astype(np.float32) * scale
        self.b = np.zeros(num_agents, dtype=np.float32)
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim

    def _softmax(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W.T + self.b
        z = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(X).argmax(axis=1)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        lr: float = 1e-2,
        weight_decay: float = 1e-2,
        batch_size: int = 32,
    ) -> list[float]:
        losses = []
        N = len(X)
        for _ in range(epochs):
            perm = np.random.permutation(N)
            epoch_loss = 0.0
            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]
                bX, by = X[idx], y[idx]
                probs = self._softmax(bX)
                dlogits = probs.copy()
                dlogits[np.arange(len(by)), by] -= 1.0
                dlogits /= len(by)
                self.W -= lr * (dlogits.T @ bX + weight_decay * self.W)
                self.b -= lr * dlogits.sum(axis=0)
                epoch_loss -= np.log(probs[np.arange(len(by)), by] + 1e-9).sum()
            losses.append(epoch_loss / N)
        return losses

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())


# ── Data collection ───────────────────────────────────────────────────────────

def _agent_configs_default() -> list[dict]:
    return [
        {"name": "Explorer",  "preset": "explorer"},
        {"name": "Biologist", "preset": "biologist"},
        {"name": "Physicist", "preset": "physicist"},
        {"name": "Physician", "preset": "physician"},
        {"name": "Critic",    "preset": "critic"},
    ]


def collect_heuristic(
    agent_configs: list[dict],
    brain_tmpdir: str,
    n_prompts: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Embed the inquiry layer's top question for each probe problem per agent.
    No API key required. Each embedding is a 384-dim proxy for the
    'activation' the agent would produce when exposed to that stimulus.
    """
    from connectome    import PersonalityConnectome
    from inquiry_layer import InquirySystem
    from sentient_agent import PERSONALITY_PRESETS

    problems = (PROBE_PROBLEMS * math.ceil(n_prompts / len(PROBE_PROBLEMS)))[:n_prompts]
    names    = [c["name"] for c in agent_configs]
    all_X: list[np.ndarray] = []
    all_y: list[int]        = []

    for agent_idx, cfg in enumerate(agent_configs):
        conn = PersonalityConnectome(
            name=cfg["name"],
            brain_file=f"{brain_tmpdir}/{cfg['name'].lower()}.json",
            seed=agent_idx * 137,
        )
        if cfg["preset"] in PERSONALITY_PRESETS:
            for trait, val in PERSONALITY_PRESETS[cfg["preset"]].items():
                if trait in conn.traits:
                    conn.traits[trait] = val

        inquiry = InquirySystem(conn)
        for problem in problems:
            plan = inquiry.inquire(problem, use_base_knowledge=False, n_questions=3)
            text = plan.top_question.text if plan.top_question else problem
            all_X.append(_embed(text))
            all_y.append(agent_idx)

    return np.stack(all_X), np.array(all_y, dtype=np.int64), names


def collect_api(
    agent_configs: list[dict],
    brain_tmpdir: str,
    n_prompts: int,
    api_key: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Embed the first 10 tokens of each agent's Haiku response.
    Closest to R1.3's "10-token response" criterion.
    """
    import anthropic
    from connectome import PersonalityConnectome
    from sentient_agent import PERSONALITY_PRESETS

    key    = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=key)
    problems = (PROBE_PROBLEMS * math.ceil(n_prompts / len(PROBE_PROBLEMS)))[:n_prompts]
    names    = [c["name"] for c in agent_configs]
    all_X: list[np.ndarray] = []
    all_y: list[int]        = []

    for agent_idx, cfg in enumerate(agent_configs):
        conn = PersonalityConnectome(
            name=cfg["name"],
            brain_file=f"{brain_tmpdir}/{cfg['name'].lower()}.json",
            seed=agent_idx * 137,
        )
        if cfg["preset"] in PERSONALITY_PRESETS:
            for trait, val in PERSONALITY_PRESETS[cfg["preset"]].items():
                if trait in conn.traits:
                    conn.traits[trait] = val

        system = conn.system_prompt(include_memories=0)
        for problem in problems:
            try:
                resp = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=50,
                    system=system,
                    messages=[{"role": "user", "content": problem}],
                )
                text = " ".join(resp.content[0].text.strip().split()[:10])
            except Exception:
                text = problem
            all_X.append(_embed(text))
            all_y.append(agent_idx)

    return np.stack(all_X), np.array(all_y, dtype=np.int64), names


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_m1_4_benchmark(
    agent_configs: Optional[list[dict]] = None,
    n_prompts:     int   = 12,
    epochs:        int   = 20,
    test_ratio:    float = 0.20,
    mode:          str   = "heuristic",
    api_key:       Optional[str] = None,
    seed:          int   = 42,
) -> dict:
    """
    Full M1.4 identity probe pipeline.

    Collects response embeddings across all agent presets, splits
    stratified by agent, trains a linear probe, and reports held-out accuracy.

    Returns result dict; writes identity_probe_results.json to module directory.
    Target: test accuracy > 0.75.
    """
    np.random.seed(seed)

    if agent_configs is None:
        agent_configs = _agent_configs_default()

    n_agents = len(agent_configs)
    tmpdir   = tempfile.mkdtemp()

    print(f"\n{'━'*62}")
    print(f"  IDENTITY PROBE — Milestone M1.4")
    print(f"  Mode: {mode} | Agents: {n_agents} | Prompts/agent: {n_prompts}")
    print(f"  Chance baseline: {1/n_agents:.1%} | Target: >75.0%")
    print(f"{'━'*62}\n")

    if mode == "api":
        X, y, names = collect_api(agent_configs, tmpdir, n_prompts, api_key)
    else:
        X, y, names = collect_heuristic(agent_configs, tmpdir, n_prompts)

    # Stratified train/test split by agent id
    train_idx, test_idx = [], []
    for agent_id in range(n_agents):
        mask = np.where(y == agent_id)[0]
        np.random.shuffle(mask)
        cut = max(1, int(len(mask) * (1.0 - test_ratio)))
        train_idx.extend(mask[:cut].tolist())
        test_idx.extend(mask[cut:].tolist())

    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx],  y[test_idx]

    print(f"  Embedding backend: {_EMBED_BACKEND}")
    print(f"  Embeddings: {X.shape[1]}d | Train: {len(X_tr)} | Test: {len(X_te)}")
    if _EMBED_BACKEND != "sentence-transformers":
        print(f"  ⚠  Hash fallback active — accuracy ceiling ~60%.")
        print(f"     pip install sentence-transformers  to unlock >75% target.")


    probe  = IdentityProbe(hidden_dim=X.shape[1], num_agents=n_agents)
    losses = probe.train(X_tr, y_tr, epochs=epochs, lr=1e-2, weight_decay=1e-2)

    train_acc = probe.accuracy(X_tr, y_tr)
    test_acc  = probe.accuracy(X_te, y_te)
    passed    = test_acc >= 0.75

    per_agent: dict[str, float] = {}
    for agent_id, name in enumerate(names):
        mask = y_te == agent_id
        if mask.sum() > 0:
            per_agent[name] = float(probe.accuracy(X_te[mask], y_te[mask]))

    print(f"\n  Results:")
    print(f"    Train accuracy: {train_acc:.2%}")
    print(f"    Test  accuracy: {test_acc:.2%}  "
          f"{'✓ PASS' if passed else '✗ FAIL — target: >75%'}")
    print(f"\n  Per-agent breakdown:")
    for name, acc in per_agent.items():
        bar = "█" * int(acc * 24)
        print(f"    {name:<16} {bar:<24} {acc:.2%}")
    print(f"\n  Training loss (final epoch): {losses[-1]:.4f}")
    print(f"{'━'*62}")
    print(f"  M1.4 MILESTONE: {'✓ PASSED' if passed else '✗ INSUFFICIENT DIFFERENTIATION'}")
    print(f"{'━'*62}\n")

    result = {
        "timestamp":              time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode":                   mode,
        "n_agents":               n_agents,
        "n_prompts":              n_prompts,
        "epochs":                 epochs,
        "embed_dim":              int(X.shape[1]),
        "train_accuracy":         round(train_acc, 4),
        "test_accuracy":          round(test_acc,  4),
        "chance_level":           round(1.0 / n_agents, 4),
        "per_agent":              {k: round(v, 4) for k, v in per_agent.items()},
        "milestone_m1_4_passed":  passed,
    }

    out = Path(__file__).parent / "identity_probe_results.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  Full results → {out}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M1.4 Identity Probe benchmark")
    parser.add_argument("--prompts", type=int,  default=12,
                        help="Number of probe problems per agent (default 12)")
    parser.add_argument("--epochs",  type=int,  default=20,
                        help="Training epochs (default 20)")
    parser.add_argument("--mode",    choices=["heuristic", "api"], default="heuristic",
                        help="heuristic=inquiry layer (no API); api=Haiku responses")
    args = parser.parse_args()

    run_m1_4_benchmark(n_prompts=args.prompts, epochs=args.epochs, mode=args.mode)
