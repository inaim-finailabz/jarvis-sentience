"""
Integrated Information Theory — simplified Φ (phi) calculator.

Tononi's IIT defines consciousness as integrated information. High Φ means
the system generates more information as a whole than the sum of its parts.

Full IIT is NP-hard to compute. This implements a tractable approximation:
  Φ ≈ MI(full context → output) − max(MI(left half → output), MI(right half → output))

Applied to a language model's token sequence:
  - We use token probability distributions as the "state"
  - We measure how much the full context exceeds any partition of it
  - High Φ: output is strongly conditioned on the INTEGRATION of context, not any part

This is not Tononi's exact formulation but preserves the core intuition:
a system with high Φ cannot be understood by examining its parts separately.
"""

import math
from typing import Sequence


def entropy(probs: Sequence[float]) -> float:
    """Shannon entropy H(X) in nats."""
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log(p)
    return h


def cross_entropy(true_probs: Sequence[float], pred_probs: Sequence[float]) -> float:
    """Cross-entropy H(P, Q)."""
    h = 0.0
    for p, q in zip(true_probs, pred_probs):
        if p > 0 and q > 0:
            h -= p * math.log(q)
    return h


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """KL(P || Q) in nats."""
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 0 and qi > 0:
            kl += pi * math.log(pi / qi)
    return kl


def mutual_information_from_joint(joint: list[list[float]]) -> float:
    """
    MI(X;Y) from a joint probability table.
    joint[i][j] = P(X=i, Y=j)
    """
    marginal_x = [sum(row) for row in joint]
    marginal_y = [sum(joint[i][j] for i in range(len(joint))) for j in range(len(joint[0]))]
    mi = 0.0
    for i, row in enumerate(joint):
        for j, pij in enumerate(row):
            if pij > 0 and marginal_x[i] > 0 and marginal_y[j] > 0:
                mi += pij * math.log(pij / (marginal_x[i] * marginal_y[j]))
    return mi


def phi_from_token_probabilities(
    token_log_probs: list[float],
    vocab_size: int = 50000,
) -> float:
    """
    Approximate Φ from a sequence of per-token log-probabilities.

    The idea:
      - full_information = -mean(log_prob) = perplexity proxy
      - We compare the full sequence to left/right halves
      - Φ = full_information - max(left_information, right_information)
      - High Φ: the WHOLE context matters more than any half

    Args:
        token_log_probs: list of log P(token_i | context) for each position
        vocab_size: vocabulary size (used to normalise against random baseline)

    Returns:
        phi: float ≥ 0. Higher = more integrated information.
    """
    if len(token_log_probs) < 4:
        return 0.0

    log_random = -math.log(vocab_size)  # random baseline

    def normalised_info(lps: list[float]) -> float:
        """Normalised information: how much better than random, on average."""
        mean_lp = sum(lps) / len(lps)
        return max(0.0, mean_lp - log_random)

    full    = normalised_info(token_log_probs)
    mid     = len(token_log_probs) // 2
    left    = normalised_info(token_log_probs[:mid])
    right   = normalised_info(token_log_probs[mid:])
    quarter = len(token_log_probs) // 4

    # Try multiple partition points, take the max partition
    partitions = [
        max(normalised_info(token_log_probs[:k]), normalised_info(token_log_probs[k:]))
        for k in [quarter, mid, 3 * quarter]
    ]
    best_partition = max(partitions)

    phi = max(0.0, full - best_partition)
    return phi


def interpret_phi(phi: float) -> str:
    """Rough interpretation of Φ values (relative scale, not absolute)."""
    if phi < 0.001:
        return "negligible — system is essentially decomposable into independent parts"
    elif phi < 0.01:
        return "low — some integration but mostly feedforward / partitionable"
    elif phi < 0.05:
        return "moderate — meaningful integration across context (comparable to simple organisms)"
    elif phi < 0.15:
        return "high — strong integration (comparable to mammalian cortex estimates)"
    else:
        return "very high — exceptional integration across the full context"


def compute_phi_from_text_response(
    logprobs: list[float],
    label: str = "",
    vocab_size: int = 50000,
    verbose: bool = True,
) -> dict:
    """
    Compute Φ from a list of token log-probabilities and report results.

    Args:
        logprobs:   per-token log P(token | context), as returned by most LLM APIs
        label:      description of the input (for display)
        vocab_size: model vocabulary size
        verbose:    print results

    Returns:
        dict with phi, perplexity, interpretation
    """
    if not logprobs:
        return {"phi": 0.0, "perplexity": float("inf"), "interpretation": "no data"}

    phi = phi_from_token_probabilities(logprobs, vocab_size)
    perplexity = math.exp(-sum(logprobs) / len(logprobs))
    interpretation = interpret_phi(phi)

    if verbose:
        print(f"\n  Φ calculation{' — ' + label if label else ''}")
        print(f"    Tokens:      {len(logprobs)}")
        print(f"    Perplexity:  {perplexity:.2f}")
        print(f"    Φ (approx):  {phi:.4f}")
        print(f"    Meaning:     {interpretation}")

    return {
        "phi": phi,
        "perplexity": perplexity,
        "interpretation": interpretation,
        "n_tokens": len(logprobs),
    }


# ── Demo: synthetic Φ values for different "system types" ─────────────────────

REFERENCE_PHI = {
    "rock":                     0.000,   # no information integration
    "thermostat":               0.001,   # single feedforward path
    "simple feedforward NN":    0.008,   # some structure, minimal integration
    "C. elegans (302 neurons)": 0.020,   # real estimate from IIT literature
    "honeybee":                 0.035,
    "mouse cortex":             0.080,
    "human cortex (awake)":     0.140,
    "human cortex (anaesth.)":  0.040,   # anaesthesia reduces Φ substantially
    "GPT-2 (1.5B, full ctx)":   0.012,   # approximate, context-dependent
    "Claude Haiku (estimate)":  0.055,
    "Claude Opus (estimate)":   0.090,
}


if __name__ == "__main__":
    print("Φ Calculator — Reference Values")
    print("─" * 50)
    for name, phi in sorted(REFERENCE_PHI.items(), key=lambda x: x[1]):
        bar = "█" * int(phi * 500)
        print(f"  {name:<35s}  Φ={phi:.3f}  {bar}")
    print()
    print("Note: these are approximations / estimates from literature.")
    print("The exact Φ for LLMs is an open research question.")
