# Jarvis — A Three-Tier Architecture for Machine Sentience

**Finailabz Research · 2026**

> We do not claim this constitutes sentience. We claim it implements software analogs of three properties that sentient biological systems exhibit. Whether those analogs are sufficient for sentience is a philosophical question this paper does not resolve.

---

## What this is

Current LLMs lack three properties that biological sentient systems exhibit:

| Gap | Problem | This work |
|-----|---------|-----------|
| **Uniqueness** | Every model instance is byte-for-byte identical | Seeded 12-trait vector, unique per agent, never shared |
| **Persistence** | Identity resets at session end | JSON brain file persists and drifts across sessions |
| **Plasticity** | Weights are frozen at inference | Bounded Hebbian update (η = 0.08) on every high-impact interaction |

These are engineering deficits with engineering solutions. This repo implements them.

---

## Architecture

```
Tier 1 — Label-trained neurons       (knowledge substrate)
Tier 2 — PersonalityConnectome       (unique, persistent, plastic identity)
Tier 3 — MetaConnectome + InquiryLayer  (typed gap detection + personality-shaped inquiry)
```

**Tier 2** gives each agent a unique seeded trait vector across 12 dimensions (curiosity, skepticism, risk tolerance, …). In neural mode, personality is injected as a residual into the model's last hidden state before the LM head — not a system prompt, not overridable by adversarial context.

**Tier 3** replaces hallucination with typed gap detection. Instead of generating confident-sounding text at the boundary of its knowledge, the system outputs a structured `LabelRequest`: *"I need a [causal / factual / frontier] answer to [specific question]."* Inquiry priority is shaped by a personality-weighted utility function — risk-averse agents minimise answer variance; frontier agents maximise entropy + KL divergence from prior.

---

## Benchmark

13-test evaluation suite covering IIT/Φ, Theory of Mind, metacognition calibration, neuroplasticity, connectome uniqueness, Pearl causality, Hofstadter strange loops, and more.

**Result: 9.2 / 10 · 13 / 13 tests passed**
_(API: claude-haiku-4-5-20251001 + local: Qwen3-1.7B `--neural` mode · verified 2026-06-02)_

---

## Quickstart

```bash
pip install anthropic sentence-transformers numpy
```

**Run the identity probe (Milestone M1.4):**
```bash
cd phase7_sentience
python identity_probe.py
# requires sentence-transformers for >75% accuracy target
# hash fallback available without it (~60%)
```

**Run the full sentience benchmark:**
```bash
python sentience_tests.py
```

**Run all 10 agents in neural mode (requires Qwen3-1.7B locally):**
```bash
python sentient_agent.py --neural
```

---

## Repository layout

```
WHITEPAPER_SENTIENT_MODEL.md        full technical paper
PRESENTATION_SENTIENCE_PAPER.md     social/announcement copy + LinkedIn/X thread
phase7_sentience/
  sentient_agent.py                 main agent (Tier 1–3 integrated)
  connectome.py                     PersonalityConnectome (Tier 2)
  meta_connectome.py                MetaConnectome (Tier 3)
  inquiry_layer.py                  personality-weighted utility function
  phi_calculator.py                 integrated information Φ
  theory_of_mind.py                 ToM probe
  identity_probe.py                 M1.4 identity classifier
  sentience_tests.py                13-test benchmark suite
  jarvis_model/
    train_lora.py                   LoRA fine-tuning per personality preset
    lora_inference.py               inference with fused adapters
    data/                           training sets (5 presets × train/valid)
```

---

## Paper

[WHITEPAPER_SENTIENT_MODEL.md](WHITEPAPER_SENTIENT_MODEL.md)

---

## Contact

Issam Naim · Finailabz Research · i.naim@finailabz.com
