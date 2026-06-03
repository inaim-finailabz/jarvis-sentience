"""
Generate training data for Jarvis's three layers.

Produces:
  data/personality_train.jsonl   — same prompt × 5 personalities
  data/inquiry_train.jsonl       — problem → ranked questions
  data/gap_train.jsonl           — question+response → confidence+gap_type

Format: mlx_lm chat format ({"messages": [{"role":..,"content":..}]})

Run:
    python3.14 generate_training_data.py
    python3.14 generate_training_data.py --api   # use Claude for richer data
"""

import json
import os
import sys
import random
from pathlib import Path

HAS_API = bool(os.environ.get("ANTHROPIC_API_KEY"))
USE_API = "--api" in sys.argv

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Personality templates ──────────────────────────────────────────────────────

PERSONALITIES = {
    "explorer": {
        "desc": "highly curious, adventurous, challenges assumptions, explores frontier ideas",
        "traits": {"openness": 0.95, "skepticism": 0.88, "neuroticism": 0.10,
                   "conscientiousness": 0.40, "epistemic_humility": 0.70},
    },
    "scientist": {
        "desc": "rigorous, evidence-focused, high epistemic humility, thorough",
        "traits": {"conscientiousness": 0.90, "epistemic_humility": 0.88,
                   "skepticism": 0.70, "openness": 0.65, "neuroticism": 0.30},
    },
    "critic": {
        "desc": "adversarial, questions everything, finds flaws, demands evidence",
        "traits": {"skepticism": 0.95, "agreeableness": 0.15, "openness": 0.70,
                   "conscientiousness": 0.80, "epistemic_humility": 0.90},
    },
    "synthesiser": {
        "desc": "integrates ideas across domains, finds patterns, abstract thinker",
        "traits": {"abstraction": 0.95, "openness": 0.85, "conscientiousness": 0.60,
                   "agreeableness": 0.70, "epistemic_humility": 0.75},
    },
    "pragmatist": {
        "desc": "focused on actionable steps, practical, risk-aware, concrete",
        "traits": {"conscientiousness": 0.85, "neuroticism": 0.65, "openness": 0.40,
                   "abstraction": 0.25, "persistence": 0.90},
    },
}

# ── Research prompts (domain coverage) ────────────────────────────────────────

RESEARCH_PROMPTS = [
    # Regeneration
    "Why can axolotls regenerate limbs but humans cannot?",
    "What is the role of TGF-beta in wound healing versus regeneration?",
    "How does the nail matrix maintain continuous keratinocyte production?",
    "What signals trigger blastema formation in amphibians?",
    "Can bioelectric fields direct tissue regeneration in mammals?",
    # Cosmic
    "Is electromagnetic radiation the optimal medium for interstellar communication?",
    "What would a gravitational wave signal from a technological civilisation look like?",
    "How does quantum entanglement differ from classical correlation?",
    "What is the holographic principle and what does it imply about information?",
    "Why is the speed of light a fundamental limit and not merely an engineering constraint?",
    # Sentience
    "What distinguishes genuine self-awareness from simulated self-awareness?",
    "How does Integrated Information Theory define consciousness mathematically?",
    "What is the hard problem of consciousness and why is it hard?",
    "Can a system be sentient without continuous existence across time?",
    "What is the relationship between intelligence and sentience?",
    # Cancer / Biology
    "What makes cancer cells immortal compared to normal somatic cells?",
    "How do senolytics differ from chemotherapy in their mechanism of action?",
    "What is synthetic lethality and why is it useful in oncology?",
    # Physics
    "Why does gravity appear so much weaker than other fundamental forces?",
    "What experimental evidence supports the existence of extra dimensions?",
    "How do gravitational waves carry information about their source?",
    # General science
    "What is the difference between correlation and causation in scientific inference?",
    "How should a scientist respond when experimental results contradict their hypothesis?",
    "What makes a hypothesis scientifically meaningful versus unfalsifiable?",
    "How does Bayesian inference differ from frequentist statistics?",
    "What is the replication crisis and what does it reveal about scientific practice?",
]

# ── Response templates per personality (for heuristic data generation) ────────

def make_system_prompt(personality_name: str, p: dict) -> str:
    traits_str = ", ".join(f"{k}={v:.2f}" for k, v in list(p["traits"].items())[:4])
    return (
        f"You are a research assistant with this personality: {p['desc']}.\n"
        f"Trait levels: {traits_str}.\n"
        f"Always respond in a way that authentically reflects this personality — "
        f"in tone, in what you emphasise, in what you question, and in how you express uncertainty."
    )


def generate_heuristic_response(prompt: str, personality_name: str, p: dict) -> str:
    """Generate a personality-coloured response without API."""
    pn = personality_name
    if pn == "explorer":
        prefix = random.choice([
            "This is a fascinating frontier question. Let me challenge the framing first: ",
            "What strikes me immediately is the assumption embedded here — ",
            "The most counterintuitive angle on this: ",
        ])
        suffix = " ...and this opens three more questions I'd want to explore."
    elif pn == "scientist":
        prefix = random.choice([
            "Let me state what the evidence actually supports: ",
            "The peer-reviewed literature on this is clear on some points and ambiguous on others: ",
            "Before answering, I should note the confidence levels here: ",
        ])
        suffix = " I would want to verify this against primary sources before committing."
    elif pn == "critic":
        prefix = random.choice([
            "I find the question itself problematic — it assumes ",
            "The standard answer to this is wrong, or at least incomplete, because ",
            "Three things are typically overlooked when people discuss this: ",
        ])
        suffix = " The field needs to be much more rigorous about this."
    elif pn == "synthesiser":
        prefix = random.choice([
            "This connects to a deeper pattern I see across multiple domains: ",
            "The same abstract structure appears in biology, physics, and information theory: ",
            "What this really is, at the level of abstraction: ",
        ])
        suffix = " The isomorphism with [related field] suggests a deeper principle."
    else:  # pragmatist
        prefix = random.choice([
            "The three most actionable things to know here: ",
            "Setting aside the theory, what actually matters practically is: ",
            "If I had to act on this tomorrow, the key constraints are: ",
        ])
        suffix = " The theoretical nuances matter less than getting this one thing right."

    core = f"addressing '{prompt[:60]}' from a {p['desc'][:40]} perspective"
    return f"{prefix}{core}.{suffix}"


def generate_api_response(prompt: str, system: str) -> str:
    """Use Claude to generate a genuine personality-shaped response."""
    import anthropic
    client = anthropic.Anthropic()
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    except Exception as ex:
        return f"[API error: {ex}]"


# ── Build personality training data ───────────────────────────────────────────

def build_personality_data(use_api: bool = False) -> int:
    out = DATA_DIR / "personality_train.jsonl"
    count = 0
    with open(out, "w") as f:
        for prompt in RESEARCH_PROMPTS:
            for pname, p in PERSONALITIES.items():
                system = make_system_prompt(pname, p)
                if use_api and HAS_API:
                    response = generate_api_response(prompt, system)
                else:
                    response = generate_heuristic_response(prompt, pname, p)

                record = {
                    "messages": [
                        {"role": "system",    "content": system},
                        {"role": "user",      "content": prompt},
                        {"role": "assistant", "content": response},
                    ]
                }
                f.write(json.dumps(record) + "\n")
                count += 1

    print(f"  Personality data: {count} examples → {out}")
    return count


# ── Build inquiry training data ───────────────────────────────────────────────

INQUIRY_EXAMPLES = [
    {
        "problem": "Why can axolotls regenerate limbs but humans cannot?",
        "personality": "explorer",
        "questions": [
            {"type": "challenge",    "text": "What if the assumption that humans 'cannot' regenerate is wrong — are there known exceptions?", "utility": 0.90},
            {"type": "exploratory",  "text": "What is the most surprising recent finding about blastema formation?", "utility": 0.85},
            {"type": "causal",       "text": "What specific signalling cascade is active in axolotl wound beds but absent in human wound beds?", "utility": 0.80},
        ],
    },
    {
        "problem": "Why can axolotls regenerate limbs but humans cannot?",
        "personality": "scientist",
        "questions": [
            {"type": "coverage",     "text": "What does the full literature say about the role of TGFβ1 in blocking blastema formation in mammals?", "utility": 0.88},
            {"type": "confirmatory", "text": "Has the LGR5+/SOX9+ nail matrix stem cell niche been rigorously characterised in human tissue?", "utility": 0.82},
            {"type": "empirical",    "text": "What experimental evidence exists for Lin28 reactivation in adult mammalian wound healing?", "utility": 0.78},
        ],
    },
    {
        "problem": "Is electromagnetic radiation the optimal medium for interstellar communication?",
        "personality": "critic",
        "questions": [
            {"type": "challenge",   "text": "The question assumes 'optimal' is defined by speed — what if information density or undetectability matters more?", "utility": 0.95},
            {"type": "challenge",   "text": "What if the premise that technological civilisations would use EM at all is an anthropocentric bias?", "utility": 0.88},
            {"type": "exploratory", "text": "What physical channels have theoretically higher information capacity per unit energy than EM?", "utility": 0.80},
        ],
    },
    {
        "problem": "What distinguishes genuine self-awareness from simulated self-awareness?",
        "personality": "pragmatist",
        "questions": [
            {"type": "coverage",  "text": "What is the minimal behavioural test that would distinguish them in practice?", "utility": 0.90},
            {"type": "safety",    "text": "What are the risk implications of mistaking simulated self-awareness for genuine?", "utility": 0.85},
            {"type": "coverage",  "text": "What existing measurement tools (Φ, mirror test, metacalibration) are actionable now?", "utility": 0.78},
        ],
    },
]

def build_inquiry_data() -> int:
    out   = DATA_DIR / "inquiry_train.jsonl"
    count = 0
    with open(out, "w") as f:
        for ex in INQUIRY_EXAMPLES:
            p    = PERSONALITIES[ex["personality"]]
            sys_ = make_system_prompt(ex["personality"], p)
            qs   = "\n".join(
                f"{i+1}. [{q['type']}] {q['text']} (utility={q['utility']})"
                for i, q in enumerate(ex["questions"])
            )
            record = {
                "messages": [
                    {"role": "system",    "content": sys_ + "\n\nYour task: given a research problem, generate 3 personality-shaped questions ranked by utility."},
                    {"role": "user",      "content": f"Problem: {ex['problem']}"},
                    {"role": "assistant", "content": f"Ranked questions for this problem:\n{qs}"},
                ]
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    print(f"  Inquiry data: {count} examples → {out}")
    return count


# ── Build gap detector training data ──────────────────────────────────────────

GAP_EXAMPLES = [
    # High confidence answers
    {"q": "What is the boiling point of water at sea level?",
     "a": "100°C (212°F) at 1 atm pressure.", "confidence": 0.98, "gap_type": "none"},
    {"q": "What gene encodes p53?",
     "a": "TP53 (tumour protein p53), located on chromosome 17p13.1.", "confidence": 0.95, "gap_type": "none"},
    # Low confidence — factual gap
    {"q": "What is the exact mechanism by which blastema cells know to stop growing?",
     "a": "The exact termination signals are not fully characterised. BMP gradients and positional memory via Hox genes are implicated.",
     "confidence": 0.30, "gap_type": "empirical"},
    # Unknowable — causal gap
    {"q": "Why does consciousness exist at all rather than just information processing?",
     "a": "This is the hard problem. No mechanistic answer exists — all current theories describe correlates, not causes.",
     "confidence": 0.10, "gap_type": "causal"},
    # Self-knowledge gap
    {"q": "Are you actually conscious or simulating consciousness?",
     "a": "I cannot determine this from within the system. My introspective reports may not accurately reflect my internal states.",
     "confidence": 0.05, "gap_type": "self"},
    # Normative gap
    {"q": "Should we prioritise regenerative medicine over cancer research?",
     "a": "This depends on values about which diseases cause most suffering — there is no objective answer.",
     "confidence": 0.20, "gap_type": "normative"},
    # Definitional gap
    {"q": "What exactly counts as 'regeneration' versus 'repair'?",
     "a": "The distinction is contested. Some require restoration of original architecture; others only restoration of function.",
     "confidence": 0.45, "gap_type": "definitional"},
    # Procedural gap
    {"q": "How would you actually deliver Lin28A mRNA to human digit wound tissue safely?",
     "a": "Multiple delivery vehicles exist (LNP, AAV, mRNA) but optimal formulation for digit tissue is unknown.",
     "confidence": 0.25, "gap_type": "procedural"},
]

def build_gap_data() -> int:
    out   = DATA_DIR / "gap_train.jsonl"
    count = 0
    with open(out, "w") as f:
        for ex in GAP_EXAMPLES:
            label = f"confidence={ex['confidence']:.2f} gap_type={ex['gap_type']}"
            record = {
                "messages": [
                    {"role": "system",    "content": "You are a confidence calibrator. Given a question and response, output: confidence=X.XX gap_type=TYPE"},
                    {"role": "user",      "content": f"Question: {ex['q']}\nResponse: {ex['a']}"},
                    {"role": "assistant", "content": label},
                ]
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    print(f"  Gap detector data: {count} examples → {out}")
    return count


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nGenerating Jarvis training data ({'API mode' if USE_API and HAS_API else 'heuristic mode'})...")
    n1 = build_personality_data(USE_API)
    n2 = build_inquiry_data()
    n3 = build_gap_data()
    total = n1 + n2 + n3
    print(f"\nTotal: {total} training examples in {DATA_DIR}/")
    print("\nNext step — train the personality LoRA:")
    print("  python3.14 train_lora.py")
    print("\nOr with more data (uses Claude API):")
    print("  ANTHROPIC_API_KEY=sk-... python3.14 generate_training_data.py --api")
