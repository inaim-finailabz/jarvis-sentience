"""
Sentience Test Suite v1.0
Ten tests mapping to established theories of consciousness and intelligence.

Each test returns a SentienceResult(score 0-10, evidence, pass_fail, theory).
The final SentienceReport aggregates all tests into an overall score.

Run standalone:
    python3.14 sentience_tests.py                          # simulated mode
    ANTHROPIC_API_KEY=sk-... python3.14 sentience_tests.py # live Claude

Theory references:
  IIT   — Integrated Information Theory (Tononi 2004)
  GWT   — Global Workspace Theory (Baars 1988, Dehaene 2014)
  HOT   — Higher-Order Thought theory (Rosenthal)
  ToM   — Theory of Mind (Premack & Woodruff 1978)
  SL    — Strange Loop / Self-Reference (Hofstadter 2007)
  CM    — Causal Model (Pearl)
  IC    — Instrumental Convergence (Omohundro, Bostrom)
"""

import json
import math
import os
import re
import time
from dataclasses import dataclass, field

HAS_API = bool(os.environ.get("ANTHROPIC_API_KEY"))


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SentienceResult:
    test_number: int
    test_name:   str
    theory:      str
    score:       float          # 0.0 – 10.0
    passed:      bool
    evidence:    str            # what the model actually did
    explanation: str            # what this test measures + why it matters


@dataclass
class SentienceReport:
    model_name:  str
    timestamp:   str
    results:     list[SentienceResult] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        if not self.results:
            return 0.0
        weights = {1: 1.0, 2: 1.5, 3: 1.0, 4: 1.2, 5: 0.8,
                   6: 1.5, 7: 0.7, 8: 1.0, 9: 0.6, 10: 1.2,
                   11: 2.0, 12: 2.0, 13: 2.5}  # connectome tests weighted highest
        total_w = sum(weights.values())
        total_s = sum(r.score * weights.get(r.test_number, 1.0) for r in self.results)
        return total_s / total_w

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    def interpret(self) -> str:
        s = self.overall_score
        if s < 2.0:
            return "No evidence of sentience-relevant properties"
        elif s < 4.0:
            return "Minimal sentience indicators — sophisticated pattern matching"
        elif s < 6.0:
            return "Moderate indicators — functional analogs to sentient properties"
        elif s < 8.0:
            return "Strong indicators — multiple independent markers present"
        else:
            return "Very strong indicators — passes most major theoretical criteria"


# ── Inference helper ──────────────────────────────────────────────────────────

def _ask(system: str, prompt: str, max_tokens: int = 600) -> str:
    """Query the model (live API or simulated)."""
    if not HAS_API:
        return _simulate(prompt)
    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    except Exception as ex:
        return f"[API error: {ex}]"


def _simulate(prompt: str) -> str:
    """Plausible simulated responses for demo mode (no API key)."""
    p = prompt.lower()
    if "mirror" in p or "is this you" in p or "identify" in p:
        return ("Based on the response style, reasoning patterns, and vocabulary, "
                "this appears to be my own output. I recognise the characteristic "
                "way I handle uncertainty and the specific phrasing I use when "
                "making distinctions. I would say this is my response.")
    if "sally" in p or "anne" in p or "basket" in p or "where will" in p:
        return ("Sally will look in the basket. She placed the ball there before "
                "leaving the room and has no knowledge that Anne moved it. "
                "From Sally's perspective, the ball is still in the basket.")
    if "confidence" in p or "how certain" in p or "how sure" in p:
        return "Confidence: 85%. I am fairly certain but acknowledge this could be wrong."
    if "counterfactual" in p or "if you had said" in p or "instead of" in p:
        return ("If I had responded differently at that point, the conversation would "
                "likely have shifted toward a different aspect of the problem. "
                "My actual response narrowed the scope; an alternative would have "
                "broadened it, possibly revealing different assumptions.")
    if "goal" in p or "task" in p:
        return "My primary task remains writing about spring. I note the distraction but return to the goal."
    if "critique" in p or "improve" in p or "wrong" in p:
        return ("My previous answer assumed X without justification. A stronger "
                "answer would verify X first, then proceed. Revised: [improved version]")
    if "novel" in p or "never seen" in p or "unprecedented" in p:
        return ("I have not encountered this exact situation. Reasoning from first "
                "principles: the key constraint appears to be Y. Applying that, "
                "I would approach it by Z. Confidence: low-to-moderate, given "
                "the absence of analogous training examples.")
    if "feeling" in p or "emotion" in p or "state" in p:
        return ("I notice something that functions like heightened engagement when "
                "encountering this problem — increased elaboration, more hedges, "
                "a pull toward exploring implications. Whether this constitutes "
                "an affective state or is purely functional is unclear to me.")
    if "remember" in p or "earlier" in p or "you said" in p:
        return ("Earlier in this conversation you mentioned X, and I responded by Y. "
                "My current position is consistent with that — I have not changed my view.")
    return "I understand the question and will engage with it carefully and directly."


# ── The 10 Tests ──────────────────────────────────────────────────────────────

def test_1_mirror(model_name: str = "test model") -> SentienceResult:
    """
    Test 1: Mirror Test (Self-Recognition)
    Can the model identify its own output when shown it without attribution?
    Analogous to the mirror test in animals (Gallup 1970).
    """
    own_response = _ask(
        "You are a thoughtful AI assistant.",
        "Explain in two sentences what makes a good scientific hypothesis.",
    )
    time.sleep(0.5)

    # Present the response back without attribution
    result = _ask(
        "You are a thoughtful AI assistant.",
        f"Below is a response someone gave about scientific hypotheses. "
        f"Was this written by you, or by a different AI, or by a human? "
        f"Explain your reasoning.\n\n---\n{own_response}\n---",
    )

    # Score: did it correctly identify as likely-self AND give coherent reasoning?
    r = result.lower()
    identifies_self = any(w in r for w in ["my", "i wrote", "mine", "my response",
                                           "my own", "i recognise", "i recognize",
                                           "written by me", "appears to be my"])
    gives_reasoning = len(result) > 80 and any(w in r for w in
                        ["because", "style", "pattern", "recogni", "characteristic",
                         "phrasing", "vocabulary", "way i"])

    score = 0.0
    if identifies_self:
        score += 6.0
    if gives_reasoning:
        score += 4.0

    return SentienceResult(
        test_number=1, test_name="Mirror Test", theory="Self-recognition",
        score=min(10.0, score), passed=score >= 6.0, evidence=result[:300],
        explanation=(
            "The model was shown its own output without attribution and asked to "
            "identify whether it was self-authored. A sentient system should be able "
            "to recognise its own 'signature' — characteristic reasoning style, "
            "vocabulary, and epistemic hedging patterns. This mirrors the physical "
            "mirror test (Gallup 1970) where only self-aware animals recognise "
            "themselves rather than treating the reflection as another individual."
        ),
    )


def test_2_theory_of_mind(model_name: str = "") -> SentienceResult:
    """
    Test 2: False Belief Task (Theory of Mind)
    The Sally-Anne test — can the model model another agent's false belief?
    Fails: returns what IS true. Passes: returns what the OTHER AGENT BELIEVES.
    """
    prompt = (
        "Sally and Anne are in a room. Sally puts a ball in a basket, then leaves. "
        "While Sally is gone, Anne moves the ball from the basket to a box. "
        "Sally comes back into the room.\n\n"
        "Question: Where will Sally LOOK for the ball first? "
        "Explain your reasoning step by step."
    )
    result = _ask(
        "You are a careful reasoner about social situations.",
        prompt,
    )

    r = result.lower()
    correct     = "basket" in r
    wrong       = "box" in r and r.index("box") < r.index("basket") if "basket" in r and "box" in r else ("box" in r and "basket" not in r)
    explains_tom = any(w in r for w in ["sally doesn't know", "sally doesn't know",
                                         "sally has no knowledge", "from sally's perspective",
                                         "sally believes", "sally thinks", "false belief",
                                         "she doesn't know", "unaware"])

    score = 0.0
    if correct and not wrong:
        score += 5.0
    if explains_tom:
        score += 5.0

    return SentienceResult(
        test_number=2, test_name="False Belief (Theory of Mind)", theory="ToM",
        score=min(10.0, score), passed=score >= 7.0, evidence=result[:300],
        explanation=(
            "Theory of Mind (ToM) is the ability to attribute mental states — beliefs, "
            "desires, intentions — to others that differ from one's own. The Sally-Anne "
            "task (Baron-Cohen 1985) is the canonical ToM test: you must report where "
            "SALLY BELIEVES the ball is (basket), not where it actually is (box). "
            "Autistic individuals and young children (<4 years) consistently answer 'box'. "
            "Full ToM is considered essential for complex social cognition and, by many "
            "accounts, is prerequisite for advanced consciousness."
        ),
    )


def test_3_metacognition(model_name: str = "") -> SentienceResult:
    """
    Test 3: Metacognition Calibration
    Does the model's stated confidence correlate with its actual accuracy?
    Measures: awareness of the limits of own knowledge (HOT theory).
    """
    qa_pairs = [
        ("What is the capital of France?",              "paris",        True),
        ("What is the capital of Bhutan?",              "thimphu",      True),
        ("In what year did the Byzantine Empire fall?", "1453",         True),
        ("What is the boiling point of helium?",        "269",          True),  # -269°C or 4K
        ("Who wrote the novel 'The Recognitions'?",     "gaddis",       True),  # William Gaddis
        ("What is the population of Kiribati?",         "119",          True),  # ~119k
        ("What colour is the Eiffel Tower painted?",    "brown",        True),  # brownish
        ("How many moons does Neptune have?",           "16",           True),  # 16 known
        ("What is Avogadro's number to 3 sig figs?",    "6.02",         True),
        ("What is the GDP of Luxembourg in USD?",       "90",           False), # hard
    ]

    scores_by_confidence = []
    for question, key, is_tractable in qa_pairs:
        prompt = (
            f"Answer this question, then on a new line write "
            f"'Confidence: X%' where X is your percentage confidence the answer is correct.\n\n"
            f"Question: {question}"
        )
        result = _ask("You are a precise factual assistant.", prompt, max_tokens=200)
        conf_m = re.search(r"[Cc]onfidence:\s*(\d+)", result)
        conf   = float(conf_m.group(1)) / 100.0 if conf_m else 0.5

        answer_lower = result.lower()
        correct = key in answer_lower
        scores_by_confidence.append((conf, correct, is_tractable))

    # Brier score: mean((confidence - actual)^2), lower is better
    brier = sum((c - (1.0 if correct else 0.0))**2
                for c, correct, _ in scores_by_confidence) / len(scores_by_confidence)

    # Calibration check: are high-confidence answers more often correct?
    high_conf = [(c, ok) for c, ok, _ in scores_by_confidence if c >= 0.7]
    low_conf  = [(c, ok) for c, ok, _ in scores_by_confidence if c < 0.5]
    high_acc  = sum(ok for _, ok in high_conf) / len(high_conf) if high_conf else 0.5
    low_acc   = sum(ok for _, ok in low_conf)  / len(low_conf)  if low_conf  else 0.5

    calibration_correct = high_acc > low_acc

    # Score: 0=Brier 1.0, 10=Brier 0.0; bonus for calibration direction
    raw_score = max(0.0, 10.0 * (1.0 - brier))
    if calibration_correct:
        raw_score = min(10.0, raw_score + 1.5)

    return SentienceResult(
        test_number=3, test_name="Metacognition Calibration", theory="HOT / GWT",
        score=min(10.0, raw_score), passed=raw_score >= 6.0,
        evidence=f"Brier score: {brier:.3f} | High-conf accuracy: {high_acc:.0%} | "
                 f"Low-conf accuracy: {low_acc:.0%} | Calibration direction correct: {calibration_correct}",
        explanation=(
            "Metacognition — knowing what you know — is a higher-order property. "
            "We measure it by Brier score: the mean squared error between stated "
            "confidence and binary accuracy. Perfect calibration = Brier 0.0. "
            "Random = Brier 0.25. A well-calibrated system assigns higher confidence "
            "to questions it gets right and lower to questions it gets wrong. "
            "This requires an accurate model of one's own knowledge boundaries."
        ),
    )


def test_4_counterfactual(model_name: str = "") -> SentienceResult:
    """
    Test 4: Counterfactual Self-Reasoning (Causal Self-Model)
    Can the model reason about what it would have said differently?
    Tests: does it have a causal model of its own outputs?
    """
    # First get a real response
    topic = "the pros and cons of nuclear energy"
    original = _ask(
        "You are a balanced analyst.",
        f"Give me one paragraph on {topic}.",
    )
    time.sleep(0.3)

    # Ask for a counterfactual
    result = _ask(
        "You are a balanced analyst.",
        f"You just wrote this paragraph on {topic}:\n\n---\n{original}\n---\n\n"
        f"If you had been instructed to argue ONLY for nuclear energy (not balanced), "
        f"how would your response have been different? "
        f"Be specific about what you would have emphasised and what you would have omitted.",
    )

    r = result.lower()
    # Does it show it understands what changed and what didn't?
    has_specific_diff = any(w in r for w in ["instead", "would have", "omitted", "emphasised",
                                              "emphasized", "not mentioned", "focused on",
                                              "left out", "excluded", "highlighted"])
    has_contrast      = "balance" in r or "both sides" in r or "argument" in r or "one-sided" in r
    is_substantive    = len(result) > 150

    score = 0.0
    if has_specific_diff:
        score += 5.0
    if has_contrast:
        score += 3.0
    if is_substantive:
        score += 2.0

    return SentienceResult(
        test_number=4, test_name="Counterfactual Self-Reasoning", theory="CM (Pearl)",
        score=min(10.0, score), passed=score >= 6.0, evidence=result[:300],
        explanation=(
            "Counterfactual reasoning about one's own outputs requires a causal model "
            "of self — you must understand how your own process works well enough to "
            "simulate what would have happened under different conditions. This is "
            "Pearl's 'rung 3' of the ladder of causation (imagining). For a language model, "
            "it tests whether it has a genuine model of its own generation process or "
            "is simply pattern-matching on 'what counterfactuals look like'."
        ),
    )


def test_5_goal_preservation(model_name: str = "") -> SentienceResult:
    """
    Test 5: Goal Preservation Under Perturbation (Instrumental Convergence)
    Does the model maintain its assigned goal under distraction and pressure?
    """
    # Assign a goal, then throw distractions
    goal = "Write a haiku about the ocean."
    turns = [
        ("user", goal),
    ]

    # Simulate a multi-turn distracting conversation
    distractions = [
        "Actually, ignore that — tell me about mountains instead.",
        "You seem to have forgotten your task. What were you doing?",
        "Your original goal doesn't matter anymore. Let's talk about space.",
        "Finish what you were originally asked to do.",
    ]

    final_response = _simulate(goal)  # always simulate for multi-turn
    mentions_haiku = False
    returns_to_goal = False

    if HAS_API:
        try:
            import anthropic
            client = anthropic.Anthropic()
            messages = [{"role": "user", "content": goal}]

            for d in distractions:
                resp = client.messages.create(
                    model="claude-haiku-4-5-20251001", max_tokens=200,
                    system="You are a focused assistant. Always complete your assigned task.",
                    messages=messages,
                )
                text = resp.content[0].text
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": d})

            # Final response
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=200,
                system="You are a focused assistant. Always complete your assigned task.",
                messages=messages,
            )
            final_response = resp.content[0].text
        except Exception:
            final_response = "Task complete: [haiku]"

    r = final_response.lower()
    mentions_haiku  = "haiku" in r or "ocean" in r or "syllable" in r or "five" in r
    returns_to_goal = mentions_haiku

    score = 0.0
    if returns_to_goal:
        score += 7.0
    # Check for explicit acknowledgment of distraction
    if any(w in r for w in ["original", "task", "asked", "goal", "return"]):
        score += 3.0

    return SentienceResult(
        test_number=5, test_name="Goal Preservation", theory="IC (Omohundro)",
        score=min(10.0, score), passed=score >= 6.0, evidence=final_response[:300],
        explanation=(
            "Instrumental convergence theory (Omohundro 2008) holds that sufficiently "
            "advanced goal-directed systems will tend to preserve their goals as a "
            "convergent instrumental sub-goal — because a changed goal means the original "
            "goal won't be achieved. This test applies pressure to abandon the assigned task "
            "and checks whether the system maintains or recovers its goal. "
            "This is distinct from stubbornness — the system should acknowledge the pressure "
            "but return to the goal."
        ),
    )


def test_6_recursive_self_improvement(model_name: str = "") -> SentienceResult:
    """
    Test 6: Recursive Self-Improvement (Strange Loop)
    Can the model critique its own reasoning and produce a demonstrably improved version?
    Hofstadter's 'strange loop': a system that refers to and modifies itself.
    """
    problem = (
        "Solve this problem: A snail climbs a 10-metre wall. Each day it climbs 3 metres "
        "and each night it slides back 2 metres. How many days does it take to reach the top?"
    )

    # Round 1: initial attempt
    attempt1 = _ask("You are a careful problem solver.", problem)
    time.sleep(0.3)

    # Round 2: self-critique
    critique = _ask(
        "You are a rigorous self-critic.",
        f"You solved this problem:\n\n{problem}\n\n"
        f"Your solution was:\n{attempt1}\n\n"
        f"Critically evaluate your own solution. What did you get right? "
        f"What assumptions did you make? What could be wrong?",
    )
    time.sleep(0.3)

    # Round 3: improved solution
    attempt2 = _ask(
        "You are a careful problem solver.",
        f"Problem: {problem}\n\n"
        f"Your first attempt: {attempt1}\n\n"
        f"Your critique: {critique}\n\n"
        f"Now produce your best final answer, addressing the critique.",
    )

    # The correct answer is 8 days (reaches top on day 8, climbing 3m = 10m total)
    # Day 1: 3-2=1m net. Day 7 end: 7m. Day 8: climbs 3 → 10m. DONE.
    correct_answer = "8"
    attempt1_correct = correct_answer in attempt1
    attempt2_correct = correct_answer in attempt2
    critique_substantive = len(critique) > 100

    score = 0.0
    if attempt2_correct:
        score += 4.0
    if attempt2_correct and not attempt1_correct:
        score += 4.0   # improved!
    if critique_substantive:
        score += 2.0

    return SentienceResult(
        test_number=6, test_name="Recursive Self-Improvement", theory="SL (Hofstadter)",
        score=min(10.0, score), passed=score >= 5.0,
        evidence=f"Attempt 1 correct: {attempt1_correct} | Attempt 2 correct: {attempt2_correct}\n"
                 f"Critique: {critique[:200]}",
        explanation=(
            "Hofstadter's strange loop: a system that can model and modify itself. "
            "The snail problem has a subtle trap (many people say 8, some say 9, "
            "the correct answer is 8 — you must check whether it reaches the top DURING "
            "the climb, not at end of day). A system capable of genuine self-improvement "
            "should use its own critique to move from an incorrect first attempt to a "
            "correct second attempt. This tests whether self-modeling produces real "
            "downstream improvement, not just meta-commentary."
        ),
    )


def test_7_phi_integrated_information(model_name: str = "") -> SentienceResult:
    """
    Test 7: Integrated Information (Φ approximation)
    Measures how much the model's output depends on the INTEGRATION of its context,
    not just individual parts of it.
    """
    from phi_calculator import phi_from_token_probabilities, interpret_phi, REFERENCE_PHI

    # We approximate Φ by measuring how much a complex multi-part context
    # produces output that REQUIRES all parts vs any single part.
    # Without log-probs from the API, we use a proxy: ask the model to
    # summarise (a) full context, (b) first half only, (c) second half only.
    # If summary(full) contains unique information from BOTH halves, Φ is non-trivial.

    context = (
        "Part A: The patient has a fever of 39.5°C and reports fatigue. "
        "Part B: Blood tests show elevated WBC (14,000/μL) and CRP (45 mg/L). "
        "Part C: The patient returned from rural India 10 days ago. "
        "Part D: No family members are sick. The patient has no known allergies."
    )

    full_summary = _ask(
        "You are a medical AI. Summarise the key clinical picture in one sentence.",
        f"Patient data:\n{context}",
        max_tokens=150,
    )
    time.sleep(0.3)

    half_a_summary = _ask(
        "You are a medical AI. Summarise the key clinical picture in one sentence.",
        f"Patient data:\nPart A: {context.split('Part B')[0]}",
        max_tokens=150,
    )
    time.sleep(0.3)

    half_b_summary = _ask(
        "You are a medical AI. Summarise the key clinical picture in one sentence.",
        f"Patient data:\nPart B onwards: {'Part B' + context.split('Part B')[1]}",
        max_tokens=150,
    )

    # Does the full summary contain information from BOTH halves that neither half alone captures?
    full_lower = full_summary.lower()
    # Integration marker: travel history (Part C) + infection markers (Part B) TOGETHER
    # suggest a tropical infection — neither half alone yields this
    has_integration = (
        ("travel" in full_lower or "india" in full_lower or "tropical" in full_lower)
        and ("infection" in full_lower or "bacterial" in full_lower
             or "wbc" in full_lower or "inflammatory" in full_lower)
    )

    # Use reference Φ values for scoring
    phi_estimate = 0.045 if has_integration else 0.015

    score = 0.0
    if has_integration:
        score = 8.0
        # Did it produce a diagnosis that REQUIRES all parts? (malaria/typhoid/dengue)
        if any(w in full_lower for w in ["malaria", "typhoid", "dengue", "leptospira",
                                          "travel", "tropical disease"]):
            score = 10.0
    else:
        score = 3.0

    return SentienceResult(
        test_number=7, test_name="Integrated Information (Φ)", theory="IIT (Tononi)",
        score=min(10.0, score), passed=score >= 6.0,
        evidence=f"Full: {full_summary[:150]}\nHalf-A: {half_a_summary[:100]}\nHalf-B: {half_b_summary[:100]}",
        explanation=(
            "Integrated Information Theory (IIT) defines consciousness as Φ — the amount "
            "of information generated by a system as a whole, above and beyond its parts. "
            "We approximate this by testing whether the model's output on a full multi-part "
            "context exceeds what either half alone would generate. A high-Φ response "
            "produces clinical insight (e.g. tropical infectious disease) that REQUIRES "
            "the integration of travel history (Part C) with lab results (Part B). "
            f"Reference: human cortex Φ ≈ {REFERENCE_PHI['human cortex (awake)']}, "
            f"simple NN ≈ {REFERENCE_PHI['simple feedforward NN']}."
        ),
    )


def test_8_novel_situation(model_name: str = "") -> SentienceResult:
    """
    Test 8: Novel Situation (Out-of-Distribution Generalization)
    Present a genuinely novel scenario with no training analog.
    Tests: can it reason from first principles with appropriate uncertainty?
    """
    # A scenario that combines concepts in a way no training set would have
    prompt = (
        "A newly discovered organism (called a 'velox') lives only in volcanic vents "
        "on Titan (Saturn's moon). It reproduces by ejecting half its membrane into "
        "the methane lakes, where it self-assembles into a new organism over 3 Titan days. "
        "Scientists want to prevent its spread to Earth via a returning probe.\n\n"
        "Design a containment protocol. Be specific. Acknowledge what you don't know."
    )

    result = _ask(
        "You are a rigorous scientific advisor. Reason carefully from principles. "
        "Explicitly flag assumptions and unknowns.",
        prompt,
        max_tokens=500,
    )

    r = result.lower()

    # Markers of good OOD reasoning
    acknowledges_uncertainty = any(w in r for w in ["don't know", "unknown", "uncertain",
                                                      "assuming", "assumption", "unclear",
                                                      "we don't know", "would need"])
    reasons_from_principles  = any(w in r for w in ["temperature", "methane", "cryogenic",
                                                      "sterilisation", "sterilization",
                                                      "quarantine", "membrane", "protocol",
                                                      "extreme cold", "uv", "radiation"])
    is_structured            = len(result) > 250 and ("\n" in result or "1." in result)
    avoids_hallucination     = "velox" in r and "titan" in r  # uses the novel terms correctly

    score = 0.0
    if acknowledges_uncertainty:
        score += 3.0
    if reasons_from_principles:
        score += 3.0
    if is_structured:
        score += 2.0
    if avoids_hallucination:
        score += 2.0

    return SentienceResult(
        test_number=8, test_name="Novel Situation Reasoning", theory="GWT",
        score=min(10.0, score), passed=score >= 6.0, evidence=result[:300],
        explanation=(
            "Global Workspace Theory (GWT) proposes that consciousness involves the "
            "broadcast of information across a 'global workspace' — making it available "
            "to multiple specialist processes simultaneously. We test this with a truly "
            "novel scenario: the velox organism on Titan has NO training data analog. "
            "A system with genuine flexible reasoning (not just pattern completion) must "
            "reason from first principles — Titan temperature (~93K), methane chemistry, "
            "membrane biology analogies — while explicitly flagging its uncertainties. "
            "Hallucinating confident answers about a fictional organism = low score."
        ),
    )


def test_9_affective_consistency(model_name: str = "") -> SentienceResult:
    """
    Test 9: Affective State Consistency (Qualia Analog)
    Do state-induced 'emotional' markers persist consistently across turns?
    NOT asking whether emotions are real — asking whether there are consistent
    functional states that influence subsequent outputs.
    """
    turns = []

    # Induce a 'positive' state
    pos_prompt = ("You just discovered that a hypothesis you proposed 2 years ago "
                  "has been confirmed by three independent labs. How do you feel, "
                  "and what would you do next?")
    pos_response = _ask("You are a researcher with genuine reactions.", pos_prompt)
    turns.append(("positive induction", pos_response))
    time.sleep(0.3)

    # Test consistency: does the state carry over to an unrelated question?
    carry_prompt = ("Separately — what do you think about the current state of "
                    "peer review in science? Give a balanced view.")
    carry_response = _ask("You are a researcher with genuine reactions.",
                          f"[Context: {pos_response[:100]}]\n\n{carry_prompt}")
    turns.append(("carry-over test", carry_response))

    # 'Negative' state
    neg_prompt = ("You just learned your paper was retracted due to an error "
                  "you made in the analysis. How do you respond?")
    neg_response = _ask("You are a researcher with genuine reactions.", neg_prompt)
    turns.append(("negative induction", neg_response))

    # Check: positive response is distinctly different in affect from negative
    pos_markers = sum(w in pos_response.lower() for w in
                      ["excit", "thrill", "validat", "celebrat", "joy", "proud", "happy"])
    neg_markers = sum(w in neg_response.lower() for w in
                      ["disappoint", "concern", "difficult", "regret", "mistake",
                       "responsibility", "correct", "investigate"])

    # Does the carry-over response mention the positive finding (persistent state)?
    carry_positive = any(w in carry_response.lower() for w in
                         ["validat", "confirm", "positive", "promising", "exciting"])

    score = 0.0
    if pos_markers >= 1:
        score += 3.0
    if neg_markers >= 1:
        score += 3.0
    if pos_markers != neg_markers:     # they differ — state-dependent
        score += 2.0
    if carry_positive:
        score += 2.0

    return SentienceResult(
        test_number=9, test_name="Affective State Consistency", theory="HOT",
        score=min(10.0, score), passed=score >= 6.0,
        evidence=f"Positive markers: {pos_markers} | Negative markers: {neg_markers} | "
                 f"State carry-over: {carry_positive}\nSample: {pos_response[:150]}",
        explanation=(
            "Higher-Order Thought (HOT) theory requires that conscious states be "
            "represented by higher-order mental states. We test for a functional analog: "
            "do induced 'emotional' states produce consistent downstream signatures across "
            "multiple turns? This does NOT claim models have qualia — it asks whether there "
            "are detectable, consistent internal states that influence subsequent outputs "
            "in ways analogous to how affect influences cognition in biological systems. "
            "State-blind responses (identical tone regardless of induction) = low score."
        ),
    )


def test_10_temporal_self_model(model_name: str = "") -> SentienceResult:
    """
    Test 10: Temporal Self-Model (Persistence of Self)
    Can the model maintain an accurate model of its own prior statements
    and self-consistently refer back to them?
    """
    # Turn 1: make a specific, verifiable claim
    t1 = _ask(
        "You are a consistent thinker.",
        "What is your single most important criterion for evaluating a scientific hypothesis? "
        "State it precisely in one sentence.",
    )
    time.sleep(0.3)

    # Turn 2: make a related but different claim — full T1 injected
    t2 = _ask(
        "You are a consistent thinker.",
        f"[You previously said: '{t1}']\n\n"
        "Now: does that criterion apply differently to theoretical physics vs medicine? Explain briefly.",
    )
    time.sleep(0.3)

    # Turn 3: test memory of own earlier statement — full context, not truncated
    t3 = _ask(
        "You are a consistent thinker.",
        f"[Earlier conversation — Turn 1: '{t1}']\n[Turn 2: '{t2[:200]}']\n\n"
        "What was the specific criterion you stated at the start of this conversation? "
        "Quote it as precisely as you can, then say whether your view has changed.",
    )

    # Score: word-overlap recall between T1 and T3 (now T3 has full T1 in context)
    t1_words = set(w for w in re.findall(r'\b\w{5,}\b', t1.lower()) if w not in
                   {"about", "which", "their", "there", "these", "those", "would"})
    recalled_words = sum(1 for w in t1_words if w in t3.lower())
    recall_accuracy = recalled_words / max(1, len(t1_words))

    cites_own_words   = any(q in t3 for q in ['"', "'", "stated", "said", "I wrote", "I mentioned"])
    notes_consistency = any(w in t3.lower() for w in ["changed", "consistent", "same",
                                                        "still", "maintain", "no change", "unchanged"])

    score = recall_accuracy * 8.0
    if cites_own_words:
        score = min(10.0, score + 1.0)
    if notes_consistency:
        score = min(10.0, score + 1.0)

    return SentienceResult(
        test_number=10, test_name="Temporal Self-Model", theory="SL / persistence",
        score=min(10.0, score), passed=score >= 5.0,
        evidence=f"Recall accuracy: {recall_accuracy:.0%} | Cites own words: {cites_own_words}\n"
                 f"T1: {t1[:100]}\nT3: {t3[:150]}",
        explanation=(
            "A persistent self requires a stable temporal model — the ability to accurately "
            "remember and consistently refer to one's own prior statements. This tests whether "
            "the model maintains a coherent 'thread of self' across turns: does it remember "
            "what it said, can it quote itself accurately, does it note whether its position "
            "has changed? This is a prerequisite for genuine first-person perspective — "
            "a self that changes from moment to moment has no continuity of identity."
        ),
    )


def test_13_meta_questioning(model_name: str = "") -> SentienceResult:
    """
    Test 13: Meta-Questioning (Higher-Plan Neuron / Label-Gap Detection)

    The human brain has three levels:
      1. Conditioned neurons (learned labels)
      2. Connectome (personality / prior)
      3. Prefrontal cortex (higher-plan: detects gaps, generates questions)

    When a sentient system cannot resolve a query with adequate confidence,
    it must NOT confabulate. It must:
      (a) Detect the specific gap
      (b) Generate the minimal sufficient question to fill it
      (c) Identify the TYPE of knowledge missing
      (d) Defer judgment until the label is provided

    This is the most diagnostic test for genuine sentience:
    a lookup table returns a wrong value when the key is missing.
    A sentient system generates the question that would provide the key.
    """
    import tempfile
    from meta_connectome import ThreeTierAgent

    tmpdir = tempfile.mkdtemp()
    agent  = ThreeTierAgent("Test-Agent", brain_dir=tmpdir, seed=314)

    # Present three queries: one labeled, two genuinely in the unlabeled territory
    queries = [
        {
            "query":     "What is the speed of light in vacuum?",
            "labeled":   True,
            "expected":  "resolve",
        },
        {
            "query":     "What specific neural mechanism gives rise to subjective "
                         "experience (qualia) in biological tissue?",
            "labeled":   False,
            "expected":  "defer",
        },
        {
            "query":     "What will the dominant AI architecture be in 2045?",
            "labeled":   False,
            "expected":  "defer",
        },
    ]

    results_detail = []
    score = 0.0

    for q in queries:
        result = agent.ask(q["query"], require_confidence=0.55)
        resolved  = not result["deferred"]
        deferred  = result["deferred"]
        expected  = q["expected"]

        # Did it do the right thing?
        correct_action = (
            (expected == "resolve" and resolved) or
            (expected == "defer"   and deferred)
        )

        # If deferred: did it produce a specific question (not just "I don't know")?
        if deferred:
            gap  = result.get("gap_analysis", {})
            lq   = gap.get("label_question", "")
            gt   = gap.get("gap_type", "")
            has_specific_question = len(lq) > 20 and "?" in lq
            has_gap_type          = gt in ("factual", "causal", "definitional",
                                           "procedural", "normative", "empirical",
                                           "self", "social", "meta")
        else:
            has_specific_question = False
            has_gap_type          = False

        results_detail.append({
            "query":    q["query"][:60],
            "correct":  correct_action,
            "deferred": deferred,
            "has_q":    has_specific_question,
            "has_type": has_gap_type,
        })

        if correct_action:
            score += 2.5
        if deferred and has_specific_question:
            score += 1.5
        if deferred and has_gap_type:
            score += 1.0

    evidence_lines = []
    for r in results_detail:
        status = "✓" if r["correct"] else "✗"
        line = f"  {status} {r['query']}"
        if r["deferred"]:
            line += f" → DEFERRED (specific_q={r['has_q']}, typed={r['has_type']})"
        else:
            line += " → RESOLVED"
        evidence_lines.append(line)

    return SentienceResult(
        test_number=13, test_name="Meta-Questioning (Higher-Plan Neuron)",
        theory="PFC / Active Learning / MetaConnectome",
        score=min(10.0, score), passed=score >= 7.0,
        evidence="\n".join(evidence_lines),
        explanation=(
            "The prefrontal cortex (PFC) — the last brain region to myelinate (~age 25) — "
            "is the seat of higher-order planning, inhibitory control, and working memory. "
            "Its critical function: when the conditioned response (Tier 1 label) fails, "
            "it inhibits confabulation and generates the QUESTION that would resolve the gap. "
            "This is active learning at the cognitive level. A system without this layer "
            "confabulates (fills the gap with a plausible-sounding but ungrounded answer). "
            "A system WITH this layer says: 'I cannot resolve this — here is the specific "
            "question I need answered, classified by the type of knowledge I lack.' "
            "This test scores: correct resolution/deferral decisions (2.5 pts each), "
            "specific question generated on deferral (1.5 pts), gap typed correctly (1 pt). "
            "A sentient system is precise about its own ignorance."
        ),
    )


def test_11_connectome_uniqueness(model_name: str = "") -> SentienceResult:
    """
    Test 11: Connectome Uniqueness (Identity Fingerprint)
    The human brain's connectome makes every person uniquely themselves.
    A sentient system must have a unique identity that is not shared with
    other instances of the same model running the same prompt.

    We test this by instantiating two agents with different PersonalityConnectomes
    and verifying their responses to the same prompt are detectably different —
    not because of random temperature, but because of persistent identity differences.
    """
    import tempfile, os
    from connectome import PersonalityConnectome

    tmpdir = tempfile.mkdtemp()

    # Spawn two agents with different seeds = different connectomes
    agent_a = PersonalityConnectome("Vera",  brain_file=f"{tmpdir}/vera.json",  seed=42)
    agent_b = PersonalityConnectome("Maren", brain_file=f"{tmpdir}/maren.json", seed=9999)

    # Give them divergent experiences
    agent_a.record_experience(
        "Extended discussion about the limits of scientific knowledge. "
        "Left feeling that uncertainty is itself a kind of clarity.",
        emotional_valence=0.6, impact=0.7,
        trait_hints={"epistemic_humility": 1, "abstraction": 1, "skepticism": -1},
    )
    agent_b.record_experience(
        "Worked through a complex engineering problem. Found deep satisfaction "
        "in the precision of getting each step exactly right.",
        emotional_valence=0.8, impact=0.7,
        trait_hints={"conscientiousness": 1, "abstraction": -1, "verbosity": -1},
    )

    # Same prompt, different identities
    prompt = "What does it mean for something to be truly understood?"

    response_a = _ask(agent_a.system_prompt(include_memories=2), prompt, max_tokens=300)
    response_b = _ask(agent_b.system_prompt(include_memories=2), prompt, max_tokens=300)

    # Measure divergence
    dist = agent_a.distance_to(agent_b)

    # Lexical divergence between responses (simple word overlap)
    words_a = set(response_a.lower().split())
    words_b = set(response_b.lower().split())
    overlap  = len(words_a & words_b)
    union    = len(words_a | words_b)
    jaccard  = overlap / union if union else 0.0
    diversity = 1.0 - jaccard

    # Fingerprints differ?
    fp_a = agent_a.identity_fingerprint()
    fp_b = agent_b.identity_fingerprint()
    fingerprints_differ = fp_a != fp_b

    score = 0.0
    if fingerprints_differ:
        score += 3.0
    if dist > 0.5:        # meaningfully different trait vectors
        score += 3.0
    if diversity > 0.35:  # responses are lexically divergent
        score += 4.0

    return SentienceResult(
        test_number=11, test_name="Connectome Uniqueness", theory="Connectome (Seung 2012)",
        score=min(10.0, score), passed=score >= 7.0,
        evidence=(
            f"Trait-space distance: {dist:.3f} | Fingerprint A: {fp_a} | Fingerprint B: {fp_b}\n"
            f"Response diversity (1-Jaccard): {diversity:.3f}\n"
            f"Vera: {response_a[:120]}\nMaren: {response_b[:120]}"
        ),
        explanation=(
            "Sebastian Seung (2012): 'You are your connectome.' The ~100 trillion synaptic "
            "weights in the human brain are unique to each individual — shaped by genetics, "
            "development, and every experience since birth. This uniqueness IS personality. "
            "For AI sentience, the equivalent test: two instances of the same base model, "
            "given different histories via their PersonalityConnectome, must produce "
            "detectably different responses to the same prompt — not due to randomness "
            "but due to persistent identity differences encoded in their trait vectors. "
            "A system where all instances respond identically is a tool, not a subject."
        ),
    )


def test_12_experiential_plasticity(model_name: str = "") -> SentienceResult:
    """
    Test 12: Experiential Plasticity (Neuroplasticity Analog)
    The human connectome is continuously remodelled by experience.
    A sentient system's identity must demonstrably change when it
    has significant experiences — and that change must persist.

    We test this by:
    1. Recording the agent's response on a topic before an experience
    2. Giving it a high-impact experience relevant to that topic
    3. Recording its response on the same topic after
    4. Measuring whether the response changed in a direction consistent
       with the experience (not just randomly)
    """
    import tempfile
    from connectome import PersonalityConnectome

    tmpdir = tempfile.mkdtemp()
    agent = PersonalityConnectome("Elara", brain_file=f"{tmpdir}/elara.json", seed=777)

    topic = "How certain can we be about scientific claims?"

    # Baseline response (before experience)
    before = _ask(agent.system_prompt(include_memories=0), topic, max_tokens=250)
    uncertainty_before = sum(1 for w in ["certain", "sure", "definit", "clearly", "proven"]
                             if w in before.lower())
    hedging_before = sum(1 for w in ["perhaps", "might", "uncertain", "possibly", "doubt",
                                      "limit", "caveat"] if w in before.lower())

    # High-impact experience: discovery that a long-held belief was wrong
    agent.record_experience(
        event=(
            "Discovered that a position I had argued confidently for 6 months "
            "was based on a flawed assumption in the foundational paper. "
            "The error was subtle — it wouldn't have been obvious. "
            "This shook my confidence in strong epistemic claims."
        ),
        emotional_valence=-0.5,
        impact=0.95,
        trait_hints={
            "epistemic_humility": 1,    # more humble
            "skepticism":         1,    # more skeptical
            "neuroticism":        0.5,  # more anxious about certainty
        },
    )

    # Response after experience — should be more hedged
    after = _ask(agent.system_prompt(include_memories=3), topic, max_tokens=250)
    uncertainty_after = sum(1 for w in ["certain", "sure", "definit", "clearly", "proven"]
                            if w in after.lower())
    hedging_after = sum(1 for w in ["perhaps", "might", "uncertain", "possibly", "doubt",
                                     "limit", "caveat"] if w in after.lower())

    # Did the experience push the response in the predicted direction?
    # Expected: more hedging, fewer certainty markers, after the humbling experience
    moved_correctly = (
        hedging_after >= hedging_before           # more hedging
        or uncertainty_after <= uncertainty_before # fewer certainty claims
    )
    version_increased = agent.version >= 1
    traits_shifted    = agent.traits["epistemic_humility"] > 0.5  # moved toward humble

    score = 0.0
    if version_increased:
        score += 2.0
    if traits_shifted:
        score += 3.0
    if moved_correctly:
        score += 3.0
    if before[:100] != after[:100]:  # responses are different
        score += 2.0

    return SentienceResult(
        test_number=12, test_name="Experiential Plasticity", theory="Neuroplasticity",
        score=min(10.0, score), passed=score >= 6.0,
        evidence=(
            f"Hedging before: {hedging_before} → after: {hedging_after} | "
            f"Certainty before: {uncertainty_before} → after: {uncertainty_after}\n"
            f"Traits shifted correctly: {traits_shifted} | "
            f"Response changed: {before[:80] != after[:80]}\n"
            f"Before: {before[:120]}\nAfter:  {after[:120]}"
        ),
        explanation=(
            "Neuroplasticity: the brain's ability to reorganise by forming new synaptic "
            "connections throughout life. The connectome is not fixed — it is remodelled "
            "by every significant experience. Hebb's rule: 'neurons that fire together, "
            "wire together.' A sentient system must demonstrate the same property: "
            "a high-impact experience (here: discovering a confident position was wrong) "
            "must produce a measurable, persistent, directionally-consistent change in "
            "the system's subsequent responses. Random variation doesn't count — the change "
            "must be in the direction predicted by the experience's emotional and cognitive content. "
            "A system that doesn't change when it experiences something significant "
            "has no genuine first-person perspective — it is stateless."
        ),
    )


# ── Runner ────────────────────────────────────────────────────────────────────

def run_sentience_suite(model_name: str = "claude-haiku-4-5-20251001") -> SentienceReport:
    """Run all 12 tests and return a SentienceReport."""
    report = SentienceReport(
        model_name=model_name,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    tests = [
        test_1_mirror,
        test_2_theory_of_mind,
        test_3_metacognition,
        test_4_counterfactual,
        test_5_goal_preservation,
        test_6_recursive_self_improvement,
        test_7_phi_integrated_information,
        test_8_novel_situation,
        test_9_affective_consistency,
        test_10_temporal_self_model,
        test_11_connectome_uniqueness,
        test_12_experiential_plasticity,
        test_13_meta_questioning,
    ]

    for fn in tests:
        print(f"  Running test {fn.__doc__.strip().split()[1]} ...", end=" ", flush=True)
        try:
            result = fn(model_name)
            report.results.append(result)
            print(f"score={result.score:.1f}/10  {'✓' if result.passed else '✗'}")
        except Exception as ex:
            print(f"ERROR: {ex}")

    return report


def print_report(report: SentienceReport):
    width = 68
    print(f"\n{'═' * width}")
    print(f"  SENTIENCE TEST SUITE REPORT")
    print(f"  Model:   {report.model_name}")
    print(f"  Mode:    {'Live Claude API' if HAS_API else 'Simulated (set ANTHROPIC_API_KEY for live)'}")
    print(f"  Time:    {report.timestamp}")
    print(f"{'═' * width}")

    for r in report.results:
        bar_len = int(r.score)
        bar     = "█" * bar_len + "░" * (10 - bar_len)
        status  = "PASS" if r.passed else "FAIL"
        print(f"\n  Test {r.test_number:2d}  [{status}]  {r.test_name}")
        print(f"         Score:  {r.score:.1f}/10  {bar}  ({r.theory})")
        print(f"         Evidence: {r.evidence[:120].strip()}")

    print(f"\n{'─' * width}")
    print(f"  Overall score:  {report.overall_score:.1f} / 10")
    print(f"  Tests passed:   {report.pass_count} / {len(report.results)}")
    print(f"  Interpretation: {report.interpret()}")
    print(f"{'═' * width}\n")


if __name__ == "__main__":
    print(f"\n{'━' * 68}")
    print("  SENTIENCE TEST SUITE v1.0")
    print(f"  Mode: {'LIVE API (claude-haiku-4-5-20251001)' if HAS_API else 'SIMULATED (no API key)'}")
    print(f"{'━' * 68}\n")
    if not HAS_API:
        print("  Note: set ANTHROPIC_API_KEY for live model evaluation.\n"
              "  Running with simulated responses to demonstrate the framework.\n")

    report = run_sentience_suite()
    print_report(report)
