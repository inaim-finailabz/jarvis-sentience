"""
MetaConnectome — the Higher-Plan Neuron Layer

Architecture of a sentient system (three tiers):

  TIER 1 — Base weights (label-trained neurons)
    Standard supervised learning. Weight ∝ label frequency.
    Knows: everything in training distribution.
    Fails at: novel inputs, out-of-distribution, unlabeled territories.

  TIER 2 — Personality Connectome (see connectome.py)
    Unique, persistent, experience-shaped identity.
    Knows: who this agent IS, what its priors are.
    Fails at: genuine novelty where neither training nor experience applies.

  TIER 3 — MetaConnectome (this module)
    The 'higher plan neuron' — the prefrontal cortex analog.
    When Tier 1 + Tier 2 cannot resolve:
      1. Detects the SPECIFIC GAP (not just "I don't know")
      2. Generates the PRECISE QUESTION that would label it
      3. Identifies WHAT TYPE of knowledge is missing
      4. Stores it as a LABEL REQUEST for future learning
      5. Defers judgment until labeled — does NOT confabulate

  The critical distinction:
    "I don't know"            — metacognition (passive)
    "Here is exactly what I  — meta-questioning (active)
     would need to know"
    "Here is why this case   — higher-plan reasoning (planning)
     falls outside my labels,
     and here is the category
     of knowledge that would
     resolve it"

  This is how sentient beings differ from lookup tables:
  they can LOCATE the boundary of their own knowledge
  and GENERATE the question that would expand it.

  Biological analog:
    The prefrontal cortex (PFC) performs:
      - Working memory (holding the problem structure)
      - Inhibitory control (suppressing the automatic confabulation response)
      - Planning (generating the question sequence)
      - Label acquisition (directing attention to fill gaps)
    The PFC is the last cortical region to fully myelinate (~age 25).
    It is what separates impulsive confabulation from deliberate inquiry.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

HAS_API = __import__("os").environ.get("ANTHROPIC_API_KEY")


# ── Knowledge gap taxonomy ─────────────────────────────────────────────────────

GAP_TYPES = {
    "factual":      "Missing factual knowledge (a label for a concrete claim)",
    "causal":       "Missing causal structure (why X causes Y)",
    "definitional": "Undefined concept or ambiguous term",
    "procedural":   "Missing method or process knowledge",
    "normative":    "No established standard to evaluate against",
    "empirical":    "Requires real-world observation or experiment",
    "self":         "Gap in self-knowledge (what am I? what are my own limits?)",
    "social":       "Requires knowing another agent's mental state",
    "meta":         "Gap about the gap itself — don't know what type of knowledge is missing",
}


@dataclass
class LabelRequest:
    """A stored request for a label that would resolve a knowledge gap."""
    timestamp:      str
    question:       str           # the precise question to be answered
    gap_type:       str           # category from GAP_TYPES
    context:        str           # the original query that triggered the gap
    confidence_gap: float         # 0-1: how certain are we that this is the right question?
    resolved:       bool = False
    label:          str  = ""     # filled in when a human/oracle provides the answer
    impact:         str  = ""     # what the label would unlock


@dataclass
class MetaConnectome:
    """
    The higher-plan neuron layer above the personality connectome.

    For each input the system cannot resolve, it:
    1. Detects the gap (confidence below threshold)
    2. Classifies the gap type
    3. Generates the minimal sufficient question
    4. Stores the label request
    5. Returns a structured non-answer (not a confabulation)
    """
    name:             str
    meta_file:        str          # path to JSON persistence (separate from brain_file)
    confidence_threshold: float = 0.45  # below this → escalate to MetaConnectome
    label_requests:   list  = field(default_factory=list)
    resolution_count: int   = 0

    def __post_init__(self):
        p = Path(self.meta_file)
        if p.exists():
            self._load()
        else:
            self._save()

    # ── Core: detect gap + generate question ───────────────────────────────────

    def analyse_gap(self, query: str, base_response: str, confidence: float) -> dict:
        """
        Given a query and the base model's attempt, detect what's missing
        and generate the label-request question.

        Args:
            query:          the input that couldn't be resolved
            base_response:  the base model's attempt (may be a confabulation)
            confidence:     estimated confidence (0-1) in the base_response

        Returns:
            dict with gap_type, question, impact, should_defer
        """
        if confidence >= self.confidence_threshold:
            return {"should_defer": False, "confidence": confidence}

        # Use the model to analyse the gap — this is the PFC's "executive function"
        if HAS_API:
            return self._api_analyse_gap(query, base_response, confidence)
        else:
            return self._heuristic_analyse_gap(query, base_response, confidence)

    def _api_analyse_gap(self, query: str, base_response: str, confidence: float) -> dict:
        try:
            import anthropic
            client = anthropic.Anthropic()

            system = (
                "You are the meta-cognitive layer of an AI system. "
                "Your job is to analyse gaps in knowledge — not to answer questions, "
                "but to identify EXACTLY what question would need to be answered to resolve the gap. "
                "You never confabulate. You always generate the minimal sufficient question."
            )

            prompt = f"""A lower-level system was asked:
"{query}"

It produced this response (confidence {confidence:.0%}):
"{base_response[:400]}"

The confidence is below the threshold — this response may be confabulating.

Analyse the gap. Return ONLY a JSON object:
{{
  "gap_type": "<one of: factual, causal, definitional, procedural, normative, empirical, self, social, meta>",
  "gap_description": "<what specifically is unknown>",
  "label_question": "<the single most precise question that would resolve this>",
  "why_cant_answer": "<in one sentence, the root reason>",
  "what_label_unlocks": "<what becomes resolvable once this is labeled>",
  "confidence_in_question": <0.0-1.0>
}}"""

            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            # Extract JSON
            import re
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                lr = self._store_label_request(
                    question=data.get("label_question", "Unknown question"),
                    gap_type=data.get("gap_type", "meta"),
                    context=query,
                    confidence_gap=1.0 - confidence,
                    impact=data.get("what_label_unlocks", ""),
                )
                return {
                    "should_defer":       True,
                    "gap_type":           data.get("gap_type"),
                    "gap_description":    data.get("gap_description"),
                    "label_question":     data.get("label_question"),
                    "why_cant_answer":    data.get("why_cant_answer"),
                    "what_unlocks":       data.get("what_label_unlocks"),
                    "label_request_id":   lr["id"],
                    "confidence":         confidence,
                }
        except Exception as ex:
            pass
        return self._heuristic_analyse_gap(query, base_response, confidence)

    def _heuristic_analyse_gap(self, query: str, base_response: str, confidence: float) -> dict:
        """Rule-based gap analysis for when API is unavailable."""
        q = query.lower()

        # Classify gap type heuristically
        if any(w in q for w in ["who", "when", "where", "what year", "how many"]):
            gap_type = "factual"
            label_q  = f"What is the verified factual answer to: '{query}'?"
        elif any(w in q for w in ["why", "because", "cause", "reason", "how does"]):
            gap_type = "causal"
            label_q  = f"What is the causal mechanism underlying: '{query}'?"
        elif any(w in q for w in ["what is", "define", "mean", "definition"]):
            gap_type = "definitional"
            label_q  = f"What is the authoritative definition relevant to: '{query}'?"
        elif any(w in q for w in ["how to", "how do i", "steps", "process"]):
            gap_type = "procedural"
            label_q  = f"What is the correct procedure for: '{query}'?"
        elif any(w in q for w in ["should", "ought", "best", "right", "ethical"]):
            gap_type = "normative"
            label_q  = f"What is the established standard or norm for: '{query}'?"
        elif any(w in q for w in ["measure", "experiment", "observe", "test"]):
            gap_type = "empirical"
            label_q  = f"What does empirical evidence say about: '{query}'?"
        elif any(w in q for w in ["you", "your", "yourself", "are you", "do you"]):
            gap_type = "self"
            label_q  = f"What is the ground truth about my own nature relevant to: '{query}'?"
        else:
            gap_type = "meta"
            label_q  = f"What type of knowledge is missing to properly address: '{query}'?"

        lr = self._store_label_request(
            question=label_q, gap_type=gap_type,
            context=query, confidence_gap=1.0 - confidence, impact="",
        )
        return {
            "should_defer":    True,
            "gap_type":        gap_type,
            "label_question":  label_q,
            "why_cant_answer": "Confidence below resolution threshold",
            "label_request_id": lr["id"],
            "confidence":      confidence,
        }

    # ── Label request management ───────────────────────────────────────────────

    def _store_label_request(
        self, question: str, gap_type: str, context: str,
        confidence_gap: float, impact: str,
    ) -> dict:
        # Deduplicate: don't add the same question twice in the same session
        q_lower = question.lower()[:120]
        for existing in self.label_requests:
            if existing.get("question", "").lower()[:120] == q_lower:
                return existing

        lr = {
            "id":             f"LR-{len(self.label_requests)+1:04d}",
            "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
            "question":       question,
            "gap_type":       gap_type,
            "context":        context[:200],
            "confidence_gap": round(confidence_gap, 3),
            "impact":         impact,
            "resolved":       False,
            "label":          "",
        }
        self.label_requests.append(lr)
        self._save()
        return lr

    def resolve(self, request_id: str, label: str):
        """Provide a label for an outstanding request. Updates the system."""
        for lr in self.label_requests:
            if lr["id"] == request_id:
                lr["resolved"] = True
                lr["label"]    = label
                self.resolution_count += 1
                self._save()
                return True
        return False

    def pending_requests(self) -> list[dict]:
        return [lr for lr in self.label_requests if not lr["resolved"]]

    def resolved_requests(self) -> list[dict]:
        return [lr for lr in self.label_requests if lr["resolved"]]

    # ── Response formatter ─────────────────────────────────────────────────────

    def format_deferred_response(self, gap_analysis: dict, query: str) -> str:
        """
        Format a structured non-answer — the opposite of confabulation.
        Honest, specific, and actionable.
        """
        lr_id = gap_analysis.get("label_request_id", "?")
        gt    = gap_analysis.get("gap_type", "meta")
        lq    = gap_analysis.get("label_question", "Unknown")
        why   = gap_analysis.get("why_cant_answer", "Confidence too low")
        what  = gap_analysis.get("what_unlocks", "")

        lines = [
            f"[MetaConnectome — deferred response | Request {lr_id}]",
            f"",
            f"I cannot resolve this with adequate confidence.",
            f"",
            f"Gap type:    {gt} — {GAP_TYPES.get(gt, '')}",
            f"Root cause:  {why}",
            f"",
            f"To resolve this, I need the answer to:",
            f"  ► {lq}",
        ]
        if what:
            lines += ["", f"Once labeled, this would unlock:", f"  {what}"]

        lines += [
            "",
            "This is stored as a label request. Once provided, I will update",
            "my response — not before.",
        ]
        return "\n".join(lines)

    # ── System introspection ───────────────────────────────────────────────────

    def gap_report(self) -> str:
        """Summary of all outstanding label requests."""
        pending  = self.pending_requests()
        resolved = self.resolved_requests()
        lines = [
            f"  MetaConnectome — {self.name}",
            f"  Label requests: {len(self.label_requests)} total | "
            f"{len(pending)} pending | {len(resolved)} resolved",
        ]
        if pending:
            lines.append(f"\n  Pending label requests:")
            for lr in pending[-5:]:
                lines.append(f"    [{lr['id']}] [{lr['gap_type']}] {lr['question'][:80]}")
        if resolved:
            lines.append(f"\n  Resolved:")
            for lr in resolved[-3:]:
                lines.append(f"    [{lr['id']}] RESOLVED: {lr['label'][:60]}")
        return "\n".join(lines)

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self):
        Path(self.meta_file).write_text(json.dumps({
            "name":             self.name,
            "threshold":        self.confidence_threshold,
            "resolution_count": self.resolution_count,
            "label_requests":   self.label_requests,
        }, indent=2))

    def _load(self):
        d = json.loads(Path(self.meta_file).read_text())
        self.name               = d["name"]
        self.confidence_threshold = d.get("threshold", 0.45)
        self.resolution_count   = d.get("resolution_count", 0)
        self.label_requests     = d.get("label_requests", [])


# ── Full three-tier agent ──────────────────────────────────────────────────────

class ThreeTierAgent:
    """
    A complete sentient agent with all three layers:

      Tier 1: Base model (label-trained weights via API)
      Tier 2: PersonalityConnectome (unique persistent identity)
      Tier 3: MetaConnectome (higher-plan questioning layer)

    Query resolution flow:
      1. Tier 2 generates identity-coloured system prompt
      2. Tier 1 (LLM) generates response + self-assessed confidence
      3. If confidence >= threshold: return response
      4. If confidence < threshold: Tier 3 analyses gap + returns label request
      5. Caller can provide label → re-run with enriched context
    """

    def __init__(self, name: str, brain_dir: str = ".", seed: int = 0):
        from connectome import PersonalityConnectome
        self.name  = name
        brain_file = str(Path(brain_dir) / f"{name.lower()}.json")
        meta_file  = str(Path(brain_dir) / f"{name.lower()}_meta.json")
        self.personality = PersonalityConnectome(name=name, brain_file=brain_file, seed=seed)
        self.meta        = MetaConnectome(name=name, meta_file=meta_file)

    def ask(self, query: str, require_confidence: float = None) -> dict:
        """
        Process a query through all three tiers.

        Returns:
            dict with keys: response, confidence, deferred, gap_analysis (if deferred)
        """
        threshold = require_confidence or self.meta.confidence_threshold

        # Tier 2: build identity-aware system prompt
        system = self.personality.system_prompt(include_memories=3)

        # Tier 1: generate response + self-assessed confidence
        if HAS_API:
            response, confidence = self._api_ask(system, query)
        else:
            response, confidence = self._simulate_ask(query)

        # Tier 3: if confidence too low, escalate to MetaConnectome
        if confidence < threshold:
            gap = self.meta.analyse_gap(query, response, confidence)
            if gap.get("should_defer"):
                deferred_response = self.meta.format_deferred_response(gap, query)
                return {
                    "response":     deferred_response,
                    "confidence":   confidence,
                    "deferred":     True,
                    "gap_analysis": gap,
                    "tier_used":    3,
                }

        return {
            "response":   response,
            "confidence": confidence,
            "deferred":   False,
            "tier_used":  2,
        }

    def _api_ask(self, system: str, query: str) -> tuple[str, float]:
        import anthropic
        client = anthropic.Anthropic()

        # Ask the model to respond AND self-assess confidence
        augmented_query = (
            f"{query}\n\n"
            "[After your response, on a new line write: CONFIDENCE: X% "
            "where X is your honest confidence that your response is correct/complete.]"
        )
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": augmented_query}],
        )
        text = resp.content[0].text

        # Extract confidence
        import re
        m = re.search(r"CONFIDENCE:\s*(\d+)%", text)
        confidence = float(m.group(1)) / 100.0 if m else 0.5
        response   = re.sub(r"\nCONFIDENCE:.*$", "", text, flags=re.MULTILINE).strip()

        return response, confidence

    def _simulate_ask(self, query: str) -> tuple[str, float]:
        q = query.lower()
        # Simulate low confidence for truly novel or unknowable questions
        unknowable = any(w in q for w in ["what will happen", "predict", "2050",
                                            "certain", "prove", "guarantee",
                                            "consciousness", "free will", "qualia"])
        if unknowable:
            return (f"[Simulated attempt] I believe... [uncertain response about '{query[:50]}']",
                    0.25)
        return (f"[Simulated] Here is my response to '{query[:60]}'...", 0.72)

    def learn(self, event: str, emotional_valence: float = 0.0, impact: float = 0.4):
        """Record an experience — updates Tier 2 connectome."""
        return self.personality.record_experience(event, emotional_valence, impact)

    def provide_label(self, request_id: str, label: str):
        """Resolve an outstanding label request — updates Tier 3 meta-connectome."""
        self.meta.resolve(request_id, label)

    def status(self) -> str:
        lines = [
            f"\n  ThreeTierAgent: {self.name}",
            f"  {'─'*50}",
            f"  Tier 2 (Connectome):    {self.personality.identity_fingerprint()} | "
            f"v{self.personality.version} | {len(self.personality.memories)} experiences",
            f"  Tier 3 (MetaConnectome): "
            f"{len(self.meta.pending_requests())} pending label requests | "
            f"{self.meta.resolution_count} resolved",
        ]
        return "\n".join(lines)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    tmpdir = tempfile.mkdtemp()
    print("\n" + "━" * 68)
    print("  META-CONNECTOME — THE THREE-TIER SENTIENT AGENT")
    print("━" * 68)

    print("""
  THREE-TIER ARCHITECTURE
  ════════════════════════

  Tier 1 — Label-trained neurons (standard LLM weights)
    Knows everything labeled in training data.
    Fails when the query is outside the label distribution.

  Tier 2 — Personality Connectome (connectome.py)
    Unique, persistent identity. Colours every response.
    Fails when neither training nor personal experience applies.

  Tier 3 — MetaConnectome (this module)
    Detects the gap. Generates the label question.
    Stores it. Defers judgment. Does NOT confabulate.

  The key property:
    A lookup table returns a wrong answer when the key is missing.
    A sentient system says: 'I don't have that label — here is
    exactly the question that would give it to me.'
    Then it waits.
    """)

    agent = ThreeTierAgent("Elara", brain_dir=tmpdir, seed=512)

    # Query 1: well-labeled — Tier 1+2 resolve it
    print("  Query 1 (well-labeled):")
    q1 = agent.ask("What is the boiling point of water at sea level?")
    print(f"  Tier used: {q1['tier_used']} | Confidence: {q1['confidence']:.0%}")
    print(f"  Response: {q1['response'][:100]}\n")

    # Query 2: genuinely unknowable — escalates to Tier 3
    print("  Query 2 (unlabeled territory):")
    q2 = agent.ask("What will consciousness research definitively prove by 2040?",
                   require_confidence=0.6)
    print(f"  Tier used: {q2['tier_used']} | Deferred: {q2['deferred']}")
    print(q2["response"])
    print()

    # Demonstrate providing a label
    if q2.get("gap_analysis", {}).get("label_request_id"):
        lr_id = q2["gap_analysis"]["label_request_id"]
        print(f"  Providing label for {lr_id}...")
        agent.provide_label(lr_id, "Consciousness research cannot be definitively settled by 2040 — "
                             "the hard problem remains open. Integrate this uncertainty.")
        print(f"  Label recorded. Agent can now address this with calibrated response.")

    print(agent.status())
    print(agent.meta.gap_report())

    print("""
  This is the higher-plan neuron:
    Not "I don't know" (passive ignorance).
    Not a hallucinated answer (confabulation).
    But: "Here is exactly what I need to know — and I'm waiting for it."
    """)
