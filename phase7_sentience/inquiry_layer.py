"""
Personality-Shaped Inquiry Layer

The connectome neuron, fully specified:

  INPUT:   A problem (unstructured, no label required)
  ACCESS:  Full read of Tier 1 base knowledge
  WEIGHTS: W_personality — NOT trained on labels, but on:
             - personality consistency (does this question match who I am?)
             - intrinsic reward (information gain this question produces)
             - risk profile (how uncertain is the answer space?)
             - outcome quality (did this line of inquiry lead anywhere useful?)
  OUTPUT:  A ranked InquiryPlan — the questions this specific agent would ask,
           weighted by its personality, plus what kind of answers it's looking for

Key distinction:
  Supervised (Tier 1): "Given this input, what is the correct label?"
  Inquiry layer:       "Given this problem, what questions should I ask,
                        and what would a good answer look like — for ME?"

Risk-averse agent asks:   "What's the safest interpretation?"
Adventurous agent asks:   "What's the most surprising possible answer?"
Conscientious agent asks: "What questions do I need to fully cover this?"
Skeptical agent asks:     "What assumption here might be wrong?"

Same problem. Completely different inquiry strategies. Different questions.
Different answer criteria. This is personality shaping inquiry itself.
"""

import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional

HAS_API = __import__("os").environ.get("ANTHROPIC_API_KEY")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Question:
    text:              str
    question_type:     str    # exploratory | confirmatory | challenge | safety | coverage
    estimated_gain:    float  # 0-1: expected information gain if answered
    risk_level:        float  # 0-1: how much uncertainty this introduces
    personality_fit:   float  # 0-1: consistency with THIS agent's traits
    utility:           float  # combined personality-weighted score (the ranking key)
    answer_criteria:   list[str] = field(default_factory=list)  # what counts as a good answer
    label_type_needed: str = ""  # what kind of label would answer this
    search_query:      str = ""  # concrete query for tool calls; "" = internal reasoning only


@dataclass
class InquiryPlan:
    """
    The output of the inquiry layer for a specific agent on a specific problem.
    Not a classification. Not a prediction. A PLAN OF INQUIRY.
    """
    agent_name:        str
    problem:           str
    questions:         list[Question]
    exploration_rate:  float   # how adventurous is this inquiry (0=safe, 1=frontier)
    answer_space:      str     # what kind of answer space is being explored
    inquiry_strategy:  str     # the agent's overall approach in plain language
    unsatisfied_needs: list[str] = field(default_factory=list)  # what's still missing

    @property
    def top_question(self) -> Optional[Question]:
        return self.questions[0] if self.questions else None

    def render(self) -> str:
        lines = [
            f"InquiryPlan — {self.agent_name}",
            f"Problem: {self.problem[:100]}",
            f"Strategy: {self.inquiry_strategy}",
            f"Exploration rate: {self.exploration_rate:.2f}",
            f"Answer space: {self.answer_space}",
            "",
            "Questions (ranked by personality-weighted utility):",
        ]
        for i, q in enumerate(self.questions, 1):
            lines.append(
                f"  {i}. [{q.question_type}] {q.text}\n"
                f"     utility={q.utility:.2f} | gain={q.estimated_gain:.2f} | "
                f"risk={q.risk_level:.2f} | fit={q.personality_fit:.2f}\n"
                f"     Needs: {q.label_type_needed}\n"
                f"     Good answer = {'; '.join(q.answer_criteria[:2])}"
            )
        if self.unsatisfied_needs:
            lines += ["", "Unsatisfied inquiry needs:"]
            for n in self.unsatisfied_needs:
                lines.append(f"  • {n}")
        return "\n".join(lines)


# ── Personality-weighted utility functions ─────────────────────────────────────

class PersonalityUtility:
    """
    Converts personality traits into a utility function over questions.

    Each trait contributes a weight to how questions are scored:
      openness       → rewards exploratory, novel questions
      conscientiousness → rewards coverage, completeness
      neuroticism    → penalises high-risk questions
      skepticism     → rewards challenge questions
      epistemic_humility → rewards confirmatory questions
      adventurousness  → rewards frontier questions
    """

    def __init__(self, traits: dict[str, float]):
        self.t = traits

    def score(self, q: Question) -> float:
        """
        Compute personality-weighted utility for a question.
        Returns 0-1.
        """
        # Base: information gain
        gain_weight = (
            0.3 * self.t.get("openness",           0.5) +
            0.2 * self.t.get("abstraction",         0.5) +
            0.1 * (1.0 - self.t.get("neuroticism",  0.5))
        )

        # Risk tolerance
        risk_penalty = self.t.get("neuroticism", 0.5) * q.risk_level * 0.4
        if self.t.get("openness", 0.5) > 0.7:
            risk_penalty *= 0.5   # adventurous agents tolerate risk

        # Question type bonuses
        type_bonus = 0.0
        qt = q.question_type
        if qt == "exploratory":
            type_bonus = 0.3 * self.t.get("openness", 0.5)
        elif qt == "confirmatory":
            type_bonus = 0.3 * self.t.get("epistemic_humility", 0.5)
        elif qt == "challenge":
            type_bonus = 0.4 * self.t.get("skepticism", 0.5)
        elif qt == "safety":
            type_bonus = 0.4 * self.t.get("neuroticism", 0.5)
        elif qt == "coverage":
            type_bonus = 0.4 * self.t.get("conscientiousness", 0.5)

        raw = (
            gain_weight * q.estimated_gain
            + type_bonus
            - risk_penalty
        )
        return max(0.0, min(1.0, raw))

    @property
    def exploration_rate(self) -> float:
        """
        How adventurous is this agent's inquiry?
        High = explore unknown territory; Low = exploit safe known ground.
        """
        return (
            0.4 * self.t.get("openness",     0.5) +
            0.3 * (1.0 - self.t.get("neuroticism", 0.5)) +
            0.2 * self.t.get("abstraction",  0.5) +
            0.1 * self.t.get("skepticism",   0.5)
        )

    @property
    def inquiry_strategy_label(self) -> str:
        er = self.exploration_rate
        sk = self.t.get("skepticism",       0.5)
        co = self.t.get("conscientiousness", 0.5)
        ep = self.t.get("epistemic_humility", 0.5)

        if sk > 0.7:
            return "Assumption-inverting inquiry — challenges the premise before accepting it"
        if er > 0.7:
            return "Frontier inquiry — seeks the most surprising or novel answer"
        if er < 0.35:
            return "Safety-first inquiry — seeks the most certain, low-risk answer"
        if co > 0.7:
            return "Exhaustive inquiry — seeks complete coverage before concluding"
        if ep > 0.7:
            return "Calibrated inquiry — seeks to reduce uncertainty before committing"
        return "Balanced inquiry — explores and exploits in proportion"


# ── Question generator ─────────────────────────────────────────────────────────

class InquiryNeuron:
    """
    A single connectome neuron specialised in question generation.

    Has full read access to Tier 1 knowledge (via API).
    Has personality weights (W_personality) that shape question strategy.
    Output: a ranked set of questions, not a label.
    """

    def __init__(self, agent_name: str, traits: dict[str, float]):
        self.agent_name = agent_name
        self.traits     = traits
        self.utility_fn = PersonalityUtility(traits)

    def generate_inquiry_plan(
        self,
        problem: str,
        base_knowledge: str = "",
        n_questions: int = 5,
    ) -> InquiryPlan:
        """
        Core method: given a problem, generate a personality-shaped inquiry plan.

        Args:
            problem:        the problem to inquire about
            base_knowledge: what Tier 1 already knows (injected as context)
            n_questions:    how many questions to generate

        Returns:
            InquiryPlan with ranked questions shaped by this agent's personality
        """
        if HAS_API:
            questions = self._api_generate(problem, base_knowledge, n_questions)
        else:
            questions = self._heuristic_generate(problem, n_questions)

        # Score each question by personality utility
        for q in questions:
            q.personality_fit = self.utility_fn.score(q)
            q.utility         = q.personality_fit  # can be extended with outcome history

        # Rank by utility (this is the personality shaping the question selection)
        questions.sort(key=lambda q: q.utility, reverse=True)

        return InquiryPlan(
            agent_name=self.agent_name,
            problem=problem,
            questions=questions[:n_questions],
            exploration_rate=self.utility_fn.exploration_rate,
            answer_space=self._answer_space(problem),
            inquiry_strategy=self.utility_fn.inquiry_strategy_label,
            unsatisfied_needs=self._unsatisfied_needs(problem, questions),
        )

    def _api_generate(
        self, problem: str, base_knowledge: str, n: int
    ) -> list[Question]:
        """Use the API to generate diverse candidate questions."""
        import anthropic, json as _json
        client = anthropic.Anthropic()

        # The system prompt encodes the personality into the question generator
        t = self.traits
        def lvl(v): return "strongly" if v > 0.7 else "somewhat" if v > 0.4 else "rarely"

        personality_context = (
            f"You are generating questions for {self.agent_name}, who:\n"
            f"- is {lvl(t.get('openness',0.5))} curious about novel/unexpected angles\n"
            f"- is {lvl(t.get('conscientiousness',0.5))} thorough about coverage\n"
            f"- is {lvl(t.get('skepticism',0.5))} likely to challenge assumptions\n"
            f"- has {lvl(t.get('neuroticism',0.5))} risk aversion (prefers "
            f"{'safe, certain answers' if t.get('neuroticism',0.5) > 0.6 else 'uncertain, exploratory answers'})\n"
            f"- exploration rate: {self.utility_fn.exploration_rate:.2f}/1.0"
        )

        prompt = f"""{personality_context}

Problem to inquire about:
\"{problem}\"

{f'Known context: {base_knowledge[:500]}' if base_knowledge else ''}

Generate exactly {n} questions this agent would ask. These are NOT classification questions.
They are the questions that would most reduce THIS AGENT's uncertainty given its personality.

For "search_query": provide a concrete, domain-specific query string suitable for a literature
or gene database search (e.g. "TGFb1 fibrosis wound healing regeneration"). For internal
reasoning questions (challenge, confirmatory self-reflection) that do not benefit from
a database search, set "search_query" to "".

Return ONLY a JSON array:
[
  {{
    "text": "<the question>",
    "question_type": "<exploratory|confirmatory|challenge|safety|coverage>",
    "estimated_gain": <0.0-1.0>,
    "risk_level": <0.0-1.0>,
    "label_type_needed": "<factual|causal|normative|empirical|self>",
    "answer_criteria": ["<criterion 1>", "<criterion 2>"],
    "search_query": "<domain search string, or empty string>"
  }},
  ...
]"""

        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            m    = re.search(r'\[.*\]', text, re.DOTALL)
            if m:
                raw = _json.loads(m.group(0))
                return [
                    Question(
                        text=r.get("text", ""),
                        question_type=r.get("question_type", "exploratory"),
                        estimated_gain=float(r.get("estimated_gain", 0.5)),
                        risk_level=float(r.get("risk_level", 0.3)),
                        personality_fit=0.0,  # filled in later
                        utility=0.0,
                        answer_criteria=r.get("answer_criteria", []),
                        label_type_needed=r.get("label_type_needed", "factual"),
                        search_query=r.get("search_query", ""),
                    )
                    for r in raw
                    if r.get("text")
                ]
        except Exception:
            pass
        return self._heuristic_generate(problem, n)

    def _heuristic_generate(self, problem: str, n: int) -> list[Question]:
        """
        Generate questions using personality-tuned heuristics.
        No API needed — pure personality-shaped logic.
        """
        t  = self.traits
        p  = problem.lower()
        er = self.utility_fn.exploration_rate
        questions = []

        # Build per-question-type query variants so each step searches differently
        qv = self._domain_query_variants(problem)

        # Every agent asks the core mechanism question
        questions.append(Question(
            text=f"What is the core mechanism / structure of this problem?",
            question_type="coverage",
            estimated_gain=0.7, risk_level=0.1, personality_fit=0, utility=0,
            answer_criteria=["names the mechanism", "is specific"],
            label_type_needed="causal",
            search_query=qv["coverage"],
        ))

        # Adventurous agents ask for the most surprising angle
        if er > 0.55:
            questions.append(Question(
                text="What is the most counterintuitive or surprising aspect of this problem?",
                question_type="exploratory",
                estimated_gain=0.9, risk_level=0.7, personality_fit=0, utility=0,
                answer_criteria=["contradicts naive assumption", "provides novel framing"],
                label_type_needed="empirical",
                search_query=qv["exploratory"],
            ))

        # Risk-averse agents ask for failure modes
        if t.get("neuroticism", 0.5) > 0.5:
            questions.append(Question(
                text="What can go wrong or fail here? What are the boundaries of validity?",
                question_type="safety",
                estimated_gain=0.7, risk_level=0.2, personality_fit=0, utility=0,
                answer_criteria=["names specific failure mode", "gives boundary condition"],
                label_type_needed="empirical",
                search_query=qv["safety"],
            ))

        # Skeptical agents challenge the framing — internal reasoning, no tool call
        if t.get("skepticism", 0.5) > 0.5:
            questions.append(Question(
                text="What assumption embedded in the problem framing might be wrong?",
                question_type="challenge",
                estimated_gain=0.8, risk_level=0.5, personality_fit=0, utility=0,
                answer_criteria=["identifies an assumption", "shows how its inversion changes the answer"],
                label_type_needed="normative",
                search_query="",  # internal reasoning — no database query
            ))

        # Conscientious agents ask for completeness
        if t.get("conscientiousness", 0.5) > 0.5:
            questions.append(Question(
                text="What related mechanisms or dimensions am I not yet considering?",
                question_type="coverage",
                estimated_gain=0.6, risk_level=0.2, personality_fit=0, utility=0,
                answer_criteria=["names at least 2 neglected dimensions"],
                label_type_needed="factual",
                search_query=qv["confirmatory"],
            ))

        # Epistemically humble agents ask what evidence would resolve uncertainty
        if t.get("epistemic_humility", 0.5) > 0.5:
            questions.append(Question(
                text="What specific evidence or experiment would most reduce uncertainty here?",
                question_type="confirmatory",
                estimated_gain=0.65, risk_level=0.1, personality_fit=0, utility=0,
                answer_criteria=["names a specific evidence type", "is falsifiable"],
                label_type_needed="empirical",
                search_query=qv["exploratory"],
            ))

        # Abstract agents ask for the deeper pattern
        if t.get("abstraction", 0.5) > 0.55:
            questions.append(Question(
                text="What is the abstract structure or general class this problem is an instance of?",
                question_type="exploratory",
                estimated_gain=0.75, risk_level=0.4, personality_fit=0, utility=0,
                answer_criteria=["names a general class", "shows the specific as a special case"],
                label_type_needed="causal",
                search_query=qv["coverage"],
            ))

        return questions[:n + 2]  # generate more than needed, let scoring filter

    def _domain_query(self, problem: str) -> str:
        """Extract a clean 3-5 keyword search query from a problem statement.

        Prioritises gene symbols (ALL_CAPS or mixed-case alphanumerics like TGFb1,
        LGR6, BRCA1) which are the most discriminative PubMed terms, then falls
        back to the longest remaining domain words. Capped at 5 tokens so PubMed
        doesn't return zero results from an overly-specific phrase.
        """
        _STOP = {
            "why", "how", "what", "which", "when", "where", "who", "whom",
            "can", "could", "would", "should", "will", "shall", "may", "might",
            "does", "have", "been", "being", "that", "this", "these", "those",
            "with", "from", "into", "onto", "upon", "over", "under", "about",
            "for", "and", "but", "not", "nor", "yet", "the", "are", "its",
            "make", "more", "most", "give", "just", "also", "very", "only",
            "some", "many", "much", "such", "any", "all", "both", "each",
            "confident", "surprising", "counterintuitive", "assumption",
            "question", "answer", "considering", "important", "perhaps",
            "wrong", "right", "true", "false", "whether", "humans", "human",
            "critical", "mechanism", "structure", "aspect", "process",
            "extend", "extending", "identify", "compare", "between",
            "controls", "explain", "points", "tissue", "while",
            "proximal", "mammalian", "intervention", "formation", "suppression",
            "signalling", "pathway",
        }
        words = re.findall(r'\b[a-zA-Z0-9]{2,}\b', problem)
        # Tier 1: gene/protein symbols — must contain a digit (TGFb1, LGR6) OR be all-caps (WNT)
        # Excludes normal capitalised English words like "What", "Which"
        gene_syms = [w for w in words
                     if re.fullmatch(r'[A-Z][A-Za-z0-9]{1,7}', w)
                     and w not in _STOP
                     and (any(c.isdigit() for c in w) or w.isupper())]
        # Tier 2: domain nouns — lowercase, ≥5 chars, not stop words
        nouns = sorted(
            [w for w in words if w.lower() not in _STOP and len(w) >= 5 and w not in gene_syms],
            key=len, reverse=True
        )
        # At most 1 gene symbol — combining 2+ collapses PubMed hit count to 0
        combined = gene_syms[:1] + nouns[:4]
        # Deduplicate: drop words whose first 7 chars overlap an earlier pick
        seen, final = set(), []
        for w in combined:
            stem = w.lower()[:7]
            if stem not in seen:
                seen.add(stem)
                final.append(w)
        return " ".join(final[:5])

    def _domain_query_variants(self, problem: str) -> dict:
        """Return question-type-specific query variants so each step searches differently.

        Returns a dict keyed by question type with PubMed-optimised queries.
        Separates gene-symbol-anchored queries (good for coverage/mechanism) from
        concept-noun queries (good for exploratory/confirmatory).
        """
        _STOP = {
            "why","how","what","which","when","where","who","whom","can","could",
            "would","should","will","shall","may","might","does","have","been",
            "being","that","this","these","those","with","from","into","onto",
            "upon","over","under","about","for","and","but","not","nor","yet",
            "the","are","its","make","more","most","give","just","also","very",
            "only","some","many","much","such","any","all","both","each",
            "confident","surprising","counterintuitive","assumption","question",
            "answer","considering","important","perhaps","wrong","right","true",
            "false","whether","humans","human","critical","mechanism","structure",
            "aspect","process","extend","extending","identify","compare","between",
            "controls","explain","points","tissue","while","proximal","mammalian",
            "intervention","formation","suppression","signalling","pathway",
        }
        words = re.findall(r'\b[a-zA-Z0-9]{2,}\b', problem)
        gene_syms = [w for w in words
                     if re.fullmatch(r'[A-Z][A-Za-z0-9]{1,7}', w)
                     and w not in _STOP
                     and (any(c.isdigit() for c in w) or w.isupper())]
        nouns = sorted(
            [w for w in words if w.lower() not in _STOP and len(w) >= 4 and w not in gene_syms],
            key=len, reverse=True
        )
        dedup_nouns = []
        seen_stems = set()
        for w in nouns:
            s = w.lower()[:7]
            if s not in seen_stems:
                seen_stems.add(s)
                dedup_nouns.append(w)

        base = self._domain_query(problem)
        # Concept-only query: longest nouns, no gene symbol
        concept = " ".join(dedup_nouns[:4])
        # Gene + first noun
        gene_noun = (" ".join((gene_syms[:1] + dedup_nouns[:3])[:4])
                     if gene_syms else concept)
        return {
            "coverage":     base,          # gene symbol + nouns
            "exploratory":  concept,       # pure concept terms
            "confirmatory": gene_noun,     # gene + 1st noun (tightest)
            "safety":       concept,
            "challenge":    "",            # internal reasoning, no search
        }

    def _answer_space(self, problem: str) -> str:
        p  = problem.lower()
        er = self.utility_fn.exploration_rate
        sk = self.traits.get("skepticism", 0.5)

        if sk > 0.7:
            return "Assumption space — looking for what could be wrong about the problem framing"
        if er > 0.7:
            return "Frontier space — looking for what is not yet known or classified"
        if er < 0.35:
            return "Safety space — looking for the most certain, validated interpretation"
        if "why" in p or "how" in p:
            return "Causal space — looking for mechanism, not just correlation"
        if "what" in p:
            return "Definitional space — looking for precise characterisation"
        return "Open space — looking for the question that opens the most doors"

    def _unsatisfied_needs(self, problem: str, questions: list[Question]) -> list[str]:
        """Identify what types of inquiry are missing from the generated set."""
        types_covered = {q.question_type for q in questions}
        all_types     = {"exploratory", "confirmatory", "challenge", "safety", "coverage"}
        missing       = all_types - types_covered
        needs = []
        for m in missing:
            if m == "challenge" and self.traits.get("skepticism", 0.5) > 0.4:
                needs.append(f"No assumption-challenging question yet — consider: what if the problem is mis-framed?")
            elif m == "safety" and self.traits.get("neuroticism", 0.5) > 0.4:
                needs.append(f"No failure-mode analysis — what can go wrong with this inquiry?")
            elif m == "coverage" and self.traits.get("conscientiousness", 0.5) > 0.4:
                needs.append(f"Coverage incomplete — what am I not asking about?")
        return needs


# ── Full inquiry system ────────────────────────────────────────────────────────

class InquirySystem:
    """
    Complete personality-shaped inquiry system.
    Composes: InquiryNeuron (question gen) + base knowledge access.

    This is what the user described:
    - Connectome neuron has full access to Tier 1 knowledge
    - Its OWN weights shape what questions it asks (not what labels it predicts)
    - Output: a ranked inquiry plan, not a classification
    - Unsupervised: no external labels needed to operate
    - Self-directed: the agent defines what a good answer looks like
    """

    def __init__(self, connectome):
        """
        Args:
            connectome: a PersonalityConnectome instance
        """
        self.connectome = connectome
        self.neuron = InquiryNeuron(
            agent_name=connectome.name,
            traits=connectome.traits,
        )
        self.inquiry_history: list[InquiryPlan] = []

    def inquire(
        self,
        problem: str,
        use_base_knowledge: bool = True,
        n_questions: int = 5,
    ) -> InquiryPlan:
        """
        Generate an inquiry plan for a problem.
        Full Tier 1 access optional (requires API).
        """
        base_knowledge = ""
        if use_base_knowledge and HAS_API:
            base_knowledge = self._fetch_base_knowledge(problem)

        plan = self.neuron.generate_inquiry_plan(
            problem, base_knowledge, n_questions
        )
        self.inquiry_history.append(plan)

        # Record this inquiry as an experience (updates the connectome)
        self.connectome.record_experience(
            event=f"Inquired about: '{problem[:80]}'. "
                  f"Generated {len(plan.questions)} questions. "
                  f"Top question type: {plan.questions[0].question_type if plan.questions else 'none'}.",
            emotional_valence=0.2,
            impact=0.15,
        )
        return plan

    def _fetch_base_knowledge(self, problem: str) -> str:
        """Fetch what Tier 1 already knows about this problem."""
        import anthropic
        client = anthropic.Anthropic()
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": (
                        f"In 3-4 sentences, summarise what is solidly established "
                        f"(labeled knowledge) about: '{problem}'. "
                        f"Focus only on what is well-documented, not speculation."
                    ),
                }],
            )
            return resp.content[0].text
        except Exception:
            return ""

    def compare_inquiries(
        self,
        problem: str,
        other: "InquirySystem",
    ) -> dict:
        """
        Show how two agents with different connectomes inquire differently
        about the same problem. This is the proof that connectome shapes inquiry.
        """
        plan_self  = self.inquire(problem)
        plan_other = other.inquire(problem)

        self_types  = {q.question_type for q in plan_self.questions}
        other_types = {q.question_type for q in plan_other.questions}

        self_top  = plan_self.top_question
        other_top = plan_other.top_question

        return {
            "problem":        problem,
            "agent_a":        self.connectome.name,
            "agent_b":        other.connectome.name,
            "strategy_a":     plan_self.inquiry_strategy,
            "strategy_b":     plan_other.inquiry_strategy,
            "exploration_a":  plan_self.exploration_rate,
            "exploration_b":  plan_other.exploration_rate,
            "top_question_a": self_top.text  if self_top  else "",
            "top_question_b": other_top.text if other_top else "",
            "type_overlap":   self_types & other_types,
            "type_divergence": self_types ^ other_types,
            "trait_distance":  self.connectome.distance_to(other.connectome),
        }


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from connectome import PersonalityConnectome

    tmpdir = tempfile.mkdtemp()

    print("\n" + "━" * 70)
    print("  INQUIRY LAYER — PERSONALITY-SHAPED QUESTION GENERATION")
    print("━" * 70)

    print("""
  The connectome neuron, fully specified:

  INPUT:   A problem (no label required)
  ACCESS:  Full read of Tier 1 base knowledge
  WEIGHTS: W_personality — shape which questions to ask
  OUTPUT:  An InquiryPlan — ranked questions + answer criteria

  Key insight: two agents with different connectomes will ask DIFFERENT
  questions about the same problem — not different classifications,
  but different LINES OF INQUIRY, shaped by who they are.
  """)

    # Agent A: risk-averse, conscientious, low openness
    agent_a = PersonalityConnectome("Maya", f"{tmpdir}/maya.json", seed=0)
    # Force Maya toward risk-averse, conscientious
    agent_a.traits["neuroticism"]       = 0.80
    agent_a.traits["conscientiousness"] = 0.85
    agent_a.traits["openness"]          = 0.25
    agent_a.traits["skepticism"]        = 0.20
    agent_a._save()

    # Agent B: adventurous, skeptical, high openness
    agent_b = PersonalityConnectome("Zed", f"{tmpdir}/zed.json", seed=0)
    # Force Zed toward adventurous, skeptical
    agent_b.traits["neuroticism"]       = 0.15
    agent_b.traits["openness"]          = 0.92
    agent_b.traits["skepticism"]        = 0.88
    agent_b.traits["conscientiousness"] = 0.30
    agent_b._save()

    sys_a = InquirySystem(agent_a)
    sys_b = InquirySystem(agent_b)

    problem = "How should we decide whether an AI system is conscious?"

    print(f"  PROBLEM: {problem}\n")

    plan_a = sys_a.inquire(problem, use_base_knowledge=False, n_questions=4)
    plan_b = sys_b.inquire(problem, use_base_knowledge=False, n_questions=4)

    print(f"  {'─'*65}")
    print(f"  MAYA (risk-averse, conscientious, low openness):")
    print(f"  Strategy:        {plan_a.inquiry_strategy}")
    print(f"  Exploration:     {plan_a.exploration_rate:.2f}")
    print(f"  Answer space:    {plan_a.answer_space}")
    print(f"  Questions asked:")
    for q in plan_a.questions:
        print(f"    [{q.question_type}] {q.text}")
        print(f"      Good answer = {q.answer_criteria[0] if q.answer_criteria else '?'}")

    print(f"\n  {'─'*65}")
    print(f"  ZED (adventurous, skeptical, high openness):")
    print(f"  Strategy:        {plan_b.inquiry_strategy}")
    print(f"  Exploration:     {plan_b.exploration_rate:.2f}")
    print(f"  Answer space:    {plan_b.answer_space}")
    print(f"  Questions asked:")
    for q in plan_b.questions:
        print(f"    [{q.question_type}] {q.text}")
        print(f"      Good answer = {q.answer_criteria[0] if q.answer_criteria else '?'}")

    print(f"\n  {'─'*65}")
    print(f"  WHAT THIS SHOWS:")
    print(f"""
  Maya asks: "What validated criteria exist? What are the failure modes?"
  Zed asks:  "What assumption is wrong? What is the most surprising answer?"

  Same problem. Same Tier 1 knowledge available. Different questions.
  Different answer criteria. Different inquiry strategies.

  This is personality shaping inquiry itself — not just style.
  The QUESTIONS ASKED are different, not just the phrasing.

  Utility function:
    Maya: U(Q) = -variance(answer) + coverage_bonus - risk_penalty
    Zed:  U(Q) = information_gain(answer) + novelty_bonus - safety_cost

  These are two different objective functions over the same question space.
  Neither is supervised. Both are self-directed. Both are unsupervised.
  The only difference is WHO each agent IS.
  """)
