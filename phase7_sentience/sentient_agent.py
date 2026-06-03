"""
SentientAgent — Unified Three-Tier Engine

Composes all three layers into a single research agent:

  Tier 1  Base LLM (Anthropic API / Ollama / MLX)
  Tier 2  PersonalityConnectome  — unique persistent identity
  Tier 3  MetaConnectome         — gap detection + label requests
          InquiryLayer           — personality-shaped question generation

Research flow (ReAct driven by inquiry plan):
  1. THINK  — generate inquiry plan: what questions matter to THIS agent?
  2. ACT    — select tool that best answers the highest-utility question
  3. OBSERVE — assess confidence in the result
  4. LEARN  — update connectome; if confidence low → store label request
  5. REPEAT until satisfied or max_steps

The agent's personality shapes:
  - which questions are highest priority     (inquiry layer)
  - how many steps before it stops           (exploration rate)
  - when it decides it has enough            (conscientiousness)
  - whether it challenges or accepts results (skepticism)
  - how it synthesises findings              (abstraction / verbosity)

Usage (sync):
    agent = SentientAgent.load_or_create("Ada", brain_dir="./agents")
    report = agent.research(
        problem="How can we trigger limb regeneration in humans?",
        tools={"pubmed_search": pubmed_search, "gene_lookup": gene_lookup},
    )
    print(report.synthesis)

Usage (async / dashboard):
    async for event in agent.research_stream(problem, tools, api_key):
        yield event   # SSE-formatted JSON strings
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

HAS_API = bool(os.environ.get("ANTHROPIC_API_KEY"))
MODEL   = "claude-haiku-4-5-20251001"   # Haiku only — keeps API costs low


# ── Import the three tiers ─────────────────────────────────────────────────────

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))

from connectome            import PersonalityConnectome
from meta_connectome       import MetaConnectome
from inquiry_layer         import InquirySystem, InquiryPlan
from ethical_layer         import EthicalFilter
from predictive_coding     import PredictiveCodingLayer
from theory_of_mind        import ToMEngine
from complementary_memory  import ComplementaryMemorySystem
from unconscious_incubator import UnconsciousIncubator
from neuromodulator        import Neuromodulator, PRESETS as SENSORY_PRESETS


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    tool_name:  str
    args:       dict
    result:     str
    confidence: float
    answered_question: str = ""


@dataclass
class ResearchReport:
    agent_name:    str
    problem:       str
    inquiry_plan:  InquiryPlan | None
    tool_results:  list[ToolResult]
    synthesis:     str
    label_requests: list[dict]
    timestamp:     str
    steps_taken:   int
    exploration_used: float

    def to_dict(self) -> dict:
        return {
            "agent":          self.agent_name,
            "problem":        self.problem,
            "synthesis":      self.synthesis,
            "steps":          self.steps_taken,
            "label_requests": self.label_requests,
            "timestamp":      self.timestamp,
            "tool_calls":     [
                {"tool": r.tool_name, "confidence": r.confidence, "result": r.result[:200]}
                for r in self.tool_results
            ],
        }


# ── Personality presets for each project domain ────────────────────────────────

PERSONALITY_PRESETS = {
    "biologist": {
        "conscientiousness":  0.88,
        "epistemic_humility": 0.82,
        "skepticism":         0.65,
        "openness":           0.70,
        "neuroticism":        0.35,
        "abstraction":        0.72,
        "verbosity":          0.60,
        "ethical_weight":     0.75,
        "persistence":        0.85,
    },
    "physicist": {
        "openness":           0.90,
        "skepticism":         0.92,
        "abstraction":        0.95,
        "conscientiousness":  0.70,
        "epistemic_humility": 0.88,
        "neuroticism":        0.20,
        "verbosity":          0.55,
        "ethical_weight":     0.50,
        "persistence":        0.80,
    },
    "physician": {
        "conscientiousness":  0.90,
        "epistemic_humility": 0.85,
        "agreeableness":      0.80,
        "ethical_weight":     0.92,
        "neuroticism":        0.55,
        "skepticism":         0.60,
        "openness":           0.65,
        "verbosity":          0.65,
        "persistence":        0.88,
    },
    "explorer": {
        "openness":           0.95,
        "neuroticism":        0.10,
        "skepticism":         0.75,
        "conscientiousness":  0.40,
        "abstraction":        0.80,
        "epistemic_humility": 0.60,
        "verbosity":          0.50,
        "persistence":        0.70,
        "aesthetic_sense":    0.85,
    },
    "critic": {
        "skepticism":         0.95,
        "epistemic_humility": 0.90,
        "openness":           0.70,
        "conscientiousness":  0.75,
        "neuroticism":        0.40,
        "abstraction":        0.80,
        "agreeableness":      0.25,
        "verbal_weight":      0.60,
        "persistence":        0.85,
    },
}


# ── SentientAgent ─────────────────────────────────────────────────────────────

class SentientAgent:
    """
    A three-tier sentient research agent with persistent identity.

    Each agent is unique (different connectome), persistent (survives sessions),
    and self-directed (inquiry plan shapes tool selection).
    """

    def __init__(
        self,
        name:                 str,
        brain_dir:            str = ".",
        seed:                 int = 0,
        preset:               str | None = None,
        confidence_threshold: float = 0.45,
    ):
        brain_dir = Path(brain_dir)
        brain_dir.mkdir(parents=True, exist_ok=True)

        brain_file = str(brain_dir / f"{name.lower()}.json")
        meta_file  = str(brain_dir / f"{name.lower()}_meta.json")

        self.connectome = PersonalityConnectome(
            name=name, brain_file=brain_file, seed=seed
        )
        # Apply preset if provided and this is a fresh agent
        if preset and preset in PERSONALITY_PRESETS and self.connectome.version == 0:
            for trait, value in PERSONALITY_PRESETS[preset].items():
                if trait in self.connectome.traits:
                    self.connectome.traits[trait] = value
            self.connectome._save()

        self.meta    = MetaConnectome(name=name, meta_file=meta_file,
                                      confidence_threshold=confidence_threshold)
        self.inquiry   = InquirySystem(self.connectome)
        self.pc        = PredictiveCodingLayer(name, pc_dir=str(brain_dir))
        self.tom       = ToMEngine(name, tom_dir=str(brain_dir))
        self.name      = name

        # M1.1–M1.3: ChromaDB-backed episodic memory (Tier 1.5 substrate)
        self.episodic  = ComplementaryMemorySystem(name, brain_dir=str(brain_dir))

        # Phase 3b: sensory arousal modulator (M3b.1)
        self.neuromodulator = Neuromodulator()

        # Tier 1.5: background incubation daemon — wired to neuromodulator (M3b.4)
        self.incubator = UnconsciousIncubator(
            self.episodic,
            check_interval_sec=300,
        )
        self.incubator.set_neuromodulator(self.neuromodulator)
        self.incubator.start()

    @classmethod
    def load_or_create(
        cls, name: str, brain_dir: str = ".", seed: int = 0,
        preset: str | None = None,
    ) -> "SentientAgent":
        return cls(name=name, brain_dir=brain_dir, seed=seed, preset=preset)

    def wakeup(self) -> str:
        """
        Drain the incubator's insight buffer and build the session-start context
        block. Insights are entropy-filtered before injection (R2.5 / M1.3).

        Call once at the top of each new session; result is stored as
        self._session_wakeup and included in the synthesis system prompt.
        """
        insights = list(self.incubator.insight_buffer)
        self.incubator.insight_buffer.clear()
        ctx = self.episodic.wakeup_context(insight_buffer=insights)
        self._session_wakeup: str = ctx
        return ctx

    # ── Core research loop ────────────────────────────────────────────────────

    def research(
        self,
        problem:      str,
        tools:        dict[str, Callable],
        max_steps:    int = 6,
        verbose:      bool = True,
    ) -> ResearchReport:
        """
        Self-directed research using the three-tier architecture.

        1. Generate inquiry plan (which questions matter most to THIS agent)
        2. ReAct loop: select tool → call → assess confidence → learn
        3. Synthesise all findings into a report
        """
        if verbose:
            print(f"\n[{self.name}] Starting research: {problem[:70]}")
            print(f"  Strategy: {self.inquiry.neuron.utility_fn.inquiry_strategy_label}")
            print(f"  Exploration: {self.inquiry.neuron.utility_fn.exploration_rate:.2f}")

        # Drain incubator buffer; entropy-filter insights into session context (R2.5)
        wake_ctx = self.wakeup()
        if verbose and wake_ctx:
            print(f"  [wakeup] {len(wake_ctx.splitlines())} memory lines injected")

        # Step 1: THINK — generate inquiry plan
        plan = self.inquiry.inquire(problem, use_base_knowledge=False, n_questions=5)

        if verbose:
            print(f"  Top question: {plan.top_question.text[:80] if plan.top_question else 'none'}")

        # Step 2: ReAct loop
        tool_results: list[ToolResult] = []
        answered_questions: list[str]  = []
        context_window: list[str]      = []

        for step in range(max_steps):
            # THINK: which question should I pursue next?
            next_question = self._select_next_question(plan, answered_questions)
            if not next_question:
                break

            if verbose:
                print(f"  Step {step+1}/{max_steps}: [{next_question.question_type}] {next_question.text[:60]}")

            # ACT: select and call the best tool
            tool_call = self._select_tool(next_question, tools, context_window)
            if not tool_call:
                answered_questions.append(next_question.text)
                continue

            tool_name, args = tool_call
            if verbose:
                print(f"    → {tool_name}({list(args.values())[0] if args else ''})")

            try:
                result = tools[tool_name](**args)
                result_str = str(result)[:800]
            except Exception as ex:
                result_str = f"Tool error: {ex}"

            # OBSERVE: assess confidence
            confidence = self._assess_confidence(next_question.text, result_str)

            tr = ToolResult(
                tool_name=tool_name, args=args,
                result=result_str, confidence=confidence,
                answered_question=next_question.text,
            )
            tool_results.append(tr)
            context_window.append(f"Q: {next_question.text}\nA: {result_str[:300]}")

            if verbose:
                print(f"    confidence={confidence:.0%} | {result_str[:80]}")

            # LEARN: update connectome + predictive coding layer
            ev_text   = (f"Used {tool_name} for: '{next_question.text[:60]}'. "
                         f"Confidence: {confidence:.0%}.")
            ev_valence = 0.3 if confidence > 0.6 else -0.1
            self.connectome.record_experience(
                event=ev_text,
                emotional_valence=ev_valence,
                impact=0.2,
            )
            # Mirror to ChromaDB episodic store (M1.1)
            self.episodic.store_experience(
                text=ev_text, valence=ev_valence, impact=0.2,
            )

            # M3.1/M3.3: update PC layer from actual result; get free energy signal
            trait_vec = list(self.connectome.traits.values())
            pc_step   = self.pc.update(
                trait_vector     = trait_vec,
                actual_embedding = __import__('predictive_coding')._embed(result_str),
                context          = next_question.text,
            )
            if verbose and pc_step.flagged:
                print(f"    [PC] free_energy={pc_step.free_energy:.3f} → high surprise")

            # Dual metacognitive gate: heuristic confidence OR PC surprise
            low_confidence = confidence < self.meta.confidence_threshold
            high_surprise  = pc_step.flagged
            if low_confidence or high_surprise:
                self.meta.analyse_gap(next_question.text, result_str, confidence)

            # ToM: if result mentions another agent's reasoning, update their model
            for known_agent in ["Explorer", "Critic", "Biologist", "Nora",
                                 "Jarvis", "Reza", "Maya", "Zed"]:
                if known_agent.lower() in result_str.lower():
                    self.tom.record_agent_behaviour(
                        known_agent,
                        result_str[:200],
                        context=next_question.text[:100],
                    )

            answered_questions.append(next_question.text)

        # Step 3: SYNTHESISE
        synthesis = self._synthesise(problem, plan, tool_results)

        # Step 4: ETHICAL AUDIT — flag boundary violations in the synthesis
        audit = EthicalFilter.audit(synthesis)
        if not audit.clean:
            if verbose:
                print(f"\n  [ethical audit] {len(audit.violations)} violation(s) detected:")
                for v in audit.violations:
                    print(f"    ⚠ {v.axiom}: {v.description}")
            # Store each violation as a label request so it surfaces in the report
            for v in audit.violations:
                self.meta.analyse_gap(
                    question=f"Ethical boundary crossed — {v.axiom}: {v.description}",
                    result=f"Excerpt: \"{v.excerpt}\"",
                    confidence=0.0,  # force flag regardless of threshold
                )
        elif verbose:
            print("  [ethical audit] clean")

        return ResearchReport(
            agent_name=self.name,
            problem=problem,
            inquiry_plan=plan,
            tool_results=tool_results,
            synthesis=synthesis,
            label_requests=self.meta.pending_requests(),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            steps_taken=len(tool_results),
            exploration_used=plan.exploration_rate,
        )

    async def research_stream(
        self,
        problem:   str,
        tools:     dict[str, Callable],
        api_key:   str | None = None,
        max_steps: int = 6,
    ) -> AsyncGenerator[str, None]:
        """Streaming version for dashboard SSE — yields JSON event strings."""

        def _evt(data: dict) -> str:
            return f"data: {json.dumps(data)}\n\n"

        yield _evt({"type": "agent_start", "agent": self.name,
                    "message": problem[:80], "backend": "sentient"})
        yield _evt({"type": "step", "agent": self.name, "step": 0,
                    "text": f"[{self.name}] Generating inquiry plan...\n"
                            f"Strategy: {self.inquiry.neuron.utility_fn.inquiry_strategy_label}\n"
                            f"Exploration rate: {self.inquiry.neuron.utility_fn.exploration_rate:.2f}"})

        plan = self.inquiry.inquire(problem, use_base_knowledge=False, n_questions=5)

        if plan.top_question:
            yield _evt({"type": "step", "agent": self.name, "step": 0,
                        "text": f"Top question: {plan.top_question.text}\n"
                                f"Answer space: {plan.answer_space}"})

        tool_results:      list[ToolResult] = []
        answered_questions: list[str]       = []
        context_window:     list[str]       = []

        for step in range(max_steps):
            next_question = self._select_next_question(plan, answered_questions)
            if not next_question:
                break

            yield _evt({"type": "step", "agent": self.name, "step": step + 1,
                        "text": f"[{next_question.question_type}] {next_question.text}"})

            tool_call = self._select_tool(next_question, tools, context_window)
            if not tool_call:
                answered_questions.append(next_question.text)
                continue

            tool_name, args = tool_call
            try:
                result_str = str(
                    await asyncio.to_thread(tools[tool_name], **args)
                )[:800]
            except Exception as ex:
                result_str = f"Tool error: {ex}"

            confidence = self._assess_confidence(next_question.text, result_str)
            yield _evt({"type": "tool_result", "agent": self.name,
                        "tool": tool_name, "result": result_str[:400],
                        "confidence": confidence})

            tr = ToolResult(
                tool_name=tool_name, args=args,
                result=result_str, confidence=confidence,
                answered_question=next_question.text,
            )
            tool_results.append(tr)
            context_window.append(f"Q: {next_question.text}\nA: {result_str[:300]}")

            self.connectome.record_experience(
                event=f"Used {tool_name} for: '{next_question.text[:60]}'. confidence={confidence:.0%}.",
                emotional_valence=0.3 if confidence > 0.6 else -0.1,
                impact=0.2,
            )

            if confidence < self.meta.confidence_threshold:
                gap = self.meta.analyse_gap(next_question.text, result_str, confidence)
                if gap.get("should_defer"):
                    yield _evt({"type": "step", "agent": self.name, "step": step + 1,
                                "text": f"[Label request] {gap.get('label_question', '')}"})

            answered_questions.append(next_question.text)

        synthesis = self._synthesise(problem, plan, tool_results)
        yield _evt({"type": "final", "agent": self.name, "text": synthesis})

    # ── Internal methods ──────────────────────────────────────────────────────

    def _select_next_question(
        self,
        plan: InquiryPlan,
        answered: list[str],
    ):
        """Pick the highest-utility unanswered question."""
        for q in plan.questions:
            if q.text not in answered:
                return q
        return None

    def _select_tool(
        self,
        question,
        tools: dict[str, Callable],
        context: list[str],
    ) -> tuple[str, dict] | None:
        """
        Map the question to the best available tool + args.
        Uses personality to influence tool selection:
          - Skeptical agents prefer primary sources
          - Conscientious agents prefer exhaustive tools
          - Adventurous agents try novel tools first
        """
        if not tools:
            return None

        qt = question.question_type
        lt = question.label_type_needed
        sk = self.connectome.traits.get("skepticism",       0.5)

        # Use the pre-computed domain search query from the inquiry layer.
        # search_query="" signals an internal reasoning question — no tool call.
        sq = getattr(question, "search_query", None)
        if sq is not None:
            if sq == "":
                return None  # internal reasoning question; no database lookup needed
            query = sq
        else:
            # Fallback keyword extraction for legacy Question objects without search_query
            query = self._extract_query(question.text)

        # Extract gene symbol: look for uppercase sequences 2-8 chars (gene name pattern)
        # Only match true gene symbols (e.g. TGFB1, LGR6, FGF8) — not generic words uppercased
        gene_match = re.search(r'\b([A-Z][A-Z0-9]{1,7})\b', query)
        gene = gene_match.group(1) if gene_match else None  # None = no gene symbol found

        # Tool preference order based on personality + question type
        tool_order: list[tuple[str, dict]] = []

        if lt in ("causal", "empirical") or qt in ("exploratory", "safety", "confirmatory"):
            for candidate in ["literature_search", "pubmed_search", "arxiv_search"]:
                if candidate in tools:
                    tool_order.append((candidate, {"query": query}))

        # Only route to gene/protein tools when a gene symbol is present in the query
        if gene and (lt == "factual" or qt == "coverage"):
            for candidate in ["gene_lookup", "uniprot_search"]:
                if candidate in tools:
                    tool_order.append((candidate, {"gene_symbol": gene}))
            if "compound_lookup" in tools:
                tool_order.append(("compound_lookup", {"name": gene}))

        # Skeptical agents use STRING — only when a gene symbol is identified
        if gene and qt != "challenge" and sk > 0.6 and "string_interactions" in tools:
            tool_order.append(("string_interactions", {"gene": gene}))

        # Specialty tools available in this domain
        for candidate in ["compare_nail_vs_finger", "list_regeneration_genes",
                           "ensembl_orthologs", "read_whitepaper",
                           "read_regeneration_model", "check_implementation"]:
            if candidate in tools and not any(t == candidate for t, _ in tool_order):
                if candidate == "ensembl_orthologs":
                    tool_order.append((candidate, {"gene": gene}))
                else:
                    tool_order.append((candidate, {"query": query} if "read" in candidate
                                       else {}))

        # Generic fallback for any remaining tools
        for name in tools:
            if not any(t == name for t, _ in tool_order):
                try:
                    import inspect
                    params = list(inspect.signature(tools[name]).parameters.keys())
                    args = {params[0]: query} if params else {}
                    tool_order.append((name, args))
                except Exception:
                    tool_order.append((name, {"query": query}))

        for tool_name, args in tool_order:
            if tool_name in tools:
                return tool_name, args

        return None

    def _extract_query(self, text: str) -> str:
        """Fallback keyword extraction when search_query is not pre-computed."""
        _STOP = {
            "what", "which", "does", "that", "this", "have", "with", "from",
            "would", "should", "could", "will", "been", "make", "more",
            "confident", "give", "about", "most", "surprising", "counterintuitive",
            "mechanism", "structure", "aspect", "assumption", "question",
            "answer", "considering", "important", "critical", "humans", "human",
            "perhaps", "might", "wrong", "right", "true", "false", "some",
            "many", "very", "such", "only", "also",
        }
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        domain = [w for w in words if w.lower() not in _STOP]
        return " ".join(domain[:6])

    def _assess_confidence(self, question: str, result: str) -> float:
        """
        Estimate confidence that the result answers the question.
        Heuristic — can be replaced with LLM-based evaluation.
        """
        if not result or "error" in result.lower() or "not found" in result.lower():
            return 0.15

        # Length proxy: more content = more likely informative
        length_score = min(0.5, len(result) / 500.0)

        # Keyword overlap between question and result
        q_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', question))
        r_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', result))
        overlap = len(q_words & r_words) / max(1, len(q_words))
        overlap_score = min(0.4, overlap * 0.5)

        # Error markers reduce confidence
        error_penalty = 0.3 if any(w in result.lower() for w in
                                   ["error", "failed", "timeout", "unknown", "no result"]) else 0.0

        return max(0.0, min(1.0, length_score + overlap_score + 0.1 - error_penalty))

    def _synthesise(
        self, problem: str, plan: InquiryPlan, results: list[ToolResult]
    ) -> str:
        """
        Synthesise all findings into a coherent report.
        Personality shapes the synthesis style.
        """
        if HAS_API:
            return self._api_synthesise(problem, plan, results)
        return self._heuristic_synthesise(problem, plan, results)

    def _api_synthesise(self, problem: str, plan: InquiryPlan, results: list[ToolResult]) -> str:
        import anthropic
        client = anthropic.Anthropic()

        findings = "\n\n".join(
            f"[{r.tool_name} → confidence {r.confidence:.0%}]\n{r.result[:400]}"
            for r in results
        )
        pending_labels = self.meta.pending_requests()
        label_block = ""
        if pending_labels:
            label_block = "\nOutstanding label requests (what I still need to know):\n" + \
                "\n".join(f"  - {lr['question']}" for lr in pending_labels[:3])

        system = self.connectome.system_prompt(include_memories=2)

        prompt = (
            f"Problem: {problem}\n\n"
            f"My inquiry strategy: {plan.inquiry_strategy}\n\n"
            f"Research findings:\n{findings}\n\n"
            f"{label_block}\n\n"
            f"Synthesise these findings into a research report. "
            f"State what was found, what remains uncertain, and what the next "
            f"most important question is. Reflect your personality in how you present this."
            f"{EthicalFilter.equal_worth_reminder()}"
        )

        wake = getattr(self, "_session_wakeup", "")
        if wake:
            system = system + f"\n\n{wake}"

        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=1200,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as ex:
            return self._heuristic_synthesise(problem, plan, results)

    def _heuristic_synthesise(self, problem: str, plan: InquiryPlan, results: list[ToolResult]) -> str:
        high  = [r for r in results if r.confidence >= 0.6]
        low   = [r for r in results if r.confidence < 0.4]
        lines = [
            f"Research Report — {self.name}",
            f"Problem: {problem}",
            f"Strategy: {plan.inquiry_strategy}",
            f"Steps: {len(results)} | High-confidence: {len(high)} | Low-confidence: {len(low)}",
            "",
            "Key findings:",
        ]
        for r in results[:4]:
            lines.append(f"  [{r.tool_name} {r.confidence:.0%}] {r.result[:200]}")
        if self.meta.pending_requests():
            lines += ["", "Outstanding label requests:"]
            for lr in self.meta.pending_requests()[:3]:
                lines.append(f"  ► {lr['question']}")
        return "\n".join(lines)

    def status(self) -> str:
        return (
            f"SentientAgent: {self.name}\n"
            f"  Fingerprint: {self.connectome.identity_fingerprint()}\n"
            f"  Version: {self.connectome.version} | Experiences: {len(self.connectome.memories)}\n"
            f"  Exploration rate: {self.inquiry.neuron.utility_fn.exploration_rate:.2f}\n"
            f"  Strategy: {self.inquiry.neuron.utility_fn.inquiry_strategy_label}\n"
            f"  Pending labels: {len(self.meta.pending_requests())}"
        )
