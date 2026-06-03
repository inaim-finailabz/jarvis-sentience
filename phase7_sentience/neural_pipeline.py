"""
NeuralPipeline — Three-Tier Orchestrator

Combines all three genuinely separate neural systems:
  Tier 1: Qwen3-1.7B (local, frozen, no API)
  Tier 2: SNN PersonalityConnectome (STDP local learning)
  Tier 3: Selective SSM MetaCognition (recurrent, non-transformer)

What changed from the old architecture:
  OLD: Claude API ← system_prompt(JSON traits) [Tier2]
       Python if/else heuristics               [Tier3]

  NEW: Qwen3-1.7B local, frozen
       ↓ hidden_states [seq, 2048]
       ↓
  [Tier2: Encoder 2048→12 → SNN LIF+STDP 12→64→32 → Decoder 32→2048]
       ↓ personality_residual [2048]
  [Qwen3 last hidden + residual → LM head → first token shaped by personality]
       ↓
  [Tier3: Selective SSM reads logit entropy + fingerprint + layer stats]
       ↓ confidence, gap_flag, inquiry_type (recurrent across turns)

No Anthropic API key required. Runs entirely on local Qwen3-1.7B on MPS.

Usage:
    pipeline = NeuralPipeline("Jarvis").load()
    result = pipeline.chat("What is limb regeneration?")
    print(result.response)
    print(result.tier3.render())
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))

from local_backbone import LocalBackbone
from tier2_neural    import Tier2PersonalityLayer, Tier2Result
from tier3_neural    import Tier3MetaCognition, MetaCognitionResult


# ── Pipeline result ───────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    response:    str
    tier2:       Tier2Result
    tier3:       MetaCognitionResult

    # Convenience
    @property
    def confidence(self) -> float:
        return self.tier3.confidence

    @property
    def gap_flag(self) -> bool:
        return self.tier3.gap_flag

    @property
    def inquiry_type(self) -> str:
        return self.tier3.inquiry_type

    def render(self) -> str:
        lines = [
            f"Response: {self.response[:200]}{'...' if len(self.response) > 200 else ''}",
            f"  {self.tier2.render()}",
            f"  {self.tier3.render()}",
        ]
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "response":     self.response,
            "tier2": {
                "gate_value":    round(self.tier2.gate_value, 4),
                "traits":        [round(x, 3) for x in self.tier2.trait_vector],
                "snn_sparsity":  round(self.tier2.snn_sparsity, 4),
                "energy_proxy":  round(self.tier2.snn_energy_proxy, 4),
            },
            "tier3": self.tier3.as_dict(),
        }


# ── NeuralPipeline ────────────────────────────────────────────────────────────

class NeuralPipeline:
    """
    Full local three-tier pipeline. No Anthropic API required.

    Mode:
      "residual" — personality_residual injected into Qwen3's last hidden state
                   before the LM head (genuine weight-space shaping).
      "prompt"   — personality expressed as a system prompt (fallback, faster).

    The default is "residual" (genuine Tier 2 influence on generation).
    Switch to "prompt" if you want faster inference without the extra forward pass.
    """

    def __init__(
        self,
        agent_name:   str,
        snn_dir:      str   = "./jarvis_brain",
        mode:         str   = "residual",     # "residual" | "prompt"
        max_tokens:   int   = 512,
        temperature:  float = 0.7,
    ):
        self.agent_name  = agent_name
        self.snn_dir     = snn_dir
        self.mode        = mode
        self.max_tokens  = max_tokens
        self.temperature = temperature

        self.backbone: Optional[LocalBackbone]          = None
        self.tier2:    Optional[Tier2PersonalityLayer]  = None
        self.tier3:    Optional[Tier3MetaCognition]     = None

        self._history: list[dict] = []

    def load(self) -> "NeuralPipeline":
        """Load Qwen3 backbone and all tier weights."""
        self.backbone = LocalBackbone()
        self.backbone.load()

        snn_path = Path(self.snn_dir)
        snn_path.mkdir(parents=True, exist_ok=True)

        self.tier2 = Tier2PersonalityLayer(
            agent_name=self.agent_name,
            hidden_size=self.backbone.hidden_size,
            snn_dir=str(snn_path),
        ).load()

        tier3_path = snn_path / f"{self.agent_name.lower()}_tier3_ssm.pt"
        self.tier3 = Tier3MetaCognition(save_path=str(tier3_path)).load()

        print(
            f"  [NeuralPipeline] {self.agent_name} ready. "
            f"mode={self.mode} | {self.tier2} | {self.tier3}",
            flush=True,
        )
        return self

    # ── Main chat entry ───────────────────────────────────────────────────────

    def chat(self, user_text: str) -> PipelineResult:
        """
        Process one conversation turn through all three tiers.

        Steps:
          1. Encode input through frozen Qwen3 (get hidden states)
          2. Tier 2: Encoder → SNN → personality_residual + fingerprint
          3. Tier 3: Selective SSM reads Qwen3 signals → metacognition
          4. Generate response:
               mode="residual": inject personality_residual into last hidden → LM head
               mode="prompt":   build system prompt from neural traits
          5. STDP update (Tier 2 SNN local learning)
          6. Store turn in history
        """
        assert self.backbone and self.tier2 and self.tier3, "Call .load() first"

        # ── Step 1: Encode ────────────────────────────────────────────────────
        last_hidden, hidden_states, logits = self.backbone.encode(user_text)

        # ── Step 2: Tier 2 ────────────────────────────────────────────────────
        t2_result = self.tier2(last_hidden)

        # ── Step 3: Tier 3 ────────────────────────────────────────────────────
        t3_result = self.tier3.forward(
            logits        = logits,
            hidden_states = hidden_states,
            fingerprint   = t2_result.fingerprint,
            gate_value    = t2_result.gate_value,
        )

        # ── Step 4: Generate ──────────────────────────────────────────────────
        if self.mode == "residual":
            response, _ = self.backbone.generate_with_residual(
                user_text             = user_text,
                personality_residual  = t2_result.personality_residual,
                max_new_tokens        = self.max_tokens,
                temperature           = self.temperature,
            )
        else:
            system_prompt = self.tier2.traits_as_system_prompt(t2_result.trait_vector)
            response = self.backbone.generate(
                user_text     = user_text,
                system_prompt = system_prompt,
                max_new_tokens = self.max_tokens,
                temperature    = self.temperature,
            )

        # ── Step 5: STDP update ───────────────────────────────────────────────
        self.tier2.stdp_update()

        # ── Step 6: Record ────────────────────────────────────────────────────
        turn = {
            "user":       user_text,
            "response":   response,
            "confidence": t3_result.confidence,
            "gap_flag":   t3_result.gap_flag,
            "inquiry":    t3_result.inquiry_type,
        }
        self._history.append(turn)

        return PipelineResult(
            response = response,
            tier2    = t2_result,
            tier3    = t3_result,
        )

    # ── Follow-up inquiry (when gap_flag is set) ──────────────────────────────

    def generate_inquiry(self, context: str, t3: MetaCognitionResult) -> str:
        """
        Generate a precise inquiry question when Tier 3 detects a knowledge gap.
        This replaces the old InquiryLayer Haiku call — uses Qwen3 locally.
        """
        assert self.backbone, "Call .load() first"

        inquiry_instructions = {
            "factual":              "Ask a precise factual question that would resolve this uncertainty.",
            "causal":               "Ask a causal question: what mechanism explains this?",
            "frontier":             "Ask a frontier question: what experiment would test this?",
            "assumption_inversion": "Invert a key assumption: what if the opposite were true?",
        }
        instruction = inquiry_instructions.get(t3.inquiry_type, "Ask a clarifying question.")

        system = (
            f"You are {self.agent_name}. You detected a knowledge gap (confidence={t3.confidence:.2f}). "
            f"{instruction} Be concise — one sentence only."
        )
        return self.backbone.generate(
            user_text     = f"Context: {context}",
            system_prompt = system,
            max_new_tokens = 80,
            temperature    = 0.6,
        )

    # ── Session management ────────────────────────────────────────────────────

    def reset_conversation(self):
        """Reset Tier 3 recurrent state between conversations."""
        if self.tier3:
            self.tier3.reset_state()
        self._history = []

    def save_weights(self):
        """Persist Tier 2 encoder/decoder and Tier 3 SSM weights."""
        if self.tier2:
            self.tier2.save()
        if self.tier3:
            self.tier3.save()

    def history_summary(self) -> list[dict]:
        return self._history[-10:]   # last 10 turns

    def __repr__(self) -> str:
        return (
            f"NeuralPipeline({self.agent_name!r}, "
            f"mode={self.mode!r}, "
            f"turns={len(self._history)})"
        )


# ── Interactive REPL ──────────────────────────────────────────────────────────

def run_repl(agent_name: str = "Jarvis", mode: str = "residual"):
    """
    Interactive REPL: talk to the pipeline from the terminal.
    No Anthropic API needed.
    """
    pipeline = NeuralPipeline(agent_name, mode=mode).load()
    print(f"\n  [{agent_name}] Three-tier neural pipeline ready.")
    print(f"  Tier 1: Qwen3-1.7B (local)  |  Tier 2: SNN+STDP  |  Tier 3: Selective SSM")
    print(f"  Type 'quit' to exit, 'save' to persist weights, 'reset' to clear session.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user:
            continue
        if user.lower() == "quit":
            break
        if user.lower() == "save":
            pipeline.save_weights()
            print("  [Saved]\n")
            continue
        if user.lower() == "reset":
            pipeline.reset_conversation()
            print("  [Session reset]\n")
            continue

        result = pipeline.chat(user)
        print(f"\n{agent_name}: {result.response}")
        print(f"  [{result.tier2.render()}]")
        print(f"  [{result.tier3.render()}]")

        # Auto-generate inquiry if gap detected
        if result.gap_flag:
            inquiry = pipeline.generate_inquiry(user, result.tier3)
            print(f"  [Gap inquiry]: {inquiry}")

        print()

    pipeline.save_weights()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    agent = sys.argv[1] if len(sys.argv) > 1 else "Jarvis"
    mode  = sys.argv[2] if len(sys.argv) > 2 else "residual"
    run_repl(agent_name=agent, mode=mode)
