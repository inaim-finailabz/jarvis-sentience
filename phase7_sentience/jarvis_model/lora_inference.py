"""
JarvisLoRA — inference with trained personality adapters.

Once LoRA adapters are trained (train_lora.py), this replaces
the JSON trait-injection approach with actual learned weights.

The personality IS the weights, not a system prompt.

Usage:
    from lora_inference import JarvisLoRA, JarvisEnsemble

    # Single personality
    jarvis = JarvisLoRA("explorer")
    response = jarvis.generate("Why can axolotls regenerate limbs but humans cannot?")

    # All personalities answer the same question (shows divergence)
    ensemble = JarvisEnsemble()
    responses = ensemble.all("What is consciousness?")
    for p, r in responses.items():
        print(f"[{p}] {r[:150]}")
"""

import os
import sys
from pathlib import Path

_ROOT        = Path(__file__).parent
ADAPTERS_DIR = _ROOT / "lora_adapters"

MODEL_PATH = (
    "/Volumes/ExternalDisk/huggingface/hub"
    "/models--Qwen--Qwen3-1.7B/snapshots"
    "/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)

PERSONALITIES = ["explorer", "scientist", "critic", "synthesiser", "pragmatist"]


class JarvisLoRA:
    """
    Single-personality inference using a trained LoRA adapter.
    Lazy-loads the model on first call.
    """

    def __init__(self, personality: str = "explorer", temperature: float = 0.7):
        self.personality = personality
        self.temperature = temperature
        self._model      = None
        self._tokenizer  = None
        self._adapter    = str(ADAPTERS_DIR / personality)

    def _load(self):
        if self._model is not None:
            return
        try:
            from mlx_lm import load
            print(f"  [JarvisLoRA] Loading {self.personality} adapter...")
            self._model, self._tokenizer = load(
                MODEL_PATH,
                adapter_path=self._adapter,
            )
            print(f"  [JarvisLoRA] {self.personality} ready.")
        except Exception as ex:
            raise RuntimeError(
                f"Could not load {self.personality} adapter: {ex}\n"
                f"Run: python3.14 train_lora.py --personality {self.personality}"
            )

    def generate(self, prompt: str, max_tokens: int = 400) -> str:
        self._load()
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = f"<|user|>\n{prompt}\n<|assistant|>\n"

        return generate(
            self._model, self._tokenizer,
            prompt=text,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=self.temperature),
            verbose=False,
        )

    def adapter_exists(self) -> bool:
        p = Path(self._adapter)
        return (p / "adapters.safetensors").exists() or (p / "adapters.npz").exists()


class JarvisEnsemble:
    """
    All five personality adapters loaded together.
    Ask the same question to all — shows how personality shapes response.
    """

    def __init__(self):
        self.agents = {p: JarvisLoRA(p) for p in PERSONALITIES}

    def all(self, prompt: str, max_tokens: int = 300) -> dict[str, str]:
        return {p: agent.generate(prompt, max_tokens)
                for p, agent in self.agents.items()
                if agent.adapter_exists()}

    def show_divergence(self, prompt: str):
        """Print all responses side by side to visualise personality divergence."""
        print(f"\n{'━'*70}")
        print(f"  Prompt: {prompt}")
        print(f"{'━'*70}")
        for p, response in self.all(prompt).items():
            print(f"\n  [{p.upper()}]")
            for line in response[:400].split("\n"):
                print(f"    {line}")


def adapters_available() -> list[str]:
    return [p for p in PERSONALITIES
            if (ADAPTERS_DIR / p / "adapters.safetensors").exists()
            or (ADAPTERS_DIR / p / "adapters.npz").exists()]


if __name__ == "__main__":
    available = adapters_available()
    if not available:
        print("\n  No trained adapters found.")
        print("  Run: python3.14 generate_training_data.py && python3.14 train_lora.py")
        sys.exit(0)

    print(f"\n  Available adapters: {available}")
    print(f"  Testing on: 'Why can humans not regenerate limbs?'\n")

    prompt = "Why can humans not regenerate limbs the way axolotls can?"
    ensemble = JarvisEnsemble()
    ensemble.show_divergence(prompt)
