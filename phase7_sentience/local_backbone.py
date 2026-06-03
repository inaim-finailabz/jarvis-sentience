"""
Local Backbone — Qwen3-1.7B inference with hidden state exposure.

Loads the model once, frozen. Exposes:
  encode()   → last_hidden [2048], all hidden_states, logits
  generate() → response text (optionally with personality system prompt)
  chat()     → encode + generate in one call, returns full BackboneResult

No Anthropic API required. Runs on MPS (Apple Silicon) or CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Model path ────────────────────────────────────────────────────────────────

_HF_CACHE = Path("/Volumes/ExternalDisk/huggingface/hub")
QWEN3_PATH = str(
    _HF_CACHE
    / "models--Qwen--Qwen3-1.7B"
    / "snapshots"
    / "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)

# ── Device ────────────────────────────────────────────────────────────────────

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class BackboneResult:
    response:      str
    last_hidden:   torch.Tensor          # [hidden_size] mean-pooled last layer
    hidden_states: tuple                 # all layer hidden states (layer, 1, seq, d)
    logits:        torch.Tensor          # [1, seq_len, vocab_size] from encoding pass
    prompt_len:    int                   # number of input tokens

    def token_entropy(self) -> float:
        """Shannon entropy of the last-token logit distribution (nats)."""
        last = self.logits[0, -1, :].float()
        probs = torch.softmax(last, dim=-1)
        return float(-torch.sum(probs * torch.log(probs + 1e-9)).item())

    def top1_prob(self) -> float:
        last = self.logits[0, -1, :].float()
        return float(torch.softmax(last, dim=-1).max().item())

    def perplexity_proxy(self) -> float:
        return self.token_entropy() / math.log(2)  # bits


# ── Backbone ──────────────────────────────────────────────────────────────────

class LocalBackbone:
    """
    Frozen Qwen3-1.7B on local disk.

    Usage:
        bb = LocalBackbone().load()
        result = bb.chat("What is photosynthesis?")
        print(result.response)
        print(result.last_hidden.shape)   # torch.Size([2048])
    """

    def __init__(
        self,
        model_path: str = QWEN3_PATH,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_path = model_path
        self.device = device or DEVICE
        self.dtype = dtype
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> "LocalBackbone":
        if self._model is not None:
            return self
        print(f"  [Backbone] Loading Qwen3-1.7B on {self.device} ({self.dtype})...", flush=True)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map=str(self.device),
            trust_remote_code=True,
        )
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        n = sum(p.numel() for p in self._model.parameters()) / 1e9
        print(f"  [Backbone] Ready. {n:.2f}B params, hidden={self.hidden_size}", flush=True)
        return self

    @property
    def hidden_size(self) -> int:
        return self._model.config.hidden_size

    # ── Encode ────────────────────────────────────────────────────────────────

    def encode(self, text: str) -> tuple[torch.Tensor, tuple, torch.Tensor]:
        """
        Forward pass only (no generation). Returns:
          last_hidden   [hidden_size] — mean-pooled final layer, float32, on DEVICE
          hidden_states — tuple of [1, seq_len, hidden_size] per layer
          logits        — [1, seq_len, vocab_size]
        """
        assert self._model is not None, "Call .load() first"
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)

        # Mean-pool over sequence dimension for the last layer
        last_hidden = out.hidden_states[-1][0].float().mean(dim=0)  # [hidden_size]
        return last_hidden, out.hidden_states, out.logits

    # ── Generate ──────────────────────────────────────────────────────────────

    def generate(
        self,
        user_text: str,
        system_prompt: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a response. If system_prompt is provided, uses chat template.
        Returns decoded response text (new tokens only).
        """
        assert self._model is not None, "Call .load() first"

        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_text},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = user_text

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_ids = out[0, prompt_len:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True)

    # ── Full chat (encode + generate) ─────────────────────────────────────────

    def chat(
        self,
        user_text: str,
        system_prompt: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> BackboneResult:
        """
        Runs encode (for hidden states / Tier 2/3 input) then generate (for response).
        Two forward passes — encode is cheap (no KV cache growth), generate is the main cost.
        """
        prompt_for_encode = (
            f"System: {system_prompt}\n\nUser: {user_text}"
            if system_prompt else user_text
        )
        last_hidden, hidden_states, logits = self.encode(prompt_for_encode)

        inputs_len = self._tokenizer(prompt_for_encode, return_tensors="pt")["input_ids"].shape[1]
        response = self.generate(user_text, system_prompt=system_prompt,
                                 max_new_tokens=max_new_tokens, temperature=temperature)

        return BackboneResult(
            response=response,
            last_hidden=last_hidden,
            hidden_states=hidden_states,
            logits=logits,
            prompt_len=inputs_len,
        )

    # ── Inject residual and resample last token ────────────────────────────────

    def generate_with_residual(
        self,
        user_text: str,
        personality_residual: torch.Tensor,    # [hidden_size]
        system_prompt: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> tuple[str, BackboneResult]:
        """
        Generate response with personality residual injected into the last hidden state
        before the LM head decides the first token. Subsequent tokens generate normally.

        This is the genuine weight-space personality shaping:
          modified_logits = LM_head(last_hidden + personality_residual)
          first_token ← sample(modified_logits)
          then continue autoregressive generation normally.
        """
        assert self._model is not None, "Call .load() first"

        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_text},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = user_text

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Forward pass to get hidden states
        with torch.no_grad():
            fwd = self._model(**inputs, output_hidden_states=True)

        last_hidden_all = fwd.hidden_states[-1]          # [1, seq, hidden_size]
        last_token_h    = last_hidden_all[0, -1, :].float()  # [hidden_size]

        # Apply personality residual
        residual_f = personality_residual.float().to(self.device)
        modified_h = last_token_h + residual_f            # [hidden_size]

        # Re-run LM head on the modified last token hidden state
        modified_h_bf = modified_h.to(self.dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
        with torch.no_grad():
            modified_logits = self._model.lm_head(modified_h_bf).float()    # [1, 1, vocab]

        # Sample first token from modified distribution
        logits_1 = modified_logits[0, 0, :] / max(temperature, 1e-6)
        probs_1  = torch.softmax(logits_1, dim=-1)
        first_token_id = torch.multinomial(probs_1, num_samples=1)          # [1]

        # Continue generation from the first sampled token
        continued_input_ids = torch.cat(
            [inputs["input_ids"], first_token_id.unsqueeze(0)], dim=1
        )
        with torch.no_grad():
            out = self._model.generate(
                input_ids=continued_input_ids,
                max_new_tokens=max_new_tokens - 1,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_ids  = out[0, prompt_len:]
        response = self._tokenizer.decode(new_ids, skip_special_tokens=True)

        result = BackboneResult(
            response=response,
            last_hidden=last_token_h,
            hidden_states=fwd.hidden_states,
            logits=fwd.logits,
            prompt_len=prompt_len,
        )
        return response, result


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    bb = LocalBackbone().load()
    result = bb.chat("What is the role of mitochondria in cellular respiration?")
    print("\nResponse:", result.response[:400])
    print(f"\nhidden shape: {result.last_hidden.shape}")
    print(f"entropy: {result.token_entropy():.3f}  top1: {result.top1_prob():.4f}")
    print(f"perplexity proxy: {result.perplexity_proxy():.2f} bits")
