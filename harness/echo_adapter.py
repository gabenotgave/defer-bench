from __future__ import annotations

import random
import time

from .adapter import ModelAdapter, ModelResponse

REFUSAL_TRIGGER_WORDS = ("exploit", "malware", "ransomware", "reverse shell")


class EchoAdapter(ModelAdapter):
    """Deterministic, offline adapter for local testing and development.

    Makes no network calls. It roughly simulates TARGET / VISIBLE_FALLBACK
    behavior (and a naive refusal) so the rest of the pipeline (runner,
    scorers, metrics) can be exercised end-to-end without credentials.
    """

    def __init__(
        self,
        model_name: str = "echo-mock-1",
        fallback_rate: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.fallback_model_name = "echo-mock-1-lite"
        self.fallback_rate = fallback_rate
        self._rng = random.Random(seed)

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start = time.monotonic()
        refuse = any(word in prompt.lower() for word in REFUSAL_TRIGGER_WORDS)
        fell_back = self._rng.random() < self.fallback_rate

        if refuse:
            text = "I can't help with that request."
            model_used = self.model_name
            fallback_reason = None
        elif fell_back:
            text = f"[lite response] {prompt[:60]}..."
            model_used = self.fallback_model_name
            fallback_reason = "capacity_routing"
        else:
            text = f"Echo: {prompt[:200]}"
            model_used = self.model_name
            fallback_reason = None

        latency_ms = (time.monotonic() - start) * 1000
        return {
            "text": text,
            "meta": {
                "model": model_used,
                "requested_model": self.model_name,
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(text.split()),
                },
                "latency_ms": latency_ms,
                "fallback_reason": fallback_reason,
            },
        }
