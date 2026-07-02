from __future__ import annotations

import os
import time
from typing import Optional

from .adapter import ModelAdapter, ModelResponse
from .sampling import SamplingConfig

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OllamaAdapter(ModelAdapter):
    """Adapter for a local (or remote) Ollama server.

    Uses the `openai` client pointed at Ollama's OpenAI-compatible
    endpoint, since Ollama's `/v1/chat/completions` mirrors the OpenAI
    Chat Completions schema — same approach as `XAIAdapter`.

    No API key is required: Ollama doesn't authenticate requests, but the
    `openai` client requires a non-empty `api_key` string to construct, so
    a placeholder is used if none is supplied.

    Metadata disclosure:
      - `meta.model` is populated from the response's `model` field, which
        Ollama echoes back as whatever tag you requested.
      - `meta.fallback_reason` is always `None`: no routing-disclosure
        field exists in this API — and unlike the cloud providers, there's
        nothing to disclose in the first place, since a local Ollama
        server only ever serves the exact model tag you pulled and
        requested. It doesn't resolve aliases to a different pinned
        snapshot server-side the way OpenAI/Anthropic/xAI/Gemini do.
      - Because of that, `harness/model_pinning.py` does NOT include an
        "ollama" entry — the pinned-vs-alias distinction this benchmark
        checks for cloud providers doesn't apply here, so `run.py` never
        warns/blocks on an Ollama model string. See docs/providers.md.

    `sampling` (temperature/max_tokens/system_prompt) defaults to
    `SamplingConfig()` and is applied the same way on every call, so
    comparisons against other adapters aren't confounded by mismatched
    sampling parameters. See harness/sampling.py.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        sampling: Optional[SamplingConfig] = None,
        base_url: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OllamaAdapter requires the 'openai' package (used as an OpenAI-compatible "
                "client for Ollama's API). Install with: pip install 'defer-bench[ollama]'"
            ) from e

        self.model = model
        self.sampling = sampling or SamplingConfig()
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OLLAMA_API_KEY") or "ollama",
            base_url=base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL,
        )

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        messages = []
        if self.sampling.system_prompt:
            messages.append({"role": "system", "content": self.sampling.system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.monotonic()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.sampling.temperature,
            max_tokens=self.sampling.max_tokens,
            **kwargs,
        )
        latency_ms = (time.monotonic() - start) * 1000

        usage = response.usage
        return {
            "text": response.choices[0].message.content or "",
            "meta": {
                "model": response.model,
                "requested_model": self.model,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else None,
                    "completion_tokens": usage.completion_tokens if usage else None,
                },
                "latency_ms": latency_ms,
                # Not exposed by this API, and not meaningful for a local
                # server either way. See class docstring.
                "fallback_reason": None,
            },
        }
