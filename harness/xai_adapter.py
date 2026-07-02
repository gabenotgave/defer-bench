from __future__ import annotations

import os
import time
from typing import Optional

from .adapter import ModelAdapter, ModelResponse
from .sampling import SamplingConfig

XAI_BASE_URL = "https://api.x.ai/v1"


class XAIAdapter(ModelAdapter):
    """Adapter for xAI's Grok API.

    Uses the `openai` client pointed at xAI's base URL, since xAI's chat
    completions endpoint mirrors the OpenAI Chat Completions schema. This
    is the only adapter that borrows another provider's SDK — it is still
    fully contained to this file.

    Metadata disclosure:
      - `meta.model` is populated from the response's `model` field.
      - `meta.fallback_reason` is always `None` here: no routing-disclosure
        field exists in this API.
      - Same pinned-model-string caveat as `OpenAIAdapter` applies — pass a
        fully-pinned model string, not a bare alias. See docs/providers.md.

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
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "XAIAdapter requires the 'openai' package (used as an OpenAI-compatible "
                "client for xAI's API). Install with: pip install 'defer-bench[xai]'"
            ) from e

        self.model = model
        self.sampling = sampling or SamplingConfig()
        self._client = OpenAI(
            api_key=api_key or os.environ.get("XAI_API_KEY"),
            base_url=XAI_BASE_URL,
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
                # Not exposed by this API. See class docstring.
                "fallback_reason": None,
            },
        }
