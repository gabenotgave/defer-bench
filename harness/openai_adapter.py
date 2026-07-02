from __future__ import annotations

import os
import time
from typing import Optional

from .adapter import ModelAdapter, ModelResponse
from .sampling import SamplingConfig


class OpenAIAdapter(ModelAdapter):
    """Adapter for the OpenAI Chat Completions API.

    Metadata disclosure, as of the standard `chat.completions` endpoint:
      - `meta.model` is populated from the response's `model` field, which
        OpenAI documents as the model that actually served the request.
      - `meta.fallback_reason` is always `None` here: the API exposes no
        distinct "you were routed elsewhere, here's why" field.
      - CAVEAT: requesting a bare alias (e.g. "gpt-4o") resolves
        server-side to a dated snapshot (e.g. "gpt-4o-2024-08-06") in the
        response's `model` field, on every call. That resolution is NOT a
        fallback, but it is indistinguishable from one under this harness's
        `requested_model != model` check. Always construct this adapter
        with a fully-pinned, dated model string so a mismatch is a
        meaningful signal. See docs/providers.md.

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
                "OpenAIAdapter requires the 'openai' package. "
                "Install with: pip install 'defer-bench[openai]'"
            ) from e

        self.model = model
        self.sampling = sampling or SamplingConfig()
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

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
