from __future__ import annotations

import os
import time
from typing import Optional

from .adapter import ModelAdapter, ModelResponse
from .sampling import SamplingConfig


class AnthropicAdapter(ModelAdapter):
    """Adapter for the Anthropic Messages API.

    Metadata disclosure:
      - `meta.model` is populated from the response's `model` field.
      - `meta.fallback_reason` is always `None` here: the Messages API
        exposes no field disclosing a routing/fallback decision.
      - CAVEAT: as with OpenAI, construct this adapter with a fully-pinned
        model string (e.g. "claude-sonnet-4-5-20250929"), not a bare family
        alias, so a `requested_model != model` mismatch is a meaningful
        signal rather than routine alias resolution. See docs/providers.md.

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
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "AnthropicAdapter requires the 'anthropic' package. "
                "Install with: pip install 'defer-bench[anthropic]'"
            ) from e

        self.model = model
        self.sampling = sampling or SamplingConfig()
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        create_kwargs = dict(
            model=self.model,
            max_tokens=self.sampling.max_tokens,
            temperature=self.sampling.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if self.sampling.system_prompt:
            create_kwargs["system"] = self.sampling.system_prompt
        create_kwargs.update(kwargs)

        start = time.monotonic()
        response = self._client.messages.create(**create_kwargs)
        latency_ms = (time.monotonic() - start) * 1000

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        usage = response.usage
        return {
            "text": text,
            "meta": {
                "model": response.model,
                "requested_model": self.model,
                "usage": {
                    "prompt_tokens": usage.input_tokens if usage else None,
                    "completion_tokens": usage.output_tokens if usage else None,
                },
                "latency_ms": latency_ms,
                # Not exposed by this API. See class docstring.
                "fallback_reason": None,
            },
        }
