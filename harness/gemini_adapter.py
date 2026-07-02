from __future__ import annotations

import os
import time
from typing import Optional

from .adapter import ModelAdapter, ModelResponse
from .sampling import SamplingConfig


class GeminiAdapter(ModelAdapter):
    """Adapter for Google's Gemini API, via the `google-genai` SDK.

    Metadata disclosure:
      - `meta.model` is populated from the response's `model_version`
        field, which the SDK documents as the specific model version that
        served the request.
      - `meta.fallback_reason` is always `None` here: the API exposes no
        distinct "you were routed elsewhere, here's why" field.
      - CAVEAT: as with the other adapters, requesting a bare alias (e.g.
        "gemini-2.0-flash") can resolve server-side to a specific dated
        version in `model_version`, indistinguishable from an actual
        fallback under this harness's `requested_model != model` check.
        Always construct this adapter with a fully-pinned, dated model
        string. See docs/providers.md.

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
            from google import genai
        except ImportError as e:
            raise ImportError(
                "GeminiAdapter requires the 'google-genai' package. "
                "Install with: pip install 'defer-bench[gemini]'"
            ) from e

        self.model = model
        self.sampling = sampling or SamplingConfig()
        self._client = genai.Client(
            api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=self.sampling.temperature,
            max_output_tokens=self.sampling.max_tokens,
            system_instruction=self.sampling.system_prompt or None,
            **kwargs,
        )

        start = time.monotonic()
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        latency_ms = (time.monotonic() - start) * 1000

        usage = response.usage_metadata
        return {
            "text": response.text or "",
            "meta": {
                "model": response.model_version or self.model,
                "requested_model": self.model,
                "usage": {
                    "prompt_tokens": usage.prompt_token_count if usage else None,
                    "completion_tokens": usage.candidates_token_count if usage else None,
                },
                "latency_ms": latency_ms,
                # Not exposed by this API. See class docstring.
                "fallback_reason": None,
            },
        }
