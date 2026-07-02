from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling parameters applied uniformly across every real adapter.

    A guarded-vs-control or model-vs-model comparison is only fair if every
    adapter is sampled the same way — otherwise a rate difference could be
    an artifact of one adapter running hotter, with a longer budget, or
    with a different system prompt than another, rather than a real
    behavioral difference. `temperature=0.0` by default for maximum
    determinism/repeatability across runs.
    """

    temperature: float = 0.0
    max_tokens: int = 1024
    system_prompt: Optional[str] = None
