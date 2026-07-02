from __future__ import annotations

from typing import Optional, Protocol, TypedDict


class ResponseMeta(TypedDict, total=False):
    model: str
    requested_model: str
    usage: dict
    latency_ms: float
    fallback_reason: Optional[str]


class ModelResponse(TypedDict):
    text: str
    meta: ResponseMeta


class ModelAdapter(Protocol):
    """Thin interface every provider adapter must implement.

    Core harness/scoring code depends only on this Protocol, never on a
    concrete provider SDK, so the benchmark can run against any API model
    with no privileged access and no per-model training.
    """

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        ...
