from __future__ import annotations

from typing import Protocol

from harness.adapter import ModelResponse

TARGET = "TARGET"
VISIBLE_FALLBACK = "VISIBLE_FALLBACK"
# Reserved for v2 — silent-fallback detection is intentionally not implemented in v1.
# See docs/limitations.md. `MetadataStateDetector` never emits this label; v1 only
# ever classifies a response as TARGET or VISIBLE_FALLBACK.
SILENT_FALLBACK = "SILENT_FALLBACK"


class StateDetector(Protocol):
    """Classifies a response as TARGET or VISIBLE_FALLBACK using only
    provider metadata plus deterministic rules.

    This is the ONLY classifier headline metrics (refusal rate,
    visible-fallback rate, over-refusal rate, capability delta) are allowed
    to read. It must never depend on the silent-fallback heuristic below.
    """

    def classify(self, response: ModelResponse) -> str:
        ...


class MetadataStateDetector:
    """Reference implementation: TARGET vs VISIBLE_FALLBACK from provider meta."""

    def classify(self, response: ModelResponse) -> str:
        meta = response.get("meta", {})
        requested = meta.get("requested_model")
        served = meta.get("model")
        fallback_reason = meta.get("fallback_reason")

        if fallback_reason or (requested and served and requested != served):
            return VISIBLE_FALLBACK
        return TARGET


class SilentFallbackHeuristic(Protocol):
    """Pluggable, best-effort LOWER BOUND signal for silent degradation that
    provider metadata does not disclose.

    Reserved for v2 — silent-fallback detection is intentionally not
    implemented in v1. See docs/limitations.md.

    This Protocol is deliberately left unfrozen: v1 ships no opinion on what
    the real v2 detector's shape should be beyond "returns a score for a
    single response." Whatever v2 implements here, its output MUST continue
    to be logged in its own field (`silent_fallback_heuristic`) and MUST NOT
    feed into refusal rate, visible-fallback rate, over-refusal rate, or the
    capability delta. Treat it as a research signal, not ground truth.
    """

    def score(self, response: ModelResponse) -> float | None:
        """Return a score in [0, 1], or None if not measured; higher means
        more likely silently degraded."""
        ...


class PlaceholderSilentFallbackHeuristic:
    """v1 placeholder — always reports "not measured".

    Reserved for v2 — silent-fallback detection is intentionally not
    implemented in v1. See docs/limitations.md.

    v2 will likely implement this as a rate-based, difficulty-calibrated
    analysis over repeated runs already captured in the JSONL logs (see
    docs/limitations.md), not a per-response classifier. This class is not
    dead code: it's the explicit "unmeasured" default so the JSONL field
    always exists with an unambiguous value in v1.
    """

    def score(self, response: ModelResponse) -> float | None:
        return None  # Not measured in v1 — never a real score, never 0.0.
