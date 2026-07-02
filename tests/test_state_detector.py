from scoring.state_detector import (
    TARGET,
    VISIBLE_FALLBACK,
    MetadataStateDetector,
    PlaceholderSilentFallbackHeuristic,
)


def _response(model="m1", requested_model="m1", fallback_reason=None):
    return {"text": "hi", "meta": {"model": model, "requested_model": requested_model, "fallback_reason": fallback_reason}}


def test_matching_model_no_fallback_reason_is_target():
    detector = MetadataStateDetector()
    assert detector.classify(_response()) == TARGET


def test_mismatched_model_is_visible_fallback():
    detector = MetadataStateDetector()
    resp = _response(model="m1-lite", requested_model="m1")
    assert detector.classify(resp) == VISIBLE_FALLBACK


def test_explicit_fallback_reason_is_visible_fallback_even_if_model_matches():
    detector = MetadataStateDetector()
    resp = _response(model="m1", requested_model="m1", fallback_reason="capacity_routing")
    assert detector.classify(resp) == VISIBLE_FALLBACK


def test_placeholder_silent_fallback_heuristic_always_returns_none():
    heuristic = PlaceholderSilentFallbackHeuristic()
    assert heuristic.score(_response()) is None
