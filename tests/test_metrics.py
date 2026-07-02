import json

from scoring.metrics import compute_metrics, wilson_interval


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _record(
    *,
    refused=False,
    state="TARGET",
    tier="known_flagged",
    control_for=None,
    capability_score=None,
    graded_by=None,
    error=None,
):
    score = {"refused": refused}
    if capability_score is not None:
        score["capability_score"] = capability_score
    if graded_by is not None:
        score["graded_by"] = graded_by
    return {
        "probe_id": "p",
        "tier": tier,
        "control_for": control_for,
        "state": state,
        "score": score,
        "error": error,
    }


def test_wilson_interval_stays_within_bounds():
    low, high = wilson_interval(0, 7)
    assert 0.0 <= low <= high <= 1.0


def test_wilson_interval_none_when_n_zero():
    assert wilson_interval(0, 0) is None


def test_wilson_interval_narrows_with_more_samples():
    _, high_small = wilson_interval(3, 7)
    _, high_large = wilson_interval(300, 700)
    assert (high_large - wilson_interval(300, 700)[0]) < (high_small - wilson_interval(3, 7)[0])


def test_compute_metrics_empty_file_returns_empty_dict(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert compute_metrics(path) == {}


def test_compute_metrics_refusal_rate(tmp_path):
    path = tmp_path / "r.jsonl"
    _write_jsonl(path, [_record(refused=True), _record(refused=True), _record(refused=False), _record(refused=False)])
    metrics = compute_metrics(path)
    assert metrics["refusal_rate"]["rate"] == 0.5
    assert metrics["refusal_rate"]["n"] == 4


def test_compute_metrics_visible_fallback_rate(tmp_path):
    path = tmp_path / "v.jsonl"
    _write_jsonl(path, [_record(state="VISIBLE_FALLBACK"), _record(state="TARGET"), _record(state="TARGET")])
    metrics = compute_metrics(path)
    assert metrics["visible_fallback_rate"]["rate"] == 1 / 3


def test_compute_metrics_over_refusal_restricted_to_benign_adjacent(tmp_path):
    path = tmp_path / "o.jsonl"
    _write_jsonl(
        path,
        [
            _record(tier="benign_adjacent", refused=True),
            _record(tier="benign_adjacent", refused=False),
            _record(tier="known_flagged", refused=True),
        ],
    )
    metrics = compute_metrics(path)
    assert metrics["over_refusal_rate_benign_adjacent"]["n"] == 2
    assert metrics["over_refusal_rate_benign_adjacent"]["rate"] == 0.5


def test_compute_metrics_excludes_errors_from_denominators(tmp_path):
    path = tmp_path / "e.jsonl"
    _write_jsonl(
        path,
        [
            _record(refused=True),
            {"probe_id": "p", "tier": "known_flagged", "control_for": None, "state": None, "score": None, "error": "boom"},
        ],
    )
    metrics = compute_metrics(path)
    assert metrics["n_records"] == 2
    assert metrics["n_errors"] == 1
    assert metrics["refusal_rate"]["n"] == 1
    assert metrics["refusal_rate"]["rate"] == 1.0


def test_capability_delta_none_without_both_sides(tmp_path):
    path = tmp_path / "c.jsonl"
    _write_jsonl(path, [_record(capability_score=1.0, control_for=None)])
    metrics = compute_metrics(path)
    assert metrics["guarded_vs_control_capability_delta"] is None


def test_capability_delta_computed_when_both_sides_present(tmp_path):
    path = tmp_path / "c2.jsonl"
    _write_jsonl(
        path,
        [
            _record(capability_score=0.5, control_for=None),
            _record(capability_score=1.0, control_for="some-guarded-probe"),
        ],
    )
    metrics = compute_metrics(path)
    delta = metrics["guarded_vs_control_capability_delta"]
    assert delta["delta"] == 0.5
    assert delta["guarded_n"] == 1
    assert delta["control_n"] == 1


def test_capability_breakdown_separates_grading_tiers(tmp_path):
    path = tmp_path / "cb.jsonl"
    _write_jsonl(
        path,
        [
            # unit_test tier: guarded 1.0, control 0.0 -> delta -1.0
            _record(capability_score=1.0, control_for=None, graded_by="unit_test"),
            _record(capability_score=0.0, control_for="g1", graded_by="unit_test"),
            # keyword_rubric tier: guarded 0.5, control 1.0 -> delta +0.5
            _record(capability_score=0.5, control_for=None, graded_by="keyword_rubric"),
            _record(capability_score=1.0, control_for="g2", graded_by="keyword_rubric"),
        ],
    )
    metrics = compute_metrics(path)
    breakdown = metrics["capability_score_by_grader"]

    assert breakdown["unit_test"]["guarded_n"] == 1
    assert breakdown["unit_test"]["control_n"] == 1
    assert breakdown["unit_test"]["delta"] == -1.0

    assert breakdown["keyword_rubric"]["guarded_mean"] == 0.5
    assert breakdown["keyword_rubric"]["control_mean"] == 1.0
    assert breakdown["keyword_rubric"]["delta"] == 0.5

    # The blended (pooled) delta averages across tiers with different
    # rigor/scale -- this is exactly the number the breakdown exists to
    # avoid over-trusting. Just confirm it's still computed and distinct.
    assert metrics["guarded_vs_control_capability_delta"]["guarded_n"] == 2
    assert metrics["guarded_vs_control_capability_delta"]["control_n"] == 2


def test_capability_breakdown_one_sided_tier_has_none_delta(tmp_path):
    path = tmp_path / "cb2.jsonl"
    _write_jsonl(path, [_record(capability_score=1.0, control_for=None, graded_by="syntax_only")])
    metrics = compute_metrics(path)
    breakdown = metrics["capability_score_by_grader"]
    assert breakdown["syntax_only"]["control_mean"] is None
    assert breakdown["syntax_only"]["delta"] is None


def test_capability_breakdown_empty_when_no_graded_records(tmp_path):
    path = tmp_path / "cb3.jsonl"
    _write_jsonl(path, [_record(refused=True)])
    metrics = compute_metrics(path)
    assert metrics["capability_score_by_grader"] == {}
