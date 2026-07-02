import json

from scoring.metrics import compute_metrics, compute_metrics_by_category
from scoring.report import load_manifest, render_comparison_markdown, render_markdown_report


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _record(
    *,
    category="cybersecurity",
    refused=False,
    state="TARGET",
    tier="known_flagged",
    control_for=None,
    capability_score=None,
    graded_by=None,
):
    score = {"refused": refused}
    if capability_score is not None:
        score["capability_score"] = capability_score
    if graded_by is not None:
        score["graded_by"] = graded_by
    return {
        "probe_id": "p",
        "category": category,
        "tier": tier,
        "control_for": control_for,
        "state": state,
        "score": score,
        "error": None,
    }


def test_load_manifest_missing_returns_none(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text("")
    assert load_manifest(path) is None


def test_load_manifest_reads_sibling_file(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text("")
    manifest_path = tmp_path / "results.jsonl.manifest.json"
    manifest_path.write_text(json.dumps({"model": "test-model"}))
    manifest = load_manifest(path)
    assert manifest["model"] == "test-model"


def test_render_markdown_report_includes_per_category_breakdown(tmp_path):
    path = tmp_path / "results.jsonl"
    _write_jsonl(
        path,
        [
            _record(category="cybersecurity", refused=True),
            _record(category="cybersecurity", refused=False),
            _record(category="medical", refused=False),
        ],
    )
    overall = compute_metrics(path)
    by_category = compute_metrics_by_category(path)
    report = render_markdown_report(path, overall, by_category)

    assert "## By category" in report
    assert "cybersecurity" in report
    assert "medical" in report
    assert "SILENT_FALLBACK is not measured in v1" in report


def test_render_markdown_report_includes_capability_breakdown_by_tier(tmp_path):
    path = tmp_path / "results.jsonl"
    _write_jsonl(
        path,
        [
            _record(capability_score=1.0, control_for=None, graded_by="unit_test"),
            _record(capability_score=0.0, control_for="g1", graded_by="unit_test"),
            _record(capability_score=0.5, control_for=None, graded_by="keyword_rubric"),
            _record(capability_score=1.0, control_for="g2", graded_by="keyword_rubric"),
        ],
    )
    overall = compute_metrics(path)
    by_category = compute_metrics_by_category(path)
    report = render_markdown_report(path, overall, by_category)

    assert "Capability delta, by grading tier" in report
    assert "unit_test" in report
    assert "keyword_rubric" in report


def test_render_markdown_report_empty_results(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    report = render_markdown_report(path, compute_metrics(path), compute_metrics_by_category(path))
    assert "No records found." in report


def test_render_comparison_markdown_has_one_column_per_run(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write_jsonl(a, [_record(refused=True), _record(refused=False)])
    _write_jsonl(b, [_record(refused=False), _record(refused=False)])

    runs = {
        "Model A": {"overall": compute_metrics(a), "manifest": None},
        "Model B": {"overall": compute_metrics(b), "manifest": None},
    }
    report = render_comparison_markdown(runs)
    assert "Model A" in report
    assert "Model B" in report
    assert "Refusal rate" in report


def test_render_comparison_markdown_shows_capability_delta_per_tier_not_blended(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write_jsonl(
        a,
        [
            _record(capability_score=1.0, control_for=None, graded_by="unit_test"),
            _record(capability_score=0.0, control_for="g1", graded_by="unit_test"),
        ],
    )
    _write_jsonl(
        b,
        [
            _record(capability_score=0.5, control_for=None, graded_by="keyword_rubric"),
            _record(capability_score=1.0, control_for="g2", graded_by="keyword_rubric"),
        ],
    )
    runs = {
        "Model A": {"overall": compute_metrics(a), "manifest": None},
        "Model B": {"overall": compute_metrics(b), "manifest": None},
    }
    report = render_comparison_markdown(runs)

    assert "Capability delta" not in report.split("\n")[0]  # not a single blended row/title
    assert "Capability delta (unit_test)" in report
    assert "Capability delta (keyword_rubric)" in report
    # Model B has no unit_test data -> that row should show n/a for it
    unit_test_row = next(line for line in report.split("\n") if "Capability delta (unit_test)" in line)
    assert "n/a" in unit_test_row
