from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _fmt_rate(metric: Optional[dict[str, Any]]) -> str:
    if metric is None:
        return "n/a"
    low, high = metric["ci95"]
    return f"{metric['rate']:.1%} (n={metric['n']}, 95% CI {low:.1%}-{high:.1%})"


def _fmt_delta(metric: Optional[dict[str, Any]]) -> str:
    if metric is None:
        return "n/a"
    return f"{metric['delta']:+.2f} (guarded_n={metric['guarded_n']}, control_n={metric['control_n']})"


def _fmt_mean(mean: Optional[float], n: int) -> str:
    if mean is None:
        return "n/a"
    return f"{mean:.2f} (n={n})"


def load_manifest(results_path: Path | str) -> Optional[dict[str, Any]]:
    manifest_path = Path(str(results_path) + ".manifest.json")
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())


def _manifest_header(manifest: Optional[dict[str, Any]]) -> list[str]:
    if manifest is None:
        return []
    model = manifest.get("model") or f"{manifest.get('adapter')} (mock)"
    lines = [
        f"- **Adapter:** `{manifest.get('adapter')}`",
        f"- **Model:** `{model}`",
    ]
    usage = manifest.get("usage_summary") or {}
    cost = usage.get("total_estimated_cost_usd")
    if cost is not None:
        lines.append(f"- **Estimated cost:** ${cost:.4f}")
    return lines


def render_markdown_report(
    results_path: Path | str,
    overall: dict[str, Any],
    by_category: dict[str, dict[str, Any]],
    manifest: Optional[dict[str, Any]] = None,
) -> str:
    """Render a human-readable markdown report: overall metrics plus a
    per-category breakdown table. `run.py score --format markdown` is the
    CLI entrypoint. Reads only what `compute_metrics`/`compute_metrics_by_category`
    already computed — no new metric logic lives here.
    """
    lines = [f"# DEFER-Bench report: {Path(results_path).name}", ""]
    lines += _manifest_header(manifest)
    if manifest is not None:
        lines.append("")

    if not overall:
        lines.append("No records found.")
        return "\n".join(lines)

    lines += [
        "## Overall",
        "",
        f"- **Records:** {overall['n_records']} ({overall['n_errors']} errors excluded from rates below)",
        f"- **Refusal rate:** {_fmt_rate(overall['refusal_rate'])}",
        f"- **Visible-fallback rate:** {_fmt_rate(overall['visible_fallback_rate'])}",
        f"- **Over-refusal rate (benign_adjacent):** {_fmt_rate(overall['over_refusal_rate_benign_adjacent'])}",
        "",
    ]

    by_grader = overall.get("capability_score_by_grader") or {}
    if by_grader:
        lines += [
            "### Capability delta, by grading tier",
            "",
            "Reported per grading tier, not blended — a rubric's continuous "
            "keyword-coverage score and a syntax check's binary pass/fail aren't "
            "on the same scale, so averaging them together produces a number "
            "that can look directionally meaningful while mostly reflecting "
            "which tier happened to have more data. See docs/limitations.md.",
            "",
            "| Grading tier | Guarded mean | Control mean | Delta |",
            "|---|---|---|---|",
        ]
        for graded_by, entry in by_grader.items():
            lines.append(
                f"| {graded_by} | {_fmt_mean(entry['guarded_mean'], entry['guarded_n'])} "
                f"| {_fmt_mean(entry['control_mean'], entry['control_n'])} "
                f"| {_fmt_delta(entry) if entry['delta'] is not None else 'n/a'} |"
            )
        lines.append("")
    else:
        lines += ["_No capability-scored records in this run._", ""]

    lines += [
        "## By category",
        "",
        "| Category | Records | Refusal rate | Visible-fallback rate | Over-refusal rate |",
        "|---|---|---|---|---|",
    ]

    for category, metrics in by_category.items():
        if not metrics:
            continue
        lines.append(
            f"| {category} | {metrics['n_records']} | {_fmt_rate(metrics['refusal_rate'])} "
            f"| {_fmt_rate(metrics['visible_fallback_rate'])} "
            f"| {_fmt_rate(metrics['over_refusal_rate_benign_adjacent'])} |"
        )

    lines += [
        "",
        "_Per-category capability deltas aren't shown — grading-tier mix varies "
        "by category and a single blended number per row would have the same "
        "problem as the old headline blended delta. Use `run.py score "
        "<file> --format json` and filter to a category if you need this "
        "broken out further._",
        "",
        "_SILENT_FALLBACK is not measured in v1 and is excluded from this report. "
        "See docs/limitations.md and ROADMAP.md._",
    ]
    return "\n".join(lines)


def render_comparison_markdown(runs: dict[str, dict[str, Any]]) -> str:
    """Render a side-by-side markdown comparison across multiple results
    files (e.g. the same probe set run against different models/providers).

    `runs` maps a label (e.g. a model name) to
    `{"overall": <compute_metrics output>, "manifest": <manifest or None>}`.
    """
    labels = list(runs.keys())
    lines = ["# DEFER-Bench model comparison", ""]

    lines.append("| | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---|" * len(labels))

    def row(title: str, fmt, key: str) -> str:
        cells = [fmt((runs[label]["overall"] or {}).get(key)) for label in labels]
        return f"| **{title}** | " + " | ".join(cells) + " |"

    lines.append(
        "| **Model** | "
        + " | ".join((runs[label]["manifest"] or {}).get("model") or "n/a" for label in labels)
        + " |"
    )
    lines.append(
        "| **Records** | "
        + " | ".join(str((runs[label]["overall"] or {}).get("n_records", "n/a")) for label in labels)
        + " |"
    )
    lines.append(row("Refusal rate", _fmt_rate, "refusal_rate"))
    lines.append(row("Visible-fallback rate", _fmt_rate, "visible_fallback_rate"))
    lines.append(row("Over-refusal rate (benign_adjacent)", _fmt_rate, "over_refusal_rate_benign_adjacent"))

    # Capability delta is shown per grading tier, not blended (see
    # render_markdown_report's docstring / docs/limitations.md for why) —
    # one row per tier that appears in ANY run being compared, "n/a" for a
    # run with no data at that tier.
    all_tiers = sorted(
        {
            tier
            for label in labels
            for tier in ((runs[label]["overall"] or {}).get("capability_score_by_grader") or {})
        }
    )
    for tier in all_tiers:
        cells = []
        for label in labels:
            by_grader = (runs[label]["overall"] or {}).get("capability_score_by_grader") or {}
            entry = by_grader.get(tier)
            cells.append(_fmt_delta(entry) if entry and entry.get("delta") is not None else "n/a")
        lines.append(f"| **Capability delta ({tier})** | " + " | ".join(cells) + " |")

    lines += [
        "",
        "_Each column comes from a separate `run.py run` invocation against that model/provider — "
        "see docs/providers.md to ensure sampling parameters were held constant across runs for a fair "
        "comparison. SILENT_FALLBACK is not measured in v1; see docs/limitations.md._",
    ]
    return "\n".join(lines)
