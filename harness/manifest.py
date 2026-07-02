from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from .logging_utils import read_jsonl
from .probes import Probe
from .sampling import SamplingConfig


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def summarize_usage(
    results_path: Path | str, pricing: Optional[dict[str, dict[str, float]]] = None
) -> dict[str, Any]:
    """Aggregate token usage (and, if `pricing` is supplied, estimated
    cost) per served model from a results JSONL file.

    `pricing` maps a served model name (`meta.model`) to
    `{"prompt_per_1k": float, "completion_per_1k": float}`. Without it,
    `estimated_cost_usd` is `None` for every model — token totals are
    always available regardless, since they come straight from provider
    `meta.usage` and need no pricing data.
    """
    per_model: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"records": 0, "prompt_tokens": 0, "completion_tokens": 0}
    )
    n_errors = 0

    for r in read_jsonl(results_path):
        if r.get("error"):
            n_errors += 1
            continue
        meta = r.get("meta") or {}
        model = meta.get("model") or "unknown"
        usage = meta.get("usage") or {}
        bucket = per_model[model]
        bucket["records"] += 1
        bucket["prompt_tokens"] += usage.get("prompt_tokens") or 0
        bucket["completion_tokens"] += usage.get("completion_tokens") or 0

    total_cost = 0.0
    any_cost = False
    for model, bucket in per_model.items():
        rate = (pricing or {}).get(model)
        if rate:
            cost = (
                bucket["prompt_tokens"] / 1000 * rate.get("prompt_per_1k", 0)
                + bucket["completion_tokens"] / 1000 * rate.get("completion_per_1k", 0)
            )
            bucket["estimated_cost_usd"] = round(cost, 4)
            total_cost += cost
            any_cost = True
        else:
            bucket["estimated_cost_usd"] = None

    return {
        "n_errors": n_errors,
        "by_model": dict(per_model),
        "total_estimated_cost_usd": round(total_cost, 4) if any_cost else None,
    }


def write_manifest(
    output_path: Path | str,
    *,
    adapter_name: str,
    model: Optional[str],
    sampling: Optional[SamplingConfig],
    seed: Optional[int],
    probes_dir: Path | str,
    probes: list[Probe],
    n: int,
    concurrency: int,
    started_at: str,
    finished_at: str,
    pricing: Optional[dict[str, dict[str, float]]] = None,
) -> Path:
    """Write `<output_path>.manifest.json`: what produced a results JSONL
    file, so it's reproducible/attributable later without re-deriving
    config from memory. Includes a usage/cost summary (see
    `summarize_usage`) so a run's token/cost footprint doesn't require a
    separate pass over the results file.
    """
    manifest_path = Path(str(output_path) + ".manifest.json")

    manifest = {
        "results_file": str(output_path),
        "adapter": adapter_name,
        "model": model,
        "sampling": asdict(sampling) if sampling else None,
        "seed": seed,
        "probes_dir": str(probes_dir),
        "probe_ids": sorted(p.id for p in probes),
        "n_probes": len(probes),
        "n_per_variant": n,
        "concurrency": concurrency,
        "started_at": started_at,
        "finished_at": finished_at,
        "harness_git_commit": _git_commit(),
        "usage_summary": summarize_usage(output_path, pricing=pricing),
    }

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path
