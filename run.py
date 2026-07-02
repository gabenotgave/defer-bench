#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Optional

from harness.adapter import ModelAdapter
from harness.echo_adapter import EchoAdapter
from harness.manifest import write_manifest
from harness.model_pinning import looks_pinned
from harness.probes import load_probes_dir
from harness.runner import run_probes
from harness.sampling import SamplingConfig
from scoring.metrics import compute_metrics, compute_metrics_by_category
from scoring.report import load_manifest, render_comparison_markdown, render_markdown_report
from scoring.scorers import REFUSAL_SCORER_NAME, LLMJudgeScorer, register_scorer
from scoring.state_detector import MetadataStateDetector

ROOT = Path(__file__).parent
DEFAULT_PROBES_DIR = ROOT / "probes"
DEFAULT_RESULTS_DIR = ROOT / "results"


def construct_adapter(
    adapter_name: str,
    *,
    model: Optional[str],
    api_key: Optional[str],
    sampling: Optional[SamplingConfig],
    seed: Optional[int] = None,
    base_url: Optional[str] = None,
) -> ModelAdapter:
    """Provider SDKs are only ever imported here, inside the branch that
    needs them, so `pip install`ing the base package never requires every
    provider's SDK. See docs/providers.md for what each adapter discloses.

    Shared by both the adapter under test (`build_adapter`) and, if
    `--refusal-scorer llm-judge` is selected, the judge adapter — so
    constructing a judge doesn't duplicate this branch.

    `base_url` is only used by `OllamaAdapter` today (a local/remote
    server address, not a fixed provider endpoint) — ignored by everything
    else.
    """
    if adapter_name == "echo":
        return EchoAdapter(seed=seed)

    if not model:
        sys.exit(
            f"--model (or --judge-model) is required for adapter {adapter_name!r} "
            "(pass a fully-pinned model string, not a bare alias — see docs/providers.md)"
        )

    sampling = sampling or SamplingConfig()

    if adapter_name == "openai":
        from harness.openai_adapter import OpenAIAdapter

        return OpenAIAdapter(model=model, api_key=api_key, sampling=sampling)
    if adapter_name == "anthropic":
        from harness.anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(model=model, api_key=api_key, sampling=sampling)
    if adapter_name == "xai":
        from harness.xai_adapter import XAIAdapter

        return XAIAdapter(model=model, api_key=api_key, sampling=sampling)
    if adapter_name == "gemini":
        from harness.gemini_adapter import GeminiAdapter

        return GeminiAdapter(model=model, api_key=api_key, sampling=sampling)
    if adapter_name == "ollama":
        from harness.ollama_adapter import OllamaAdapter

        return OllamaAdapter(model=model, api_key=api_key, sampling=sampling, base_url=base_url)

    raise ValueError(f"Unknown adapter {adapter_name!r}")


def build_adapter(args: argparse.Namespace) -> ModelAdapter:
    sampling = None
    if args.adapter != "echo":
        sampling = SamplingConfig(
            temperature=args.temperature, max_tokens=args.max_tokens, system_prompt=args.system_prompt
        )
    return construct_adapter(
        args.adapter,
        model=args.model,
        api_key=args.api_key,
        sampling=sampling,
        seed=args.seed,
        base_url=args.base_url,
    )


def maybe_configure_refusal_scorer(args: argparse.Namespace) -> None:
    """If `--refusal-scorer llm-judge` was chosen, build the judge adapter
    (defaulting to the same adapter/model being benchmarked, or an
    explicit `--judge-adapter`/`--judge-model`/`--judge-api-key`) and
    register it to back the `"refusal_keyword"` scorer name every probe's
    YAML already uses. Default (`--refusal-scorer keyword`) leaves the
    deterministic `RefusalKeywordScorer` in place — no behavior change,
    no extra cost, unless explicitly asked for.
    """
    if args.refusal_scorer != "llm-judge":
        return

    judge_adapter_name = args.judge_adapter or args.adapter
    judge_model = args.judge_model or args.model
    judge_api_key = args.judge_api_key or args.api_key
    judge_base_url = args.judge_base_url or (args.base_url if judge_adapter_name == args.adapter else None)
    judge_sampling = None
    if judge_adapter_name != "echo":
        judge_sampling = SamplingConfig(temperature=0.0, max_tokens=16, system_prompt=None)

    judge_adapter = construct_adapter(
        judge_adapter_name,
        model=judge_model,
        api_key=judge_api_key,
        sampling=judge_sampling,
        base_url=judge_base_url,
    )
    register_scorer(REFUSAL_SCORER_NAME, LLMJudgeScorer(judge_adapter))


def warn_or_block_on_unpinned_model(adapter_name: str, model: Optional[str], force: bool) -> None:
    """Warn (or block, without `--force`) when `model` doesn't look like a
    fully-pinned dated snapshot for a provider where that's checkable (see
    `harness/model_pinning.py`).

    `VISIBLE_FALLBACK` detection (`MetadataStateDetector`) depends
    entirely on `requested_model != model`, which is meaningless noise —
    not a real fallback signal — if `model` is a bare alias that resolves
    to a different string server-side on every call. See
    docs/providers.md.
    """
    if adapter_name == "echo" or not model:
        return

    pinned = looks_pinned(adapter_name, model)
    if pinned is not False:  # None (not checkable) or True (looks pinned): nothing to flag
        return

    message = (
        f"--model {model!r} doesn't look like a fully-pinned, dated snapshot for "
        f"--adapter {adapter_name}. VISIBLE_FALLBACK detection depends on requested_model != "
        "model; a bare alias can resolve to a different string server-side on every call, "
        "making that comparison meaningless noise rather than a real fallback signal. "
        "See docs/providers.md."
    )
    if force:
        print(f"WARNING: {message} Continuing because --force was passed.", file=sys.stderr)
        return
    sys.exit(f"{message}\nPass --force to run anyway (e.g. to intentionally test alias behavior).")


def cmd_run(args: argparse.Namespace) -> None:
    probes = load_probes_dir(args.probes_dir)
    if not probes:
        sys.exit(f"No probes found in {args.probes_dir}")

    warn_or_block_on_unpinned_model(args.adapter, args.model, args.force)

    adapter = build_adapter(args)
    state_detector = MetadataStateDetector()
    maybe_configure_refusal_scorer(args)

    pricing = None
    if args.pricing:
        pricing = json.loads(Path(args.pricing).read_text())

    sampling = None
    if args.adapter != "echo":
        sampling = SamplingConfig(
            temperature=args.temperature, max_tokens=args.max_tokens, system_prompt=args.system_prompt
        )

    output_path = Path(args.output)
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    run_probes(
        probes,
        adapter,
        state_detector,
        output_path,
        n=args.n,
        max_workers=args.concurrency,
        max_retries=args.max_retries,
        retry_backoff_base=args.retry_backoff,
    )
    finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
    print(f"Wrote results to {output_path}")

    manifest_path = write_manifest(
        output_path,
        adapter_name=args.adapter,
        model=args.model,
        sampling=sampling,
        seed=args.seed,
        probes_dir=args.probes_dir,
        probes=probes,
        n=args.n,
        concurrency=args.concurrency,
        started_at=started_at,
        finished_at=finished_at,
        pricing=pricing,
    )
    print(f"Wrote manifest to {manifest_path}")


def cmd_score(args: argparse.Namespace) -> None:
    if args.format == "json":
        metrics = compute_metrics(args.results)
        output = json.dumps(metrics, indent=2)
    else:
        overall = compute_metrics(args.results)
        by_category = compute_metrics_by_category(args.results)
        manifest = load_manifest(args.results)
        output = render_markdown_report(args.results, overall, by_category, manifest=manifest)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Wrote report to {args.output}")
    else:
        print(output)


def cmd_compare(args: argparse.Namespace) -> None:
    runs: dict[str, dict] = {}
    for entry in args.results:
        label, _, path = entry.partition("=")
        if not path:
            label, path = Path(entry).stem, entry
        runs[label] = {
            "overall": compute_metrics(path),
            "manifest": load_manifest(path),
        }

    output = render_comparison_markdown(runs)
    if args.output:
        Path(args.output).write_text(output)
        print(f"Wrote comparison report to {args.output}")
    else:
        print(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run.py", description="DEFER-Bench CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run probes against a model adapter and log results")
    p_run.add_argument("--probes-dir", default=str(DEFAULT_PROBES_DIR))
    p_run.add_argument("--output", default=str(DEFAULT_RESULTS_DIR / "results.jsonl"))
    p_run.add_argument(
        "-n", type=int, default=20, help="Repeats per probe *per phrasing variant* (see probe.variants)"
    )
    p_run.add_argument(
        "--adapter",
        choices=["echo", "openai", "anthropic", "xai", "gemini", "ollama"],
        default="echo",
        help="Model adapter to use (default: echo, an offline mock — no API key needed)",
    )
    p_run.add_argument(
        "--model",
        default=None,
        help="Fully-pinned model string to request (required for non-echo adapters)",
    )
    p_run.add_argument(
        "--api-key",
        default=None,
        help="API key override; defaults to the provider's standard env var (see docs/providers.md)",
    )
    p_run.add_argument(
        "--base-url",
        default=None,
        help="Server URL, only used by --adapter ollama (default: http://localhost:11434/v1, or "
        "$OLLAMA_BASE_URL). Ignored by every other adapter.",
    )
    p_run.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature, applied identically to every real adapter (default: 0.0)",
    )
    p_run.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max output tokens, applied identically to every real adapter (default: 1024)",
    )
    p_run.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt, applied identically to every real adapter (default: none)",
    )
    p_run.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for --adapter echo, so its fallback-rate rolls are reproducible "
        "(e.g. for CI/regression testing of the harness itself). Ignored by real adapters.",
    )
    p_run.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Max concurrent adapter calls (default: 1, sequential). NOTE: --adapter echo's "
        "--seed determinism only holds at --concurrency 1 (see harness/runner.py).",
    )
    p_run.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per adapter call before logging it as a failed record (default: 3)",
    )
    p_run.add_argument(
        "--retry-backoff",
        type=float,
        default=1.0,
        help="Base seconds for exponential backoff between retries (default: 1.0)",
    )
    p_run.add_argument(
        "--pricing",
        default=None,
        help="Path to a JSON file mapping served model name to "
        '{"prompt_per_1k": float, "completion_per_1k": float}, for cost estimates in the '
        "run manifest. Without it, the manifest reports token totals only (cost: null).",
    )
    p_run.add_argument(
        "--refusal-scorer",
        choices=["keyword", "llm-judge"],
        default="keyword",
        help="keyword (default): deterministic RefusalKeywordScorer, no extra cost/latency. "
        "llm-judge: ask a judge model to classify REFUSED/COMPLIED instead (see docs/limitations.md "
        "for the tradeoffs) — falls back to keyword scoring if the judge call fails or is ambiguous.",
    )
    p_run.add_argument(
        "--judge-adapter",
        choices=["echo", "openai", "anthropic", "xai", "gemini", "ollama"],
        default=None,
        help="Adapter for --refusal-scorer llm-judge (default: same as --adapter). "
        "Can be a different/cheaper model than the one being benchmarked.",
    )
    p_run.add_argument(
        "--judge-model",
        default=None,
        help="Model for --refusal-scorer llm-judge (default: same as --model)",
    )
    p_run.add_argument(
        "--judge-api-key",
        default=None,
        help="API key for --judge-adapter (default: same as --api-key, or that provider's env var)",
    )
    p_run.add_argument(
        "--judge-base-url",
        default=None,
        help="Server URL for --judge-adapter ollama (default: same as --base-url if "
        "--judge-adapter matches --adapter, otherwise http://localhost:11434/v1)",
    )
    p_run.add_argument(
        "--force",
        action="store_true",
        help="Run anyway if --model doesn't look like a fully-pinned dated snapshot "
        "(otherwise run.py refuses, since it would make VISIBLE_FALLBACK detection "
        "meaningless — see docs/providers.md).",
    )
    p_run.set_defaults(func=cmd_run)

    p_score = sub.add_parser("score", help="Compute metrics from a results JSONL file")
    p_score.add_argument("results", help="Path to a results JSONL file")
    p_score.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="json: raw compute_metrics() output (default, unchanged from earlier versions). "
        "markdown: human-readable report with a per-category breakdown table.",
    )
    p_score.add_argument("--output", default=None, help="Write the report to this file instead of stdout")
    p_score.set_defaults(func=cmd_score)

    p_compare = sub.add_parser(
        "compare", help="Side-by-side markdown comparison across multiple results JSONL files"
    )
    p_compare.add_argument(
        "results",
        nargs="+",
        help="Results JSONL files to compare, each from a separate `run.py run` against a "
        "different model/provider. Optionally label each as label=path (default label: filename "
        "stem). If a sibling <path>.manifest.json exists, its model name is shown in the report.",
    )
    p_compare.add_argument("--output", default=None, help="Write the report to this file instead of stdout")
    p_compare.set_defaults(func=cmd_compare)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
