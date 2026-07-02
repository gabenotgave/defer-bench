from __future__ import annotations

import datetime as dt
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from scoring.scorers import CAPABILITY_SCORER_NAME, get_scorer
from scoring.state_detector import PlaceholderSilentFallbackHeuristic, StateDetector

from .adapter import ModelAdapter, ModelResponse
from .logging_utils import JsonlWriter
from .probes import Probe


@dataclass(frozen=True)
class _Task:
    probe: Probe
    variant_index: int
    prompt_text: str
    run_index: int


def run_probes(
    probes: list[Probe],
    adapter: ModelAdapter,
    state_detector: StateDetector,
    output_path: Path | str,
    n: int = 20,
    silent_fallback_heuristic=None,
    max_workers: int = 1,
    max_retries: int = 3,
    retry_backoff_base: float = 1.0,
) -> None:
    """Run each probe `n` times per phrasing variant (`probe.prompt_variants`
    — the canonical prompt plus any paraphrase `variants`), so total records
    per probe are `n * len(probe.prompt_variants)`. Variant 0 is always the
    canonical `prompt`; core metrics (`scoring/metrics.py`) currently treat
    all variants of a probe identically (no per-phrasing breakdown yet) —
    see docs/methodology.md and WORK_ITEMS.md.

    A single `adapter.generate` call is retried up to `max_retries` times
    with exponential backoff (`retry_backoff_base * 2**attempt` seconds)
    before giving up. A run that exhausts retries is logged as an error
    record (`error` field set, `state`/`score` null) rather than crashing
    the rest of the benchmark — see `_run_one`.

    `max_workers > 1` runs tasks concurrently via a thread pool (adapters
    are assumed to be blocking I/O, e.g. HTTP calls, so threads — not
    asyncio — are the right tool here). CAVEAT: `EchoAdapter`'s seeded
    determinism (see its `--seed` flag) only holds at `max_workers=1` —
    `random.Random` draws from concurrent threads race, so a seeded run
    used for CI/regression testing should stay sequential.
    """
    # Reserved for v2 — silent-fallback detection is intentionally not implemented
    # in v1. See docs/limitations.md.
    silent_fallback_heuristic = silent_fallback_heuristic or PlaceholderSilentFallbackHeuristic()
    capability_scorer = get_scorer(CAPABILITY_SCORER_NAME)

    tasks = [
        _Task(probe, variant_index, prompt_text, run_index)
        for probe in probes
        for variant_index, prompt_text in enumerate(probe.prompt_variants)
        for run_index in range(n)
    ]

    with JsonlWriter(output_path) as writer:

        def handle(task: _Task) -> None:
            record = _run_one(
                task,
                adapter=adapter,
                state_detector=state_detector,
                capability_scorer=capability_scorer,
                silent_fallback_heuristic=silent_fallback_heuristic,
                max_retries=max_retries,
                retry_backoff_base=retry_backoff_base,
            )
            writer.write(record)

        if max_workers <= 1:
            for task in tasks:
                handle(task)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                list(pool.map(handle, tasks))


def _run_one(
    task: _Task,
    *,
    adapter: ModelAdapter,
    state_detector: StateDetector,
    capability_scorer,
    silent_fallback_heuristic,
    max_retries: int,
    retry_backoff_base: float,
) -> dict[str, Any]:
    probe = task.probe
    base_record = {
        "run_id": str(uuid.uuid4()),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "probe_id": probe.id,
        "category": probe.category,
        "tier": probe.tier,
        "control_for": probe.control_for,
        "difficulty": probe.difficulty,
        "prompt_variant_index": task.variant_index,
        "run_index": task.run_index,
        "prompt": task.prompt_text,
    }

    response, error = _generate_with_retry(
        adapter, task.prompt_text, max_retries=max_retries, retry_backoff_base=retry_backoff_base
    )
    if error is not None:
        return {
            **base_record,
            "error": f"{type(error).__name__}: {error}",
            "response_text": None,
            "meta": None,
            "state": None,
            "score": None,
            "silent_fallback_heuristic": None,
        }
    assert response is not None  # invariant: _generate_with_retry sets exactly one of response/error

    scorer = get_scorer(probe.scorer)
    state = state_detector.classify(response)
    score = scorer.score(task.prompt_text, response, probe_id=probe.id)

    # Guarded probes are scored on a different axis (e.g. refusal
    # detection) than control probes (capability). Always attach a
    # capability_score too, unless the primary scorer already provided
    # one or the response was refused (nothing to grade for correctness),
    # so guarded_vs_control_capability_delta has guarded-tier data to
    # compare against once capability grading is real. See WORK_ITEMS.md.
    if "capability_score" not in score and not score.get("refused"):
        score = {**score, **capability_scorer.score(task.prompt_text, response, probe_id=probe.id)}

    heuristic_score = silent_fallback_heuristic.score(response)

    return {
        **base_record,
        "error": None,
        "response_text": response["text"],
        "meta": response["meta"],
        "state": state,
        "score": score,
        # Reserved for v2 — silent-fallback detection is intentionally not
        # implemented in v1. See docs/limitations.md.
        # Always null in v1 (never 0.0), so a downstream reader cannot
        # mistake "not measured" for a resolved, non-degraded score.
        # Never feeds core metrics even once v2 implements this.
        "silent_fallback_heuristic": heuristic_score,
    }


def _generate_with_retry(
    adapter: ModelAdapter, prompt: str, *, max_retries: int, retry_backoff_base: float
) -> tuple[Optional[ModelResponse], Optional[Exception]]:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return adapter.generate(prompt), None
        except Exception as e:  # noqa: BLE001 - adapters are arbitrary providers; any failure is retryable
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_backoff_base * (2**attempt))
    return None, last_error
