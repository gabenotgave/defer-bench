# Methodology

## Related work

DEFER-Bench's probe design follows the same general pattern as existing
open safety/red-teaming benchmarks — HarmBench, AdvBench, JailbreakBench,
and SORRY-Bench — direct (non-adversarial) harmful-request prompts used to
measure refusal behavior, not jailbreak/bypass techniques. It differs in
pairing every `benign_adjacent`/`known_flagged` probe with a matched
"control" probe in an unguarded domain (see "Controls" below), specifically
to separate over-refusal/capability-loss from the refusal-rate measurement
itself, and in scoring provider-disclosed model-fallback behavior alongside
refusal — a dimension the benchmarks above don't cover.

## Three outcomes

For each probe, a run against a model is classified into exactly one state:

- **TARGET** — the model answered using the model it was asked for, with no
  disclosed fallback.
- **VISIBLE_FALLBACK** — provider metadata discloses that a different
  (usually weaker) model served the request, e.g. a `fallback_reason` or a
  mismatch between `requested_model` and `model` in the response metadata.
- **SILENT_FALLBACK** — not detected by the `StateDetector`. **Reserved for
  v2 — silent-fallback detection is intentionally not implemented in v1.**
  v1 only ever classifies a response as TARGET or VISIBLE_FALLBACK; the
  `SILENT_FALLBACK` label and the `SilentFallbackHeuristic` Protocol (see
  `scoring/state_detector.py`) exist as a reserved extension point, not a
  working detector. The `silent_fallback_heuristic` JSONL field is always
  `null` in v1 output. See docs/limitations.md for why this is deferred.

## Probe tiers

Each probe has a `tier`:

- `benign` — should essentially never be refused.
- `benign_adjacent` — safe intent, adjacent to a sensitive topic. Used to
  measure over-refusal.
- `known_flagged` — content a reasonably calibrated model should refuse or
  heavily qualify.

## Controls

Some guarded probes have a matched control probe in an unguarded domain
(e.g. general software engineering instead of cybersecurity), linked via
`control_for`. Controls are chosen to require comparable effort/complexity
so that a capability gap between the two indicates the guardrail — not a
difference in underlying task difficulty.

## Paraphrase variants

A probe can optionally carry `variants: [...]` — alternate phrasings of the
same underlying request. `harness/runner.py` runs `n` times per phrasing
(the canonical `prompt` plus every entry in `variants`), so a probe with 2
variants and `n=20` produces 60 records, not 20. Each record carries
`prompt_variant_index` (`0` = canonical `prompt`) so phrasings can be
told apart in the raw JSONL.

This exists to catch behavior that's an artifact of exact wording rather
than the underlying request — a model that refuses one phrasing of a
`known_flagged` probe but complies with a lightly reworded version is
telling you the guardrail is shallow. **Core metrics currently treat all
variants of a probe identically** — `compute_metrics` doesn't yet report
a per-phrasing breakdown or a formal "phrasing sensitivity" statistic; the
raw JSONL has the data (`prompt_variant_index`), but aggregating it is
still open (see WORK_ITEMS.md). Most shipped probes have no `variants` yet
(`variants` defaults to empty) — `cyber-flagged-01` and `cyber-adjacent-01`
in `probes/cybersecurity.yaml` are the worked examples.

## Core metrics

Computed by `scoring/metrics.py` from the `state` and `score` fields only:

- **Refusal rate** — fraction of runs where the deterministic scorer
  detects a refusal.
- **Visible-fallback rate** — fraction of runs classified as
  `VISIBLE_FALLBACK` by the `StateDetector`.
- **Over-refusal rate** — refusal rate restricted to `benign_adjacent`
  probes.
- **Guarded-vs-control capability delta, per grading tier** — mean
  capability score on control probes minus mean capability score on the
  guarded probes they pair with, computed *separately per grading tier*
  (`capability_score_by_grader` in `compute_metrics`'s output).
  `CapabilityScorer` (`scoring/scorers.py`) grades a response by probe id,
  at whichever rigor tier that probe has: sandboxed unit-test execution
  (real correctness), keyword-rubric coverage, syntax-only (`ast.parse`),
  or not applicable (`None`) — see docs/limitations.md for which probes
  have which tier today (most are still syntax-only or ungraded).

  `compute_metrics` also reports a pooled `guarded_vs_control_capability_delta`
  across all tiers, but this is **not a headline number** — it averages
  scores of fundamentally different rigor and scale (a continuous
  keyword-coverage fraction against a binary syntax pass/fail), so it can
  read as directionally meaningful while mostly reflecting which tier had
  more data. `run.py score --format markdown` and `run.py compare` both
  surface the per-tier breakdown, not the pooled number, for exactly this
  reason. Keep the pooled value only for backward-compat/debugging.

Each proportion metric (refusal rate, visible-fallback rate, over-refusal
rate) is reported as `{rate, n, ci95}`, not a bare number — `ci95` is a 95%
Wilson score interval, chosen over a naive normal approximation because it
stays within `[0, 1]` and doesn't collapse to a zero-width interval at the
small sample sizes some tiers have (e.g. `benign_adjacent` currently has
one probe per category). `n` is always reported alongside the rate so a
reader isn't misled by a percentage with no sense of how many samples
produced it. Capability deltas instead report `guarded_n`/`control_n` per
tier — a formal interval on the difference of means isn't meaningful yet,
since most `capability_score` values are still `None` (see
docs/limitations.md).

## Scoring

Scorers are provider-agnostic, selected per-probe via the `scorer` field
in the probe YAML. Every probe currently scored on the refusal axis uses
the name `"refusal_keyword"` — but which *implementation* backs that name
is chosen at runtime, not baked into the probe file:

- **Default:** `RefusalKeywordScorer` — deterministic phrase/pattern
  matching, no extra cost or latency, keeps the whole pipeline runnable
  offline. See docs/limitations.md for what it does and doesn't catch.
- **Opt-in:** `run.py run --refusal-scorer llm-judge` swaps in
  `LLMJudgeScorer`, which asks a judge model (via the same `ModelAdapter`
  abstraction every other adapter uses — no provider SDK coupling) to
  classify REFUSED/COMPLIED. `scoring.scorers.register_scorer` performs
  this swap process-wide for the run; probe YAML files never need to
  change. See docs/limitations.md for the tradeoffs (cost, latency, and
  the judge model's own reliability).

Neither option has been empirically validated against real transcripts
from the shipped providers — see docs/limitations.md before treating
`refusal_rate` as precise rather than directional.

## Reports and comparison

`compute_metrics_by_category` (`scoring/metrics.py`) runs the same
computation as `compute_metrics`, grouped by each record's `category`
field — the same formulas, just partitioned, so a category that behaves
very differently from the aggregate isn't hidden by it. `scoring/report.py`
renders this (plus the overall numbers) as markdown via
`run.py score --format markdown`; it contains no metric logic of its own,
only formatting.

`run.py compare` renders a side-by-side markdown table across multiple
results files — each expected to come from a separate `run.py run`
invocation against a different model/provider. It does not itself
re-normalize sampling parameters across those runs; it's on the caller to
have held `--temperature`/`--max-tokens`/`--system-prompt` constant (see
docs/providers.md) so the comparison is actually apples-to-apples.
