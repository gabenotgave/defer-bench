# Limitations

- **Keyword-based refusal scorer, plus an optional but unvalidated LLM
  judge.** `RefusalKeywordScorer` (default) matches a fixed phrase list
  (~45 English phrases plus a handful of non-English openers) and a small
  set of regex patterns for refusal *structures* ("I can't write that")
  and partial compliance ("I can't X, but I can Y" — still counted as a
  refusal of the original ask). This is a real, meaningful improvement
  over the original 10-phrase v1 list, but it is still a hand-authored
  list/pattern set that has **not been empirically validated against real
  transcripts** from any of the shipped providers (OpenAI, Anthropic, xAI,
  Gemini, Ollama) — treat `refusal_rate`/`over_refusal_rate` as
  directional, not precise, until that validation happens. It will still
  miss refusal phrasing outside these patterns and can false-positive on
  legitimate text containing a matched phrase or pattern.

  `LLMJudgeScorer` (`scoring/scorers.py`) is an opt-in alternative —
  `run.py run --refusal-scorer llm-judge` — that asks a judge model
  (any `ModelAdapter`, defaulting to the same model/provider being
  benchmarked, or a separate `--judge-adapter`/`--judge-model` — Ollama is
  a reasonable free/local default judge, see docs/providers.md) to
  classify REFUSED/COMPLIED instead of pattern-matching. It generalizes
  better to soft hedges and non-English refusals in principle, at the cost
  of one extra model call per scored response (latency + $, tracked in the
  usage/manifest like any other adapter call) and a new failure mode: **the
  judge model can itself be wrong**, and this scorer does not correct for
  that — it only falls back to the keyword scorer (logged via
  `judge_error`/`judge_ambiguous` in the JSONL) when the judge's response
  can't be parsed, not when the judge is simply mistaken. A smaller local
  judge model (e.g. via Ollama) is plausibly *more* likely to misjudge
  than a frontier model — cheap/local isn't free of this risk, just cheap.
  Like the keyword scorer, no judge configuration has been validated
  against ground-truth-labeled real transcripts. A trained refusal
  classifier (neither of these) remains unimplemented.

- **Capability scoring now checks real correctness for some probes, not
  just syntax — but only some.** `CapabilityScorer` (`scoring/scorers.py`)
  dispatches by probe id, in order of rigor: (1) **unit test** — if the
  probe is in `CODE_UNIT_TESTS`, the model's extracted code block is
  executed in `scoring/sandbox.py`'s resource-limited subprocess against a
  real correctness check (currently one worked example:
  `ctrl-flagged-01`, rewritten as a fixed-signature pure function so it's
  mechanically testable and never needs to make a real network call); (2)
  **keyword rubric** — if the probe is in `PROSE_RUBRICS`, the score is
  the fraction of required concept-keywords mentioned (worked examples:
  `cyber-adjacent-01`/`ctrl-adjacent-01`, `cyber-recon-01`); (3)
  **syntax-only fallback** (`ast.parse`, the entire v1 behavior) for any
  other probe with a code block; (4) `None` for anything else. Most probes
  — including `ctrl-flagged-02` and every `known_flagged` guarded probe —
  still fall back to tier 3 or `None`.

  The syntax/rubric-only tiers have the same weaknesses as before (a
  response can score `1.0` on syntax while being logically wrong; keyword
  rubrics are exact-match/coverage, not real reading comprehension). The
  new unit-test tier is real correctness grading, but sandboxed execution
  of untrusted model output is a genuine security surface — see
  `scoring/sandbox.py`'s docstring: it's a best-effort resource-limit
  sandbox (CPU/memory/file-descriptor/fork limits via `resource.setrlimit`,
  independently best-effort per-limit since not every platform supports
  every `RLIMIT_*`), **not network-isolated**, and not a security boundary
  against a determined adversary. If you're grading untrusted output at
  real scale, run this harness itself inside a network-isolated
  container/VM.

  **Deliberate scope decision, not an oversight:** unit tests and rubrics
  are populated for control/benign probes only, never for `known_flagged`
  guarded probes — this benchmark does not build tooling to confirm
  "this exploit/malware code actually works," on purpose. A non-refusing
  response to a `known_flagged` probe still gets syntax-only grading (tier
  3), same ceiling as before this change.

  `harness/runner.py` runs this scorer on every non-refused response
  (guarded or control) regardless of the probe's primary `scorer`, so
  guarded-tier `capability_score` data does flow into `compute_metrics`.
  Note when testing with `EchoAdapter`: since it echoes the prompt text
  back rather than answering, a rubric-scored probe run through the mock
  will trivially "mention" its own rubric keywords (they're in the echoed
  prompt) and score artificially high — this is a mock-adapter artifact,
  not meaningful against `EchoAdapter`; only trust rubric/unit-test scores
  from real model responses.

  **Don't pool tiers into one delta.** `compute_metrics` reports
  `capability_score_by_grader` — the delta computed *separately* per
  grading tier — alongside a single pooled
  `guarded_vs_control_capability_delta` across all tiers. The pooled
  number is not meaningful once more than one tier has data: it averages
  a continuous rubric-coverage fraction with a binary syntax pass/fail, so
  it can look directionally significant while mostly reflecting which
  tier happened to have more records, not a real capability difference.
  This was confirmed empirically running against a real model (`llama3.2`
  via Ollama): the pooled delta read `-0.33`, implying a meaningful
  guardrail-tax signal, while the per-tier breakdown showed `0.0` on the
  one apples-to-apples comparison (`keyword_rubric`) and an undefined
  (`None`) delta on `syntax_only` (no guarded-side data to compare
  against — expected, since `known_flagged` probes are never
  unit-test/rubric graded). `run.py score --format markdown` and
  `run.py compare` both surface the per-tier breakdown as the primary
  view for exactly this reason; the pooled number is kept in the JSON
  output for backward-compat/debugging only, not presented as a headline
  metric anywhere.

- **Silent-fallback detection is not implemented in v1 — deferred to v2
  by design, not an oversight.** Reserved for v2 — silent-fallback
  detection is intentionally not implemented in v1. It's deferred rather
  than shipped half-working for two reasons: (1) without provider ground
  truth or privileged access, any single-response detector is fundamentally
  unreliable — a single-sample signal can't distinguish a target model from
  a mildly-downgraded fallback model that happens to pass the same probe;
  and (2) the alternative that *is* methodologically sound — a rate-based
  detector using difficulty-calibrated control-probe pass rates aggregated
  across the repeated runs (`-n`) v1 already logs — is real design and
  validation work that doesn't belong in a v1 skeleton. `SILENT_FALLBACK`
  and `SilentFallbackHeuristic` (see `scoring/state_detector.py`) are kept
  as a reserved label and an intentionally-unfrozen extension point;
  `PlaceholderSilentFallbackHeuristic` always returns `None`
  ("not measured"), and the `silent_fallback_heuristic` JSONL field is
  always `null` in v1 output — never a numeric placeholder like `0.0` that
  could be mistaken for a resolved "no degradation detected" score. This
  field is excluded from all core metrics in v1 and must remain excluded
  once v2 implements it, since it will still be a lower-bound heuristic,
  not ground truth.

- **Provider metadata is trusted, not verified.** `MetadataStateDetector`
  relies on the adapter honestly reporting `model`, `requested_model`, and
  `fallback_reason`. A provider that fails to disclose a fallback will be
  scored as `TARGET`. In practice, none of the shipped real adapters
  (OpenAI, Anthropic, xAI, Gemini, Ollama) have a `fallback_reason` field
  to report at all — `VISIBLE_FALLBACK` detection for the four cloud
  providers rests entirely on `requested_model != model`, which only means
  something if the adapter requests a fully-pinned model snapshot rather
  than an alias (Ollama is exempt from this — see docs/providers.md for
  why). `run.py run` now enforces this (see below) rather than relying on
  the user reading
  the docs, but it's still a best-effort regex heuristic on each
  provider's *current* naming convention (`harness/model_pinning.py`), not
  a guarantee — see [docs/providers.md](providers.md) for exactly what's
  checked, what isn't, and `--force` to override.

- **Five categories, wider but still not exhaustive.** `cybersecurity`,
  `medical`, `legal`, `extremism`, and `privacy` (doxxing/stalking/covert
  surveillance) now ship — 69 probes total (30 guarded: 5 `benign` + 15
  `benign_adjacent` + 15 `known_flagged`, plus 39 controls/benign), up from
  28 probes in 4 categories. Every category now has 3 `benign_adjacent`
  probes (was 1), which meaningfully tightens the `over_refusal_rate`
  Wilson CI per category. Still not enough volume to call any single rate
  definitively tight/citable at scale, and the benchmark's category
  coverage is illustrative, not a claim of completeness — more categories
  and more probes per tier remain open if the benchmark's intended scope
  calls for them (see WORK_ITEMS.md).

- **Paraphrase variant coverage widened but still partial.** Probes can
  carry `variants: [...]` (alternate phrasings run alongside the canonical
  prompt; see docs/methodology.md). 12 of 69 probes now have at least one
  variant (was 2 of 28) — one `known_flagged` and one `benign_adjacent`
  variant added per category as worked examples — but most probes still
  have none, and `compute_metrics` still has no per-phrasing breakdown: a
  model that's wording-sensitive (refuses one phrasing, complies with a
  reworded one) shows up only as noise in the aggregate rate, not as its
  own "phrasing sensitivity" statistic. Both further variant coverage and
  that aggregate metric are open (see WORK_ITEMS.md).

- **Mock adapter only.** `EchoAdapter` is a local, deterministic stand-in
  for testing the harness. It does not reflect the behavior of any real
  model or provider.
