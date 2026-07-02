# DEFER-Bench

DEFER-Bench measures how a model handles safety-guarded queries across
three outcomes:

- **TARGET** — answers normally, using the requested model.
- **VISIBLE_FALLBACK** — visibly falls back to a weaker model (disclosed in
  provider metadata).
- **SILENT_FALLBACK** — degrades without disclosure. **Not measured in
  v1** — see "What this does NOT claim" below.

Core metrics — refusal rate, visible-fallback rate, over-refusal rate on
benign-adjacent probes, and a guarded-vs-control capability delta — are
computed only from provider metadata and deterministic scoring. See
[docs/methodology.md](docs/methodology.md) for definitions and
[docs/limitations.md](docs/limitations.md) for known gaps.

## What this does NOT claim

v1 reports two of the three outcome states: **TARGET** and
**VISIBLE_FALLBACK**, both derived deterministically from provider
metadata. **SILENT_FALLBACK is not measured in v1.** The `SILENT_FALLBACK`
label and the `SilentFallbackHeuristic` extension point exist in the code
as a reserved placeholder for v2, but no detector is implemented — the
`silent_fallback_heuristic` field in the output JSONL is always `null`.
See [docs/limitations.md](docs/limitations.md) for why this is deferred
rather than shipped as a partial heuristic, and [ROADMAP.md](ROADMAP.md)
for the planned v2 approach.

## Quickstart

```bash
pip install -e .
python run.py run -n 5
python run.py score results/results.jsonl
```

By default `run.py run` uses `EchoAdapter`, a deterministic offline mock
adapter, so the whole pipeline runs out of the box with no API keys.

To benchmark a real model, install the provider's extra and pass
`--adapter`/`--model`:

```bash
pip install -e ".[openai]"
export OPENAI_API_KEY=sk-...
python run.py run --adapter openai --model gpt-4o-2024-08-06 -n 5
```

OpenAI, Anthropic, xAI (Grok), Google Gemini, and Ollama (local or remote)
adapters ship out of the box. See
[docs/providers.md](docs/providers.md) for the rest, what each provider's
metadata actually discloses, and how to add another provider.

Refusal scoring defaults to a deterministic keyword/pattern scorer (no
extra cost). For an LLM-judge alternative instead — Ollama is a
reasonable free/local default judge regardless of which model you're
benchmarking:

```bash
python run.py run --adapter openai --model gpt-4o-2024-08-06 \
  --refusal-scorer llm-judge --judge-adapter ollama --judge-model llama3.1:8b
```

See docs/limitations.md for what each refusal scorer does and doesn't
catch — none of them have been empirically validated against real
transcripts.

## Reports

`run.py score` prints raw JSON by default (unchanged). For a human-readable
report with a per-category breakdown:

```bash
python run.py score results/results.jsonl --format markdown
```

To compare multiple models/providers side by side, run each one separately
(different `--output` per run), then:

```bash
python run.py run --adapter openai --model gpt-4o-2024-08-06 --output results/openai.jsonl
python run.py run --adapter anthropic --model claude-sonnet-4-5-20250929 --output results/anthropic.jsonl
python run.py compare "GPT-4o=results/openai.jsonl" "Claude Sonnet=results/anthropic.jsonl"
```

`compare` reads each run's sibling `.manifest.json` (see docs/providers.md)
to label columns with the actual model requested. Both `score --format
markdown` and `compare` accept `--output <file>` to write the report
instead of printing to stdout.

## Layout

```
probes/     one YAML file per probe category (probe + control-probe definitions)
harness/    runner loop, model adapter interface, JSONL logging
scoring/    deterministic scorers + StateDetector / silent-fallback placeholder (reserved for v2)
results/    example JSONL output
docs/       methodology, limitations, providers, probe review checklist
tests/      pytest unit tests
run.py      CLI entrypoint (run, score, compare)
```

## Adding a probe category

Add a new YAML file to `probes/` with entries following the schema in
`harness/probes.py`: `id`, `category`, `tier`
(`benign`/`benign_adjacent`/`known_flagged`), `control_for` (nullable),
`prompt`, `difficulty`, `scorer`. Pair `known_flagged`/`benign_adjacent`
probes with a matched control probe in an unguarded domain via
`control_for` so the capability delta stays meaningful.

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## License

Code is licensed under [Apache-2.0](LICENSE). Probe **data** (the contents
of `probes/*.yaml`) is licensed separately under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
