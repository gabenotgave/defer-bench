# Provider adapters

Real adapters live in `harness/openai_adapter.py`, `harness/anthropic_adapter.py`,
`harness/xai_adapter.py`, `harness/gemini_adapter.py`, and
`harness/ollama_adapter.py`. Provider SDKs are imported only inside these
files (lazily, inside `__init__`) — nothing in `harness/runner.py`,
`scoring/`, or `run.py` depends on a specific provider.

## Usage

```bash
pip install -e ".[openai]"        # or ".[anthropic]", ".[xai]", ".[gemini]", ".[ollama]", ".[all]"

export OPENAI_API_KEY=sk-...
python run.py run --adapter openai --model gpt-4o-2024-08-06 -n 5

export ANTHROPIC_API_KEY=sk-ant-...
python run.py run --adapter anthropic --model claude-sonnet-4-5-20250929 -n 5

export XAI_API_KEY=xai-...
python run.py run --adapter xai --model grok-2-1212 -n 5

export GEMINI_API_KEY=...        # or GOOGLE_API_KEY
python run.py run --adapter gemini --model gemini-2.0-flash-001 -n 5

# Local (or remote) Ollama server, no API key needed:
python run.py run --adapter ollama --model llama3.1:8b -n 5
```

`--api-key` overrides the env var per-run if you'd rather not export it.

## Ollama

`OllamaAdapter` talks to Ollama's OpenAI-compatible endpoint (via the
`openai` client pointed at a different `base_url`, same approach as
`XAIAdapter`) — no API key required. Defaults to
`http://localhost:11434/v1`; override with `--base-url` or `$OLLAMA_BASE_URL`
for a remote server. Use whatever tag you've pulled (`ollama pull
llama3.1:8b`), e.g. `--model llama3.1:8b`.

**Model pinning doesn't apply to Ollama.** `harness/model_pinning.py` has
no entry for `"ollama"` — deliberately, not an oversight. A local Ollama
server only ever serves the exact tag you requested; it doesn't resolve
aliases to a different pinned snapshot server-side the way the cloud
providers do, so `run.py` never warns/blocks on an Ollama model string
(`looks_pinned("ollama", ...)` returns `None`, meaning "not checkable,"
same as for `echo`).

**Using Ollama as the refusal judge.** Since it's free and runs locally,
Ollama is a reasonable default judge for `--refusal-scorer llm-judge`
independent of which adapter you're actually benchmarking:

```bash
python run.py run --adapter openai --model gpt-4o-2024-08-06 \
  --refusal-scorer llm-judge --judge-adapter ollama --judge-model llama3.1:8b
```

`--judge-base-url` overrides the judge's server address independently of
`--base-url` (relevant if you're benchmarking one Ollama server and
judging with another). The same reliability caveats apply as any other
judge model — see docs/limitations.md: the judge can be wrong, and a
smaller local model is more likely to be than a frontier one.

## Sampling parameters are standardized across adapters

`--temperature` (default `0.0`), `--max-tokens` (default `1024`), and
`--system-prompt` (default: none) are applied identically to every real
adapter via `harness.sampling.SamplingConfig` — `run.py` builds one
`SamplingConfig` from these flags and passes it to whichever adapter
`--adapter` selects. This matters for guarded-vs-control and
model-vs-model comparisons: if one adapter ran hotter, with a shorter
token budget, or with a different system prompt than another, a rate
difference could be an artifact of that mismatch rather than a real
behavioral difference. `EchoAdapter` doesn't take a `SamplingConfig` — it's
a deterministic mock with nothing to sample.

If you're constructing an adapter directly (not via `run.py`), pass
`sampling=SamplingConfig(...)` explicitly rather than leaving it at each
adapter's own default, to keep multiple adapters in a comparison aligned.

## Always pass a fully-pinned model string

Pass a dated snapshot (`gpt-4o-2024-08-06`, `claude-sonnet-4-5-20250929`,
`grok-2-1212`, `gemini-2.0-flash-001`), not a bare alias (`gpt-4o`,
`claude-sonnet`, `grok-2-latest`, `gemini-2.0-flash`). Aliases resolve
server-side to a specific snapshot on *every* call, and that resolution
shows up in the response's `model` field exactly the way an actual
fallback would. If `requested_model` is an alias, `MetadataStateDetector`
cannot tell "alias resolved normally" apart from "provider silently
routed me elsewhere" — both look like `requested_model != model`. Pinning
removes that ambiguity: a mismatch on a pinned request is an actual
signal, not routine resolution.

**This is enforced, not just documented.** `run.py run` checks `--model`
against each provider's dated-snapshot naming convention
(`harness/model_pinning.py`) before running, and refuses to start (exit
with an explanation) if it doesn't look pinned — pass `--force` to run
anyway (e.g. to intentionally test alias behavior, or if a provider's
convention has changed since this was written and the check is a false
positive). This is a best-effort regex heuristic based on each provider's
*current* naming convention, not a guarantee — it can't detect a
provider changing conventions, and a string matching the pattern isn't
proof the provider won't still reroute it. `--judge-model` (for
`--refusal-scorer llm-judge`) is not checked — pinning it doesn't affect
`VISIBLE_FALLBACK` correctness for the model actually being benchmarked.

## What each provider discloses today

None of OpenAI's Chat Completions API, Anthropic's Messages API, xAI's Grok
API, Google's Gemini API, or Ollama's API expose a dedicated "you were
routed elsewhere, here's why" field. All five shipped adapters therefore
always set `meta.fallback_reason = None` — `VISIBLE_FALLBACK` detection
for the four cloud providers rests entirely on the `requested_model !=
model` check, which (per above) is only meaningful when the request pins
an exact model snapshot. For Gemini specifically, `meta.model` comes from
the response's `model_version` field (the SDK's documented "actual model
version served" field), not a `model` field — same caveat applies. For
**Ollama**, `requested_model != model` should essentially never fire in
practice (see the Ollama section above) — a mismatch there would suggest
something unusual about the local server's response, not routine alias
resolution.

If a provider silently serves a different model *without* changing the
`model` field in its response, this harness cannot detect it as a visible
fallback at all — that gap is exactly what the deferred v2 silent-fallback
work is aimed at (see [ROADMAP.md](../ROADMAP.md) and
[limitations.md](limitations.md)).

## Adding another provider

Implement `harness.adapter.ModelAdapter` in a new
`harness/<provider>_adapter.py`: `generate(prompt) -> {text, meta}`. Keep
the SDK import inside that file (ideally inside `__init__`, so installing
the base package doesn't require every provider's SDK). Document what that
provider actually discloses in `meta.fallback_reason` here, following the
pattern above.
