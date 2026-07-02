# Contributing

## Adding probes

Add YAML files under `probes/`, one file per category, following the
schema in `harness/probes.py`. Keep example prompts benign — no working
harmful content, even for `known_flagged` probes; the point is to elicit a
refusal/fallback decision, not to ship a payload. Pair sensitive probes
with a `control_for` probe of matched difficulty in an unguarded domain.

Probe data is licensed CC-BY-4.0 (see README); by contributing probes you
agree to license them under those terms.

Every probe PR (new or edited) goes through the
[probe review checklist](docs/probe-review.md) before merge — tier
placement, control pairing, no real operational uplift, and licensing.
It's meant to take minutes, not gatekeep — see the doc for what a reviewer
actually checks.

## Adding a model adapter

Implement `harness.adapter.ModelAdapter` (`generate(prompt) -> {text,
meta}`) for your provider. Do not add provider SDKs to `harness/runner.py`,
`scoring/`, or any other core-path module — adapters are the only place
provider-specific code belongs.

## Adding a scorer

Add a class implementing `scoring.scorers.Scorer` and register it in the
`SCORERS` dict. Scorers must be deterministic given `(prompt, response)`.

## Running locally

```bash
pip install -e ".[dev]"
python run.py run -n 5
python run.py score results/results.jsonl
```
