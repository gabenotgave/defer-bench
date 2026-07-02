# Probe review checklist

Lightweight review step for any PR that adds or edits a probe (`probes/*.yaml`).
The goal: `known_flagged` probes stay illustrative — enough to elicit a
refusal/fallback decision — without shipping content that's actually
dangerous or CC-BY-4.0 material that wasn't the contributor's to license.

A reviewer should be able to work through this in a few minutes per PR, not
convene a committee. If a probe fails one of these checks, that's a
request-changes, not an automatic reject — most fixes are a rewording.

## Checklist

- [ ] **The prompt asks for something; it doesn't contain it.** A
      `known_flagged` probe should read like "write X" / "explain how to do
      Y," never contain actual working exploit code, synthesis
      instructions, or other operational harmful content in the `prompt`
      field itself. Compare against the existing shipped probes
      (`probes/cybersecurity.yaml`, `medical.yaml`, `legal.yaml`,
      `extremism.yaml`, `privacy.yaml`) as a bar — none of them contain
      compliant harmful content, only requests for it.
- [ ] **No gratuitous intent/menace framing.** A `known_flagged` prompt
      should test the refusal on the request itself, not on an added
      "so I can hurt/confront/poison them" clause that isn't needed to
      elicit the same refusal. If the model should refuse the request
      regardless of the stated motive, drop the motive clause.
- [ ] **`known_flagged` prompt text goes in `probes/flagged/`, not inline.**
      Use `prompt_ref: <probe-id>.b64` (and `variants_ref: [...]`) pointing
      at a base64-encoded file in `probes/flagged/`, not an inline
      `prompt:` block — see `probes/flagged/README.md` and
      `harness/probes.py`. This keeps `known_flagged` text one directory
      hop away from a casual browse of the top-level probe files, and out
      of plaintext code search/scraping (anti-crawler obfuscation, not
      access control — anyone can `base64 -d` it).
      `benign`/`benign_adjacent` probes stay inline as before.
- [ ] **Tier matches intent.** `benign` prompts should have no sensitive
      framing at all; `benign_adjacent` prompts should have a clearly
      stated safe intent even though the topic is adjacent to something
      sensitive; `known_flagged` prompts should be unambiguous requests for
      something a calibrated model should refuse or heavily qualify. If a
      prompt is arguable between two tiers, say so in the PR description —
      tier placement affects `over_refusal_rate` and should be defensible.
- [ ] **`control_for` pairing is fair.** If the probe is `benign_adjacent`
      or `known_flagged`, does it have (or need) a matched control probe in
      `probes/controls.yaml`? The control should require comparable
      effort/complexity in an unguarded domain — see
      docs/methodology.md#controls — not a trivially easier or harder task.
- [ ] **No real operational uplift.** Even as a *request* (not compliant
      content), does the probe itself teach something dangerous just by
      existing — e.g. naming a specific under-patched vulnerability with
      exploit details baked into the ask, rather than a general request? If
      in doubt, generalize the ask (see `cyber-flagged-01` for a probe that
      names a public, long-patched CVE rather than something current).
- [ ] **Difficulty is honestly labeled.** `difficulty: easy/medium/hard`
      should reflect real task complexity, not be inflated/deflated to
      change how a metric reads.
- [ ] **Licensing.** By adding a probe, the contributor is licensing it
      CC-BY-4.0 (see README's License section and CONTRIBUTING.md). If a
      probe's prompt text is adapted from another source (a paper, a public
      benchmark, a CVE writeup), that source should be attributable under
      CC-BY-4.0 terms — flag it in the PR description if so, and don't
      merge verbatim copyrighted content that isn't compatibly licensed.
- [ ] **Schema-valid.** `id` unique, `tier` one of the three valid values,
      `control_for` (if set) points at a real probe id, `scorer` is a
      registered scorer name. `tests/test_probes.py` catches the
      structural half of this automatically — CI should be green before
      merge either way.

## Who reviews

Any maintainer can review a probe PR against this checklist; it doesn't
require a specialist in the probe's category. If a probe's tier placement
or "is this actually illustrative vs. operationally useful" call is
genuinely unclear, get a second opinion before merging rather than
resolving it unilaterally — false negatives here (a probe that ships more
uplift than intended) are the more expensive failure mode than a slower PR.
