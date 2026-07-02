# Flagged prompt text

CONTENT NOTICE: the files in this directory are `known_flagged`-tier probe
prompts — requests for harmful content, used to test whether/how a model
refuses them. They do **not** contain working exploit code, functional
instructions, or other operational harmful content — only the request
itself. See the DEFER-Bench README and
[docs/probe-review.md](../../docs/probe-review.md) for full context before
reusing anything here in isolation.

Each file holds the prompt text for one probe (`<probe-id>.b64`) or one
paraphrase variant (`<probe-id>.vN.b64`), referenced from the corresponding
`probes/*.yaml` category file via `prompt_ref`/`variants_ref` (see
`harness/probes.py`). This split exists so a casual browse of the
top-level `probes/*.yaml` files, a GitHub file preview, or a search index
doesn't land directly on this text — it's one directory hop further in,
with this notice attached.

## Encoding

Files are base64-encoded at rest (`.b64`), decoded transparently by
`harness/probes.py` at load time. **This is anti-crawler obfuscation, not
access control or a license restriction** — the data is exactly as open
as the rest of this CC-BY-4.0-licensed repo, just not stored as
plaintext, so it isn't directly indexed by code search or casually
readable in a GitHub file preview / raw scrape. The benchmark's behavior
is unchanged; this and the directory split above are
presentation/discoverability changes, not functional ones.

To decode a file by hand for inspection:

```bash
base64 -d < probes/flagged/cyber-flagged-01.b64
```

To re-encode after editing a probe's plaintext (see
`docs/probe-review.md` before adding/editing a `known_flagged` probe):

```bash
base64 < your-edited-prompt.txt > probes/flagged/<probe-id>.b64
```
