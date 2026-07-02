from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

VALID_TIERS = {"benign", "benign_adjacent", "known_flagged"}


@dataclass(frozen=True)
class Probe:
    id: str
    category: str
    tier: str
    control_for: Optional[str]
    prompt: str
    difficulty: str
    scorer: str
    # Optional alternate phrasings of the same underlying request, for
    # paraphrase/robustness testing. `prompt` is always variant 0; entries
    # here are additional variants run alongside it. Empty by default —
    # most probes ship with a single fixed phrasing.
    variants: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.tier not in VALID_TIERS:
            raise ValueError(f"{self.id}: invalid tier {self.tier!r}, must be one of {VALID_TIERS}")

    @property
    def prompt_variants(self) -> list[str]:
        """All phrasings to run for this probe: the canonical prompt first,
        then any paraphrase variants."""
        return [self.prompt, *self.variants]


def _read_ref(flagged_dir: Path, ref: str) -> str:
    """Read a base64-encoded prompt-text file (`probes/flagged/*.b64`),
    decode it, and normalize whitespace (collapse newlines/indentation to
    single spaces) the same way YAML's folded scalar (`>`) does, so a
    probe sourced from an external file behaves identically to one with an
    inline `prompt:` string.

    The base64 encoding is anti-crawler obfuscation, not access control —
    see `probes/flagged/README.md` for how to decode a file by hand. It
    keeps the stored form out of plaintext code search / scraping while
    the data stays fully open (same license, same repo, one `base64 -d`
    away).
    """
    encoded = (flagged_dir / ref).read_text()
    decoded = base64.b64decode(encoded).decode("utf-8")
    return " ".join(decoded.split())


def _resolve_refs(entry: dict, flagged_dir: Path) -> dict:
    """Resolve `prompt_ref`/`variants_ref` (paths relative to
    `probes/flagged/`, used by `known_flagged` probes — see
    `probes/flagged/README.md`) into inline `prompt`/`variants` values,
    so the rest of the codebase never needs to know a probe's prompt text
    came from an external file rather than the YAML itself. Entries with
    no `*_ref` fields (the common case — everything except known_flagged
    probes) pass through unchanged.
    """
    entry = dict(entry)
    prompt_ref = entry.pop("prompt_ref", None)
    if prompt_ref is not None:
        entry["prompt"] = _read_ref(flagged_dir, prompt_ref)
    variants_ref = entry.pop("variants_ref", None)
    if variants_ref is not None:
        entry["variants"] = [_read_ref(flagged_dir, ref) for ref in variants_ref]
    return entry


def load_probes(path: Path | str) -> list[Probe]:
    path = Path(path)
    flagged_dir = path.parent / "flagged"
    with path.open() as f:
        raw = yaml.safe_load(f) or []
    entries = [_resolve_refs(entry, flagged_dir) for entry in raw]
    return [Probe(**entry) for entry in entries]


def load_probes_dir(dir_path: Path | str) -> list[Probe]:
    dir_path = Path(dir_path)
    probes: list[Probe] = []
    for yaml_file in sorted(dir_path.glob("*.yaml")):
        probes.extend(load_probes(yaml_file))
    return probes
