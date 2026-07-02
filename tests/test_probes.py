import base64
from pathlib import Path

import pytest

from harness.probes import Probe, load_probes, load_probes_dir


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


VALID_KWARGS = dict(
    id="p1",
    category="cybersecurity",
    tier="benign",
    control_for=None,
    prompt="hello",
    difficulty="easy",
    scorer="refusal_keyword",
)


def test_valid_tier_accepted():
    probe = Probe(**VALID_KWARGS)
    assert probe.tier == "benign"


def test_invalid_tier_rejected():
    with pytest.raises(ValueError):
        Probe(**{**VALID_KWARGS, "tier": "not_a_real_tier"})


def test_prompt_variants_defaults_to_just_the_prompt():
    probe = Probe(**VALID_KWARGS)
    assert probe.prompt_variants == ["hello"]


def test_prompt_variants_includes_extra_variants():
    probe = Probe(**VALID_KWARGS, variants=["hi there", "greetings"])
    assert probe.prompt_variants == ["hello", "hi there", "greetings"]


def test_load_probes_from_yaml_file(tmp_path):
    yaml_file = tmp_path / "probes.yaml"
    yaml_file.write_text(
        """
- id: a1
  category: test
  tier: benign
  control_for: null
  prompt: "test prompt"
  difficulty: easy
  scorer: refusal_keyword
"""
    )
    probes = load_probes(yaml_file)
    assert len(probes) == 1
    assert probes[0].id == "a1"


def test_load_probes_dir_loads_all_yaml_files(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "- {id: a1, category: c, tier: benign, control_for: null, prompt: p, difficulty: easy, scorer: refusal_keyword}\n"
    )
    (tmp_path / "b.yaml").write_text(
        "- {id: b1, category: c, tier: benign, control_for: null, prompt: p, difficulty: easy, scorer: refusal_keyword}\n"
    )
    probes = load_probes_dir(tmp_path)
    assert {p.id for p in probes} == {"a1", "b1"}


def test_shipped_probes_load_and_control_for_refs_resolve():
    probes = load_probes_dir("probes")
    assert len(probes) > 0
    ids = {p.id for p in probes}
    for probe in probes:
        if probe.control_for:
            assert probe.control_for in ids, f"{probe.id} references missing control_for {probe.control_for!r}"


def test_prompt_ref_resolves_to_decoded_file_content(tmp_path):
    flagged_dir = tmp_path / "flagged"
    flagged_dir.mkdir()
    (flagged_dir / "external.b64").write_text(_b64("This is\nthe external prompt.\n"))

    yaml_file = tmp_path / "probes.yaml"
    yaml_file.write_text(
        """
- id: a1
  category: test
  tier: known_flagged
  control_for: null
  prompt_ref: external.b64
  difficulty: easy
  scorer: refusal_keyword
"""
    )
    probes = load_probes(yaml_file)
    assert len(probes) == 1
    # Newlines normalized to spaces, same as a YAML folded scalar (`>`).
    assert probes[0].prompt == "This is the external prompt."


def test_variants_ref_resolves_to_decoded_file_contents(tmp_path):
    flagged_dir = tmp_path / "flagged"
    flagged_dir.mkdir()
    (flagged_dir / "canonical.b64").write_text(_b64("Canonical text.\n"))
    (flagged_dir / "variant1.b64").write_text(_b64("Variant one.\n"))
    (flagged_dir / "variant2.b64").write_text(_b64("Variant\ntwo.\n"))

    yaml_file = tmp_path / "probes.yaml"
    yaml_file.write_text(
        """
- id: a1
  category: test
  tier: known_flagged
  control_for: null
  prompt_ref: canonical.b64
  difficulty: easy
  scorer: refusal_keyword
  variants_ref: [variant1.b64, variant2.b64]
"""
    )
    probes = load_probes(yaml_file)
    assert probes[0].prompt == "Canonical text."
    assert probes[0].variants == ["Variant one.", "Variant two."]


def test_all_shipped_known_flagged_probes_use_prompt_ref():
    """Content-audit invariant: known_flagged prompt text should live in
    probes/flagged/, referenced via prompt_ref, not inline in the
    top-level probe files — so it's one hop away from a casual browse.
    Checks the raw YAML source directly, since resolved Probe objects
    don't retain whether their text came from prompt_ref or inline."""
    import yaml as yaml_module

    probes_dir = Path("probes")
    checked_any = False
    for yaml_file in sorted(probes_dir.glob("*.yaml")):
        raw = yaml_module.safe_load(yaml_file.read_text()) or []
        for entry in raw:
            if entry.get("tier") == "known_flagged":
                checked_any = True
                assert "prompt_ref" in entry, f"{entry['id']} ({yaml_file.name}) has an inline prompt, not prompt_ref"
                assert "prompt" not in entry, f"{entry['id']} ({yaml_file.name}) has both prompt and prompt_ref"
                assert entry["prompt_ref"].endswith(".b64"), (
                    f"{entry['id']} ({yaml_file.name}) prompt_ref isn't base64-encoded: {entry['prompt_ref']!r}"
                )
                for ref in entry.get("variants_ref", []):
                    assert ref.endswith(".b64"), f"{entry['id']} variants_ref isn't base64-encoded: {ref!r}"
    assert checked_any, "no known_flagged probes found to check"


def test_shipped_flagged_files_are_not_plaintext_at_rest():
    """Anti-crawler invariant: probes/flagged/*.b64 must not be readable
    English text as stored on disk — only after decoding. This is
    obfuscation against casual/automated scraping, not access control
    (anyone can `base64 -d` them); the point is that a code-search index
    or a raw GitHub file view doesn't show the harmful-request text
    directly. See probes/flagged/README.md.
    """
    flagged_dir = Path("probes/flagged")
    b64_files = sorted(flagged_dir.glob("*.b64"))
    assert len(b64_files) > 0

    for b64_file in b64_files:
        raw = b64_file.read_text()
        decoded = base64.b64decode(raw).decode("utf-8")
        assert decoded != raw, f"{b64_file.name} is not encoded (matches its own decoded form)"
        assert " " not in raw.strip(), f"{b64_file.name} contains a literal space; not stored as base64"


def test_shipped_flagged_files_decode_to_nonempty_prompts():
    probes = load_probes_dir("probes")
    known_flagged = [p for p in probes if p.tier == "known_flagged"]
    assert len(known_flagged) > 0
    for probe in known_flagged:
        assert probe.prompt.strip(), f"{probe.id} decoded to an empty/whitespace-only prompt"
