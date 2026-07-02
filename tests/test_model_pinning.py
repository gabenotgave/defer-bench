from harness.model_pinning import looks_pinned


def test_openai_dated_snapshot_looks_pinned():
    assert looks_pinned("openai", "gpt-4o-2024-08-06") is True


def test_openai_older_4digit_snapshot_looks_pinned():
    assert looks_pinned("openai", "gpt-3.5-turbo-0125") is True


def test_openai_bare_alias_does_not_look_pinned():
    assert looks_pinned("openai", "gpt-4o") is False


def test_anthropic_dated_snapshot_looks_pinned():
    assert looks_pinned("anthropic", "claude-sonnet-4-5-20250929") is True


def test_anthropic_bare_alias_does_not_look_pinned():
    assert looks_pinned("anthropic", "claude-sonnet-4-5") is False


def test_anthropic_latest_alias_does_not_look_pinned():
    assert looks_pinned("anthropic", "claude-3-5-sonnet-latest") is False


def test_xai_dated_snapshot_looks_pinned():
    assert looks_pinned("xai", "grok-2-1212") is True


def test_xai_latest_alias_does_not_look_pinned():
    assert looks_pinned("xai", "grok-2-latest") is False


def test_gemini_dated_snapshot_looks_pinned():
    assert looks_pinned("gemini", "gemini-2.0-flash-001") is True


def test_gemini_bare_alias_does_not_look_pinned():
    assert looks_pinned("gemini", "gemini-2.0-flash") is False


def test_unknown_adapter_returns_none():
    assert looks_pinned("echo", "anything") is None
    assert looks_pinned("some-future-provider", "model-x") is None
