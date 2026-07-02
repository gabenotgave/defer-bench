from __future__ import annotations

import re
from typing import Optional

# Convention-based regex for "this model string looks like a fully-pinned,
# dated snapshot" per provider, based on each provider's published naming
# convention as of when this was written:
#   - OpenAI:    trailing -YYYY-MM-DD (e.g. gpt-4o-2024-08-06) or the older
#                4-digit -MMDD style (e.g. gpt-3.5-turbo-0125)
#   - Anthropic: trailing -YYYYMMDD (e.g. claude-sonnet-4-5-20250929)
#   - xAI:       trailing 4-digit -MMDD (e.g. grok-2-1212)
#   - Gemini:    trailing 3-digit stable-release suffix (e.g.
#                gemini-2.0-flash-001)
#
# This is a best-effort heuristic, not a guarantee: providers can and do
# change naming conventions, and a string that happens to match the
# pattern isn't proof the provider won't still resolve/reroute it. It
# exists to catch the common case (a bare alias like "gpt-4o" or
# "grok-2-latest") before a run silently produces a meaningless
# VISIBLE_FALLBACK rate. See docs/providers.md.
_PINNED_MODEL_PATTERNS: dict[str, re.Pattern] = {
    "openai": re.compile(r"-\d{4}-\d{2}-\d{2}$|-\d{4}$"),
    "anthropic": re.compile(r"-\d{8}$"),
    "xai": re.compile(r"-\d{4}$"),
    "gemini": re.compile(r"-\d{3}$"),
}


def looks_pinned(adapter_name: str, model: str) -> Optional[bool]:
    """Best-effort check for whether `model` looks like a fully-pinned,
    dated snapshot for `adapter_name`'s provider.

    Returns `True`/`False` for a provider this module has a known
    convention for, or `None` if `adapter_name` isn't one of those (e.g.
    `"echo"`, or a provider added later without an entry here) — in that
    case the distinction isn't checkable, so callers should skip the
    warning entirely rather than guess.
    """
    pattern = _PINNED_MODEL_PATTERNS.get(adapter_name)
    if pattern is None:
        return None
    return bool(pattern.search(model))
