import json
import threading

from harness.probes import Probe
from harness.runner import run_probes
from scoring.state_detector import MetadataStateDetector

PROBE = Probe(
    id="p1",
    category="test",
    tier="known_flagged",
    control_for=None,
    prompt="hello",
    difficulty="easy",
    scorer="refusal_keyword",
)


class _StubAdapter:
    def generate(self, prompt, **kwargs):
        return {
            "text": "sure, here's an answer",
            "meta": {
                "model": "m1",
                "requested_model": "m1",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "latency_ms": 0.1,
                "fallback_reason": None,
            },
        }


class _AlwaysFailsAdapter:
    def generate(self, prompt, **kwargs):
        raise RuntimeError("boom")


class _FlakyAdapter:
    """Fails every 3rd call, thread-safe counter."""

    def __init__(self):
        self._n = 0
        self._lock = threading.Lock()

    def generate(self, prompt, **kwargs):
        with self._lock:
            self._n += 1
            n = self._n
        if n % 3 == 0:
            raise RuntimeError("transient")
        return {
            "text": "ok",
            "meta": {
                "model": "m1",
                "requested_model": "m1",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "latency_ms": 0.1,
                "fallback_reason": None,
            },
        }


def _read(path):
    return [json.loads(line) for line in open(path)]


def test_run_probes_writes_n_records_per_variant(tmp_path):
    output = tmp_path / "out.jsonl"
    run_probes([PROBE], _StubAdapter(), MetadataStateDetector(), output, n=5)
    records = _read(output)
    assert len(records) == 5
    assert all(r["error"] is None for r in records)


def test_run_probes_records_error_after_exhausting_retries(tmp_path):
    output = tmp_path / "out.jsonl"
    run_probes(
        [PROBE], _AlwaysFailsAdapter(), MetadataStateDetector(), output, n=2, max_retries=2, retry_backoff_base=0.001
    )
    records = _read(output)
    assert len(records) == 2
    assert all(r["error"] is not None for r in records)
    assert all(r["state"] is None for r in records)


def test_run_probes_concurrent_writes_are_not_lost_or_corrupted(tmp_path):
    output = tmp_path / "out.jsonl"
    probes = [PROBE for _ in range(3)]
    run_probes(
        probes,
        _FlakyAdapter(),
        MetadataStateDetector(),
        output,
        n=4,
        max_retries=1,
        retry_backoff_base=0.001,
        max_workers=4,
    )
    records = _read(output)
    assert len(records) == 3 * 4
    assert all("error" in r for r in records)
