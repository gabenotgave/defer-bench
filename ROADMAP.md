# Roadmap

## v2: silent-fallback detection

Reserved for v2 — silent-fallback detection is intentionally not
implemented in v1 (see [docs/limitations.md](docs/limitations.md)). It's
planned as a rate-based, difficulty-calibrated analysis over the repeated
per-probe runs (`-n`) and control-probe pairings (`control_for`) v1 already
produces in its JSONL logs — not a per-response classifier, and not a
trained cross-model fingerprinting model. No interface for it is frozen
yet; `scoring.state_detector.SilentFallbackHeuristic` is a placeholder
Protocol, not a spec.
