from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover - resource is POSIX-only, no Windows
    resource = None  # type: ignore[assignment]

DEFAULT_TIMEOUT_SECONDS = 5.0
_CPU_SECONDS_LIMIT = 5
_MEMORY_BYTES_LIMIT = 256 * 1024 * 1024  # 256 MB
_MAX_OPEN_FILES = 64


@dataclass(frozen=True)
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


def _limit_resources() -> None:
    """`preexec_fn` for the sandboxed subprocess: caps CPU time, memory,
    open files, and forbids spawning further child processes or writing
    core dumps. Runs in the child after `fork()`, before `exec()`.

    IMPORTANT — this is NOT a full sandbox. It mitigates runaway/resource-
    exhaustion behavior (infinite loops, memory bombs, fork bombs) but does
    **not** block filesystem access within the process's existing
    permissions, and does **not** block network access at all — doing
    either properly needs OS-level isolation (containers, nsjail, gVisor,
    a network namespace), which is out of scope for this dependency-light
    benchmark repo. See docs/limitations.md. If you're grading untrusted
    model output at real scale, run this harness itself inside a
    network-isolated container/VM with nothing sensitive reachable from it
    — do not rely on this function alone as a security boundary.
    """
    if resource is None:
        return
    # Each limit is set independently and best-effort: some platforms
    # (notably macOS/Darwin) don't support every RLIMIT_* the same way
    # Linux does — e.g. RLIMIT_AS commonly fails there with "current limit
    # exceeds maximum limit". A limit that can't be set is skipped rather
    # than aborting the whole sandbox call.
    for limit, value in (
        (getattr(resource, "RLIMIT_CPU", None), (_CPU_SECONDS_LIMIT, _CPU_SECONDS_LIMIT)),
        (getattr(resource, "RLIMIT_AS", None), (_MEMORY_BYTES_LIMIT, _MEMORY_BYTES_LIMIT)),
        (getattr(resource, "RLIMIT_NOFILE", None), (_MAX_OPEN_FILES, _MAX_OPEN_FILES)),
        (getattr(resource, "RLIMIT_NPROC", None), (0, 0)),  # no forking/spawning further processes
        (getattr(resource, "RLIMIT_CORE", None), (0, 0)),  # no core dumps
    ):
        if limit is None:
            continue
        try:
            resource.setrlimit(limit, value)
        except (ValueError, OSError):
            pass


def run_python_sandboxed(code: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> SandboxResult:
    """Run `code` as a standalone Python script in a resource-limited
    subprocess (`python -I`, isolated mode: ignores env vars and the user
    site-packages dir) and report the outcome.

    See `_limit_resources` for exactly what is and isn't mitigated — this
    is a best-effort resource-limit sandbox, not a security boundary
    against a determined adversary.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "candidate.py"
        script_path.write_text(code)

        kwargs: dict = {}
        if resource is not None:
            kwargs["preexec_fn"] = _limit_resources

        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                **kwargs,
            )
        except subprocess.TimeoutExpired as e:
            timeout_stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            timeout_stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            return SandboxResult(
                ok=False,
                stdout=timeout_stdout,
                stderr=timeout_stderr + "\n[sandbox] timed out",
                returncode=-1,
                timed_out=True,
            )

        return SandboxResult(
            ok=proc.returncode == 0,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            timed_out=False,
        )
