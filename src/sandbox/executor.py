"""Runs agent-authored analysis code in an isolated subprocess.

Isolation strategy: the child process runs ``python -I -S`` (isolated mode --
ignores PYTHONPATH/PYTHONHOME/the user site-packages directory, so it never
sees this repository or its dependencies on sys.path), gets a minimal
environment and a throwaway working directory, and is bounded by best-effort
CPU/memory/file-descriptor rlimits plus a wall-clock timeout as a backstop
that does not depend on rlimit support.

This is not equivalent to a container or VM boundary -- there is no seccomp
filter, chroot, or network namespace here, only process-level resource
limits and a restricted execution surface (see src/sandbox/runner.py). That
tradeoff was made deliberately in favor of not requiring Docker in every
environment this project runs in.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from src.core.logger import get_logger

try:
    import resource
except ImportError:  # pragma: no cover - resource is POSIX-only
    resource = None

logger = get_logger(__name__)

_RUNNER_PATH = Path(__file__).resolve().parent / "runner.py"

# Deliberately minimal: no inherited API keys, DB path overrides, etc. The
# sandboxed code cannot reach environment variables anyway (no `os` module
# is bound into its globals), but keeping the subprocess's own environment
# minimal is a second, independent layer of the same protection.
_MINIMAL_ENV = {"PATH": "/usr/bin:/bin"}


def _limit_resources(memory_mb: int, cpu_seconds: int):
    """Build a preexec_fn applying best-effort POSIX resource limits.

    Each limit is set independently and failures are swallowed: some
    platforms/sandboxes refuse to lower certain rlimits, and a refusal here
    should degrade gracefully rather than crash the whole tool call -- the
    wall-clock subprocess timeout in run_sandboxed is the backstop that does
    not depend on any of this succeeding.
    """

    def _apply() -> None:
        if resource is None:
            return
        for limit_id, value in (
            (resource.RLIMIT_CPU, cpu_seconds),
            (resource.RLIMIT_AS, memory_mb * 1024 * 1024),
            (resource.RLIMIT_NOFILE, 32),
            (resource.RLIMIT_CORE, 0),
        ):
            try:
                resource.setrlimit(limit_id, (value, value))
            except (ValueError, OSError):
                logger.warning("Could not apply sandbox resource limit %s", limit_id)

    return _apply


def run_sandboxed(
    code: str,
    data,
    *,
    timeout_s: int,
    memory_mb: int,
    max_output_chars: int,
    max_code_length: int,
) -> dict:
    """Execute `code` in the sandbox subprocess with `data` bound as a variable.

    Returns a dict with `ok` and either `result`/`stdout` (success) or
    `error` (any failure: validation, timeout, crash, non-serializable
    result). Never raises -- callers get a structured outcome either way.
    """
    payload = json.dumps(
        {"code": code, "data": data, "max_code_length": max_code_length}
    )

    with tempfile.TemporaryDirectory(prefix="analysis-sandbox-") as scratch_dir:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(_RUNNER_PATH)],
                input=payload,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=scratch_dir,
                env=dict(_MINIMAL_ENV),
                preexec_fn=(
                    _limit_resources(memory_mb, timeout_s + 1)
                    if resource is not None
                    else None
                ),
            )
        except subprocess.TimeoutExpired:
            logger.warning("Sandboxed analysis timed out after %ss", timeout_s)
            return {
                "ok": False,
                "error": f"Analysis timed out after {timeout_s} seconds.",
            }
        except OSError as exc:
            logger.warning("Sandboxed analysis failed to start: %s", exc)
            return {"ok": False, "error": "Analysis process could not be started."}

    output = completed.stdout.strip()
    if not output:
        logger.warning(
            "Sandboxed analysis exited %s with no output", completed.returncode
        )
        return {"ok": False, "error": "Analysis process produced no output."}

    try:
        parsed = json.loads(output.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"ok": False, "error": "Analysis process returned an unreadable result."}

    if not isinstance(parsed, dict) or "ok" not in parsed:
        return {"ok": False, "error": "Analysis process returned an unexpected result."}

    stdout_text = parsed.get("stdout")
    if isinstance(stdout_text, str) and len(stdout_text) > max_output_chars:
        parsed["stdout"] = stdout_text[:max_output_chars] + "…"

    return parsed
