"""Guarded process execution.

Every external command in EvoLayer goes through :func:`run`. The guard exists
because this layer executes commands that were generated from findings, and a
mistake must fail loudly instead of reaching a shell:

* argv is a list, never a string — no shell is ever involved;
* arguments containing shell metacharacters are rejected (defence in depth, in
  case a command is ever copy-pasted from a report into a shell);
* the executable must be on an explicit allowlist;
* the environment is rebuilt from scratch, with only documented pass-throughs;
* a wall-clock timeout is mandatory;
* stdout/stderr are captured and, optionally, streamed to a log file.
"""

from __future__ import annotations

import os
import shutil
import sys
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

DEFAULT_ALLOWED_BINARIES = frozenset(
    {
        "bash",
        "colcon",
        "ffmpeg",
        "ffprobe",
        "git",
        "nvidia-smi",
        "python",
        "python3",
        "rsync",
        "screen",
        "uv",
    }
)

DENIED_FRAGMENTS = (";", "&&", "||", "`", "$(", "\n", "\r", "|", ">", "<")

PASSTHROUGH_ENV = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TERM",
        "TMPDIR",
        "CUDA_VISIBLE_DEVICES",
        "MUJOCO_GL",
        "HF_TOKEN",
        "WANDB_MODE",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    }
)


class ExecutionError(RuntimeError):
    """The command could not be started, or was rejected by the guard."""


class CommandRejected(ExecutionError):
    """The guard refused to run the command."""


@dataclass
class Result:
    argv: list[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    log_path: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def tail(self, n: int = 40) -> str:
        lines = (self.stdout + "\n" + self.stderr).strip().splitlines()
        return "\n".join(lines[-n:])


def sanitized_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in PASSTHROUGH_ENV}
    if "PATH" not in env:
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    for key, value in (extra or {}).items():
        if not isinstance(value, str):
            raise CommandRejected(f"environment value for {key} must be a string")
        env[key] = value
    return env


def check_argv(argv: Sequence[str], allowed: Iterable[str] | None = None) -> list[str]:
    if not argv:
        raise CommandRejected("empty argv")
    if isinstance(argv, (str, bytes)):
        raise CommandRejected("argv must be a list of arguments, not a string")
    argv = [str(part) for part in argv]
    allowed_set = frozenset(allowed) if allowed is not None else DEFAULT_ALLOWED_BINARIES
    exe = Path(argv[0]).name
    is_self = False
    try:
        is_self = Path(argv[0]).resolve() == Path(sys.executable).resolve()
    except OSError:  # pragma: no cover - unresolvable path
        is_self = False
    if exe not in allowed_set and not is_self:
        raise CommandRejected(f"executable {exe!r} is not allowed (allowed: {sorted(allowed_set)})")
    for part in argv:
        for fragment in DENIED_FRAGMENTS:
            if fragment in part:
                raise CommandRejected(f"argument {part!r} contains shell fragment {fragment!r}")
    return argv


def resolve_executable(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise ExecutionError(f"executable not found on PATH: {name}")
    return found


def run(
    argv: Sequence[str],
    *,
    cwd: str | Path,
    timeout: float,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
    allowed: Iterable[str] | None = None,
    check: bool = False,
) -> Result:
    """Run ``argv`` in ``cwd`` with a hard timeout. No shell, no exceptions for exit codes."""
    argv = check_argv(argv, allowed)
    if timeout <= 0:
        raise CommandRejected("timeout must be positive")
    cwd_path = Path(cwd)
    if not cwd_path.is_dir():
        raise ExecutionError(f"working directory does not exist: {cwd_path}")

    started = time.time()
    timed_out = False
    log_file = None
    if log_path is not None:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd_path),
            env=sanitized_env(env),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            check=False,
        )
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        code = -9
    duration = time.time() - started

    if log_file is not None:
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"$ {' '.join(argv)}\n# cwd={cwd_path} exit={code} timed_out={timed_out}\n")
            if stdout:
                handle.write(stdout if stdout.endswith("\n") else stdout + "\n")
            if stderr:
                handle.write("# stderr\n")
                handle.write(stderr if stderr.endswith("\n") else stderr + "\n")

    result = Result(
        argv=argv,
        cwd=str(cwd_path),
        returncode=code,
        stdout=stdout,
        stderr=stderr,
        duration_s=duration,
        timed_out=timed_out,
        log_path=str(log_file) if log_file else "",
    )
    if check and not result.ok:
        raise ExecutionError(f"command failed ({code}): {' '.join(argv)}\n{result.tail()}")
    return result
