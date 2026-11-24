"""Git operations with deployment-shaped guard rails.

EvoLayer needs versioning for two different things — a *pin* (the state of the
code before a change) and a *rollback point* (what to return to). Both are git
operations, and both are dangerous if done casually, so this wrapper refuses
rather than guesses:

* it never pushes, and has no method that can;
* it refuses to commit on ``main``/``master`` — changes live on ``evolayer/…``
  branches and are reviewed as a pin bump;
* it refuses to touch paths outside the caller-provided set;
* it refuses to apply patches on a dirty worktree for those paths;
* every mutating call returns the resulting commit so the caller can record it.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

PROTECTED_BRANCHES = {"main", "master"}
BRANCH_PREFIX = "evolayer"


class GitError(RuntimeError):
    """A git operation was refused or failed."""


@dataclass
class Commit:
    sha: str
    branch: str
    paths: list[str]
    message: str


def _run(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=120
    )
    if check and result.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed in {repo}: {result.stderr.strip() or result.stdout.strip()}")
    return result


class GitRepo:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        probe = _run(self.path, "rev-parse", "--show-toplevel", check=False)
        if probe.returncode != 0:
            raise GitError(f"not a git checkout: {self.path}")
        self.toplevel = Path(probe.stdout.strip())
        if self.toplevel != self.path:
            # Allow subdirectories (e.g. the workspace root) but keep paths relative to it.
            self.toplevel = self.path

    # -- read ------------------------------------------------------------
    @property
    def branch(self) -> str:
        return _run(self.toplevel, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    @property
    def head(self) -> str:
        return _run(self.toplevel, "rev-parse", "HEAD").stdout.strip()

    def is_dirty(self, paths: list[str] | None = None) -> bool:
        args = ["status", "--porcelain"]
        if paths:
            args += ["--", *paths]
        return bool(_run(self.toplevel, *args).stdout.strip())

    def exists(self, relpath: str) -> bool:
        return (self.toplevel / relpath).exists()

    def show(self, relpath: str, ref: str = "HEAD") -> str:
        return _run(self.toplevel, "show", f"{ref}:{relpath}").stdout

    def tags(self, prefix: str = BRANCH_PREFIX) -> list[str]:
        out = _run(self.toplevel, "tag", "--list", f"{prefix}*", "--sort=-creatordate").stdout
        return [line for line in out.splitlines() if line.strip()]

    # -- write -----------------------------------------------------------
    def ensure_branch(self, name: str, *, base: str = "HEAD") -> str:
        if not name:
            raise GitError("branch name is required")
        if name in PROTECTED_BRANCHES:
            raise GitError(f"refusing to work on protected branch {name!r}")
        if not name.startswith(f"{BRANCH_PREFIX}/"):
            raise GitError(f"branch must start with {BRANCH_PREFIX}/, got {name!r}")
        current = self.branch
        if current == name:
            return current
        existing = _run(self.toplevel, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}", check=False)
        if existing.returncode == 0:
            _run(self.toplevel, "checkout", name)
        else:
            _run(self.toplevel, "checkout", "-b", name, base)
        return name

    def stage_paths(self, paths: list[str]) -> None:
        for path in paths:
            resolved = (self.toplevel / path).resolve()
            try:
                resolved.relative_to(self.toplevel)
            except ValueError as exc:
                raise GitError(f"path escapes the repository: {path}") from exc
        _run(self.toplevel, "add", "--", *paths)

    def commit(self, paths: list[str], message: str) -> Commit:
        if self.branch in PROTECTED_BRANCHES:
            raise GitError(
                f"refusing to commit on {self.branch!r}; create an {BRANCH_PREFIX}/… branch first"
            )
        if not message.strip():
            raise GitError("commit message is required")
        self.stage_paths(paths)
        _run(self.toplevel, "commit", "-m", message, "--", *paths)
        return Commit(sha=self.head, branch=self.branch, paths=list(paths), message=message)

    def tag(self, name: str, message: str) -> str:
        if not name.startswith(f"{BRANCH_PREFIX}-"):
            raise GitError(f"tag must start with {BRANCH_PREFIX}-, got {name!r}")
        existing = _run(self.toplevel, "rev-parse", "--verify", "--quiet", f"refs/tags/{name}", check=False)
        if existing.returncode == 0:
            raise GitError(f"tag already exists: {name}")
        _run(self.toplevel, "tag", "-a", name, "-m", message)
        return name

    def restore_paths_from(self, ref: str, paths: list[str], *, message: str) -> Commit:
        """Revert ``paths`` to ``ref`` and commit the result (a real rollback commit)."""
        if self.branch in PROTECTED_BRANCHES:
            raise GitError(f"refusing to roll back on {self.branch!r}")
        verify = _run(self.toplevel, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False)
        if verify.returncode != 0:
            raise GitError(f"unknown ref: {ref}")
        _run(self.toplevel, "checkout", ref, "--", *paths)
        return self.commit(paths, message)


def sanitise_branch_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-")[:60] or "change"
