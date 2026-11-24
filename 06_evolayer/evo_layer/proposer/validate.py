"""Proposal validation — the gate between "generated" and "rehearsable".

A proposal is only allowed to reach the simulator if all of the following hold:

1. every edit targets a writable repo, and the path is on that repo's allowlist;
2. the patch applies cleanly to the *current* file content (proved in a sandbox);
3. the resulting file still parses (``compile`` for .py, ``yaml.safe_load`` for .yaml);
4. added lines in Python files do not introduce process/network/dynamic-exec
   primitives (no ``subprocess``, ``os.system``, ``eval``, ``exec``, ``socket``…);
5. the patch is small enough to review (line and byte caps);
6. any training command is argv-only, uses an allowed binary and allowed flags.

Anything else is a hard validation failure with a machine-readable reason, which
the orchestrator records and Hermes sees in the report.
"""

from __future__ import annotations

import py_compile
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import EvoConfig
from ..execution import CommandRejected, check_argv
from ..models import Edit, Proposal

MAX_DIFF_LINES = 200
MAX_DIFF_BYTES = 128 * 1024

ALLOWED_TRAIN_FLAGS = frozenset(
    {
        "--task",
        "--num_envs",
        "--max_iterations",
        "--learning_rate",
        "--seed",
        "--viz",
        "--logger",
        "--physics",
        "--checkpoint",
        "--resume",
        "--video",
        "--headless",
    }
)

FORBIDDEN_PY_ADDITIONS = (
    "subprocess",
    "os.system",
    "os.popen",
    "os.exec",
    "eval(",
    "exec(",
    "__import__",
    "socket",
    "urllib",
    "requests.",
    "shutil.rmtree",
    "ctypes",
)


@dataclass
class CheckOutcome:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class ValidationReport:
    proposal_id: str
    revision: int
    ok: bool
    checks: list[CheckOutcome] = field(default_factory=list)
    applied: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "proposal_id": self.proposal_id,
            "revision": self.revision,
            "checks": [c.to_dict() for c in self.checks],
            "applied": self.applied,
        }

    @property
    def failures(self) -> list[CheckOutcome]:
        return [c for c in self.checks if not c.ok]


class Validator:
    def __init__(self, cfg: EvoConfig, *, max_lines: int = MAX_DIFF_LINES, max_bytes: int = MAX_DIFF_BYTES):
        self.cfg = cfg
        self.max_lines = max_lines
        self.max_bytes = max_bytes

    def validate(self, proposal: Proposal) -> ValidationReport:
        report = ValidationReport(proposal_id=proposal.id, revision=proposal.revision, ok=True)

        for index, edit in enumerate(proposal.edits):
            prefix = f"edit[{index}] {edit.repo}:{edit.path}"
            report.checks.append(self._check_repo(edit, prefix))
            report.checks.append(self._check_path(edit, prefix))
            report.checks.append(self._check_size(edit, prefix))
            report.checks.append(self._check_semantics(edit, prefix))

        report.checks.append(self._check_train_command(proposal))
        report.checks.append(self._check_overlay_shape(proposal))

        if proposal.edits:
            applied = self._apply_in_sandbox(proposal, report)
            report.applied = applied
        for check in report.checks:
            if not check.ok:
                report.ok = False
        return report

    # -- individual checks ----------------------------------------------
    def _check_repo(self, edit: Edit, prefix: str) -> CheckOutcome:
        spec = self.cfg.repos.get(edit.repo)
        if spec is None:
            return CheckOutcome(f"{prefix}:repo", False, f"unknown repo {edit.repo!r}")
        if not spec.writable:
            return CheckOutcome(f"{prefix}:repo", False, f"repo {edit.repo!r} is read-only")
        return CheckOutcome(f"{prefix}:repo", True, f"{spec.path}")

    def _check_path(self, edit: Edit, prefix: str) -> CheckOutcome:
        spec = self.cfg.repos.get(edit.repo)
        if spec is None:
            return CheckOutcome(f"{prefix}:allowlist", False, "unknown repo")
        if not spec.allows(edit.path):
            return CheckOutcome(
                f"{prefix}:allowlist", False, f"{edit.path!r} is not on the allowlist for {edit.repo!r}"
            )
        return CheckOutcome(f"{prefix}:allowlist", True, edit.path)

    def _check_size(self, edit: Edit, prefix: str) -> CheckOutcome:
        lines = edit.added_lines + edit.removed_lines
        size = len(edit.diff.encode("utf-8"))
        if lines > self.max_lines:
            return CheckOutcome(f"{prefix}:size", False, f"{lines} changed lines > {self.max_lines}")
        if size > self.max_bytes:
            return CheckOutcome(f"{prefix}:size", False, f"{size} bytes > {self.max_bytes}")
        if "\x00" in edit.diff:
            return CheckOutcome(f"{prefix}:size", False, "patch contains NUL bytes (binary?)")
        return CheckOutcome(f"{prefix}:size", True, f"{lines} lines, {size} bytes")

    def _check_semantics(self, edit: Edit, prefix: str) -> CheckOutcome:
        if not edit.path.endswith(".py"):
            return CheckOutcome(f"{prefix}:semantics", True, "no python-specific checks")
        added = [
            line[1:]
            for line in edit.diff.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ]
        for line in added:
            for token in FORBIDDEN_PY_ADDITIONS:
                if token in line:
                    return CheckOutcome(
                        f"{prefix}:semantics",
                        False,
                        f"added line introduces forbidden primitive {token!r}: {line.strip()[:80]}",
                    )
        return CheckOutcome(f"{prefix}:semantics", True, "no forbidden primitives in added lines")

    def _check_train_command(self, proposal: Proposal) -> CheckOutcome:
        if not proposal.train_command:
            return CheckOutcome("train_command", True, "none")
        argv = proposal.train_command
        if any(not isinstance(part, str) for part in argv):
            return CheckOutcome("train_command", False, "argv must be strings")
        try:
            check_argv([Path(argv[0]).name, *argv[1:]], allowed={"python", "python3", "uv", "bash"})
        except CommandRejected as exc:
            return CheckOutcome("train_command", False, str(exc))
        if len(argv) < 2 or not argv[1].endswith("train.py"):
            return CheckOutcome("train_command", False, "only train.py entrypoints may be proposed")
        for part in argv[1:]:
            if part.startswith("--"):
                flag = part.split("=", 1)[0]
                if flag not in ALLOWED_TRAIN_FLAGS:
                    return CheckOutcome("train_command", False, f"flag {flag!r} is not allowed")
        return CheckOutcome("train_command", True, " ".join(argv))

    def _check_overlay_shape(self, proposal: Proposal) -> CheckOutcome:
        overlay = proposal.config_overlay
        if not isinstance(overlay, dict):
            return CheckOutcome("overlay", False, "config_overlay must be a mapping")
        try:
            yaml.safe_dump(overlay)
        except yaml.YAMLError as exc:
            return CheckOutcome("overlay", False, f"not YAML-serialisable: {exc}")
        return CheckOutcome("overlay", True, f"{len(overlay)} section(s)")

    # -- sandbox ---------------------------------------------------------
    def _apply_in_sandbox(self, proposal: Proposal, report: ValidationReport) -> dict[str, str]:
        """Copy the touched files into a temp tree, apply the patches, verify they parse.

        The sandbox mirrors ``<repo>/<path>`` exactly, so ``git apply -p1`` runs from
        the repo directory with the same relative paths as the real checkout. A patch
        that applies here is guaranteed to apply to the live file.
        """
        applied: dict[str, str] = {}
        sandbox = Path(tempfile.mkdtemp(prefix="evolayer-validate-"))
        try:
            for edit in proposal.edits:
                spec = self.cfg.repos[edit.repo]
                source = spec.path / edit.path
                repo_root = sandbox / edit.repo
                target = repo_root / edit.path
                if not source.is_file():
                    report.checks.append(
                        CheckOutcome(f"sandbox {edit.path}", False, f"source file missing: {source}")
                    )
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                patch_file = sandbox / f"patch-{edit.repo}.diff"
                patch_file.write_text(edit.diff, encoding="utf-8")

                if not _git_apply(repo_root, patch_file):
                    report.checks.append(
                        CheckOutcome(
                            f"sandbox {edit.path}",
                            False,
                            "git apply --check rejected the patch (context does not match the file)",
                        )
                    )
                    continue
                ok, detail = _parses(target, edit.path)
                report.checks.append(CheckOutcome(f"sandbox {edit.path}", ok, detail))
                if ok:
                    applied[f"{edit.repo}:{edit.path}"] = target.read_text(encoding="utf-8")
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)
        return applied


def _git_apply(repo_root: Path, patch_file: Path) -> bool:
    """``git apply --check`` then a real apply, both rooted at the sandbox repo."""
    from ..execution import run

    repo_root.mkdir(parents=True, exist_ok=True)
    check = run(
        ["git", "apply", "--check", "-p1", str(patch_file)],
        cwd=str(repo_root),
        timeout=30,
    )
    if not check.ok:
        return False
    real = run(["git", "apply", "-p1", str(patch_file)], cwd=str(repo_root), timeout=30)
    return real.ok


def _parses(path: Path, relpath: str) -> tuple[bool, str]:
    try:
        if relpath.endswith(".py"):
            py_compile.compile(str(path), doraise=True, cfile=str(path) + ".pyc")
            Path(str(path) + ".pyc").unlink(missing_ok=True)
            return (True, "python syntax ok")
        if relpath.endswith((".yaml", ".yml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(data, (dict, list)):
                return (False, f"yaml did not parse to a mapping/list ({type(data).__name__})")
            return (True, "yaml parses")
        if relpath.endswith(".json"):
            import json

            json.loads(path.read_text(encoding="utf-8"))
            return (True, "json parses")
        return (True, "no parser for this file type")
    except (OSError, ValueError, SyntaxError, yaml.YAMLError) as exc:
        return (False, f"parse failed: {exc}")


def diff_paths(diff: str) -> list[str]:
    """Extract file paths declared by a unified diff (for defence-in-depth checks)."""
    paths = []
    for line in diff.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            raw = line[4:].strip()
            if raw in ("/dev/null",):
                continue
            paths.append(re.sub(r"^[ab]/", "", raw))
    return sorted(set(paths))
