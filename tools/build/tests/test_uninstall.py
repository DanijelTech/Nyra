"""Tests for uninstall.sh — a destructive script must be proven, not trusted.

Every test builds a throw-away checkout in tmp_path with a fake HOME, so the real
home directory is never in scope.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "uninstall.sh"


def fake_checkout(tmp_path: Path) -> tuple[Path, Path]:
    """A checkout that looks like Nyra, plus a separate fake HOME."""
    root = tmp_path / "Nyra"
    home = tmp_path / "home"
    for rel in (
        "00_body_base/roboparty_deploy",
        "04_training/roboparty_train/.venv",
        "06_assets/build",
        "06_assets/evolayer",
        "02_motion_infra/UFO/.venv",
        "06_evolayer/__pycache__",
    ):
        (root / rel).mkdir(parents=True)
    (root / ".gitmodules").write_text("", encoding="utf-8")
    (root / "README.md").write_text("keep me\n", encoding="utf-8")
    (root / "00_body_base/roboparty_deploy/policy.onnx").write_bytes(b"onnx")
    (root / "06_assets/build/manifest.json").write_text("{}", encoding="utf-8")
    (root / "06_assets/evolayer/state.db").write_text("db", encoding="utf-8")
    (root / "04_training/roboparty_train/.venv/blob.bin").write_bytes(b"\0" * 4096)
    (root / "02_motion_infra/UFO/.venv/blob.bin").write_bytes(b"\0" * 4096)
    (root / "06_evolayer/__pycache__/a.pyc").write_text("x", encoding="utf-8")
    shutil.copy2(SCRIPT, root / "uninstall.sh")

    (home / ".venvs/nyra/bin").mkdir(parents=True)
    (home / ".venvs/nyra/bin/python").write_text("x", encoding="utf-8")
    (home / ".cache/uv").mkdir(parents=True)
    (home / ".cache/uv/blob").write_text("x", encoding="utf-8")
    return root, home


def run(root: Path, home: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "HOME": str(home), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    env.pop("NYRA_UNINSTALL_REEXEC", None)
    env.pop("NYRA_UNINSTALL_REPO", None)
    return subprocess.run(
        ["bash", str(root / "uninstall.sh"), *args],
        capture_output=True, text=True, timeout=120, env=env, cwd=str(root),
    )


def test_dry_run_deletes_nothing(tmp_path):
    root, home = fake_checkout(tmp_path)
    result = run(root, home, "--all")
    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
    assert (root / "06_assets/build/manifest.json").is_file()
    assert (root / "04_training/roboparty_train/.venv/blob.bin").is_file()
    assert (home / ".venvs/nyra/bin/python").is_file()
    assert (root / "README.md").is_file()


def test_build_and_envs_removed_repo_kept(tmp_path):
    root, home = fake_checkout(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("must survive\n", encoding="utf-8")

    result = run(root, home, "--yes", "--build", "--envs")
    assert result.returncode == 0, result.stdout + result.stderr

    # removed
    assert not (root / "06_assets/build").exists()
    assert not (root / "06_assets/evolayer").exists()
    assert not (root / "04_training/roboparty_train/.venv").exists()
    assert not (root / "02_motion_infra/UFO/.venv").exists()
    assert not (root / "06_evolayer/__pycache__").exists()
    assert not (home / ".venvs/nyra").exists()
    # kept
    assert (root / "README.md").is_file()
    assert (root / "00_body_base/roboparty_deploy/policy.onnx").is_file()
    assert outside.read_text(encoding="utf-8") == "must survive\n"


def test_all_removes_the_checkout_and_the_cache(tmp_path):
    root, home = fake_checkout(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("must survive\n", encoding="utf-8")

    tmpdir = Path(os.environ.get("TMPDIR", "/tmp"))
    before = {p.name for p in tmpdir.glob("nyra-uninstall-*.sh")}
    result = run(root, home, "--yes", "--all")
    after = {p.name for p in tmpdir.glob("nyra-uninstall-*.sh")}

    assert result.returncode == 0, result.stdout + result.stderr
    assert not root.exists(), "the checkout should be gone"
    assert not (home / ".venvs/nyra").exists()
    assert not (home / ".cache/uv").exists()
    assert outside.read_text(encoding="utf-8") == "must survive\n"
    # the re-exec'd copy removes itself (trap on EXIT)
    assert after - before == set(), f"leftover script copies: {after - before}"


def test_refuses_dangerous_paths(tmp_path):
    root, home = fake_checkout(tmp_path)
    for target in ("/", str(home), str(tmp_path)):
        result = run(root, home, "--yes", "--all", "--path", target)
        assert result.returncode == 2, f"{target}: {result.stdout} {result.stderr}"
        assert "refusing" in result.stderr
    assert root.is_dir()


def test_refuses_a_directory_without_gitmodules(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "keep.txt").write_text("x", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    result = subprocess.run(
        ["bash", str(SCRIPT), "--yes", "--all", "--path", str(plain)],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "HOME": str(home)},
    )
    assert result.returncode == 2
    assert "no .gitmodules" in result.stderr
    assert (plain / "keep.txt").is_file()


def test_unknown_flag_is_rejected(tmp_path):
    root, home = fake_checkout(tmp_path)
    result = run(root, home, "--nuke-everything")
    assert result.returncode == 2
    assert "unknown argument" in result.stderr


def test_writes_a_log(tmp_path):
    root, home = fake_checkout(tmp_path)
    log = tmp_path / "uninstall.log"
    result = run(root, home, "--yes", "--build", "--log", str(log))
    assert result.returncode == 0
    assert log.is_file()
    assert "removed" in log.read_text(encoding="utf-8")


def test_package_json_exposes_purge():
    import json

    package = json.loads((REPO_ROOT / "package.json").read_text(encoding="utf-8"))
    assert "purge" in package["scripts"]
    assert "uninstall.sh" in package["scripts"]["purge"]
    assert (REPO_ROOT / "uninstall.sh").is_file()
