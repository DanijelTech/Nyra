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


def test_symlinked_uv_cache_is_removed_with_its_target(tmp_path):
    """~/.cache is often a symlink to another disk; the space must actually be freed."""
    root, home = fake_checkout(tmp_path)
    target = tmp_path / "cache-on-another-disk" / "uv"
    target.mkdir(parents=True)
    (target / "blob").write_text("x", encoding="utf-8")
    shutil.rmtree(home / ".cache")
    (home / ".cache").symlink_to(tmp_path / "cache-on-another-disk")

    result = run(root, home, "--yes", "--uv-cache")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not target.exists(), "the symlink target (the actual data) must be removed"
    assert not (home / ".cache/uv").exists()


def test_symlinked_home_paths_are_still_allowed(tmp_path):
    """A repo reached through a symlink must not be refused either."""
    root, home = fake_checkout(tmp_path)
    link = tmp_path / "link-to-repo"
    link.symlink_to(root)
    result = run(link, home, "--yes", "--build")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (root / "06_assets/build").exists()


def _stub_apt(tmp_path: Path) -> Path:
    """A fake dpkg/apt-get so the package-selection logic can be tested for real."""
    bin_dir = tmp_path / "stubbin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "dpkg-query").write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *Status-Abbrev*)\n"
        "    printf 'ii \\tros-jazzy-rclcpp\\nii \\tros-jazzy-urdf\\nii \\tros-jazzy-ros-base\\n' ;;\n"
        "  *Installed-Size*)\n"
        "    printf '20000\\n20000\\n20000\\n' ;;\n"
        "  *) exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    (bin_dir / "dpkg").write_text(
        "#!/bin/sh\n"
        "case \"$2\" in ros-jazzy-ros-base|ros-jazzy-joy|python3-colcon-common-extensions) exit 0 ;; esac\n"
        "exit 1\n",
        encoding="utf-8",
    )
    (bin_dir / "apt-get").write_text(
        "#!/bin/sh\n"
        "echo 'Remv ros-jazzy-rclcpp [1.0]'\n"
        "echo 'Remv ros-jazzy-urdf [1.0]'\n"
        "echo 'After this operation, 4096 MB disk space will be freed.'\n",
        encoding="utf-8",
    )
    for name in ("dpkg-query", "dpkg", "apt-get"):
        (bin_dir / name).chmod(0o755)
    return bin_dir


def test_apt_lists_all_ros_packages_and_the_autoremove_preview(tmp_path):
    root, home = fake_checkout(tmp_path)
    stub = _stub_apt(tmp_path)
    env = {**os.environ, "HOME": str(home), "PATH": f"{stub}:/usr/bin:/bin"}
    result = subprocess.run(
        ["bash", str(root / "uninstall.sh"), "--apt", "--apt-ros-all", "--apt-autoremove"],
        capture_output=True, text=True, timeout=120, env=env, cwd=str(root),
    )
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "ALL 3 installed ros-jazzy-* packages" in out
    assert "58.6 MB" in out, out            # 60000 KB
    assert "autoremove" in out
    assert "4096 MB" in out
    # still a dry run: nothing was executed
    assert "EXECUTING" not in out


def test_apt_without_ros_all_points_at_the_flag(tmp_path):
    root, home = fake_checkout(tmp_path)
    stub = _stub_apt(tmp_path)
    env = {**os.environ, "HOME": str(home), "PATH": f"{stub}:/usr/bin:/bin"}
    result = subprocess.run(
        ["bash", str(root / "uninstall.sh"), "--apt"],
        capture_output=True, text=True, timeout=120, env=env, cwd=str(root),
    )
    assert result.returncode == 0
    assert "3 ros-jazzy-* package(s) remain" in result.stdout
    assert "--apt-ros-all" in result.stdout


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
