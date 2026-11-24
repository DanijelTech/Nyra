"""Tests for nyra-build (the single build entry point behind the npm package).

Real subprocesses, real git, real files: a throw-away workspace is built and the
builder is driven through its public ``main(argv)`` API. The Node launcher is
tested as a subprocess when ``node`` is available.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BUILD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BUILD_DIR.parents[1]
sys.path.insert(0, str(BUILD_DIR))

import nyra_build  # noqa: E402

MINIMAL_URDF = """<?xml version="1.0"?>
<robot name="mini">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>
  <joint name="joint_a" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <limit lower="-1" upper="1" effort="10" velocity="5"/>
  </joint>
  <joint name="joint_b" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <limit lower="-1" upper="1" effort="10" velocity="5"/>
  </joint>
</robot>
"""

MINIMAL_MJCF = """<?xml version="1.0"?>
<mujoco model="mini">
  <asset><mesh name="body" file="body.stl"/></asset>
  <worldbody>
    <body name="base_link">
      <freejoint name="floating_base_joint"/>
      <geom type="mesh" mesh="body"/>
      <body name="link1">
        <joint name="joint_a" type="hinge" axis="0 0 1"/>
        <body name="link2">
          <joint name="joint_b" type="hinge" axis="0 0 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="joint_a" joint="joint_a"/>
    <motor name="joint_b" joint="joint_b"/>
  </actuator>
</mujoco>
"""

SKILL_YAML = """\
id: locomotion
title: Locomotion
backend: mujoco
repo: training
deploy_policy: amp
eval_suite:
  episodes: 4
limits:
  min_episodes: 4
"""


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True)


@pytest.fixture
def ws(tmp_path) -> Path:
    """A minimal but structurally valid Nyra workspace."""
    root = tmp_path / "ws"
    (root / "00_body_base/rpo_description/urdf").mkdir(parents=True)
    (root / "00_body_base/rpo_description/mjcf").mkdir(parents=True)
    (root / "00_body_base/rpo_description/meshes").mkdir(parents=True)
    (root / "00_body_base/roboparty_deploy/src/inference").mkdir(parents=True)
    (root / "05_integration/scripts").mkdir(parents=True)
    (root / "06_evolayer/config/skills").mkdir(parents=True)

    (root / "00_body_base/rpo_description/urdf/rpo.urdf").write_text(MINIMAL_URDF, encoding="utf-8")
    (root / "00_body_base/rpo_description/mjcf/rpo.xml").write_text(MINIMAL_MJCF, encoding="utf-8")
    (root / "00_body_base/rpo_description/meshes/body.stl").write_text("solid body\nendsolid\n",
                                                                     encoding="utf-8")
    (root / "06_evolayer/config/skills/locomotion.yaml").write_text(SKILL_YAML, encoding="utf-8")
    (root / "05_integration/scripts/setup_all.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (root / "05_integration/scripts/setup_all.sh").chmod(0o755)
    (root / ".gitmodules").write_text("", encoding="utf-8")

    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True, capture_output=True)
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@e", "commit", "-qm", "init")
    return root


def run_build(*args: str) -> int:
    return nyra_build.main([*args])


# --------------------------------------------------------------------------- graph


def test_plan_lists_stages_with_dependencies_in_order(capsys):
    assert run_build("--root", str(REPO_ROOT), "plan") == 0
    out = capsys.readouterr().out
    order = [line.split(".")[1].strip().split()[0] for line in out.splitlines() if line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8."))]
    assert order[0] == "submodules"
    assert order[-1] == "manifest"
    for stage in nyra_build.STAGES:
        for dep in stage.deps:
            assert order.index(dep) < order.index(stage.name)


def test_plan_json_is_machine_readable(capsys):
    assert run_build("--root", str(REPO_ROOT), "plan", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["builder"] == nyra_build.BUILDER_VERSION
    names = [stage["name"] for stage in payload["stages"]]
    assert names == list(nyra_build.STAGE_ORDER)


def test_unknown_stage_is_rejected():
    with pytest.raises(SystemExit) as excinfo:
        nyra_build.select_stages(["nope"], None)
    assert "unknown stage" in str(excinfo.value)


def test_only_pulls_in_dependencies_and_manifest():
    stages = nyra_build.select_stages(["contract"], None)
    names = [stage.name for stage in stages]
    assert names == ["submodules", "assets", "contract", "manifest"]


def test_skip_keeps_the_manifest():
    names = [stage.name for stage in nyra_build.select_stages(None, ["evolayer"])]
    assert "evolayer" not in names
    assert names[-1] == "manifest"


# --------------------------------------------------------------------------- assets stage


def test_assets_stage_validates_a_consistent_description(ws, capsys):
    rc = run_build("--root", str(ws), "build", "--only", "assets")
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "assets" in out and "2 joints" in out


def test_assets_stage_detects_a_joint_mismatch(ws, capsys):
    mjcf = ws / "00_body_base/rpo_description/mjcf/rpo.xml"
    mjcf.write_text(MINIMAL_MJCF.replace('joint="joint_b" type="hinge"', 'joint="renamed_b" type="hinge"')
                    .replace('name="joint_b"', 'name="renamed_b"'), encoding="utf-8")
    rc = run_build("--root", str(ws), "build", "--only", "assets")
    out = capsys.readouterr().out
    assert rc == 1
    assert "joint mismatch" in out


def test_assets_stage_detects_a_missing_mesh(ws, capsys):
    (ws / "00_body_base/rpo_description/meshes/body.stl").unlink()
    rc = run_build("--root", str(ws), "build", "--only", "assets")
    assert rc == 1
    assert "missing meshes" in capsys.readouterr().out


# --------------------------------------------------------------------------- capabilities


def test_missing_capability_skips_with_a_reason(ws, capsys, monkeypatch):
    # A PATH with only git + python: colcon/ros2/uv/node are genuinely absent.
    python_dir = Path(sys.executable).parent
    git_dir = Path(shutil.which("git")).parent
    monkeypatch.setenv("PATH", f"{python_dir}:{git_dir}")
    rc = run_build("--root", str(ws), "build", "--only", "deploy")
    out = capsys.readouterr().out
    assert rc == 0
    assert "skipped deploy" in out
    assert "colcon" in out and "ros2" in out


def test_strict_turns_a_required_skip_into_a_failure(ws, capsys, monkeypatch):
    python_dir = Path(sys.executable).parent
    git_dir = Path(shutil.which("git")).parent
    monkeypatch.setenv("PATH", f"{python_dir}:{git_dir}")
    rc = run_build("--root", str(ws), "build", "--only", "deploy", "--strict")
    captured = capsys.readouterr()
    assert rc == 1
    assert "strict" in captured.err


# --------------------------------------------------------------------------- manifest


def test_manifest_is_written_and_history_appended(ws, capsys):
    assert run_build("--root", str(ws), "build", "--only", "assets") == 0
    manifest = json.loads((ws / "06_assets/build/manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == nyra_build.MANIFEST_SCHEMA
    assert manifest["builder"] == nyra_build.BUILDER_VERSION
    assert manifest["git"]["branch"] == "main"
    assert [stage["name"] for stage in manifest["stages"]][-1] == "manifest"
    assert manifest["summary"]["ok"] >= 1
    assert manifest["status"] in ("ok", "degraded")

    assert run_build("--root", str(ws), "build", "--only", "assets", "--force") == 0
    history = (ws / "06_assets/build/manifest-history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history) == 2
    assert json.loads(history[-1])["builder"] == nyra_build.BUILDER_VERSION


def test_status_reads_the_last_manifest(ws, capsys):
    assert run_build("--root", str(ws), "build", "--only", "assets") == 0
    capsys.readouterr()
    assert run_build("--root", str(ws), "status") == 0
    assert "last build" in capsys.readouterr().out
    assert run_build("--root", str(ws), "status", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert "stages" in payload


def test_status_without_manifest_fails(ws, capsys):
    assert run_build("--root", str(ws), "status") == 1
    assert "no build manifest" in capsys.readouterr().err


def _stage_status(ws: Path, name: str) -> str:
    manifest = json.loads((ws / "06_assets/build/manifest.json").read_text(encoding="utf-8"))
    return next(stage["status"] for stage in manifest["stages"] if stage["name"] == name)


def test_second_run_reuses_a_cached_stage(ws, capsys):
    assert run_build("--root", str(ws), "build", "--only", "assets") == 0
    assert _stage_status(ws, "assets") == "ok"

    assert run_build("--root", str(ws), "build", "--only", "assets") == 0
    assert _stage_status(ws, "assets") == "cached"
    assert "unchanged inputs" in capsys.readouterr().out

    # Change a stage input (the skill configs are part of the fingerprint).
    skill = ws / "06_evolayer/config/skills/locomotion.yaml"
    skill.write_text(SKILL_YAML + "# touched\n", encoding="utf-8")
    assert run_build("--root", str(ws), "build", "--only", "assets") == 0
    assert _stage_status(ws, "assets") == "ok"


def test_force_ignores_the_cache(ws, capsys):
    assert run_build("--root", str(ws), "build", "--only", "assets") == 0
    assert run_build("--root", str(ws), "build", "--only", "assets") == 0
    assert _stage_status(ws, "assets") == "cached"
    assert run_build("--root", str(ws), "build", "--only", "assets", "--force") == 0
    assert _stage_status(ws, "assets") == "ok"


# --------------------------------------------------------------------------- dry run / clean


def test_dry_run_writes_nothing(ws, capsys):
    rc = run_build("--root", str(ws), "build", "--only", "assets", "--dry-run")
    out = capsys.readouterr().out
    assert rc == 0
    assert "dry run" in out
    assert not (ws / "06_assets/build/manifest.json").exists()


def test_clean_removes_only_build_outputs(ws, capsys):
    (ws / "06_assets/build").mkdir(parents=True)
    (ws / "06_assets/build/manifest.json").write_text("{}", encoding="utf-8")
    for name in ("build", "install", "log"):
        (ws / f"00_body_base/roboparty_deploy/{name}").mkdir(parents=True)
    keep = ws / "00_body_base/roboparty_deploy/src/inference/keep.txt"
    keep.write_text("precious", encoding="utf-8")

    assert run_build("--root", str(ws), "clean", "--dry-run") == 0
    assert (ws / "06_assets/build").exists()

    assert run_build("--root", str(ws), "clean") == 0
    for name in ("06_assets/build", "00_body_base/roboparty_deploy/build",
                 "00_body_base/roboparty_deploy/install", "00_body_base/roboparty_deploy/log"):
        assert not (ws / name).exists(), name
    assert keep.read_text(encoding="utf-8") == "precious"


# --------------------------------------------------------------------------- failure paths


def test_env_stages_are_opt_in(ws, capsys):
    rc = run_build("--root", str(ws), "build", "--only", "motion")
    out = capsys.readouterr().out
    assert rc == 0
    assert "opt-in" in out and "--with-envs" in out
    manifest = json.loads((ws / "06_assets/build/manifest.json").read_text(encoding="utf-8"))
    motion = next(stage for stage in manifest["stages"] if stage["name"] == "motion")
    assert motion["status"] == "skipped"


def test_env_install_refuses_when_disk_is_short(ws, capsys, monkeypatch):
    def fake_usage(path):
        return shutil._ntuple_diskusage(total=20e9, used=19e9, free=1e9)  # type: ignore[attr-defined]

    monkeypatch.setattr(nyra_build.shutil, "disk_usage", fake_usage)
    rc = run_build("--root", str(ws), "build", "--only", "motion", "--with-envs")
    captured = capsys.readouterr()
    assert rc == 2
    assert "refusing to install environments" in captured.err
    assert not (ws / "06_assets/build/manifest.json").exists()


def test_clean_refuses_to_delete_tracked_files(ws, capsys):
    tracked_dir = ws / "06_evolayer/config"
    assert run_build("--root", str(ws), "clean", "--venvs") == 0
    assert tracked_dir.exists(), "a directory with tracked files must never be removed"
    assert (ws / "06_evolayer/config/skills/locomotion.yaml").exists()


def test_contract_stage_fails_when_the_checker_is_missing(ws, capsys):
    rc = run_build("--root", str(ws), "build", "--only", "contract")
    out = capsys.readouterr().out
    assert rc == 1
    assert "check_compat.py is missing" in out


def test_doctor_reports_capabilities_and_modules(ws, capsys):
    assert run_build("--root", str(ws), "doctor", "--json") == 0
    report = json.loads(capsys.readouterr().out)
    assert report["builder"] == nyra_build.BUILDER_VERSION
    assert "capabilities" in report and "git" in report["capabilities"]
    assert report["modules"]["00_body_base/rpo_description"]["populated"] is True
    assert report["modules"]["04_training/roboparty_train"]["exists"] is False


def test_find_root_walks_up_from_a_subdirectory(ws):
    assert nyra_build.find_root(ws / "06_evolayer/config") == ws


def test_find_root_refuses_a_non_workspace(tmp_path):
    with pytest.raises(SystemExit):
        nyra_build.find_root(tmp_path)


# --------------------------------------------------------------------------- npm package


def test_package_json_matches_the_builder():
    package = json.loads((REPO_ROOT / "package.json").read_text(encoding="utf-8"))
    assert package["bin"]["nyra-build"] == "tools/build/cli.js"
    assert (REPO_ROOT / package["bin"]["nyra-build"]).is_file()
    # A build tool for a robot must not pull a dependency tree.
    assert "dependencies" not in package
    assert "devDependencies" not in package
    assert "tools/build/nyra_build.py" in package["files"]
    for script in ("build", "verify", "test", "doctor", "clean"):
        assert script in package["scripts"]
        assert "tools/build/cli.js" in package["scripts"][script]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_launcher_forwards_arguments_and_json(capsys):
    result = subprocess.run(
        ["node", str(BUILD_DIR / "cli.js"), "--root", str(REPO_ROOT), "plan", "--json"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["builder"] == nyra_build.BUILDER_VERSION


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_launcher_propagates_the_exit_code(ws):
    result = subprocess.run(
        ["node", str(BUILD_DIR / "cli.js"), "--root", str(ws), "build", "--only", "contract"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 1
    assert "check_compat.py is missing" in result.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_launcher_reports_a_missing_python(tmp_path):
    node = shutil.which("node")
    result = subprocess.run(
        [node, str(BUILD_DIR / "cli.js"), "--version"],
        capture_output=True, text=True, timeout=60,
        env={"PATH": str(tmp_path), "NYRA_PYTHON": "definitely-not-python"},
    )
    assert result.returncode == 127
    assert "NYRA_PYTHON" in result.stderr
