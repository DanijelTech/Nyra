#!/usr/bin/env python3
"""nyra-build — build the whole Nyra workspace with one command.

This is the single source of truth for "build everything". The npm package at the
repository root is a thin launcher for it (``npx nyra-build``), because the stack
is ROS 2 + C++ + Python + uv: Node cannot drive ``colcon`` or ``uv`` better than a
process launcher can, and duplicating the pipeline in JavaScript would create two
definitions of "built" that drift apart.

Design rules
------------
* **Delegate, never duplicate.** Stages call the existing entry points
  (``setup_all.sh``, ``colcon``, ``check_compat.py``, ``pytest``) instead of
  re-implementing them.
* **Say what you skipped and why.** A missing capability (no ROS 2, no Isaac Lab,
  no GPU) yields ``skipped`` with a reason, not a fake success. ``--strict`` turns
  a skipped *required* stage into a failure, for CI.
* **No shell.** Every command is an argv list with a timeout; output is captured
  and mirrored to a log file.
* **Reproducible record.** Each run writes ``06_assets/build/manifest.json`` plus
  an append-only history, with git and toolchain versions, per-stage fingerprints
  and artifact paths.
* **Cheap re-runs.** A stage whose fingerprint (repo HEADs + builder version +
  stage inputs) is unchanged and whose last status was ``ok`` is reported as
  ``cached`` unless ``--force`` is given.

Usage
-----
    nyra-build plan      [--json]
    nyra-build build     [--jobs N] [--dry-run] [--strict] [--only …] [--skip …]
    nyra-build verify                              assets + contract only
    nyra-build test                                both pytest suites
    nyra-build doctor                              toolchain, capabilities, modules
    nyra-build status    [--json]
    nyra-build clean     [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

BUILDER_VERSION = "1.0.6"
MANIFEST_SCHEMA = 1
BUILD_DIR = Path("06_assets/build")
#: Local, git-ignored record of the interpreter this workspace should build with.
LOCAL_CONFIG = Path(".nyra-build.json")

OK, SKIPPED, FAILED, CACHED = "ok", "skipped", "failed", "cached"


# --------------------------------------------------------------------------- model


@dataclass(frozen=True)
class Stage:
    name: str
    title: str
    deps: tuple[str, ...] = ()
    caps: tuple[str, ...] = ()
    optional: bool = False
    #: CLI flag that must be present for this stage to run (heavy/opt-in stages).
    flag: str = ""
    #: Capabilities whose presence changes the stage's outcome; they are part of the
    #: fingerprint, so installing mujoco/pytest invalidates the cache.
    fingerprint_caps: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "deps": list(self.deps),
            "caps": list(self.caps),
            "optional": self.optional,
            "flag": self.flag,
        }


#: Stages that install Python environments. They are opt-in because they download
#: ~15 GB (motion infra) and ~10 GB (Isaac Lab / training) — a "build" that fills
#: the disk is not a build, it is an incident.
ENV_STAGES = frozenset({"motion", "training"})


#: The build graph. ``manifest`` always runs last and writes the record.
STAGES: tuple[Stage, ...] = (
    Stage("submodules", "initialise submodules (nested)", caps=("git",)),
    Stage("assets", "validate the robot description (URDF/MJCF/meshes)"),
    Stage("deploy", "colcon build of roboparty_inference", deps=("submodules",), caps=("colcon", "ros2")),
    Stage("motion", "uv environments for the motion infra (~15 GB)", deps=("submodules",),
          caps=("uv",), flag="with_envs", optional=True),
    Stage("training", "uv environment for roboparty_train (~10 GB)", deps=("submodules",),
          caps=("uv", "isaaclab"), optional=True, flag="with_envs"),
    Stage("contract", "module compatibility contract", deps=("submodules", "assets"), caps=("pyyaml",)),
    Stage("evolayer", "EvoLayer test suite", deps=("submodules",), caps=("pytest",), optional=True,
          fingerprint_caps=("pytest", "numpy", "mujoco", "onnxruntime")),
    Stage("manifest", "write the build manifest"),
)
STAGE_ORDER = tuple(stage.name for stage in STAGES)
STAGE_MAP = {stage.name: stage for stage in STAGES}


@dataclass
class StageResult:
    name: str
    status: str
    reason: str = ""
    duration_s: float = 0.0
    commands: list[list[str]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    fingerprint: str = ""
    log: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "duration_s": round(self.duration_s, 3),
            "commands": self.commands,
            "artifacts": self.artifacts,
            "fingerprint": self.fingerprint,
        }


@dataclass
class Context:
    root: Path
    build_dir: Path
    jobs: int = 1
    dry_run: bool = False
    force: bool = False
    strict: bool = False
    with_data: bool = False
    with_advanced: bool = False
    with_envs: bool = False
    skip_tests: bool = False
    verbose: bool = False
    capabilities: dict[str, dict[str, Any]] = field(default_factory=dict)
    log_path: Path | None = None
    cached: dict[str, StageResult] = field(default_factory=dict)

    @property
    def deploy(self) -> Path:
        return self.root / "00_body_base/roboparty_deploy"

    @property
    def description(self) -> Path:
        return self.root / "00_body_base/rpo_description"

    @property
    def integration(self) -> Path:
        return self.root / "05_integration/scripts"

    @property
    def evolayer(self) -> Path:
        return self.root / "06_evolayer"

    def has(self, capability: str) -> bool:
        return bool(self.capabilities.get(capability, {}).get("available"))

    def missing(self, caps: tuple[str, ...]) -> list[str]:
        return [cap for cap in caps if not self.has(cap)]

    def explain(self, caps: tuple[str, ...]) -> str:
        parts = []
        for cap in caps:
            entry = self.capabilities.get(cap, {})
            parts.append(f"{cap}: {entry.get('detail', 'unknown')}")
        return "; ".join(parts)


# --------------------------------------------------------------------------- helpers


def find_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / ".gitmodules").is_file():
            return candidate
    raise SystemExit("nyra-build: not inside a Nyra workspace (no .gitmodules found); pass --root")


def which(name: str) -> str | None:
    return shutil.which(name)


def probe_import(module: str, python: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [python, "-c", f"import {module}"], capture_output=True, text=True, timeout=60, shell=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "detail": f"import {module} failed: {exc}"}
    if result.returncode == 0:
        return {"available": True, "detail": f"import {module} ok"}
    return {"available": False, "detail": f"import {module} failed: {(result.stderr or '').strip()[:120]}"}


def local_config_path(root: Path) -> Path:
    return root / LOCAL_CONFIG


def read_local_python(root: Path) -> str:
    path = local_config_path(root)
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    candidate = str(data.get("python", ""))
    return candidate if candidate and Path(candidate).is_file() else ""


def write_local_python(root: Path, python: str) -> Path:
    path = local_config_path(root)
    payload = {"python": python, "set_at": time.time(), "builder": BUILDER_VERSION}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def probe_python(path: str) -> tuple[bool, str]:
    """A candidate interpreter must be Python 3.10+ and runnable."""
    try:
        result = subprocess.run(
            [path, "-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"],
            capture_output=True, text=True, timeout=60, shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return (False, f"{path}: {exc}")
    if result.returncode != 0:
        return (False, f"{path}: needs Python 3.10+ (or is not runnable)")
    version = subprocess.run([path, "-V"], capture_output=True, text=True, timeout=60, shell=False)
    return (True, (version.stdout or version.stderr or "").strip() or path)


def python_candidates(root: Path) -> list[str]:
    """Places a user plausibly installed the rehearsal dependencies."""
    home = Path.home()
    candidates = [
        root / ".venv/bin/python",
        root / "venv/bin/python",
        home / ".venvs/nyra/bin/python",
        root / "06_evolayer/.venv/bin/python",
    ]
    return [str(path) for path in candidates if path.is_file()]


def suggest_python(ctx: Context, needed: tuple[str, ...] = ("numpy", "mujoco", "onnxruntime")) -> str:
    """If another interpreter has the deps this one lacks, name it (with the command)."""
    if not ctx.missing(needed):
        return ""
    for candidate in python_candidates(ctx.root):
        if Path(candidate).resolve() == Path(sys.executable).resolve():
            continue
        probes = [probe_import(module, candidate) for module in needed]
        if all(probe.get("available") for probe in probes):
            return (
                f"{', '.join(needed)} are available in {candidate} — "
                f"run: nyra-build use-python {candidate}"
            )
    return ""


def detect_capabilities(ctx: Context) -> dict[str, dict[str, Any]]:
    caps: dict[str, dict[str, Any]] = {}
    sink_ros2 = which("ros2")
    for name in ("git", "uv", "colcon", "node", "npm", "ffmpeg"):
        path = which(name)
        caps[name] = {"available": bool(path), "detail": path or f"{name} not found on PATH"}
    for capability, module in (
        ("pyyaml", "yaml"),
        ("pytest", "pytest"),
        ("numpy", "numpy"),
        ("mujoco", "mujoco"),
        ("onnxruntime", "onnxruntime"),
    ):
        caps[capability] = probe_import(module, sys.executable)
    isaac = ctx.root / "04_training/IsaacLab-3.0.0"
    caps["isaaclab"] = {
        "available": isaac.is_dir(),
        "detail": str(isaac) if isaac.is_dir() else "missing 04_training/IsaacLab-3.0.0",
    }
    setup = find_ros_setup()
    caps["ros2"] = {
        "available": bool(setup) or sink_ros2 is not None,
        "detail": str(setup) if setup else (sink_ros2 or "not found under /opt/ros"),
    }
    caps["python3"] = {"available": True, "detail": sys.version.split()[0]}
    return caps


def run(ctx: Context, argv: list[str], *, cwd: Path | None = None, timeout: float = 3600.0) -> tuple[bool, int, str]:
    """Run one command. argv only, no shell, captured and mirrored to the build log."""
    if isinstance(argv, (str, bytes)):
        raise TypeError("argv must be a list of arguments")
    workdir = cwd or ctx.root
    printable = " ".join(argv)
    if ctx.dry_run:
        print(f"    would run: {printable}")
        return (True, 0, "dry-run")
    if ctx.verbose:
        print(f"    $ {printable}")
    started = time.time()
    try:
        completed = subprocess.run(
            argv, cwd=str(workdir), capture_output=True, text=True, timeout=timeout, shell=False
        )
    except subprocess.TimeoutExpired:
        _append_log(ctx, f"$ {printable}\n# TIMEOUT after {timeout:.0f}s\n")
        return (False, -9, "timeout")
    except OSError as exc:
        _append_log(ctx, f"$ {printable}\n# ERROR {exc}\n")
        return (False, -1, str(exc))
    _append_log(
        ctx,
        f"$ {printable}\n# cwd={workdir} exit={completed.returncode} {time.time() - started:.1f}s\n"
        + (completed.stdout or "")
        + (("# stderr\n" + completed.stderr) if completed.stderr else ""),
    )
    tail = (completed.stdout or "").strip().splitlines()
    return (completed.returncode == 0, completed.returncode, "\n".join(tail[-8:]))


def _append_log(ctx: Context, text: str) -> None:
    if ctx.log_path is None:
        return
    ctx.log_path.parent.mkdir(parents=True, exist_ok=True)
    with ctx.log_path.open("a", encoding="utf-8") as handle:
        handle.write(text if text.endswith("\n") else text + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                             timeout=60, shell=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip()


def git_info(root: Path) -> dict[str, Any]:
    if which("git") is None:
        return {"available": False}
    return {
        "available": True,
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "head": _git(root, "rev-parse", "HEAD"),
        "dirty": bool(_git(root, "status", "--porcelain")),
    }


def stage_fingerprint(ctx: Context, stage: Stage) -> str:
    """Cheap, honest fingerprint: repos, configs, builder version **and toolchain**.

    The toolchain matters: installing mujoco turns a "136 passed, 10 skipped" run
    into "146 passed". Without the interpreter and capability flags in the
    fingerprint, a cached result would hide exactly that change.
    """
    payload: dict[str, Any] = {
        "builder": BUILDER_VERSION,
        "stage": stage.name,
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "caps": {
            cap: bool(ctx.capabilities.get(cap, {}).get("available"))
            for cap in (stage.fingerprint_caps or stage.caps)
        },
    }
    for repo in ("00_body_base/roboparty_deploy", "00_body_base/rpo_description",
                 "04_training/roboparty_train", "05_integration", "06_evolayer"):
        path = ctx.root / repo
        git_dir = path / ".git"
        if git_dir.exists():
            payload[repo] = _git(path, "rev-parse", "HEAD")
        elif (ctx.root / ".git").exists() and path.exists():
            payload[repo] = _git(ctx.root, "rev-parse", f"HEAD:{repo}") or _git(ctx.root, "rev-parse", "HEAD")
    skills = ctx.evolayer / "config/skills"
    if skills.is_dir():
        digests = sorted(sha256_file(path) for path in skills.glob("*.yaml"))
        payload["skills"] = hashlib.sha256("".join(digests).encode()).hexdigest()[:16]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- stages


def stage_submodules(ctx: Context) -> StageResult:
    ok_sync, _, tail = run(ctx, ["git", "submodule", "sync", "--recursive"])
    if not ok_sync:
        return StageResult("submodules", FAILED, f"git submodule sync failed: {tail}")
    ok, code, tail = run(
        ctx, ["git", "submodule", "update", "--init", "--recursive", "--jobs", str(ctx.jobs)], timeout=7200
    )
    if not ok:
        return StageResult("submodules", FAILED, f"git submodule update failed ({code}): {tail}")
    mimic = ctx.root / "02_motion_infra/MimicLite"
    if mimic.exists():
        # MimicLite declares its submodules over SSH; map to HTTPS for keyless clones.
        run(ctx, ["git", "-C", str(mimic), "config", "url.https://github.com/.insteadOf", "git@github.com:"])
        run(ctx, ["git", "-C", str(mimic), "submodule", "update", "--init", "--recursive"])
    status = _git(ctx.root, "submodule", "status", "--recursive")
    uninitialised = [line.split()[1] for line in status.splitlines() if line.startswith("-")]
    if uninitialised:
        return StageResult(
            "submodules", FAILED,
            f"{len(uninitialised)} submodule(s) not initialised: {uninitialised[:6]}",
        )
    count = len([line for line in status.splitlines() if line.strip()])
    return StageResult("submodules", OK, f"{count} submodule(s) initialised")


def stage_assets(ctx: Context) -> StageResult:
    """Validate the robot description with no external toolchain at all."""
    import xml.etree.ElementTree as ET

    urdf = ctx.description / "urdf/rpo.urdf"
    mjcf = ctx.description / "mjcf/rpo.xml"
    if not urdf.is_file() or not mjcf.is_file():
        return StageResult("assets", SKIPPED, "00_body_base/rpo_description is missing or empty")
    try:
        urdf_root = ET.parse(urdf).getroot()
        mjcf_root = ET.parse(mjcf).getroot()
    except ET.ParseError as exc:
        return StageResult("assets", FAILED, f"description is not valid XML: {exc}")

    urdf_joints = [
        joint.get("name") for joint in urdf_root.findall("joint")
        if joint.get("type") not in (None, "fixed", "floating")
    ]
    mjcf_joints = [
        joint.get("name") for joint in mjcf_root.iter("joint")
        if joint.get("name") and joint.get("name") != "floating_base_joint"
    ]
    if not urdf_joints:
        return StageResult("assets", FAILED, "URDF has no actuated joints")
    if urdf_joints != mjcf_joints:
        missing = sorted(set(urdf_joints) - set(mjcf_joints))
        return StageResult(
            "assets", FAILED,
            f"URDF/MJCF joint mismatch ({len(urdf_joints)} vs {len(mjcf_joints)}; missing in MJCF: {missing or '-'})",
        )

    meshes_dir = ctx.description / "meshes"
    referenced = {
        Path(mesh.get("file", "")).name
        for mesh in mjcf_root.iter("mesh")
        if mesh.get("file")
    }
    missing_meshes = sorted(name for name in referenced if not (meshes_dir / name).is_file())
    if missing_meshes:
        return StageResult("assets", FAILED, f"missing meshes: {missing_meshes[:6]}")
    return StageResult(
        "assets", OK, f"{len(urdf_joints)} joints, {len(referenced)} meshes verified",
        artifacts=["00_body_base/rpo_description/urdf/rpo.urdf",
                   "00_body_base/rpo_description/mjcf/rpo.xml"],
    )


#: Environment variables a sourced ROS 2 overlay needs (colcon, ament, DDS).
ROS_ENV_VARS = (
    "AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH", "CMAKE_PREFIX_PATH", "LD_LIBRARY_PATH",
    "ROS_DISTRO", "ROS_VERSION", "ROS_PYTHON_VERSION", "ROS_LOCALHOST_ONLY",
    "RMW_IMPLEMENTATION", "FASTRTPS_DEFAULT_PROFILES_FILE", "ROS_AUTOMATIC_DISCOVERY_RANGE",
)


def find_ros_setup(base: Path = Path("/opt/ros")) -> Path | None:
    """The ROS 2 setup.bash this machine has, if any (``base`` is injectable for tests)."""
    import os

    distro = os.environ.get("ROS_DISTRO", "")
    if distro and (base / distro / "setup.bash").is_file():
        return base / distro / "setup.bash"
    candidates = sorted(base.glob("*/setup.bash")) if base.is_dir() else []
    return candidates[0] if candidates else None


def ros_script(setup: Path, argv: list[str]) -> str:
    """A tiny script that sources ROS and execs an already-allowlisted argv.

    The pipeline keeps "no shell" as its rule, so the only place a shell exists is
    this generated file: every argument is quoted with shlex, so nothing from user
    data can become a command. It is written to the build directory and removed
    afterwards.
    """
    import shlex

    quoting = " ".join(shlex.quote(part) for part in argv)
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"source {shlex.quote(str(setup))}\n"
        f"exec {quoting}\n"
    )


def run_in_ros_env(ctx: Context, argv: list[str], *, cwd: Path) -> tuple[bool, int, str]:
    """Run ``argv`` with ROS 2 sourced, without handing the guard a shell string."""
    import os

    setup = find_ros_setup()
    if setup is None or os.environ.get("AMENT_PREFIX_PATH"):
        # Already sourced in this shell (or no ROS): plain run, env passed through.
        return run(ctx, argv, cwd=cwd)
    assert setup is not None
    ctx.build_dir.mkdir(parents=True, exist_ok=True)
    script = ctx.build_dir / "ros-env.sh"
    script.write_text(ros_script(setup, argv), encoding="utf-8")
    script.chmod(0o700)
    try:
        return run(ctx, ["bash", str(script)], cwd=cwd)
    finally:
        script.unlink(missing_ok=True)


def stage_deploy(ctx: Context) -> StageResult:
    missing = ctx.missing(("colcon", "ros2"))
    if missing:
        return StageResult("deploy", SKIPPED, f"missing capability — {ctx.explain(tuple(missing))}")
    inference = ctx.deploy / "src/inference"
    if not inference.is_dir() or not any(inference.iterdir()):
        return StageResult("deploy", SKIPPED, "roboparty_deploy/src/inference is empty (submodule not initialised)")
    result = StageResult("deploy", OK)
    argv = [
        "colcon", "build", "--symlink-install", "--base-paths", "src",
        "--packages-up-to", "roboparty_inference",
    ]
    result.commands.append(argv)
    ok, code, tail = run_in_ros_env(ctx, argv, cwd=ctx.deploy)
    if not ok:
        setup = find_ros_setup()
        hint = "" if setup else " (no ROS 2 setup.bash found under /opt/ros — install ros-jazzy-ros-base)"
        result.status, result.reason = FAILED, f"colcon build failed ({code}):{hint} {tail}"
        return result
    result.reason = "roboparty_inference built"
    if (ctx.deploy / "install/setup.bash").is_file():
        result.artifacts.append("00_body_base/roboparty_deploy/install/setup.bash")
    return result


def _setup_all(ctx: Context, phases: list[str]) -> StageResult:
    script = ctx.integration / "setup_all.sh"
    if not script.is_file():
        return StageResult("?", FAILED, "05_integration/scripts/setup_all.sh is missing")
    argv = [str(script), *phases, "--yes"]
    if ctx.with_data:
        argv.append("--with-data")
    ok, code, tail = run(ctx, argv, cwd=ctx.integration, timeout=7200)
    result = StageResult("?", OK if ok else FAILED)
    result.commands.append(argv)
    result.reason = (
        "environments ready" if ok
        else f"setup_all.sh {' '.join(phases)} failed ({code}): {tail}"
    )
    return result


def stage_motion(ctx: Context) -> StageResult:
    missing = ctx.missing(("uv",))
    if missing:
        return StageResult("motion", SKIPPED, f"missing capability — {ctx.explain(tuple(missing))}")
    result = _setup_all(ctx, ["--motion"])
    result.name = "motion"
    return result


def stage_training(ctx: Context) -> StageResult:
    missing = ctx.missing(("uv", "isaaclab"))
    if missing:
        return StageResult("training", SKIPPED, f"missing capability — {ctx.explain(tuple(missing))}")
    result = _setup_all(ctx, ["--train"])
    result.name = "training"
    return result


def stage_contract(ctx: Context) -> StageResult:
    missing = ctx.missing(("pyyaml",))
    if missing:
        return StageResult("contract", SKIPPED, f"missing capability — {ctx.explain(tuple(missing))}")
    script = ctx.integration / "check_compat.py"
    if not script.is_file():
        return StageResult("contract", FAILED, "05_integration/scripts/check_compat.py is missing")
    argv = [sys.executable, str(script), "--no-write"]
    ok, code, tail = run(ctx, argv, timeout=600)
    result = StageResult("contract", OK if ok else FAILED)
    result.commands.append(argv)
    if ok:
        result.reason = tail.splitlines()[-1] if tail else "compat contract holds"
    else:
        result.reason = f"compatibility contract failed ({code}): {tail}"
        if (ctx.root / "06_assets/compat_report.json").is_file():
            result.artifacts.append("06_assets/compat_report.json")
    return result


def stage_evolayer(ctx: Context) -> StageResult:
    if ctx.skip_tests:
        return StageResult("evolayer", SKIPPED, "--skip-tests")
    missing = ctx.missing(("pytest",))
    if missing:
        return StageResult("evolayer", SKIPPED, f"missing capability — {ctx.explain(tuple(missing))}")
    tests = ctx.evolayer / "tests"
    if not tests.is_dir():
        return StageResult("evolayer", SKIPPED, "06_evolayer/tests not present")
    argv = [
        sys.executable, "-m", "pytest", "-o", "addopts=", "-q", "-p", "no:cacheprovider", str(tests)
    ]
    ok, code, tail = run(ctx, argv, cwd=ctx.evolayer, timeout=2400)
    result = StageResult("evolayer", OK if ok else FAILED)
    result.commands.append(argv)
    missing_sim = ctx.missing(("numpy", "mujoco", "onnxruntime"))
    note = (
        f" — rehearsal tests skipped ({', '.join(missing_sim)} missing: "
        "python3 -m pip install mujoco numpy onnxruntime)"
        if missing_sim and ok else ""
    )
    if ok:
        result.reason = (summary_line(tail) or "tests passed") + note
    else:
        result.reason = f"pytest failed ({code}): {tail}"
    return result


STAGE_FUNCS: dict[str, Callable[[Context], StageResult]] = {
    "submodules": stage_submodules,
    "assets": stage_assets,
    "deploy": stage_deploy,
    "motion": stage_motion,
    "training": stage_training,
    "contract": stage_contract,
    "evolayer": stage_evolayer,
}


# --------------------------------------------------------------------------- orchestration


def select_stages(only: list[str] | None, skip: list[str] | None) -> list[Stage]:
    names = list(STAGE_ORDER)
    for label, values in (("--only", only), ("--skip", skip)):
        unknown = [value for value in (values or []) if value not in STAGE_MAP]
        if unknown:
            raise SystemExit(f"nyra-build: unknown stage(s) in {label}: {unknown} (known: {list(STAGE_ORDER)})")
    if only:
        chosen = set(only)
        changed = True
        while changed:
            changed = False
            for name in list(chosen):
                for dep in STAGE_MAP[name].deps:
                    if dep not in chosen:
                        chosen.add(dep)
                        changed = True
        chosen.add("manifest")
        names = [name for name in names if name in chosen]
    if skip:
        names = [name for name in names if name not in skip]
    if "manifest" not in names:
        names.append("manifest")
    return [STAGE_MAP[name] for name in names]


_SUMMARY_RE = None


def summary_line(text: str) -> str:
    """The most informative last line of a captured run (pytest prints progress last)."""
    import re

    global _SUMMARY_RE
    if _SUMMARY_RE is None:
        _SUMMARY_RE = re.compile(r"\b\d+ (passed|failed|error|errors|skipped|xfailed|xpassed|warning|warnings)\b")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if _SUMMARY_RE.search(line):
            return line
    return lines[-1] if lines else ""


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / 1e9


#: Where the environment stages put their virtualenvs (used to tell "install" from "verify").
ENV_MARKERS = {
    "training": ("04_training/roboparty_train/.venv", "04_training/roboparty_train/.venv-physx"),
    "motion": ("02_motion_infra/human-humanoid-tools/.venv", "02_motion_infra/PartyMimic/.venv",
               "02_motion_infra/UFO/.venv", "02_motion_infra/any4hdmi/.venv"),
}


def env_already_installed(ctx: Context, stage_name: str) -> bool:
    return any((ctx.root / marker).is_dir() for marker in ENV_MARKERS.get(stage_name, ()))


def preflight_disk(ctx: Context, stages: list[Stage], min_free_gb: float) -> str | None:
    """Refuse an environment install that cannot fit — but do not block a verification.

    ``uv sync`` on an existing environment downloads almost nothing, so requiring the
    full install budget again would block a legitimate `--with-envs` run right after
    the install.
    """
    needs = [stage.name for stage in stages if stage.name in ENV_STAGES]
    if not needs:
        return None
    pending = [name for name in needs if not env_already_installed(ctx, name)]
    free = disk_free_gb(ctx.root)
    if not pending:
        return None if free >= 2.0 else (
            f"{', '.join(needs)} already installed, but only {free:.1f} GB free — "
            "a verification run needs ~2 GB"
        )
    if free >= min_free_gb:
        return None
    return (
        f"refusing to install environments: {', '.join(pending)} needs roughly 15 GB (motion) and "
        f"10 GB (training), but only {free:.1f} GB is free (limit {min_free_gb:.0f} GB). "
        "Free space, lower --min-free-gb, or skip these stages."
    )


def load_previous_manifest(ctx: Context) -> dict[str, Any] | None:
    path = ctx.build_dir / "manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def previous_results(manifest: dict[str, Any] | None) -> dict[str, StageResult]:
    results: dict[str, StageResult] = {}
    for entry in (manifest or {}).get("stages", []):
        if not isinstance(entry, dict) or "name" not in entry:
            continue
        results[entry["name"]] = StageResult(
            name=entry["name"],
            status=entry.get("status", SKIPPED),
            reason=entry.get("reason", ""),
            fingerprint=entry.get("fingerprint", ""),
        )
    return results


def run_graph(ctx: Context, stages: list[Stage]) -> list[StageResult]:
    results: dict[str, StageResult] = {}
    pending = {stage.name: stage for stage in stages if stage.name != "manifest"}
    running: dict[str, Any] = {}
    started = 0
    total = len(stages)

    with ThreadPoolExecutor(max_workers=max(1, ctx.jobs)) as pool:
        while pending or running:
            changed = False
            for name in list(pending):
                stage = pending[name]
                if any(dep not in results for dep in stage.deps):
                    continue
                if stage.flag and not getattr(ctx, stage.flag, False):
                    results[name] = StageResult(
                        name, SKIPPED,
                        f"opt-in: pass --{stage.flag.replace('_', '-')} to install this environment",
                    )
                    del pending[name]
                    changed = True
                    _report_stage(results[name])
                    continue
                missing = ctx.missing(stage.caps)
                if missing:
                    results[name] = StageResult(
                        name, SKIPPED, f"missing capability — {ctx.explain(tuple(missing))}"
                    )
                    del pending[name]
                    changed = True
                    _report_stage(results[name])
                    continue
                fingerprint = stage_fingerprint(ctx, stage)
                cached = ctx.cached.get(name)
                if (
                    not ctx.force
                    and cached is not None
                    and cached.status == OK
                    and cached.fingerprint == fingerprint
                ):
                    results[name] = StageResult(
                        name, CACHED, "unchanged inputs since the last build", fingerprint=fingerprint
                    )
                    del pending[name]
                    changed = True
                    _report_stage(results[name])
                    continue
                started += 1
                print(f"[{started}/{total}] {name}: {stage.title}")
                running[name] = pool.submit(_run_stage, ctx, stage, fingerprint)
                del pending[name]
                changed = True
            for name in list(running):
                if running[name].done():
                    results[name] = running[name].result()
                    del running[name]
                    _report_stage(results[name])
                    changed = True
            if not changed:
                if running:
                    time.sleep(0.05)
                    continue
                raise SystemExit(f"nyra-build: unresolvable dependencies for: {sorted(pending)}")
    return [results[name] for name in STAGE_ORDER if name in results]


def _run_stage(ctx: Context, stage: Stage, fingerprint: str) -> StageResult:
    started = time.time()
    free_before = disk_free_gb(ctx.root)
    try:
        result = STAGE_FUNCS[stage.name](ctx)
    except Exception as exc:  # noqa: BLE001 - a crashing stage must be reported, not hide the build
        result = StageResult(stage.name, FAILED, f"{type(exc).__name__}: {exc}")
    result.duration_s = time.time() - started
    result.fingerprint = fingerprint
    used = free_before - disk_free_gb(ctx.root)
    if used > 0.5:
        result.reason = f"{result.reason} (disk -{used:.1f} GB)"
    return result


def _report_stage(result: StageResult) -> None:
    label = {OK: "ok   ", SKIPPED: "skip ", FAILED: "FAIL ", CACHED: "cache"}.get(result.status, result.status)
    print(f"    {label} {result.name}: {result.reason}")


# --------------------------------------------------------------------------- manifest


def write_manifest(ctx: Context, results: list[StageResult], *, started_at: float) -> Path:
    summary = {
        status: sum(1 for result in results if result.status == status)
        for status in (OK, SKIPPED, FAILED, CACHED)
    }
    overall = FAILED if summary[FAILED] else ("degraded" if summary[SKIPPED] else OK)
    payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "builder": BUILDER_VERSION,
        "created_at": time.time(),
        "started_at": started_at,
        "duration_s": round(time.time() - started_at, 3),
        "root": str(ctx.root),
        "git": git_info(ctx.root),
        "toolchain": {
            "python": sys.version.split()[0],
            "node": _version_of("node"),
            "npm": _version_of("npm"),
        },
        "capabilities": {
            name: {"available": bool(entry.get("available")), "detail": entry.get("detail", "")}
            for name, entry in sorted(ctx.capabilities.items())
        },
        "stages": [result.to_dict() for result in results],
        "summary": summary,
        "status": overall,
    }
    ctx.build_dir.mkdir(parents=True, exist_ok=True)
    manifest = ctx.build_dir / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (ctx.build_dir / "manifest-history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "created_at": payload["created_at"],
            "status": overall,
            "summary": summary,
            "git": payload["git"],
            "builder": BUILDER_VERSION,
        }, sort_keys=True) + "\n")
    return manifest


def _version_of(name: str) -> str:
    if which(name) is None:
        return ""
    try:
        out = subprocess.run([name, "--version"], capture_output=True, text=True, timeout=30, shell=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (out.stdout or "").strip().splitlines()[0] if out.stdout.strip() else ""


# --------------------------------------------------------------------------- commands


def cmd_plan(args: argparse.Namespace) -> int:
    ctx = _context(args)
    ctx.capabilities = detect_capabilities(ctx)
    stages = select_stages(getattr(args, "only", None), getattr(args, "skip", None))
    if args.json:
        print(json.dumps({
            "builder": BUILDER_VERSION,
            "root": str(ctx.root),
            "stages": [
                {**stage.to_dict(), "caps_available": {cap: ctx.has(cap) for cap in stage.caps}}
                for stage in stages
            ],
        }, indent=2))
        return 0
    print(f"nyra-build {BUILDER_VERSION} · {ctx.root}")
    for index, stage in enumerate(stages, start=1):
        detail = []
        if stage.caps:
            states = ", ".join(f"{cap}{'' if ctx.has(cap) else ' (missing)'}" for cap in stage.caps)
            detail.append(f"needs: {states}")
        if stage.deps:
            detail.append(f"after: {', '.join(stage.deps)}")
        if stage.optional:
            detail.append("optional")
        if stage.flag:
            detail.append(f"needs --{stage.flag.replace('_', '-')}")
        suffix = f"  [{'; '.join(detail)}]" if detail else ""
        print(f"  {index}. {stage.name:12} {stage.title}{suffix}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    ctx = _context(args)
    started = time.time()
    ctx.capabilities = detect_capabilities(ctx)
    ctx.cached = previous_results(load_previous_manifest(ctx))
    if not ctx.dry_run:
        ctx.build_dir.mkdir(parents=True, exist_ok=True)
        ctx.log_path = ctx.build_dir / f"build-{time.strftime('%Y%m%d-%H%M%S')}.log"

    stages = select_stages(getattr(args, "only", None), getattr(args, "skip", None))
    blocked = preflight_disk(ctx, stages, float(getattr(args, "min_free_gb", 12.0)))
    if blocked:
        print(f"nyra-build: {blocked}", file=sys.stderr)
        return 2
    if not args.json:
        print(
            f"nyra-build {BUILDER_VERSION} · {ctx.root} · "
            f"python {sys.version.split()[0]} ({sys.executable})"
            f"{' (dry run)' if ctx.dry_run else ''}"
        )
        available = sum(1 for entry in ctx.capabilities.values() if entry.get("available"))
        print(f"capabilities {available}/{len(ctx.capabilities)} · stages {len(stages)}")

    results = run_graph(ctx, stages)

    # The manifest records itself, so its entry is decided before the write.
    manifest_stage = StageResult(
        "manifest",
        OK,
        reason="06_assets/build/manifest.json",
        artifacts=["06_assets/build/manifest.json"],
        duration_s=0.0,
    )
    if ctx.dry_run:
        manifest_stage.status = SKIPPED
        manifest_stage.reason = "dry run: no manifest written"
        manifest_stage.artifacts = []
    results.append(manifest_stage)
    if not ctx.dry_run:
        try:
            write_manifest(ctx, results, started_at=started)
        except OSError as exc:
            manifest_stage.status = FAILED
            manifest_stage.reason = f"cannot write manifest: {exc}"
            manifest_stage.artifacts = []

    failed = [result for result in results if result.status == FAILED]
    skipped = [result for result in results if result.status == SKIPPED]
    blocked = [r for r in skipped if not STAGE_MAP[r.name].optional]

    if args.json:
        print(json.dumps({
            "builder": BUILDER_VERSION,
            "root": str(ctx.root),
            "status": "failed" if failed else ("degraded" if skipped else OK),
            "stages": [result.to_dict() for result in results],
            "summary": {
                status: sum(1 for result in results if result.status == status)
                for status in (OK, SKIPPED, FAILED, CACHED)
            },
        }, indent=2))
    else:
        print()
        print(
            f"{time.time() - started:.1f}s · "
            f"{sum(1 for r in results if r.status == OK)} ok · "
            f"{sum(1 for r in results if r.status == CACHED)} cached · "
            f"{len(skipped)} skipped · {len(failed)} failed"
        )
        for result in skipped:
            print(f"  skipped {result.name}: {result.reason}")
        for result in failed:
            print(f"  FAILED  {result.name}: {result.reason}")
        if ctx.log_path is not None:
            print(f"log: {ctx.log_path.relative_to(ctx.root)}")
        print("manifest: 06_assets/build/manifest.json")

    if failed:
        return 1
    if ctx.strict and blocked:
        print(f"nyra-build: --strict and required stages were skipped: {[r.name for r in blocked]}",
              file=sys.stderr)
        return 1
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    args.only = ["assets", "contract"]
    args.skip = getattr(args, "skip", None)
    return cmd_build(args)


def cmd_test(args: argparse.Namespace) -> int:
    ctx = _context(args)
    ctx.capabilities = detect_capabilities(ctx)
    if not ctx.has("pytest"):
        print("nyra-build: pytest is not available (python3 -m pip install pytest)", file=sys.stderr)
        return 2
    suites = [ctx.evolayer / "tests", ctx.root / "tools/build/tests"]
    failures = 0
    for suite in suites:
        if not suite.is_dir():
            print(f"  skip {suite} (missing)")
            continue
        print(f"  {suite.relative_to(ctx.root)}")
        ok, code, tail = run(
            ctx,
            [sys.executable, "-m", "pytest", "-o", "addopts=", "-q", "-p", "no:cacheprovider", str(suite)],
            cwd=suite.parent,
            timeout=2400,
        )
        last = tail.splitlines()[-1] if tail else ""
        print(f"    {'ok' if ok else f'FAILED ({code})'}: {last}")
        failures += 0 if ok else 1
    return 1 if failures else 0


def cmd_doctor(args: argparse.Namespace) -> int:
    ctx = _context(args)
    ctx.capabilities = detect_capabilities(ctx)
    modules: dict[str, Any] = {}
    for module in ("00_body_base/roboparty_deploy", "00_body_base/rpo_description",
                   "04_training/roboparty_train", "02_motion_infra/human-humanoid-tools",
                   "06_evolayer"):
        path = ctx.root / module
        modules[module] = {
            "exists": path.is_dir(),
            "populated": path.is_dir() and any(path.iterdir()),
        }
    evolayer: dict[str, Any] | None = None
    if (ctx.evolayer / "evo_layer").is_dir():
        ok, code, tail = run(ctx, [sys.executable, "-m", "evo_layer", "doctor"], cwd=ctx.evolayer, timeout=300)
        evolayer = {"ok": ok, "exit": code, "detail": tail.splitlines()[-1] if tail else ""}
    report = {
        "builder": BUILDER_VERSION,
        "root": str(ctx.root),
        "git": git_info(ctx.root),
        "capabilities": ctx.capabilities,
        "modules": modules,
        "evolayer": evolayer,
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"nyra-build {BUILDER_VERSION} · {ctx.root}")
    print(f"interpreter: {sys.executable} (python {sys.version.split()[0]})")
    configured = read_local_python(ctx.root)
    if configured and Path(configured).resolve() != Path(sys.executable).resolve():
        print(f"  note: {LOCAL_CONFIG} says {configured}, but this run uses {sys.executable}")
    print(f"git: {report['git'].get('branch')} {str(report['git'].get('head'))[:10]} "
          f"{'dirty' if report['git'].get('dirty') else 'clean'}")
    for name, entry in sorted(report["capabilities"].items()):
        mark = "ok  " if entry.get("available") else "MISS"
        print(f"  {mark} {name:10} {entry.get('detail', '')}")
    for name, entry in modules.items():
        mark = "ok  " if entry["exists"] and entry["populated"] else "MISS"
        print(f"  {mark} {name}")
    if evolayer is not None:
        print(f"  {'ok  ' if evolayer['ok'] else 'FAIL'} evolayer doctor")
    hint = suggest_python(ctx)
    if hint:
        print(f"  hint: {hint}")
    required = ("git", "python3")
    missing = [name for name in required if not ctx.has(name)]
    return 1 if missing else 0


def cmd_use_python(args: argparse.Namespace) -> int:
    ctx = _context(args)
    if args.unset:
        path = local_config_path(ctx.root)
        if path.is_file():
            path.unlink()
            print(f"removed {path.relative_to(ctx.root)} (back to the default python3 on PATH)")
        else:
            print("no local interpreter configured")
        return 0
    candidate = args.python or ""
    if not candidate:
        found = python_candidates(ctx.root)
        if not found:
            print(
                "no candidate interpreter found. Create one first:\n"
                "  python3 -m venv ~/.venvs/nyra && ~/.venvs/nyra/bin/pip install pyyaml pytest "
                "mujoco numpy onnxruntime onnx",
                file=sys.stderr,
            )
            return 2
        candidate = found[0]
    # Do NOT resolve symlinks: a venv's bin/python is a symlink to the base
    # interpreter, and following it would lose the venv's site-packages.
    candidate = str(Path(candidate).expanduser().absolute())
    if not Path(candidate).is_file():
        print(f"nyra-build: not a file: {candidate}", file=sys.stderr)
        return 2
    ok, detail = probe_python(candidate)
    if not ok:
        print(f"nyra-build: {detail}", file=sys.stderr)
        return 2
    path = write_local_python(ctx.root, candidate)
    caps = {module: probe_import(module, candidate).get("available") for module in
            ("yaml", "pytest", "numpy", "mujoco", "onnxruntime", "onnx")}
    print(f"using {candidate} ({detail})")
    print(f"written to {path.relative_to(ctx.root)} — every nyra-build run will use it")
    for module, available in caps.items():
        print(f"  {'ok  ' if available else 'MISS'} {module}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    ctx = _context(args)
    manifest = load_previous_manifest(ctx)
    if manifest is None:
        print("no build manifest yet — run `nyra-build build`", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    age = max(0.0, time.time() - manifest.get("created_at", 0))
    print(f"last build {age / 60:.1f} min ago · status {manifest.get('status')} · "
          f"summary {manifest.get('summary')}")
    for entry in manifest.get("stages", []):
        print(f"  {entry.get('status', '?'):7} {entry.get('name', '?'):12} {str(entry.get('reason', ''))[:100]}")
    return 0


#: Multi-gigabyte environments recreated by ``--with-envs``. The MimicLite entry is
#: deliberately the *inner* uv environments: ``active-adaptation/venv`` also holds
#: tracked manifests, so deleting the directory would destroy tracked files.
VENV_DIRS = (
    "02_motion_infra/human-humanoid-tools/.venv",
    "02_motion_infra/any4hdmi/.venv",
    "02_motion_infra/PartyMimic/.venv",
    "02_motion_infra/UFO/.venv",
    "02_motion_infra/MimicLite/active-adaptation/venv/mjlab/.venv",
    "02_motion_infra/MimicLite/active-adaptation/venv/isaaclab/.venv",
    "04_training/roboparty_train/.venv",
    "04_training/roboparty_train/.venv-physx",
)


def contains_tracked_files(path: Path) -> bool:
    """True when a git repository tracks anything inside ``path``."""
    probe = _git(path if path.is_dir() else path.parent, "rev-parse", "--show-toplevel")
    if not probe:
        return False
    repo_root = Path(probe)
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    return bool(_git(repo_root, "ls-files", "--", str(rel)).strip())


def cmd_clean(args: argparse.Namespace) -> int:
    ctx = _context(args)
    targets = [
        ctx.build_dir,
        ctx.deploy / "build",
        ctx.deploy / "install",
        ctx.deploy / "log",
        ctx.evolayer / ".pytest_cache",
        ctx.root / ".pytest_cache",
        ctx.root / "tools/build/.pytest_cache",
    ]
    if args.venvs:
        # Explicit: these are multi-gigabyte installs, recreated by --with-envs.
        targets.extend(ctx.root / rel for rel in VENV_DIRS)
    removed = 0
    for target in targets:
        if not target.exists():
            continue
        try:
            label = target.relative_to(ctx.root)
        except ValueError:
            label = target
        if contains_tracked_files(target):
            print(f"refusing to remove {label}: it contains tracked files")
            continue
        if args.dry_run:
            print(f"would remove {label}")
            continue
        shutil.rmtree(target, ignore_errors=True)
        print(f"removed {label}")
        removed += 1
    if removed == 0 and not args.dry_run:
        print("nothing to remove")
    return 0


# --------------------------------------------------------------------------- entrypoint


def _context(args: argparse.Namespace) -> Context:
    root = Path(args.root).resolve() if args.root else find_root()
    return Context(
        root=root,
        build_dir=root / BUILD_DIR,
        jobs=max(1, int(getattr(args, "jobs", 1) or 1)),
        dry_run=bool(getattr(args, "dry_run", False)),
        force=bool(getattr(args, "force", False)),
        strict=bool(getattr(args, "strict", False)),
        with_data=bool(getattr(args, "with_data", False)),
        with_advanced=bool(getattr(args, "with_advanced", False)),
        with_envs=bool(getattr(args, "with_envs", False)),
        skip_tests=bool(getattr(args, "skip_tests", False)),
        verbose=bool(getattr(args, "verbose", False)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nyra-build", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help="workspace root (default: discovered via .gitmodules)")
    parser.add_argument("--version", action="version", version=f"nyra-build {BUILDER_VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func: Callable[[argparse.Namespace], int], help_text: str) -> argparse.ArgumentParser:
        child = sub.add_parser(name, help=help_text)
        child.set_defaults(func=func)
        return child

    plan = add("plan", cmd_plan, "print the stage graph and capability availability")
    plan.add_argument("--json", action="store_true")
    plan.add_argument("--only", nargs="*")
    plan.add_argument("--skip", nargs="*")

    build = add("build", cmd_build, "build every stage this machine supports")
    build.add_argument("--only", nargs="*", help="run these stages (dependencies are added)")
    build.add_argument("--skip", nargs="*", help="skip these stages")
    build.add_argument("--jobs", type=int, default=1, help="parallel stages (default 1)")
    build.add_argument("--dry-run", action="store_true", help="print commands, change nothing")
    build.add_argument("--force", action="store_true", help="ignore cached stage results")
    build.add_argument("--strict", action="store_true", help="fail when a required stage is skipped")
    build.add_argument("--with-data", action="store_true", help="also download motion datasets")
    build.add_argument("--with-envs", action="store_true",
                       help="also install the Python environments (~15 GB motion, ~10 GB training)")
    build.add_argument("--min-free-gb", type=float, default=12.0,
                       help="free space required before installing environments (default 12)")
    build.add_argument("--with-advanced", action="store_true")
    build.add_argument("--skip-tests", action="store_true")
    build.add_argument("--json", action="store_true")
    build.add_argument("--verbose", action="store_true")

    verify = add("verify", cmd_verify, "assets + compatibility contract (fast pre-flight)")
    verify.add_argument("--json", action="store_true")
    verify.add_argument("--dry-run", action="store_true")
    verify.add_argument("--skip", nargs="*")

    test = add("test", cmd_test, "run both pytest suites")
    test.add_argument("--json", action="store_true")

    doctor = add("doctor", cmd_doctor, "toolchain, capabilities and module state")
    doctor.add_argument("--json", action="store_true")

    use_python = add("use-python", cmd_use_python,
                     "remember which Python interpreter (and its venv) nyra-build should use")
    use_python.add_argument("python", nargs="?", help="path to a python 3.10+ interpreter")
    use_python.add_argument("--unset", action="store_true", help="forget the local choice")

    status = add("status", cmd_status, "show the last build manifest")
    status.add_argument("--json", action="store_true")

    clean = add("clean", cmd_clean, "remove build outputs (and, with --venvs, the environments)")
    clean.add_argument("--dry-run", action="store_true")
    clean.add_argument("--venvs", action="store_true", help="also delete the module .venv directories")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("nyra-build: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
