"""Shared fixtures.

Two kinds of workspace are used:

``real_repo``  the actual Nyra checkout, read-only. Used to prove that templates
               read real files and that the URDF/ONNX contract checks pass on the
               shipped configuration.

``workspace``  a throw-away git workspace with a fake deploy repo, a fake
               training repo and its own skill configs. It lets the end-to-end
               test exercise real git, real snapshots, real MuJoCo and the real
               installer without touching the checkout.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EVOLAYER_ROOT = REPO_ROOT / "06_evolayer"
sys.path.insert(0, str(EVOLAYER_ROOT))

from evo_layer.config import EvoConfig  # noqa: E402
from evo_layer.models import Episode  # noqa: E402


# --------------------------------------------------------------------------- onnx helpers


def build_policy(path: Path, obs_dim: int, *, action_value: float = 0.0) -> Path:
    """A deterministic ONNX 'policy' that emits a constant action.

    ``action_value=0`` holds the default pose (a benign, standing policy). The
    observation input is declared but unused, so the graph serves any obs width.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    actions = helper.make_tensor(
        "actions", TensorProto.FLOAT, [1, 23], np.full((1, 23), action_value, dtype=np.float32)
    )
    node = helper.make_node("Constant", [], ["action"], value=actions)
    graph = helper.make_graph(
        [node],
        "fixture_policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_dim])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 23])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


@pytest.fixture(scope="session")
def zero_policy(tmp_path_factory) -> Path:
    """Policy matching the `amp` deploy config (3-frame stack, 234 inputs)."""
    target = tmp_path_factory.mktemp("policies") / "policy_amp_zero.onnx"
    return build_policy(target, 234)


# --------------------------------------------------------------------------- episodes


def make_episodes(
    count: int,
    *,
    skill: str = "locomotion",
    falls: int = 0,
    start: float = 1_760_000_000.0,
    step: float = 120.0,
    torque: float = 40.0,
    velocity: float = 8.0,
    tracking: float | None = 0.1,
    torque_ratio: float | None = 0.3,
    vel_ratio: float | None = 0.3,
    source: str = "real",
    policy_version: str = "test-pol",
) -> list[Episode]:
    out = []
    for index in range(count):
        falls_here = 1 if index < falls else 0
        out.append(
            Episode(
                id=f"ep_{skill}_{index:05d}",
                skill=skill,
                source=source,
                policy_version=policy_version,
                started_at=start + index * step,
                duration_s=8.0,
                success=falls_here == 0,
                falls=falls_here,
                torque_peak_nm=torque,
                joint_vel_peak_rps=velocity,
                collisions=0,
                battery_min_v=48.0,
                tracking_error_rad=tracking,
                torque_ratio=torque_ratio,
                vel_ratio=vel_ratio,
                notes="test episode",
            )
        )
    return out


def write_jsonl(path: Path, episodes: list[Episode]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(
                json.dumps(
                    {
                        "record": "episode",
                        "schema_version": 1,
                        "episode_id": episode.id,
                        "skill": episode.skill,
                        "policy_version": episode.policy_version,
                        "source": episode.source,
                        "started_at": episode.started_at,
                        "duration_s": episode.duration_s,
                        "success": episode.success,
                        "falls": episode.falls,
                        "torque_peak_nm": episode.torque_peak_nm,
                        "joint_vel_peak_rps": episode.joint_vel_peak_rps,
                        "collisions": episode.collisions,
                        "battery_min_v": episode.battery_min_v,
                        "tracking_error_rad": episode.tracking_error_rad,
                        "torque_ratio": episode.torque_ratio,
                        "vel_ratio": episode.vel_ratio,
                        "notes": episode.notes,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return path


# --------------------------------------------------------------------------- workspaces


@pytest.fixture(scope="session")
def real_repo() -> Path:
    return REPO_ROOT


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True)


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True, capture_output=True)
    _git(path, "add", "-A")
    _git(path, "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-qm", "init")


SKILL_YAML = """\
id: locomotion
title: Locomotion (fixture)
title_override: null
backend: mujoco
repo: training
deploy_robot: rpo
deploy_policy: amp
train_task: RPO-AMP
metrics: [success_rate, fall_rate]
eval_suite:
  episodes: 6
  seeds: [0, 1]
  duration_s: 2.0
  commands:
    - [0.0, 0.0, 0.0]
  init_poses: [stand]
limits:
  fall_rate_max: 0.05
  success_drop_max: 0.02
  min_episodes: 5
"""


@pytest.fixture
def workspace(tmp_path, zero_policy) -> dict:
    """A fake Nyra workspace with deploy + training + evolayer repos and a config."""
    root = tmp_path / "ws"
    deploy = root / "00_body_base/roboparty_deploy"
    robot = deploy / "src/inference/robots/rpo"
    (robot / "configs").mkdir(parents=True)
    (robot / "models").mkdir(parents=True)
    (robot / "latents").mkdir(parents=True)
    for name in ("amp.yaml", "default.yaml"):
        shutil.copy2(
            REPO_ROOT / f"00_body_base/roboparty_deploy/src/inference/robots/rpo/configs/{name}",
            robot / "configs" / name,
        )
    shutil.copy2(
        REPO_ROOT / "00_body_base/roboparty_deploy/src/inference/robots/rpo/models/policy_amp.onnx",
        robot / "models" / "policy_amp.onnx",
    )
    _init_repo(deploy)

    training = root / "04_training/roboparty_train"
    (training / "robolab/robolab/tasks/manager_based/amp").mkdir(parents=True)
    (training / "robolab/robolab/assets/robots").mkdir(parents=True)
    shutil.copy2(
        REPO_ROOT / "04_training/roboparty_train/robolab/robolab/tasks/manager_based/amp/amp_env_cfg.py",
        training / "robolab/robolab/tasks/manager_based/amp/amp_env_cfg.py",
    )
    shutil.copy2(
        REPO_ROOT / "04_training/roboparty_train/robolab/robolab/assets/robots/roboparty.py",
        training / "robolab/robolab/assets/robots/roboparty.py",
    )
    _init_repo(training)

    skills_dir = root / "06_evolayer/config/skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "locomotion.yaml").write_text(SKILL_YAML, encoding="utf-8")
    _init_repo(root)

    cfg = EvoConfig.load(
        root=root,
        overrides={
            "state_dir": "state",
            "hermes": {"require_auth": False, "outbound": {"mode": "pull"}},
            "simulator": {"max_episodes": 12, "timeout_s": 600},
            "skills": [{"id": "locomotion", "title": "Locomotion (fixture)", "backend": "mujoco",
                        "repo": "training", "deploy_policy": "amp", "train_task": "RPO-AMP",
                        "eval_suite": {"episodes": 6, "seeds": [0, 1], "duration_s": 2.0,
                                       "commands": [[0.0, 0.0, 0.0]], "init_poses": ["stand"]},
                        "limits": {"fall_rate_max": 0.05, "min_episodes": 5}}],
        },
    )
    # Point the simulator at the real robot description and the fixture policy.
    cfg.simulator.backends["mujoco"] = dict(cfg.simulator.backends["mujoco"])
    cfg.simulator.backends["mujoco"].update(
        {
            "cwd": str(EVOLAYER_ROOT),
            "mjcf": str(REPO_ROOT / "00_body_base/rpo_description/mjcf/rpo.xml"),
            "urdf": str(REPO_ROOT / "00_body_base/rpo_description/urdf/rpo.urdf"),
        }
    )
    cfg.simulator.backends["fixture"] = dict(cfg.simulator.backends["fixture"])
    cfg.simulator.backends["fixture"].update(
        {
            "cwd": str(EVOLAYER_ROOT),
            "dataset": str(REPO_ROOT / "06_evolayer/examples/telemetry/real_locomotion.jsonl"),
        }
    )
    cfg.validate()
    cfg.ensure_dirs()
    return {"root": root, "cfg": cfg, "deploy": deploy, "training": training,
            "root_repo": root, "policy": zero_policy}


@pytest.fixture
def loop(workspace):
    from evo_layer.orchestrator.loop import EvoLoop

    return EvoLoop.from_config(workspace["cfg"], secret="unit-test-secret-value-0123456789", env={})


@pytest.fixture
def real_cfg() -> EvoConfig:
    """Config over the real checkout (read-only tests)."""
    return EvoConfig.load(root=REPO_ROOT, env={})
