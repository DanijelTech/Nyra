"""Simulator backends.

Each backend answers two questions: *can I run here?* and *what is the exact
command?* The runner owns execution and parsing, so a backend stays a pure
function from (proposal, plan, workdir) to an argv — which makes every command
inspectable in a report before anything runs, and testable without a GPU.

``evidence_class`` is the honesty mechanism:

``rehearsal``  the backend actually executes the candidate and measures it.
               Only this class can certify a proposal for deployment.
``replay``     the backend replays recorded episodes (the fixture backend). It
               exercises the loop and can seed a baseline, but it cannot validate
               a change, so the orchestrator refuses to deploy on it.
``producer``   the backend trains/exports a candidate artifact but does not
               measure episodes. It is chained in front of a rehearsal backend.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..config import EvoConfig
from ..models import Proposal, SimPlan

EVIDENCE_CLASSES = ("rehearsal", "replay", "producer")


@dataclass
class Plan:
    argv: list[str]
    cwd: Path
    timeout_s: int
    artifacts: list[str] = field(default_factory=list)
    emits_metrics: bool = False
    evidence_class: str = "rehearsal"
    env: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    followup: "Plan | None" = None

    def describe(self) -> dict[str, Any]:
        return {
            "argv": self.argv,
            "cwd": str(self.cwd),
            "timeout_s": self.timeout_s,
            "artifacts": self.artifacts,
            "emits_metrics": self.emits_metrics,
            "evidence_class": self.evidence_class,
            "notes": self.notes,
            "followup": self.followup.describe() if self.followup else None,
        }


class Backend(Protocol):
    name: str

    def availability(self, cfg: EvoConfig) -> tuple[bool, str]: ...

    def plan(
        self,
        cfg: EvoConfig,
        proposal: Proposal,
        sim_plan: SimPlan,
        workdir: Path,
        candidate: "Candidate",
    ) -> Plan: ...


@dataclass
class Candidate:
    """The file set a rehearsal should evaluate, already patched."""

    policy_yaml: Path
    policy_onnx: Path | None
    mjcf: Path | None
    patched: dict[str, Path] = field(default_factory=dict)
    policy_version: str = ""
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- backends


class MujocoBackend:
    """Our in-repo MuJoCo evaluator: real physics, real episodes, GPU-free."""

    name = "mujoco"
    evidence_class = "rehearsal"

    def _config(self, cfg: EvoConfig) -> dict[str, Any]:
        return cfg.simulator.backend("mujoco")

    def availability(self, cfg: EvoConfig) -> tuple[bool, str]:
        conf = self._config(cfg)
        # Default to the interpreter running the builder: that is where the user
        # installs mujoco/onnxruntime, and probing anything else lies about it.
        python = str(conf.get("python") or sys.executable)
        if not Path(python).is_file() and shutil.which(python) is None:
            return (False, f"python interpreter not found: {python}")
        for module in ("mujoco", "onnxruntime", "numpy"):
            probe = __import__("subprocess").run(
                [python, "-c", f"import {module}"], capture_output=True, text=True
            )
            if probe.returncode != 0:
                return (False, f"{module} not importable by {python}: pip install {module}")
        mjcf = cfg.resolve(conf.get("mjcf", "00_body_base/rpo_description/mjcf/rpo.xml"))
        if not mjcf.is_file():
            return (False, f"MJCF not found: {mjcf}")
        return (True, f"{Path(python).name} + mujoco, model {cfg.rel(mjcf)}")

    def plan(
        self,
        cfg: EvoConfig,
        proposal: Proposal,
        sim_plan: SimPlan,
        workdir: Path,
        candidate: Candidate,
    ) -> Plan:
        conf = self._config(cfg)
        python = str(conf.get("python") or sys.executable)
        mjcf = candidate.mjcf or cfg.resolve(conf.get("mjcf", "00_body_base/rpo_description/mjcf/rpo.xml"))
        argv = [
            python,
            "-m",
            "evo_layer.simulator.mujoco_runner",
            "--mjcf",
            str(mjcf),
            "--policy-yaml",
            str(candidate.policy_yaml),
            "--episodes",
            str(sim_plan.episodes),
            "--seeds",
            ",".join(str(s) for s in (sim_plan.seeds or [0])),
            "--duration",
            str(float(sim_plan.params.get("duration_s", 8.0))),
            "--skill",
            proposal.skill,
            "--policy-version",
            candidate.policy_version or proposal.id,
            "--out",
            str(workdir / "episodes.jsonl"),
            "--quiet",
        ]
        commands = sim_plan.params.get("commands") or [[0.6, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.6]]
        for cmd in commands:
            argv += ["--command", ",".join(f"{float(v):g}" for v in cmd)]
        poses = sim_plan.params.get("init_poses") or ["stand"]
        argv += ["--init-poses", ",".join(str(p) for p in poses)]
        urdf = cfg.resolve(conf.get("urdf", "00_body_base/rpo_description/urdf/rpo.urdf"))
        if urdf.is_file():
            argv += ["--urdf", str(urdf)]
        if candidate.policy_onnx is not None:
            argv += ["--policy", str(candidate.policy_onnx)]
        if sim_plan.params.get("video", True):
            argv += ["--video-dir", str(workdir / "video")]
        return Plan(
            argv=argv,
            cwd=cfg.resolve(conf.get("cwd", "06_evolayer")),
            timeout_s=sim_plan.timeout_s,
            artifacts=["episodes.jsonl"],
            emits_metrics=True,
            evidence_class="rehearsal",
            notes="MuJoCo rehearsal of the candidate policy with deploy-equivalent PD control",
        )


class FixtureBackend:
    """Replay recorded episodes. Exercises the loop; cannot certify a change."""

    name = "fixture"
    evidence_class = "replay"

    def _config(self, cfg: EvoConfig) -> dict[str, Any]:
        return cfg.simulator.backend("fixture")

    def availability(self, cfg: EvoConfig) -> tuple[bool, str]:
        conf = self._config(cfg)
        dataset = cfg.resolve(conf.get("dataset", "06_evolayer/examples/telemetry/sim_recovery.jsonl"))
        if not dataset.is_file():
            return (False, f"replay dataset not found: {dataset}")
        return (True, f"replay dataset {cfg.rel(dataset)}")

    def plan(
        self,
        cfg: EvoConfig,
        proposal: Proposal,
        sim_plan: SimPlan,
        workdir: Path,
        candidate: Candidate,
    ) -> Plan:
        conf = self._config(cfg)
        source = Path(sim_plan.params.get("dataset") or cfg.resolve(conf.get("dataset", "")))
        if not source.is_file():
            raise FileNotFoundError(f"replay dataset not found: {source}")
        python = str(conf.get("python") or sys.executable)
        argv = [
            python,
            "-m",
            "evo_layer.simulator.fixture_runner",
            "--dataset",
            str(source),
            "--skill",
            proposal.skill,
            "--episodes",
            str(sim_plan.episodes),
            "--out",
            str(workdir / "episodes.jsonl"),
        ]
        return Plan(
            argv=argv,
            cwd=cfg.resolve(conf.get("cwd", "06_evolayer")),
            timeout_s=min(sim_plan.timeout_s, 300),
            artifacts=["episodes.jsonl"],
            emits_metrics=True,
            evidence_class="replay",
            notes="replays recorded episodes; cannot certify a policy change",
        )


class IsaacLabBackend:
    """Train/export through roboparty_train (produces a candidate ONNX)."""

    name = "isaaclab"
    evidence_class = "producer"

    def _config(self, cfg: EvoConfig) -> dict[str, Any]:
        return cfg.simulator.backend("isaaclab")

    def availability(self, cfg: EvoConfig) -> tuple[bool, str]:
        conf = self._config(cfg)
        cwd = cfg.resolve(conf.get("cwd", "04_training/roboparty_train"))
        python = cwd / str(conf.get("python", ".venv/bin/python"))
        if not python.is_file():
            return (False, f"training environment missing: {cfg.rel(python)} (run setup_all.sh --train)")
        if not (cwd / "robolab/scripts/rsl_rl/train.py").is_file():
            return (False, "robolab submodule not initialised")
        return (True, f"Isaac Lab workspace at {cfg.rel(cwd)}")

    def plan(
        self,
        cfg: EvoConfig,
        proposal: Proposal,
        sim_plan: SimPlan,
        workdir: Path,
        candidate: Candidate,
    ) -> Plan:
        conf = self._config(cfg)
        cwd = cfg.resolve(conf.get("cwd", "04_training/roboparty_train"))
        python = str(conf.get("python", ".venv/bin/python"))
        argv = [python, *proposal.train_command] if proposal.train_command else [
            python,
            "robolab/scripts/rsl_rl/train.py",
            f"--task={conf.get('task', 'RPO-AMP')}",
            "--viz=none",
            "--logger=tensorboard",
        ]
        exported = cwd / "exported/policy.onnx"
        rehearsal_candidate = Candidate(
            policy_yaml=candidate.policy_yaml,
            policy_onnx=exported,
            mjcf=candidate.mjcf,
            patched=candidate.patched,
            policy_version=candidate.policy_version,
            notes=list(candidate.notes) + [f"candidate exported to {cfg.rel(exported)}"],
        )
        followup = MujocoBackend().plan(cfg, proposal, sim_plan, workdir, rehearsal_candidate)
        return Plan(
            argv=argv,
            cwd=cwd,
            timeout_s=sim_plan.timeout_s,
            artifacts=["exported/policy.onnx"],
            emits_metrics=True,
            evidence_class="rehearsal",
            notes=(
                "training run chained into a MuJoCo rehearsal of exported/policy.onnx; "
                "the measurements come from the second step"
            ),
            followup=followup,
        )


class UfoBackend:
    """UFO training / inference wrapper."""

    name = "ufo"
    evidence_class = "producer"

    def _config(self, cfg: EvoConfig) -> dict[str, Any]:
        return cfg.simulator.backend("ufo")

    def availability(self, cfg: EvoConfig) -> tuple[bool, str]:
        conf = self._config(cfg)
        cwd = cfg.resolve(conf.get("cwd", "02_motion_infra/UFO"))
        entry = cwd / str(conf.get("entry", "./run_inference.sh"))
        if not entry.is_file():
            return (False, f"UFO checkout missing: {cfg.rel(entry)}")
        if not (cwd / ".venv").is_dir():
            return (False, "UFO environment missing (uv sync --locked)")
        return (True, f"UFO at {cfg.rel(cwd)}")

    def plan(
        self,
        cfg: EvoConfig,
        proposal: Proposal,
        sim_plan: SimPlan,
        workdir: Path,
        candidate: Candidate,
    ) -> Plan:
        conf = self._config(cfg)
        cwd = cfg.resolve(conf.get("cwd", "02_motion_infra/UFO"))
        entry = str(conf.get("entry", "./run_inference.sh"))
        argv = [
            entry,
            str(conf.get("mode", "tracking")),
            "--model-folder",
            str(sim_plan.params.get("model_folder", "runs/rpo_fb")),
            "--motion-list",
            str(sim_plan.params.get("motion_list", 0)),
            "--export-onnx",
            "--save-mp4",
        ]
        return Plan(
            argv=argv,
            cwd=cwd,
            timeout_s=sim_plan.timeout_s,
            artifacts=[],
            emits_metrics=False,
            evidence_class="producer",
            notes="UFO tracking inference + ONNX export; measure with the mujoco backend",
        )


class PartyMimicBackend:
    """PartyMimic evaluation wrapper."""

    name = "partymimic"
    evidence_class = "producer"

    def _config(self, cfg: EvoConfig) -> dict[str, Any]:
        return cfg.simulator.backend("partymimic")

    def availability(self, cfg: EvoConfig) -> tuple[bool, str]:
        conf = self._config(cfg)
        cwd = cfg.resolve(conf.get("cwd", "02_motion_infra/PartyMimic"))
        entry = cwd / str(conf.get("entry", "scripts/eval.sh"))
        if not entry.is_file():
            return (False, f"PartyMimic checkout missing: {cfg.rel(entry)}")
        if not (cwd / ".venv").is_dir():
            return (False, "PartyMimic environment missing (uv sync)")
        return (True, f"PartyMimic at {cfg.rel(cwd)}")

    def plan(
        self,
        cfg: EvoConfig,
        proposal: Proposal,
        sim_plan: SimPlan,
        workdir: Path,
        candidate: Candidate,
    ) -> Plan:
        conf = self._config(cfg)
        cwd = cfg.resolve(conf.get("cwd", "02_motion_infra/PartyMimic"))
        argv = [
            "bash",
            str(conf.get("entry", "scripts/eval.sh")),
            str(sim_plan.params.get("checkpoint", "")),
            str(workdir / "eval"),
            str(sim_plan.params.get("robot", "rp1")),
        ]
        env = {}
        if conf.get("sim2real"):
            env["SIM2REAL"] = str(cfg.resolve(conf["sim2real"]))
        return Plan(
            argv=argv,
            cwd=cwd,
            timeout_s=sim_plan.timeout_s,
            artifacts=[],
            emits_metrics=False,
            evidence_class="producer",
            env=env,
            notes="PartyMimic sim2sim evaluation; measure with the mujoco backend",
        )


BACKENDS: dict[str, Backend] = {
    "mujoco": MujocoBackend(),
    "fixture": FixtureBackend(),
    "isaaclab": IsaacLabBackend(),
    "ufo": UfoBackend(),
    "partymimic": PartyMimicBackend(),
}


def get_backend(name: str) -> Backend:
    try:
        return BACKENDS[name]
    except KeyError as exc:
        raise KeyError(f"unknown backend {name!r} (known: {sorted(BACKENDS)})") from exc


def availability_report(cfg: EvoConfig) -> list[dict[str, Any]]:
    rows = []
    for name, backend in BACKENDS.items():
        try:
            ok, detail = backend.availability(cfg)
        except Exception as exc:  # noqa: BLE001 - a broken backend must not hide the others
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        rows.append({"backend": name, "available": ok, "detail": detail})
    return rows


def write_plan_json(plan: Plan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.describe(), indent=2) + "\n", encoding="utf-8")
