"""The simulation runner: materialise the candidate, execute, measure, collect.

Order of operations matters and is enforced here:

1. the candidate file set is materialised from the *validated* patch output, so
   the simulator never reads a file the validator has not seen;
2. the argv is built by the backend and recorded before execution;
3. the process runs under a hard timeout with a log file;
4. metrics are extracted from artifacts, and a run that produces no measurable
   episodes is marked ``failed`` — never silently "ok";
5. artifacts are hashed so a report cannot be relinked to different bytes.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import EvoConfig
from ..execution import Result, run
from ..ids import content_id
from ..models import Artifact, Proposal, SimulationResult
from . import metrics as metrics_mod
from .backends import BACKENDS, Candidate, Plan, get_backend, write_plan_json
from .video import artifact_for, probe

ALLOWED_BINARIES = frozenset(
    {"python", "python3", "uv", "bash", "sh", "colcon", "run_inference.sh", "eval.sh", "git"}
)

DEFAULT_POLICY_YAML = "00_body_base/roboparty_deploy/src/inference/robots/rpo/configs/{policy}.yaml"


@dataclass
class RunnerOptions:
    backend: str | None = None
    dry_run: bool = False
    keep_workdir: bool = True
    allow_replay: bool = False


class Runner:
    def __init__(self, cfg: EvoConfig):
        self.cfg = cfg

    # -- candidate -------------------------------------------------------
    def materialize(self, proposal: Proposal, applied: dict[str, str], workdir: Path) -> Candidate:
        """Write validated file contents into ``workdir/candidate`` and resolve the evaluation inputs."""
        candidate_root = workdir / "candidate"
        candidate_root.mkdir(parents=True, exist_ok=True)
        patched: dict[str, Path] = {}
        for key, content in applied.items():
            repo_key, _, relpath = key.partition(":")
            target = candidate_root / repo_key / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            patched[key] = target

        skill = self.cfg.skill(proposal.skill)
        policy_yaml_key = f"deploy:src/inference/robots/{skill.deploy_robot}/configs/{skill.deploy_policy}.yaml"
        if policy_yaml_key in patched:
            policy_yaml = patched[policy_yaml_key]
        else:
            policy_yaml = self.cfg.resolve(
                DEFAULT_POLICY_YAML.format(policy=skill.deploy_policy or "default")
            )

        mujoco_conf = self.cfg.simulator.backend("mujoco")
        onnx_param = proposal.sim_plan.params.get("policy_onnx")
        if onnx_param:
            onnx = self.cfg.resolve(onnx_param)
        elif "deploy" in patched:
            # A patched deploy tree: prefer a policy file inside it, else the live model.
            model_dir = self.cfg.repo("deploy").path / f"src/inference/robots/{skill.deploy_robot}/models"
            candidates = sorted(model_dir.glob("policy*.onnx"))
            onnx = candidates[0] if candidates else None
        else:
            onnx = None

        mjcf = self.cfg.resolve(mujoco_conf.get("mjcf", "00_body_base/rpo_description/mjcf/rpo.xml"))
        return Candidate(
            policy_yaml=policy_yaml,
            policy_onnx=onnx,
            mjcf=mjcf,
            patched=patched,
            policy_version=f"{proposal.id}@r{proposal.revision}",
            notes=[f"{len(patched)} patched file(s)"],
        )

    # -- run -------------------------------------------------------------
    def run(
        self,
        proposal: Proposal,
        *,
        applied: dict[str, str] | None = None,
        options: RunnerOptions | None = None,
    ) -> SimulationResult:
        options = options or RunnerOptions()
        skill = self.cfg.skill(proposal.skill)
        backend_name = options.backend or proposal.sim_plan.backend or skill.backend
        backend = get_backend(backend_name)

        started = time.time()
        safe_id = f"{proposal.id}-r{proposal.revision}-{int(started)}"
        workdir = self.cfg.sim_dir / safe_id
        workdir.mkdir(parents=True, exist_ok=True)

        candidate = self.materialize(proposal, applied or {}, workdir)
        sim_plan = proposal.sim_plan
        sim_plan.episodes = min(sim_plan.episodes, self.cfg.simulator.max_episodes)
        sim_plan.timeout_s = min(sim_plan.timeout_s, self.cfg.simulator.timeout_s)

        try:
            plan = backend.plan(self.cfg, proposal, sim_plan, workdir, candidate)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            return SimulationResult(
                proposal_id=proposal.id,
                revision=proposal.revision,
                backend=backend_name,
                status="unavailable",
                error=f"cannot build the simulator command: {exc}",
                started_at=started,
                finished_at=time.time(),
            )

        write_plan_json(plan, workdir / "plan.json")
        if options.dry_run:
            return SimulationResult(
                proposal_id=proposal.id,
                revision=proposal.revision,
                backend=backend_name,
                status="ok",
                command=plan.argv,
                log_path=str(workdir / "plan.json"),
                error="dry run: command not executed",
                started_at=started,
                finished_at=time.time(),
            )

        if (
            plan.evidence_class == "rehearsal"
            and plan.emits_metrics
            and candidate.policy_onnx is None
            and _requires_candidate_policy(proposal)
        ):
            return SimulationResult(
                proposal_id=proposal.id,
                revision=proposal.revision,
                backend=backend_name,
                status="unavailable",
                command=plan.argv,
                error=(
                    "this proposal changes the policy, but no candidate ONNX is available. "
                    "Produce one first (backend 'isaaclab' chains training + rehearsal), or set "
                    "sim_plan.params.policy_onnx to an exported policy."
                ),
                started_at=started,
                finished_at=time.time(),
            )

        available, detail = backend.availability(self.cfg)
        if not available:
            return SimulationResult(
                proposal_id=proposal.id,
                revision=proposal.revision,
                backend=backend_name,
                status="unavailable",
                command=plan.argv,
                error=f"backend unavailable: {detail}",
                started_at=started,
                finished_at=time.time(),
            )

        result = self._execute(plan, workdir / "run.log")
        status = "ok" if result.ok else ("timeout" if result.timed_out else "failed")
        error = "" if result.ok else result.tail(30)

        artifacts: list[Artifact] = []
        extracted = metrics_mod.Extracted(None, [], "", "not run")
        if result.ok and plan.emits_metrics:
            candidates = [workdir / name for name in plan.artifacts] + [
                workdir / "episodes.jsonl",
                workdir / "metrics.json",
            ]
            extracted = metrics_mod.extract(candidates, skill=proposal.skill)
            if extracted.metrics is None:
                status = "failed"
                error = f"rehearsal produced no measurable episodes: {extracted.notes}"

        for name in ("episodes.jsonl", "metrics.json", "plan.json", "run.log"):
            path = workdir / name
            if path.is_file():
                artifacts.append(artifact_for(path, kind="metrics" if name.endswith((".jsonl", ".json")) else "log"))
        _collect_video(workdir, artifacts)

        if plan.evidence_class == "replay" and not options.allow_replay:
            status = "failed"
            error = (error + " | " if error else "") + "replay evidence cannot certify a policy change"

        if not options.keep_workdir and status == "failed":
            shutil.rmtree(workdir, ignore_errors=True)

        return SimulationResult(
            proposal_id=proposal.id,
            revision=proposal.revision,
            backend=backend_name,
            status=status,
            metrics=extracted.metrics,
            artifacts=artifacts,
            command=plan.argv,
            log_path=str(workdir / "run.log"),
            error=error,
            started_at=started,
            finished_at=time.time(),
            episodes_n=len(extracted.episodes) or (extracted.metrics.n if extracted.metrics else 0),
        )

    def _execute(self, plan: Plan, log_path: Path) -> Result:
        result = run(
            plan.argv,
            cwd=plan.cwd,
            timeout=plan.timeout_s,
            env=plan.env,
            log_path=log_path,
            allowed=ALLOWED_BINARIES,
        )
        if plan.followup is not None and result.ok:
            secondary = run(
                plan.followup.argv,
                cwd=plan.followup.cwd,
                timeout=plan.followup.timeout_s,
                env=plan.followup.env,
                log_path=log_path,
                allowed=ALLOWED_BINARIES,
            )
            return secondary
        return result

    # -- introspection ---------------------------------------------------
    def available_backends(self) -> list[dict[str, Any]]:
        from .backends import availability_report

        return availability_report(self.cfg)

    @staticmethod
    def result_id(result: SimulationResult) -> str:
        return content_id(
            "simulation",
            {
                "proposal_id": result.proposal_id,
                "revision": result.revision,
                "backend": result.backend,
                "started_at": result.started_at,
            },
        )


def _requires_candidate_policy(proposal: Proposal) -> bool:
    return bool(proposal.train_command) or proposal.kind in ("reward_term_weight", "train_hyperparameter")


def _collect_video(workdir: Path, artifacts: list[Artifact]) -> Path | None:
    for pattern in ("video/*.mp4", "video/frames/*.png", "*.mp4"):
        matches = sorted(workdir.glob(pattern))
        if matches:
            target = matches[0]
            artifacts.append(artifact_for(target, kind="video" if target.suffix == ".mp4" else "frame"))
            return target
    return None


def summarise(result: SimulationResult) -> str:
    if result.metrics is None:
        return f"{result.status} ({result.backend}) — no metrics: {result.error[:160]}"
    m = result.metrics
    return (
        f"{result.status} ({result.backend}) — n={m.n} success={m.success_rate:.1%} "
        f"falls={m.fall_rate:.1%} torque={m.torque_peak_nm:.1f} N·m"
    )


__all__ = ["Runner", "RunnerOptions", "Candidate", "Plan", "BACKENDS", "summarise", "probe"]
