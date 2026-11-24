"""Deployment installer.

Deployment is the only write path that reaches the robot, so it is deliberately
narrow and ordered so that a failure at any step leaves a recoverable state:

1. re-check the decision (approved, latest revision, gate not rejecting);
2a. snapshot the deploy state **before** touching it (rollback point);
2b. create the git tag/branch that pins the source state;
3. apply the validated patch to the deploy repo on an ``evolayer/…`` branch;
4. copy the artifacts (ONNX + sidecars + latents) into the robot's model dir;
5. run the deploy-contract checks against the *deployed* files;
6. commit + tag, record the deployment.

If step 5 fails, the snapshot is restored immediately and the deployment is
recorded as failed — the robot's directory is left as it was found.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from ..audit import AuditLog
from ..config import EvoConfig
from ..ids import content_id
from ..models import Decision, Deployment, Proposal, SimulationResult, Verdict
from ..safety.limits import load_urdf_limits
from ..safety.policy import check_policy_contract
from ..store import Store
from ..versioning.repo import GitError, GitRepo, sanitise_branch_component
from ..versioning.snapshot import Snapshot, SnapshotError, Snapshotter
from ..simulator.video import artifact_for


class DeployError(RuntimeError):
    """Deployment refused or aborted (state has been restored)."""


@dataclass
class DeployedFile:
    path: str
    sha256: str
    size: int


class Installer:
    def __init__(
        self,
        cfg: EvoConfig,
        store: Store,
        audit: AuditLog,
        snapshots: Snapshotter,
    ):
        self.cfg = cfg
        self.store = store
        self.audit = audit
        self.snapshots = snapshots

    # -- guards ----------------------------------------------------------
    def preflight(self, proposal: Proposal, verdict: Verdict | None, decision: Decision | None) -> None:
        if decision is None:
            raise DeployError("no Hermes decision on record for this proposal revision")
        if decision.verdict != "approve":
            raise DeployError(f"decision is {decision.verdict!r}, only 'approve' can deploy")
        if decision.proposal_id != proposal.id or decision.revision != proposal.revision:
            raise DeployError("decision does not match this proposal revision")
        if verdict is None or verdict.verdict == "reject":
            raise DeployError("safety gate did not pass; deployment is impossible")
        latest = self.store.proposals_for_finding(proposal.finding_id)
        if latest and latest[-1].id != proposal.id:
            raise DeployError(
                f"proposal {proposal.id} is superseded by {latest[-1].id} revision {latest[-1].revision}"
            )
        if not proposal.validation or not proposal.validation.get("ok"):
            raise DeployError("proposal was never validated; refusing to write to the robot")

    # -- deploy ----------------------------------------------------------
    def deploy(
        self,
        proposal: Proposal,
        *,
        decision: Decision,
        verdict: Verdict,
        sim: SimulationResult,
        artifacts: list[Path] | None = None,
        note: str = "",
    ) -> Deployment:
        self.preflight(proposal, verdict, decision)
        skill = self.cfg.skill(proposal.skill)
        deploy_repo_spec = self.cfg.repo("deploy")
        robot_dir = deploy_repo_spec.path / f"src/inference/robots/{skill.deploy_robot}"

        snapshot = self._snapshot(proposal, robot_dir)
        self.audit.append(
            "deploy.snapshot",
            subject=proposal.id,
            actor=decision.actor,
            data={"snapshot": snapshot.id, "files": len(snapshot.manifest)},
        )

        repo = GitRepo(deploy_repo_spec.path)
        branch = f"evolayer/{sanitise_branch_component(skill.id)}/{sanitise_branch_component(proposal.id)}"
        touched: list[str] = []
        copied: list[Path] = []
        try:
            if proposal.edits:
                repo.ensure_branch(branch)
                for edit in proposal.edits:
                    if edit.repo != "deploy":
                        raise DeployError(
                            f"edit targets repo {edit.repo!r}; only deploy artifacts may be installed by "
                            "this path (training changes are deployed by retraining, not by copying files)"
                        )
                    self._apply_edit(repo, edit.path, edit.diff)
                    touched.append(edit.path)

            copied = self._install_artifacts(proposal, sim, artifacts or [], robot_dir)
            checks = self._contract_checks(robot_dir, proposal)
            failures = [check for check in checks if check.status == "fail"]
            self.audit.append(
                "deploy.contract_checks",
                subject=proposal.id,
                actor=decision.actor,
                data={"checks": [check.to_dict() for check in checks]},
            )
            if failures:
                raise DeployError(
                    "deploy contract checks failed: "
                    + "; ".join(f"{c.name}: {c.detail}" for c in failures)
                )

            commit_sha = ""
            if touched:
                commit = repo.commit(touched, self._commit_message(proposal, decision))
                commit_sha = commit.sha
            tag = ""
            try:
                tag = repo.tag(
                    f"evolayer-{sanitise_branch_component(skill.id)}-{time.strftime('%Y%m%d-%H%M%S')}-"
                    f"{sanitise_branch_component(proposal.id)}",
                    f"EvoLayer deployment of {proposal.id} r{proposal.revision}\n\n"
                    f"actor: {decision.actor}\nreason: {decision.reason}\n"
                    f"snapshot: {snapshot.id}\ngate: {verdict.verdict}",
                )
            except GitError as exc:
                # A missing tag must not silently pass: it is the source rollback point.
                raise DeployError(f"could not create the rollback tag: {exc}") from exc

            deployment = Deployment(
                id=content_id(
                    "deployment",
                    {
                        "proposal_id": proposal.id,
                        "revision": proposal.revision,
                        "tag": tag,
                        "snapshot": snapshot.id,
                        "created_at": round(time.time(), 3),
                    },
                ),
                proposal_id=proposal.id,
                revision=proposal.revision,
                tag=tag,
                snapshot_id=snapshot.id,
                artifacts=[artifact_for(path) for path in copied if path.is_file()],
                created_at=time.time(),
                note=(note + f" commit={commit_sha[:12]}").strip(),
            )
            self.store.add_deployment(deployment)
            self.store.set_proposal_status(proposal.id, "deployed")
            self.audit.append(
                "deploy.installed",
                subject=proposal.id,
                actor=decision.actor,
                data={
                    "deployment": deployment.id,
                    "tag": tag,
                    "snapshot": snapshot.id,
                    "files": [str(path) for path in copied],
                    "commit": commit_sha,
                },
            )
            return deployment
        except (DeployError, GitError, SnapshotError, OSError) as exc:
            self._restore(snapshot, proposal, decision, str(exc))
            self.store.set_proposal_status(proposal.id, "failed")
            raise DeployError(f"deployment aborted and rolled back: {exc}") from exc

    # -- rollback --------------------------------------------------------
    def rollback(self, deployment: Deployment, *, actor: str, reason: str = "") -> Deployment:
        skill_id = deployment.artifacts[0].name if deployment.artifacts else ""
        del skill_id  # artifacts carry no skill; the snapshot is authoritative
        restored = self.snapshots.restore(deployment.snapshot_id, target_root=self.cfg.root)
        self.audit.append(
            "deploy.rollback",
            subject=deployment.id,
            actor=actor,
            data={"snapshot": deployment.snapshot_id, "files": restored, "reason": reason},
        )
        self.store.mark_rolled_back(deployment.id)
        self.store.set_proposal_status(deployment.proposal_id, "rejected")
        updated = self.store.get_deployment(deployment.id)
        return updated

    # -- internals -------------------------------------------------------
    def _snapshot(self, proposal: Proposal, robot_dir: Path) -> Snapshot:
        paths = [robot_dir / "configs", robot_dir / "models", robot_dir / "latents"]
        try:
            return self.snapshots.create(
                paths,
                root=self.cfg.root,
                meta={"proposal": proposal.id, "revision": proposal.revision, "kind": "pre-deploy"},
            )
        except SnapshotError as exc:
            raise DeployError(f"cannot create a rollback snapshot: {exc}") from exc

    def _apply_edit(self, repo: GitRepo, relpath: str, diff: str) -> None:
        if repo.is_dirty([relpath]):
            raise DeployError(f"{relpath} has uncommitted changes; refusing to stack a deployment on top")
        patch = self.cfg.state_dir / "pending.patch"
        patch.parent.mkdir(parents=True, exist_ok=True)
        patch.write_text(diff, encoding="utf-8")
        from ..execution import run

        check = run(["git", "apply", "--check", "-p1", str(patch)], cwd=str(repo.toplevel), timeout=60)
        if not check.ok:
            raise DeployError(f"patch no longer applies to {relpath}: {check.tail(10)}")
        applied = run(["git", "apply", "-p1", str(patch)], cwd=str(repo.toplevel), timeout=60)
        if not applied.ok:
            raise DeployError(f"failed to apply patch to {relpath}: {applied.tail(10)}")

    def _install_artifacts(
        self, proposal: Proposal, sim: SimulationResult, extra: list[Path], robot_dir: Path
    ) -> list[Path]:
        models_dir = robot_dir / "models"
        latents_dir = robot_dir / "latents"
        models_dir.mkdir(parents=True, exist_ok=True)
        candidates: list[Path] = list(extra)
        for artifact in sim.artifacts:
            candidates.append(Path(artifact.path))
        onnx = next((p for p in candidates if p.is_file() and p.suffix == ".onnx"), None)
        if onnx is None:
            if self._requires_new_policy(proposal):
                raise DeployError(
                    "this proposal changes the policy, but the rehearsal produced no ONNX artifact"
                )
            return []
        # Keep the name the policy YAML already references, so the config and the
        # artifact cannot drift apart; fall back to a derived name when absent.
        target = models_dir / self._policy_file_name(proposal, robot_dir)
        shutil.copy2(onnx, target)
        copied = [target]
        for suffix in (".data",):
            sidecar = onnx.with_name(onnx.name + suffix)
            if sidecar.is_file():
                destination = models_dir / f"{target.name}{suffix}"
                shutil.copy2(sidecar, destination)
                copied.append(destination)
        latents_dir.mkdir(parents=True, exist_ok=True)
        for path in candidates:
            if path.suffix in (".npz", ".npy") and path.is_file():
                destination = latents_dir / path.name
                shutil.copy2(path, destination)
                copied.append(destination)
        return copied

    def _requires_new_policy(self, proposal: Proposal) -> bool:
        return bool(proposal.train_command) or proposal.kind in ("reward_term_weight", "train_hyperparameter")

    def _policy_file_name(self, proposal: Proposal, robot_dir: Path) -> str:
        skill = self.cfg.skill(proposal.skill)
        yaml_path = robot_dir / "configs" / f"{skill.deploy_policy}.yaml"
        if yaml_path.is_file():
            try:
                import yaml as _yaml

                params = _yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["inference_node"]["ros__parameters"]
                names = params.get("model_names") or []
                if names:
                    return str(names[0])
            except (KeyError, TypeError, _yaml.YAMLError):  # type: ignore[name-defined]
                pass
        return f"policy_{sanitise_branch_component(proposal.skill)}.onnx"

    def _contract_checks(self, robot_dir: Path, proposal: Proposal):
        skill = self.cfg.skill(proposal.skill)
        urdf = self.cfg.resolve(self.cfg.simulator.backend("mujoco").get("urdf", "00_body_base/rpo_description/urdf/rpo.urdf"))
        limits = load_urdf_limits(urdf)
        yaml_path = robot_dir / "configs" / f"{skill.deploy_policy}.yaml"
        models_dir = robot_dir / "models"
        onnx = None
        for candidate in sorted(models_dir.glob("policy*.onnx")):
            onnx = candidate
        return check_policy_contract(
            urdf_joints=list(limits.joints),
            policy_yaml=yaml_path,
            models_dir=models_dir,
            latents_dir=robot_dir / "latents",
            onnx_path=onnx,
            rpo_joint_order_file=self.cfg.repo("training").path
            / "robolab/robolab/assets/robots/roboparty.py",
        )

    def _commit_message(self, proposal: Proposal, decision: Decision) -> str:
        return (
            f"evolayer({proposal.skill}): {proposal.kind}\n\n"
            f"proposal: {proposal.id} r{proposal.revision}\n"
            f"finding:  {proposal.finding_id}\n"
            f"risk:     {proposal.risk.risk_class} ({proposal.risk.score:.2f})\n"
            f"approved: {decision.actor} — {decision.reason}\n"
        )

    def _restore(self, snapshot: Snapshot, proposal: Proposal, decision: Decision, reason: str) -> None:
        try:
            restored = self.snapshots.restore(snapshot.id, target_root=self.cfg.root)
        except SnapshotError as exc:  # pragma: no cover - verification failure is catastrophic
            self.audit.append(
                "deploy.rollback_failed",
                subject=proposal.id,
                actor=decision.actor,
                data={"snapshot": snapshot.id, "error": str(exc)},
            )
            return
        self.audit.append(
            "deploy.automatic_rollback",
            subject=proposal.id,
            actor=decision.actor,
            data={"snapshot": snapshot.id, "restored": restored, "reason": reason},
        )
