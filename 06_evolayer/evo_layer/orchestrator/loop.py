"""The closed loop, as a resumable state machine.

    observe → critique → propose → validate → simulate → gate → await Hermes
                                                                  ↓
                                                        approve → deploy → tag/snapshot
                                                        reject  → closed
                                                        modify  → new revision, back to propose

Every step is idempotent and persisted, so ``tick`` can be run by cron, resumed
after a crash, or driven step by step from the CLI. The loop stops at
``awaiting_hermes`` and **cannot** proceed without a decision: there is no
auto-approve path in this module, by design.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..audit import AuditLog
from ..config import EvoConfig
from ..critic.analyze import Critic
from ..critic.baseline import InsufficientData, compute_baseline
from ..deploy.installer import DeployError, Installer
from ..hermes_interface.bridge import BridgeError, HermesBridge
from ..hermes_interface.interface import HermesError, HermesInterface, Report
from ..models import Decision, Deployment, Finding, Proposal, SimulationResult, Verdict
from ..observer.ingest import discover_inputs, parse_csv, parse_jsonl
from ..proposer.generator import Proposer
from ..proposer.templates import TemplateUnavailable
from ..safety.gate import GateInput, SafetyGate
from ..simulator.runner import Runner, RunnerOptions
from ..store import Store
from ..versioning.snapshot import Snapshotter


@dataclass
class TickResult:
    skill: str
    ingested: int = 0
    findings: list[str] = field(default_factory=list)
    proposals: list[str] = field(default_factory=list)
    simulated: list[str] = field(default_factory=list)
    gated: list[str] = field(default_factory=list)
    awaiting: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "ingested": self.ingested,
            "findings": self.findings,
            "proposals": self.proposals,
            "simulated": self.simulated,
            "gated": self.gated,
            "awaiting_hermes": self.awaiting,
            "errors": self.errors,
        }


class EvoLoop:
    """Owns the wiring; every collaborator is injectable for tests."""

    def __init__(
        self,
        cfg: EvoConfig,
        store: Store,
        audit: AuditLog,
        *,
        critic: Critic | None = None,
        proposer: Proposer | None = None,
        runner: Runner | None = None,
        gate: SafetyGate | None = None,
        hermes: HermesInterface | None = None,
        installer: Installer | None = None,
        snapshots: Snapshotter | None = None,
        bridge: HermesBridge | None = None,
        secret: str | None = None,
        env: dict[str, str] | None = None,
    ):
        self.cfg = cfg
        self.store = store
        self.audit = audit
        self.snapshots = snapshots or Snapshotter(cfg.snapshots_dir)
        self.critic = critic or Critic(cfg, store)
        self.proposer = proposer or Proposer(cfg, store)
        self.runner = runner or Runner(cfg)
        self.gate = gate or SafetyGate()
        self.hermes = hermes or HermesInterface(
            cfg, store, audit, self.snapshots, secret=secret, env=env
        )
        self.installer = installer or Installer(cfg, store, audit, self.snapshots)
        self.bridge = bridge if bridge is not None else HermesBridge(cfg, audit, env=env)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_config(
        cls,
        cfg: EvoConfig,
        *,
        store: Store | None = None,
        audit: AuditLog | None = None,
        secret: str | None = None,
        env: dict[str, str] | None = None,
    ) -> "EvoLoop":
        cfg.ensure_dirs()
        store = store or Store(cfg.store_path)
        audit = audit or AuditLog(cfg.audit_path)
        loop = cls(cfg, store, audit, secret=secret, env=env)
        for skill in cfg.skills.values():
            loop.store.upsert_skill(
                skill.id,
                skill.title,
                skill.backend,
                {
                    "deploy_policy": skill.deploy_policy,
                    "train_task": skill.train_task,
                    "eval_suite": skill.eval_suite,
                    "limits": dataclasses.asdict(skill.limits),
                },
            )
        return loop

    # -- 1. observe ------------------------------------------------------
    def observe(self, skill: str, inputs: list[str] | str | Path, *, source: str | None = None) -> dict[str, Any]:
        self.cfg.skill(skill)
        raw = [inputs] if isinstance(inputs, (str, Path)) else list(inputs)
        paths = discover_inputs(raw)
        accepted = 0
        rejected = 0
        reasons: list[str] = []
        for path in paths:
            if path.suffix.lower() == ".csv":
                report = parse_csv(path, skill=skill)
            else:
                report, _ = parse_jsonl(path, skill=skill, source=source or "real")
            accepted += self.store.add_episodes(report.accepted)
            rejected += len(report.rejected)
            reasons.extend(f"{path.name}:{r.line_no}:{r.reason}" for r in report.rejected[:20])
        total = self.store.episode_count(skill)
        self.audit.append(
            "observer.ingested",
            subject=skill,
            actor="observer",
            data={
                "files": [str(p) for p in paths],
                "accepted": accepted,
                "rejected": rejected,
                "total": total,
            },
        )
        return {
            "skill": skill,
            "files": len(paths),
            "accepted": accepted,
            "rejected": rejected,
            "reasons": reasons,
            "episodes_total": total,
        }

    # -- 2. critique -----------------------------------------------------
    def critique(self, skill: str, *, now: float | None = None) -> list[Finding]:
        findings = self.critic.analyze(skill, now=now)
        self.audit.append(
            "critic.findings",
            subject=skill,
            actor="critic",
            data={"findings": [{"id": f.id, "kind": f.kind, "severity": f.severity} for f in findings]},
        )
        return findings

    # -- 3. propose ------------------------------------------------------
    def propose(self, finding: Finding, *, template: str | None = None) -> tuple[Proposal, dict[str, Any]]:
        proposal, report = self.proposer.propose_and_validate(finding, template=template)
        proposal.validation = report.to_dict()
        proposal.status = "validated" if report.ok else "failed"
        self.store.add_proposal(proposal)
        self.audit.append(
            "proposer.created",
            subject=proposal.id,
            actor="proposer",
            data={
                "finding": finding.id,
                "kind": proposal.kind,
                "risk": proposal.risk.to_dict(),
                "validation_ok": report.ok,
                "failures": [c.to_dict() for c in report.failures],
            },
        )
        return proposal, report.to_dict()

    def propose_for_finding(
        self, finding_id: str, *, template: str | None = None
    ) -> tuple[Proposal, dict[str, Any]]:
        return self.propose(self.store.get_finding(finding_id), template=template)

    # -- 4. simulate -----------------------------------------------------
    def simulate(
        self,
        proposal_id: str,
        *,
        backend: str | None = None,
        dry_run: bool = False,
        allow_replay: bool = False,
        policy_onnx: str | None = None,
    ) -> SimulationResult:
        proposal = self.store.get_proposal(proposal_id)
        if proposal.status not in ("validated", "simulated", "gated", "failed"):
            raise HermesError(
                f"proposal {proposal.id} is {proposal.status!r}; it must be validated before simulation"
            )
        if not proposal.validation or not proposal.validation.get("ok"):
            raise HermesError("refusing to simulate a proposal that failed validation")
        applied = dict((proposal.validation or {}).get("applied") or {})
        if policy_onnx:
            proposal.sim_plan.params["policy_onnx"] = policy_onnx
        result = self.runner.run(
            proposal,
            applied=applied,
            options=RunnerOptions(backend=backend, dry_run=dry_run, allow_replay=allow_replay),
        )
        self.store.add_simulation(result)
        if not dry_run:
            self.store.set_proposal_status(proposal.id, "simulated" if result.status == "ok" else "failed")
        self.audit.append(
            "simulator.finished",
            subject=proposal.id,
            actor="simulator",
            data={
                "backend": result.backend,
                "status": result.status,
                "episodes": result.episodes_n,
                "metrics": result.metrics.to_dict() if result.metrics else None,
                "error": result.error[:400],
            },
        )
        return result

    # -- 5. gate ---------------------------------------------------------
    def gate_proposal(self, proposal_id: str) -> Verdict:
        proposal = self.store.get_proposal(proposal_id)
        sim = self.store.latest_simulation(proposal.id, proposal.revision)
        if sim is None:
            raise HermesError("no simulation on record; run `simulate` first")
        skill = self.cfg.skill(proposal.skill)
        limits = skill.limits
        baseline = compute_baseline(self.store, proposal.skill, min_episodes=limits.min_episodes)
        robot = self._robot_limits()
        if sim.metrics is None:
            verdict = Verdict(
                proposal_id=proposal.id,
                revision=proposal.revision,
                verdict="reject",
                checks=[
                    {
                        "name": "metrics",
                        "status": "reject",
                        "measured": None,
                        "limit": None,
                        "detail": f"no measurable episodes: {sim.error[:200]}",
                    }
                ],
                stats={"sim_status": sim.status},
                created_at=time.time(),
            )
        else:
            verdict = self.gate.evaluate(
                GateInput(
                    metrics=sim.metrics,
                    baseline=baseline.metrics,
                    limits=limits,
                    robot=robot,
                    sim_status=sim.status,
                    required_episodes=proposal.sim_plan.episodes,
                    extra={"proposal_id": proposal.id, "revision": proposal.revision},
                )
            )
        self.store.add_verdict(verdict)
        self.store.set_proposal_status(proposal.id, "gated" if verdict.verdict != "reject" else "failed")
        self.audit.append(
            "gate.verdict",
            subject=proposal.id,
            actor="gate",
            data={"verdict": verdict.verdict, "checks": verdict.stats.get("checks", {})},
        )
        return verdict

    # -- 6. request approval ---------------------------------------------
    def request_approval(
        self, proposal_id: str, *, caveats: list[str] | None = None, notify: bool = True
    ) -> Report:
        proposal = self.store.get_proposal(proposal_id)
        sim = self.store.latest_simulation(proposal.id, proposal.revision)
        verdict = self.store.latest_verdict(proposal.id, proposal.revision)
        if verdict is None:
            raise HermesError("gate has not produced a verdict; approvals cannot be requested")
        if verdict.verdict == "reject":
            raise HermesError("the gate rejected this proposal; there is nothing to approve")
        finding = self.store.get_finding(proposal.finding_id)
        try:
            baseline = compute_baseline(
                self.store,
                proposal.skill,
                min_episodes=self.cfg.skill(proposal.skill).limits.min_episodes,
            ).metrics
        except InsufficientData:
            baseline = None
        report = self.hermes.build_report(
            proposal, finding, sim, verdict, baseline, extra_caveats=caveats
        )
        path = self.hermes.write_report(report)
        self.store.set_proposal_status(proposal.id, "awaiting_hermes")
        delivery: dict[str, Any] | None = None
        if notify:
            # The fork: publish the review request to the local Hermes (webhook) or
            # leave it for polling. A delivery failure is recorded, never fatal.
            request = self.bridge.review_request(
                proposal=proposal, report=report, verdict=verdict, sim=sim, baseline=baseline
            )
            try:
                delivery = self.bridge.notify(request).to_dict()
            except BridgeError as exc:
                delivery = {"ok": False, "detail": f"bridge error: {exc}", "mode": "unknown", "attempts": 0}
                self.audit.append(
                    "bridge.error",
                    subject=proposal.id,
                    actor="evolayer",
                    data={"error": str(exc)},
                )
        report.delivery = delivery
        self.audit.append(
            "hermes.review_requested",
            subject=proposal.id,
            actor="evolayer",
            data={"report": str(path), "gate": verdict.verdict, "delivery": delivery},
        )
        return report

    # -- 7. decide -------------------------------------------------------
    def decide(
        self,
        proposal_id: str,
        *,
        verdict: str,
        actor: str,
        reason: str,
        token: str | None = None,
        modifications: dict[str, Any] | None = None,
        require_token: bool | None = None,
    ) -> dict[str, Any]:
        proposal = self.store.get_proposal(proposal_id)
        if proposal.status not in ("awaiting_hermes", "gated"):
            raise HermesError(
                f"proposal {proposal.id} is {proposal.status!r}; only a rehearsed proposal can be decided"
            )
        gate = self.store.latest_verdict(proposal.id, proposal.revision)
        sim = self.store.latest_simulation(proposal.id, proposal.revision)
        decision: Decision = self.hermes.decide(
            proposal,
            verdict=verdict,
            actor=actor,
            reason=reason,
            token=token,
            gate=gate,
            sim=sim,
            modifications=modifications,
            require_token=require_token,
        )
        result: dict[str, Any] = {"decision": decision.to_dict(), "proposal_id": proposal.id}
        if verdict == "approve":
            self.store.set_proposal_status(proposal.id, "approved")
            self.store.set_finding_status(proposal.finding_id, "approved")
        elif verdict == "reject":
            self.store.set_proposal_status(proposal.id, "rejected")
            self.store.set_finding_status(proposal.finding_id, "rejected")
        else:
            self.store.set_proposal_status(proposal.id, "superseded")
            finding = self.store.get_finding(proposal.finding_id)
            merged = merge_params(finding, (modifications or {}).get("template_params", {}))
            new_proposal, validation = self.proposer.propose_and_validate(merged)
            new_proposal.supersedes = proposal.id
            new_proposal.validation = validation.to_dict()
            new_proposal.status = "validated" if validation.ok else "failed"
            self.store.add_proposal(new_proposal)
            self.audit.append(
                "proposer.revised",
                subject=new_proposal.id,
                actor=actor,
                data={
                    "supersedes": proposal.id,
                    "reason": reason,
                    "params": merged.params,
                },
            )
            result["new_proposal"] = {"id": new_proposal.id, "revision": new_proposal.revision}
        return result

    # -- 8. deploy / rollback --------------------------------------------
    def deploy(self, proposal_id: str, *, actor: str | None = None) -> Deployment:
        proposal = self.store.get_proposal(proposal_id)
        decision = self.store.get_decision(proposal.id, proposal.revision)
        verdict = self.store.latest_verdict(proposal.id, proposal.revision)
        sim = self.store.latest_simulation(proposal.id, proposal.revision)
        if sim is None:
            raise DeployError("no simulation on record")
        if actor is not None and decision is not None and decision.actor != actor:
            raise DeployError(f"deployment actor {actor!r} does not match the approving actor")
        if decision is None or verdict is None:
            raise DeployError("no decision/verdict on record")
        return self.installer.deploy(proposal, decision=decision, verdict=verdict, sim=sim)

    def rollback(self, deployment_id: str, *, actor: str, reason: str = "") -> Deployment:
        deployment = self.store.get_deployment(deployment_id)
        return self.installer.rollback(deployment, actor=actor, reason=reason)

    # -- tick ------------------------------------------------------------
    def tick(self, skill: str | None = None, *, now: float | None = None) -> list[TickResult]:
        """Advance every skill as far as possible without a human."""
        skills = [skill] if skill else list(self.cfg.skills)
        results: list[TickResult] = []
        for skill_id in skills:
            result = TickResult(skill=skill_id)
            spec = self.cfg.skill(skill_id)
            stored = self.store.episode_count(skill_id)
            if stored < spec.limits.min_episodes:
                result.errors.append(f"only {stored} episodes; need {spec.limits.min_episodes}")
                results.append(result)
                continue
            try:
                findings = self.critique(skill_id, now=now)
            except InsufficientData as exc:
                result.errors.append(str(exc))
                results.append(result)
                continue
            result.findings = [f.id for f in findings]
            for finding in findings:
                if self.store.proposals_for_finding(finding.id):
                    continue
                try:
                    proposal, validation = self.propose(finding)
                except TemplateUnavailable as exc:
                    result.errors.append(f"{finding.id}: {exc}")
                    continue
                result.proposals.append(proposal.id)
                if not validation["ok"]:
                    result.errors.append(
                        f"{proposal.id}: validation failed: "
                        + "; ".join(c["detail"] for c in validation["checks"] if not c["ok"])
                    )
                    continue
                self.simulate(proposal.id)
                result.simulated.append(proposal.id)
                verdict = self.gate_proposal(proposal.id)
                result.gated.append(f"{proposal.id}:{verdict.verdict}")
                if verdict.verdict != "reject":
                    self.request_approval(proposal.id)
                    result.awaiting.append(proposal.id)
            results.append(result)
        return results

    # -- status ----------------------------------------------------------
    def status(self, skill: str | None = None) -> dict[str, Any]:
        skills = [skill] if skill else list(self.cfg.skills)
        out: dict[str, Any] = {
            "skills": {},
            "deployments": [d.to_dict() for d in self.store.deployments(limit=5)],
        }
        for skill_id in skills:
            proposals = self.store.proposals(skill=skill_id)
            rows = []
            for proposal in proposals:
                verdict = self.store.latest_verdict(proposal.id, proposal.revision)
                sim = self.store.latest_simulation(proposal.id, proposal.revision)
                rows.append(
                    {
                        "id": proposal.id,
                        "revision": proposal.revision,
                        "status": proposal.status,
                        "kind": proposal.kind,
                        "risk": proposal.risk.risk_class,
                        "gate": verdict.verdict if verdict else None,
                        "sim": sim.status if sim else None,
                    }
                )
            out["skills"][skill_id] = {
                "episodes": self.store.episode_count(skill_id),
                "episodes_required": self.cfg.skill(skill_id).limits.min_episodes,
                "findings_open": len(self.store.findings(skill=skill_id, status="open")),
                "proposals": rows,
            }
        return out

    # -- helpers ---------------------------------------------------------
    def _robot_limits(self):
        from ..safety.limits import load_urdf_limits

        urdf = self.cfg.resolve(
            self.cfg.simulator.backend("mujoco").get(
                "urdf", "00_body_base/rpo_description/urdf/rpo.urdf"
            )
        )
        return load_urdf_limits(urdf)


def merge_params(finding: Finding, overrides: dict[str, Any] | None) -> Finding:
    """Copy of ``finding`` with template params overridden (used by ``modify``)."""
    if not overrides:
        return finding
    params = dict(finding.params or {})
    params.update(overrides)
    return dataclasses.replace(finding, params=params)
