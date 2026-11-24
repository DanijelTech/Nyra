"""Proposal generation and risk scoring.

Risk is a transparent linear model, not a model of a model:

    score = target_weight(repo)
          + 0.05 * files_touched
          + min(changed_lines / 200, 0.20)
          + kind_weight(proposal kind)
          + severity_weight(finding severity)
          + 0.10 if the change reaches the deployed controller
    clamped to [0, 1]; classes low < 0.35 <= medium < 0.65 <= high.

The weights encode policy, not physics: touching the deploy tree is worse than
touching our own config, and a critical finding justifies a bigger change. Each
factor is stored with the score so a reviewer can challenge the arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import EvoConfig
from ..ids import content_id
from ..models import Finding, Proposal, Risk, SimPlan
from ..store import Store
from .templates import Context, TemplateUnavailable, build
from .validate import Validator

TARGET_WEIGHTS = {"training": 0.25, "evolayer": 0.15, "deploy": 0.55}
KIND_WEIGHTS = {
    "reward_term_weight": 0.05,
    "train_hyperparameter": 0.03,
    "eval_suite_expand": 0.0,
    "deploy_config_tune": 0.20,
}
SEVERITY_WEIGHTS = {"info": 0.0, "low": 0.02, "medium": 0.06, "high": 0.12, "critical": 0.20}

LOW_THRESHOLD = 0.35
HIGH_THRESHOLD = 0.65


@dataclass
class ProposalPlan:
    """How much rehearsal a proposal needs, derived from its risk class."""

    episodes: int
    seeds: list[int]
    duration_s: float
    commands: list[list[float]]
    init_poses: list[str]

    def to_sim_plan(self, backend: str, timeout_s: int) -> SimPlan:
        return SimPlan(
            backend=backend,
            episodes=self.episodes,
            seeds=self.seeds,
            timeout_s=timeout_s,
            params={
                "duration_s": self.duration_s,
                "commands": self.commands,
                "init_poses": self.init_poses,
                "video": True,
            },
        )


def rehearsal_plan(cfg: EvoConfig, skill_id: str, risk: Risk) -> ProposalPlan:
    """Bigger risk buys more evidence. Numbers come from the skill's eval suite."""
    suite = cfg.skill(skill_id).eval_suite or {}
    base_episodes = int(suite.get("episodes") or 60)
    seeds = [int(s) for s in (suite.get("seeds") or [0])]
    factor = {"low": 0.5, "medium": 1.0, "high": 1.5}.get(risk.risk_class, 1.0)
    episodes = max(10, int(round(base_episodes * factor)))
    commands = suite.get("commands") or [[0.6, 0.0, 0.0], [0.0, 0.0, 0.6]]
    poses = suite.get("init_poses") or ["stand"]
    return ProposalPlan(
        episodes=episodes,
        seeds=seeds,
        duration_s=float(suite.get("duration_s") or 8.0),
        commands=[[float(v) for v in cmd] for cmd in commands],
        init_poses=[str(p) for p in poses],
    )


def score_risk(cfg: EvoConfig, finding: Finding, kind: str, edits_count: int, changed_lines: int) -> Risk:
    factors: dict[str, float] = {}
    skill = cfg.skill(finding.skill)
    target = TARGET_WEIGHTS.get(skill.repo, 0.30)
    factors["target_repo"] = target
    factors["files_touched"] = 0.05 * max(0, edits_count - 1)
    factors["diff_size"] = min(changed_lines / 200.0, 0.20)
    factors["kind"] = KIND_WEIGHTS.get(kind, 0.10)
    factors["severity"] = SEVERITY_WEIGHTS.get(finding.severity, 0.05)
    reaches_robot = any(
        edit_repo == "deploy" for edit_repo in [skill.repo] if kind == "deploy_config_tune"
    )
    factors["reaches_controller"] = 0.10 if reaches_robot else 0.0
    score = max(0.0, min(1.0, sum(factors.values())))
    if score < LOW_THRESHOLD:
        risk_class = "low"
    elif score < HIGH_THRESHOLD:
        risk_class = "medium"
    else:
        risk_class = "high"
    return Risk(score=round(score, 4), risk_class=risk_class, factors={k: round(v, 4) for k, v in factors.items()})


class Proposer:
    def __init__(self, cfg: EvoConfig, store: Store):
        self.cfg = cfg
        self.store = store
        self.validator = Validator(cfg)

    def propose(self, finding: Finding, *, template: str | None = None) -> Proposal:
        template_id = template or finding.recommendation
        if not template_id:
            raise TemplateUnavailable("finding carries no recommendation template")
        skill = self.cfg.skill(finding.skill)
        result = build(template_id, Context(cfg=self.cfg, finding=finding))

        revision = self.store.next_revision(finding.id)
        changed_lines = sum(
            edit.added_lines + edit.removed_lines for edit in result.edits
        )
        risk = score_risk(self.cfg, finding, result.kind, len(result.edits), changed_lines)
        plan = rehearsal_plan(self.cfg, finding.skill, risk)
        sim_plan = plan.to_sim_plan(skill.backend, self.cfg.simulator.timeout_s)

        proposal_id = content_id(
            "proposal",
            {
                "finding_id": finding.id,
                "revision": revision,
                "template": template_id,
                "edits": [e.to_dict() for e in result.edits],
                "overlay": result.config_overlay,
                "train": result.train_command,
            },
        )
        return Proposal(
            id=proposal_id,
            revision=revision,
            finding_id=finding.id,
            skill=finding.skill,
            kind=result.kind,
            title=f"{finding.title} → {result.kind}",
            rationale=result.rationale or finding.summary,
            edits=result.edits,
            sim_plan=sim_plan,
            expected_effect=result.expected_effect,
            risk=risk,
            config_overlay=result.config_overlay,
            train_command=result.train_command,
            created_by="template",
            status="proposed",
        )

    def propose_and_validate(self, finding: Finding, *, template: str | None = None) -> tuple[Proposal, Any]:
        proposal = self.propose(finding, template=template)
        report = self.validator.validate(proposal)
        return proposal, report
