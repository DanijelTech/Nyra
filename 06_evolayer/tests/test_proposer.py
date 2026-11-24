"""Proposer: file-derived diffs, validation gates, risk scoring."""

from __future__ import annotations

import pytest

from evo_layer.models import Edit, Finding, Proposal, Risk, SimPlan
from evo_layer.proposer.generator import rehearsal_plan, score_risk
from evo_layer.proposer.templates import Context, TemplateUnavailable, build
from evo_layer.proposer.validate import Validator


def finding(term: str = "ang_vel_xy_l2", *, severity: str = "medium", skill: str = "locomotion",
            recommendation: str = "reward_term_weight", params: dict | None = None) -> Finding:
    return Finding(
        id="fnd_test000001", skill=skill, kind="regression", severity=severity, title="t",
        summary="s", metric="fall_rate", baseline=0.01, current=0.2, delta_rel=19.0, p_value=0.001,
        recommendation=recommendation,
        params=params or {"term": term, "direction": "increase_penalty"},
        created_at=1.0,
    )


# --------------------------------------------------------------------------- templates


def test_reward_template_edits_the_matching_env_config(real_cfg):
    result = build("reward_term_weight", Context(cfg=real_cfg, finding=finding()))
    edit = result.edits[0]
    assert edit.repo == "training"
    assert edit.path.endswith(f"{'/amp/' if False else 'amp_env_cfg.py'}")
    assert "ang_vel_xy_l2" in edit.diff
    # The diff must be a real patch of the file that exists on disk.
    assert edit.diff.startswith("--- a/")
    assert result.sim_params["old_weight"] != result.sim_params["new_weight"]


def test_reward_template_refuses_an_unknown_term(real_cfg):
    with pytest.raises(TemplateUnavailable):
        build("reward_term_weight", Context(cfg=real_cfg, finding=finding(term="not_a_term")))


def test_reward_template_refuses_an_unlisted_file(real_cfg):
    # A finding recommending an edit outside the allowlist must fail, not guess.
    ctx = Context(cfg=real_cfg, finding=finding())
    with pytest.raises(TemplateUnavailable):
        ctx.read(real_cfg.repo("training"), "setup.py")


def test_hyperparameter_template_emits_an_allowlisted_command(real_cfg):
    result = build(
        "train_hyperparameter",
        Context(cfg=real_cfg, finding=finding(recommendation="train_hyperparameter",
                                             params={"flag": "--max_iterations", "scale": 1.5})),
    )
    assert not result.edits, "a CLI override must not modify files"
    assert result.train_command[1].endswith("train.py")
    assert "--max_iterations=750" in result.train_command


def test_hyperparameter_template_rejects_unknown_flags(real_cfg):
    with pytest.raises(TemplateUnavailable):
        build(
            "train_hyperparameter",
            Context(cfg=real_cfg, finding=finding(recommendation="train_hyperparameter",
                                                  params={"flag": "--evil", "scale": 2})),
        )


def test_eval_suite_template_edits_our_own_config(workspace):
    cfg = workspace["cfg"]
    result = build(
        "eval_suite_expand",
        Context(cfg=cfg, finding=finding(recommendation="eval_suite_expand", params={})),
    )
    edit = result.edits[0]
    assert edit.repo == "evolayer"
    assert edit.path == "06_evolayer/config/skills/locomotion.yaml"
    assert "episodes: 12" in edit.diff


def test_deploy_template_only_touches_safe_bands(real_cfg):
    ok = build(
        "deploy_config_tune",
        Context(cfg=real_cfg, finding=finding(recommendation="deploy_config_tune",
                                              params={"key": "action_scale", "factor": 1.2})),
    )
    assert ok.edits[0].path == "src/inference/robots/rpo/configs/amp.yaml"
    with pytest.raises(TemplateUnavailable):
        build(
            "deploy_config_tune",
            Context(cfg=real_cfg, finding=finding(recommendation="deploy_config_tune",
                                                  params={"key": "action_scale", "factor": 9.0})),
        )
    with pytest.raises(TemplateUnavailable):
        build(
            "deploy_config_tune",
            Context(cfg=real_cfg, finding=finding(recommendation="deploy_config_tune",
                                                  params={"key": "kp", "factor": 1.2})),
        )


# --------------------------------------------------------------------------- validation


def _proposal_with(edit: Edit, *, command: list[str] | None = None) -> Proposal:
    return Proposal(
        id="prp_test000001", revision=1, finding_id="fnd_test000001", skill="locomotion",
        kind="reward_term_weight", title="t", rationale="r", edits=[edit],
        sim_plan=SimPlan(backend="mujoco", episodes=5), expected_effect="e",
        risk=Risk(score=0.2, risk_class="low"), train_command=command or [],
    )


def test_validator_accepts_a_real_template_diff(real_cfg):
    result = build("reward_term_weight", Context(cfg=real_cfg, finding=finding()))
    report = Validator(real_cfg).validate(_proposal_with(result.edits[0]))
    assert report.ok, [c.detail for c in report.failures]
    assert "training:robolab/robolab/tasks/manager_based/amp/amp_env_cfg.py" in report.applied


def test_validator_rejects_path_outside_the_allowlist(real_cfg):
    bad = Edit(repo="training", path="README.md", diff="--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-a\n+b\n")
    report = Validator(real_cfg).validate(_proposal_with(bad))
    assert not report.ok
    assert any("allowlist" in check.name for check in report.failures)


def test_validator_rejects_a_patch_that_does_not_apply(real_cfg):
    stale = Edit(
        repo="training",
        path="robolab/robolab/tasks/manager_based/amp/amp_env_cfg.py",
        diff="--- a/robolab/robolab/tasks/manager_based/amp/amp_env_cfg.py\n"
        "+++ b/robolab/robolab/tasks/manager_based/amp/amp_env_cfg.py\n"
        "@@ -1,3 +1,3 @@\n-nonexistent line\n+replacement\n context\n context\n",
    )
    report = Validator(real_cfg).validate(_proposal_with(stale))
    assert not report.ok
    assert any("git apply" in check.detail for check in report.failures)


def test_validator_rejects_dangerous_added_code(real_cfg):
    path = "robolab/robolab/tasks/manager_based/amp/amp_env_cfg.py"
    payload = (
        f"--- a/{path}\n+++ b/{path}\n@@ -1,3 +1,4 @@\n"
        "-# Copyright\n+import subprocess\n+# Copyright\n context\n context\n"
    )
    report = Validator(real_cfg).validate(_proposal_with(Edit(repo="training", path=path, diff=payload)))
    assert not report.ok
    assert any("forbidden primitive" in check.detail for check in report.failures)


def test_validator_rejects_oversized_patches(real_cfg):
    path = "robolab/robolab/tasks/manager_based/amp/amp_env_cfg.py"
    body = "".join(f"+line {i}\n" for i in range(400))
    diff = f"--- a/{path}\n+++ b/{path}\n@@ -1,1 +1,401 @@\n context\n{body}"
    report = Validator(real_cfg).validate(_proposal_with(Edit(repo="training", path=path, diff=diff)))
    assert not report.ok
    assert any("size" in check.name for check in report.failures)


def test_validator_rejects_unknown_training_flags(real_cfg):
    result = build(
        "train_hyperparameter",
        Context(cfg=real_cfg, finding=finding(recommendation="train_hyperparameter",
                                              params={"flag": "--max_iterations", "scale": 1.5})),
    )
    proposal = _proposal_with(result.edits[0] if result.edits else Edit(
        repo="training", path="robolab/a.py", diff="--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"),
        command=[".venv/bin/python", "robolab/scripts/rsl_rl/train.py", "--task=RPO-AMP", "--evil=1"])
    report = Validator(real_cfg).validate(proposal)
    assert not report.ok
    assert any("not allowed" in check.detail for check in report.failures)


# --------------------------------------------------------------------------- risk


def test_risk_grows_with_target_and_severity(real_cfg):
    low = score_risk(real_cfg, finding(severity="info"), "eval_suite_expand", 1, 2)
    high = score_risk(real_cfg, finding(severity="critical"), "deploy_config_tune", 1, 40)
    assert low.risk_class == "low"
    assert high.risk_class in ("medium", "high")
    assert high.score > low.score
    assert set(high.factors) >= {"target_repo", "diff_size", "kind", "severity"}


def test_rehearsal_plan_scales_with_risk(real_cfg):
    low = rehearsal_plan(real_cfg, "locomotion", Risk(score=0.1, risk_class="low"))
    high = rehearsal_plan(real_cfg, "locomotion", Risk(score=0.9, risk_class="high"))
    assert high.episodes > low.episodes
    plan = high.to_sim_plan("mujoco", 600)
    assert plan.backend == "mujoco"
    assert plan.params["commands"]


def test_proposer_records_validation_and_revision(workspace):
    """Revisions come from the store, so they survive a restart and never collide."""
    from evo_layer.orchestrator.loop import EvoLoop

    cfg = workspace["cfg"]
    loop = EvoLoop.from_config(cfg, secret="unit-test-secret-value-0123456789", env={})
    loop.store.add_finding(finding())
    proposal, report = loop.propose(loop.store.get_finding("fnd_test000001"))
    assert report["ok"]
    assert proposal.revision == 1
    assert proposal.risk.risk_class in ("low", "medium", "high")
    second, _ = loop.propose(loop.store.get_finding("fnd_test000001"))
    assert second.revision == 2
    assert loop.store.get_proposal(second.id).revision == 2
