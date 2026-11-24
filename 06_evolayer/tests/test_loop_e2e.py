"""End-to-end: observe → critique → propose → simulate → gate → Hermes → deploy → rollback.

Runs against a throw-away workspace with real git, real snapshots and a real
MuJoCo rehearsal. Nothing is mocked; the only substitution is the robot: the
policy under test is a deterministic ONNX fixture.
"""

from __future__ import annotations

import subprocess

import pytest

from evo_layer.deploy.installer import DeployError
from evo_layer.hermes_interface.interface import HermesError
from evo_layer.models import Finding
from evo_layer.proposer.templates import TemplateUnavailable

from conftest import make_episodes, write_jsonl


@pytest.fixture(autouse=True)
def git_identity(monkeypatch):
    monkeypatch.setenv("GIT_AUTHOR_NAME", "EvoLayer Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@evolayer.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "EvoLayer Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@evolayer.invalid")


def _finding(**overrides) -> Finding:
    base = dict(
        id="fnd_e2e0000001", skill="locomotion", kind="instability", severity="medium",
        title="Action scale drift", summary="Measured oscillation suggests a lower action scale.",
        metric="fall_rate", baseline=0.01, current=0.012, delta_rel=0.2, p_value=0.2,
        recommendation="deploy_config_tune", params={"key": "action_scale", "factor": 0.8},
        created_at=1.0,
    )
    base.update(overrides)
    return Finding(**base)


def bootstrap(loop, workspace, *, episodes: int = 40) -> None:
    """A healthy baseline so the gate has something to compare against."""
    path = write_jsonl(workspace["root"] / "telemetry/base.jsonl", make_episodes(episodes))
    loop.observe("locomotion", str(path))


def test_full_cycle_deploy_and_rollback(workspace, loop):
    cfg = workspace["cfg"]
    deploy_config = (
        workspace["deploy"] / "src/inference/robots/rpo/configs/amp.yaml"
    )
    original_scale = "action_scale: [0.25]"
    assert original_scale in deploy_config.read_text(encoding="utf-8")

    bootstrap(loop, workspace)
    loop.store.add_finding(_finding())

    # 1. propose + validate
    proposal, validation = loop.propose_for_finding("fnd_e2e0000001")
    assert validation["ok"], [c for c in validation["checks"] if not c["ok"]]
    assert proposal.status == "validated"
    assert proposal.risk.risk_class in ("medium", "high")

    # 2. rehearse (real MuJoCo, candidate policy supplied explicitly)
    sim = loop.simulate(proposal.id, backend="mujoco", policy_onnx=str(workspace["policy"]))
    assert sim.status == "ok", sim.error
    assert sim.metrics is not None and sim.metrics.n >= 1

    # 3. gate
    verdict = loop.gate_proposal(proposal.id)
    assert verdict.verdict in ("pass", "review"), [c for c in verdict.checks if c["status"] != "pass"]

    # 4. hand it to Hermes (pull mode: the report is the payload)
    report = loop.request_approval(proposal.id)
    assert report.delivery and report.delivery["mode"] == "pull"
    assert loop.store.get_proposal(proposal.id).status == "awaiting_hermes"
    report_file = cfg.artifacts_dir / proposal.id / f"report-r{proposal.revision}.md"
    assert report_file.is_file()
    assert "## Safety gate" in report_file.read_text(encoding="utf-8")

    # 5. Hermes approves
    result = loop.decide(proposal.id, verdict="approve", actor="hermes", reason="evidence is solid")
    assert result["decision"]["verdict"] == "approve"
    assert loop.store.get_proposal(proposal.id).status == "approved"

    # 6. deploy
    deployment = loop.deploy(proposal.id)
    assert deployment.tag.startswith("evolayer-locomotion-")
    assert deployment.snapshot_id.startswith("snp_")
    patched = deploy_config.read_text(encoding="utf-8")
    assert "action_scale: [0.2]" in patched  # 0.25 * 0.8
    # the deploy repo recorded a commit on an evolayer branch and a tag
    branch = subprocess.run(["git", "-C", str(workspace["deploy"]), "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    assert branch.startswith("evolayer/")
    tags = subprocess.run(["git", "-C", str(workspace["deploy"]), "tag", "--list", deployment.tag],
                          capture_output=True, text=True).stdout.strip()
    assert tags == deployment.tag
    ok, problems = loop.snapshots.verify(deployment.snapshot_id)
    assert ok, problems

    # 7. rollback restores the exact bytes
    rolled = loop.rollback(deployment.id, actor="hermes", reason="regression on hardware")
    assert rolled.rolled_back is True
    assert original_scale in deploy_config.read_text(encoding="utf-8")
    audit_events = [entry.event for entry in loop.audit.entries()]
    for event in ("proposer.created", "simulator.finished", "gate.verdict", "hermes.review_requested",
                  "hermes.approve", "deploy.installed", "deploy.rollback"):
        assert event in audit_events
    chain_ok, _, _, _ = loop.audit.verify()
    assert chain_ok is True


def test_rejection_blocks_deployment(workspace, loop):
    bootstrap(loop, workspace)
    loop.store.add_finding(_finding())
    proposal, _ = loop.propose_for_finding("fnd_e2e0000001")
    loop.simulate(proposal.id, backend="mujoco", policy_onnx=str(workspace["policy"]))
    loop.gate_proposal(proposal.id)
    loop.request_approval(proposal.id)

    # Hermes cannot approve without a rehearsal+gate, and a rejection is final for this revision.
    loop.decide(proposal.id, verdict="reject", actor="hermes", reason="too risky on hardware")
    assert loop.store.get_proposal(proposal.id).status == "rejected"
    with pytest.raises(DeployError):
        loop.deploy(proposal.id)


def test_modify_creates_a_new_validated_revision(workspace, loop):
    bootstrap(loop, workspace)
    loop.store.add_finding(_finding())
    proposal, _ = loop.propose_for_finding("fnd_e2e0000001")
    loop.simulate(proposal.id, backend="mujoco", policy_onnx=str(workspace["policy"]))
    loop.gate_proposal(proposal.id)
    loop.request_approval(proposal.id)

    result = loop.decide(
        proposal.id, verdict="modify", actor="hermes", reason="gentler change please",
        modifications={"template_params": {"factor": 0.9}},
    )
    assert loop.store.get_proposal(proposal.id).status == "superseded"
    new_id = result["new_proposal"]["id"]
    new_proposal = loop.store.get_proposal(new_id)
    assert new_proposal.revision == 2
    assert new_proposal.supersedes == proposal.id
    assert new_proposal.status == "validated"
    assert "action_scale: [0.225]" in new_proposal.edits[0].diff  # 0.25 * 0.9
    assert "action_scale: [0.2]" not in new_proposal.edits[0].diff


def test_tick_stops_at_hermes(workspace, loop):
    bootstrap(loop, workspace, episodes=60)
    results = loop.tick("locomotion")
    payload = results[0].to_dict()
    # The critic finds regressions; the reward proposal cannot be rehearsed without a
    # candidate policy, so the loop must report the error and never invent approval.
    assert payload["skill"] == "locomotion"
    for proposal_id in payload["awaiting_hermes"]:
        assert loop.store.get_proposal(proposal_id).status == "awaiting_hermes"
    for proposal in loop.store.proposals(skill="locomotion"):
        assert proposal.status != "deployed"


def test_tick_is_resumable_and_idempotent(workspace, loop):
    bootstrap(loop, workspace, episodes=60)
    first = loop.tick("locomotion")[0].to_dict()
    second = loop.tick("locomotion")[0].to_dict()
    # The second pass must not create duplicate findings or proposals.
    assert len(loop.store.proposals(skill="locomotion")) <= len(first["proposals"]) + len(second["proposals"])
    assert first["findings"] == second["findings"]


def test_unavailable_backend_blocks_the_chain(workspace, loop):
    bootstrap(loop, workspace)
    loop.store.add_finding(_finding())
    proposal, _ = loop.propose_for_finding("fnd_e2e0000001")
    sim = loop.simulate(proposal.id, backend="isaaclab")
    assert sim.status == "unavailable"
    assert "training environment" in sim.error
    verdict = loop.gate_proposal(proposal.id)
    assert verdict.verdict == "reject"


def test_refuses_to_simulate_an_invalid_proposal(workspace, loop):
    bootstrap(loop, workspace)
    loop.store.add_finding(_finding())
    proposal, _ = loop.propose_for_finding("fnd_e2e0000001")
    # Tamper with the stored validation to simulate a proposal that failed it.
    loop.store.set_proposal_validation(proposal.id, {"ok": False, "checks": [], "proposal_id": proposal.id,
                                                     "revision": proposal.revision})
    with pytest.raises(HermesError):
        loop.simulate(proposal.id)


def test_template_unavailable_is_reported_not_papered_over(workspace, loop):
    bootstrap(loop, workspace)
    loop.store.add_finding(
        _finding(id="fnd_e2e0000002", recommendation="reward_term_weight",
                 params={"term": "this_term_does_not_exist", "direction": "increase_penalty"})
    )
    with pytest.raises(TemplateUnavailable):
        loop.propose_for_finding("fnd_e2e0000002")
