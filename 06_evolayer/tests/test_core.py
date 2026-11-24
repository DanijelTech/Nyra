"""Core invariants: identifiers, domain models, store, audit chain, statistics."""

from __future__ import annotations

import json
import statistics

import pytest

from evo_layer import stats
from evo_layer.audit import AuditError, AuditLog
from evo_layer.ids import content_id, is_id, nonce
from evo_layer.models import (
    Decision,
    Edit,
    Episode,
    Finding,
    Metrics,
    ModelError,
    Proposal,
    Risk,
    SimPlan,
)
from evo_layer.store import DuplicateDecision, NotFound, Store


# --------------------------------------------------------------------------- identifiers


def test_content_id_is_deterministic_and_prefixed():
    payload = {"a": 1, "b": [1, 2]}
    first = content_id("proposal", payload)
    second = content_id("proposal", dict(reversed(list(payload.items()))))
    assert first == second
    assert is_id("proposal", first)
    assert not is_id("finding", first)
    assert content_id("proposal", {"a": 2}) != first


def test_nonce_is_unique():
    assert len({nonce() for _ in range(200)}) == 200


# --------------------------------------------------------------------------- models


def test_episode_rejects_impossible_values():
    with pytest.raises(ModelError):
        Episode(id="e", skill="s", source="real", policy_version="p", started_at=0.0,
                duration_s=0.0, success=True, falls=0, torque_peak_nm=1.0, joint_vel_peak_rps=1.0, collisions=0)
    with pytest.raises(ModelError):
        Episode(id="e", skill="s", source="teleport", policy_version="p", started_at=0.0,
                duration_s=1.0, success=True, falls=0, torque_peak_nm=1.0, joint_vel_peak_rps=1.0, collisions=0)


def test_metrics_from_episodes_is_plain_arithmetic():
    episodes = [
        Episode(id="a", skill="s", source="sim", policy_version="p", started_at=0.0, duration_s=4.0,
                success=True, falls=0, torque_peak_nm=10.0, joint_vel_peak_rps=2.0, collisions=0,
                tracking_error_rad=0.1, torque_ratio=0.2, vel_ratio=0.3),
        Episode(id="b", skill="s", source="sim", policy_version="p", started_at=1.0, duration_s=6.0,
                success=False, falls=2, torque_peak_nm=30.0, joint_vel_peak_rps=4.0, collisions=1,
                tracking_error_rad=0.3, torque_ratio=0.9, vel_ratio=0.8),
    ]
    metrics = Metrics.from_episodes(episodes)
    assert metrics.n == 2
    assert metrics.success_rate == 0.5
    assert metrics.fall_rate == 0.5
    assert metrics.torque_peak_nm == 30.0
    assert metrics.collision_rate == 0.5
    assert metrics.tracking_error_rad == pytest.approx(0.2)
    assert metrics.duration_s == 5.0
    assert metrics.torque_ratio == 0.9
    assert metrics.vel_ratio == 0.8


def test_proposal_must_change_something():
    with pytest.raises(ModelError):
        Proposal(id="p", revision=1, finding_id="f", skill="s", kind="k", title="t", rationale="r",
                 edits=[], sim_plan=SimPlan(backend="mujoco", episodes=1), expected_effect="e",
                 risk=Risk(score=0.1, risk_class="low"))


def test_edit_rejects_path_traversal():
    with pytest.raises(ModelError):
        Edit(repo="training", path="../etc/passwd", diff="--- a\n+++ b\n")


def test_decision_requires_reason():
    with pytest.raises(ModelError):
        Decision(proposal_id="p", revision=1, verdict="approve", actor="a", reason="ok", nonce="n")  # too short


# --------------------------------------------------------------------------- store


def _proposal(finding_id: str = "fnd_1") -> Proposal:
    return Proposal(
        id=content_id("proposal", {"f": finding_id, }), revision=1, finding_id=finding_id, skill="locomotion",
        kind="reward_term_weight", title="t", rationale="r",
        edits=[Edit(repo="training", path="robolab/a.py", diff="--- a/robolab/a.py\n+++ b/robolab/a.py\n@@ -1 +1 @@\n-x\n+y\n")],
        sim_plan=SimPlan(backend="mujoco", episodes=10), expected_effect="e",
        risk=Risk(score=0.3, risk_class="low"), created_at=1.0,
    )


def _finding(identifier: str = "fnd_1") -> Finding:
    return Finding(id=identifier, skill="locomotion", kind="regression", severity="high", title="t",
                   summary="s", metric="fall_rate", baseline=0.01, current=0.2, delta_rel=19.0,
                   p_value=0.001, recommendation="reward_term_weight", params={"term": "x"}, created_at=1.0)


def test_store_roundtrip(tmp_path):
    store = Store(tmp_path / "state.db")
    store.add_finding(_finding())
    proposal = _proposal()
    store.add_proposal(proposal)
    assert store.get_proposal(proposal.id).title == "t"
    assert store.get_finding("fnd_1").severity == "high"
    store.set_proposal_status(proposal.id, "validated")
    assert store.get_proposal(proposal.id).status == "validated"
    assert store.next_revision("fnd_1") == 2
    with pytest.raises(NotFound):
        store.get_proposal("prp_missing0000")
    store.close()


def test_store_rejects_second_decision(tmp_path):
    store = Store(tmp_path / "state.db")
    store.add_finding(_finding())
    proposal = _proposal()
    store.add_proposal(proposal)
    decision = Decision(proposal_id=proposal.id, revision=1, verdict="approve", actor="hermes",
                        reason="looks fine", nonce=nonce(), created_at=2.0)
    store.add_decision(decision)
    with pytest.raises(DuplicateDecision):
        store.add_decision(Decision(proposal_id=proposal.id, revision=1, verdict="reject",
                                    actor="hermes", reason="changed my mind", nonce=nonce()))
    store.close()


def test_nonce_can_only_be_consumed_once(tmp_path):
    store = Store(tmp_path / "state.db")
    value = nonce()
    assert store.consume_nonce(value) is True
    assert store.consume_nonce(value) is False
    store.close()


def test_store_migration_is_idempotent(tmp_path):
    path = tmp_path / "state.db"
    store = Store(path)
    store.add_episodes([Episode(id="e1", skill="s", source="sim", policy_version="p", started_at=0.0,
                                duration_s=1.0, success=True, falls=0, torque_peak_nm=1.0,
                                joint_vel_peak_rps=1.0, collisions=0, torque_ratio=0.1, vel_ratio=0.1)])
    store.close()
    reopened = Store(path)
    episodes = reopened.episodes("s")
    assert len(episodes) == 1
    assert episodes[0].torque_ratio == 0.1
    assert reopened.kv_get("schema_version") == "2"
    reopened.close()


def test_episode_dedup_on_reingest(tmp_path):
    store = Store(tmp_path / "state.db")
    episode = Episode(id="e-dup", skill="s", source="sim", policy_version="p", started_at=0.0,
                      duration_s=1.0, success=True, falls=0, torque_peak_nm=1.0,
                      joint_vel_peak_rps=1.0, collisions=0)
    assert store.add_episodes([episode]) == 1
    assert store.add_episodes([episode]) == 0
    assert store.episode_count("s") == 1
    store.close()


# --------------------------------------------------------------------------- audit


def test_audit_chain_detects_tampering(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for index in range(5):
        log.append("event.test", subject=f"s{index}", actor="test", data={"i": index})
    ok, count, broken, reason = log.verify()
    assert (ok, count, broken) == (True, 5, None)
    assert "intact" in reason

    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[2])
    record["data"]["i"] = 99
    lines[2] = json.dumps(record, sort_keys=True)
    (tmp_path / "audit.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, count, broken, reason = log.verify()
    assert ok is False
    assert broken == 3
    assert "hash" in reason


def test_audit_chain_detects_removed_record(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for index in range(4):
        log.append("event.test", subject=f"s{index}", actor="test", data={})
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    (tmp_path / "audit.jsonl").write_text("\n".join([lines[0], lines[2], lines[3]]) + "\n", encoding="utf-8")
    ok, count, broken, reason = log.verify()
    assert ok is False
    assert "sequence" in reason or "previous-hash" in reason


def test_audit_append_is_append_only(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append("a", subject="x", actor="t", data={})
    before = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    log.append("b", subject="y", actor="t", data={})
    after = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert after.startswith(before)
    assert log.count() == 2


def test_audit_rejects_malformed_file(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text('{"seq": 1}\n', encoding="utf-8")
    log = AuditLog(path)
    with pytest.raises(AuditError):
        list(log.entries())


# --------------------------------------------------------------------------- statistics


def test_wilson_interval_matches_reference_values():
    lo, hi = stats.wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)
    assert stats.wilson_interval(0, 0) == (0.0, 1.0)
    assert stats.wilson_interval(0, 20)[0] == 0.0


def test_two_proportion_z_detects_a_real_difference():
    z, p = stats.two_proportion_z(11, 50, 2, 150)
    assert z > 3
    assert p < 0.01
    _, p_same = stats.two_proportion_z(5, 50, 15, 150)
    assert p_same > 0.05


def test_bootstrap_median_ci_is_deterministic():
    values = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
    first = stats.bootstrap_median_ci(values, iterations=500)
    second = stats.bootstrap_median_ci(values, iterations=500)
    assert first == second
    assert first[0] <= statistics.median(values) <= first[1]


def test_cusum_finds_a_step_change():
    series = [10.0] * 40 + [18.0] * 40
    index = stats.cusum_changepoint(series)
    assert index is not None
    assert 30 <= index <= 45
    assert stats.cusum_changepoint([10.0] * 40) is None


def test_relative_delta_and_summarise():
    assert stats.relative_delta(0.0, 1.0) is None
    assert stats.relative_delta(2.0, 3.0) == pytest.approx(0.5)
    summary = stats.summarise([1, 2, 3, 4, 5])
    assert summary["n"] == 5
    assert summary["median"] == 3
    assert summary["max"] == 5
