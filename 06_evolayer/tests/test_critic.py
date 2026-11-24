"""Critic: real fixtures, real statistics, deterministic findings."""

from __future__ import annotations

import pytest

from evo_layer.critic.baseline import InsufficientData, compute_baseline, split_recent
from evo_layer.store import Store

from conftest import make_episodes


@pytest.fixture
def loaded(workspace, real_repo):
    """Store with the shipped locomotion fixture ingested."""
    from evo_layer.orchestrator.loop import EvoLoop

    cfg = workspace["cfg"]
    loop = EvoLoop.from_config(cfg, secret="unit-test-secret-value-0123456789", env={})
    loop.observe("locomotion", str(real_repo / "06_evolayer/examples/telemetry/real_locomotion.jsonl"))
    return loop


def test_regression_is_detected_with_statistics(loaded):
    findings = loaded.critique("locomotion")
    kinds = {(finding.kind, finding.metric) for finding in findings}
    assert ("regression", "fall_rate") in kinds
    assert ("regression", "success_rate") in kinds

    fall = next(f for f in findings if f.kind == "regression" and f.metric == "fall_rate")
    assert fall.p_value is not None and fall.p_value < 0.05
    assert fall.current > fall.baseline
    assert fall.severity in ("high", "critical")
    assert fall.recommendation in ("reward_term_weight", "train_hyperparameter", "eval_suite_expand")
    # evidence carries the raw counts so a reviewer can recompute the test
    assert fall.evidence["current_falls"] > fall.evidence["baseline_falls"]
    assert fall.evidence["wilson_upper"] >= fall.current


def test_weak_spot_reports_absolute_limit(loaded):
    findings = loaded.critique("locomotion")
    weak = [f for f in findings if f.kind == "weak_spot"]
    limit = loaded.cfg.skill("locomotion").limits.fall_rate_max
    assert weak, "fixture fall rate exceeds the skill limit, so a weak spot must be reported"
    assert weak[0].baseline == pytest.approx(limit)
    assert weak[0].current > limit


def test_instability_detects_the_torque_shift(loaded):
    findings = loaded.critique("locomotion")
    inst = [f for f in findings if f.kind == "instability"]
    assert inst, "fixture raises torque by ~1.25x halfway through"
    assert inst[0].evidence["ratio"] > 1.1
    assert 120 <= inst[0].evidence["changepoint_index"] <= 175


def test_findings_are_deterministic_within_the_same_hour(loaded):
    first = {f.id for f in loaded.critique("locomotion")}
    second = {f.id for f in loaded.critique("locomotion")}
    assert first == second


def test_critic_refuses_without_enough_episodes(workspace):
    from evo_layer.orchestrator.loop import EvoLoop

    loop = EvoLoop.from_config(workspace["cfg"], secret="unit-test-secret-value-0123456789", env={})
    loop.store.add_episodes(make_episodes(2))
    with pytest.raises(InsufficientData):
        loop.critique("locomotion")


def test_baseline_is_anchored_to_the_newest_episode(workspace):
    cfg = workspace["cfg"]
    store = Store(cfg.store_path)
    # Old data: a year in the past. A wall-clock window would be empty.
    store.add_episodes(make_episodes(80, start=1_700_000_000.0))
    baseline = compute_baseline(store, "locomotion", min_episodes=10)
    assert len(baseline.episodes) == 80
    assert baseline.window[1] == pytest.approx(1_700_000_000.0 + 79 * 120.0)
    store.close()


def test_split_recent_keeps_the_tail(workspace):
    episodes = make_episodes(40)
    older, recent = split_recent(episodes, recent_fraction=0.25, min_recent=5)
    assert len(recent) == 10
    assert len(older) == 30
    assert recent[0].started_at > older[-1].started_at
    with pytest.raises(ValueError):
        split_recent(episodes, recent_fraction=1.5)


def test_no_regression_when_the_policy_is_stable(workspace):
    from evo_layer.orchestrator.loop import EvoLoop

    loop = EvoLoop.from_config(workspace["cfg"], secret="unit-test-secret-value-0123456789", env={})
    loop.store.add_episodes(make_episodes(200, falls=0, torque=30.0, tracking=0.1))
    findings = loop.critique("locomotion")
    assert not [f for f in findings if f.kind == "regression"]
    assert not [f for f in findings if f.kind == "weak_spot"]


def test_opportunity_is_reported_for_a_clean_policy(workspace):
    from evo_layer.orchestrator.loop import EvoLoop

    cfg = workspace["cfg"]
    loop = EvoLoop.from_config(cfg, secret="unit-test-secret-value-0123456789", env={})
    loop.store.add_episodes(make_episodes(200, falls=0, torque=25.0, tracking=0.08))
    findings = loop.critique("locomotion")
    opportunities = [f for f in findings if f.kind == "opportunity"]
    assert opportunities, "a 100% success / 0% fall policy should surface headroom"
    assert opportunities[0].severity == "info"
