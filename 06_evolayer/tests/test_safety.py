"""Safety: URDF limits, the gate, and the deploy contract."""

from __future__ import annotations

import pytest

from evo_layer.config import Limits
from evo_layer.models import Metrics
from evo_layer.safety.gate import PASS, REJECT, REVIEW, GateInput, SafetyGate
from evo_layer.safety.limits import LimitsError, load_urdf_limits
from evo_layer.safety.policy import FAIL, PASS as POLICY_PASS, SKIP, check_policy_contract
from evo_layer.store import Store

from conftest import make_episodes


@pytest.fixture(scope="module")
def robot_limits(real_repo):
    return load_urdf_limits(real_repo / "00_body_base/rpo_description/urdf/rpo.urdf")


def test_urdf_limits_are_read_from_the_description(robot_limits):
    assert robot_limits.count == 23
    assert robot_limits.max_effort_nm == pytest.approx(120.0)
    assert robot_limits.weakest_effort_nm == pytest.approx(27.0)
    assert robot_limits.weakest_velocity_rps == pytest.approx(8.0)
    assert robot_limits.max_velocity_rps == pytest.approx(25.0)
    assert "left_knee_joint" in robot_limits.joints


def test_urdf_limits_reject_a_broken_file(tmp_path):
    broken = tmp_path / "broken.urdf"
    broken.write_text("<robot><joint name='a' type='revolute'/></robot>", encoding="utf-8")
    with pytest.raises(LimitsError):
        load_urdf_limits(broken)
    with pytest.raises(LimitsError):
        load_urdf_limits(tmp_path / "missing.urdf")


def metrics(**overrides) -> Metrics:
    base = dict(n=200, success_rate=0.99, fall_rate=0.005, torque_peak_nm=45.0,
                joint_vel_peak_rps=9.0, collision_rate=0.0, battery_min_v=48.0,
                tracking_error_rad=0.11, duration_s=8.0, torque_ratio=0.35, vel_ratio=0.4)
    base.update(overrides)
    return Metrics(**base)


BASELINE = Metrics(n=200, success_rate=0.99, fall_rate=0.005, torque_peak_nm=46.0,
                   joint_vel_peak_rps=9.2, collision_rate=0.0, battery_min_v=48.0,
                   tracking_error_rad=0.11, duration_s=8.0, torque_ratio=0.36, vel_ratio=0.41)


def evaluate(candidate: Metrics, *, limits: Limits | None = None, robot=None, sim_status: str = "ok",
             required: int = 20):
    return SafetyGate().evaluate(
        GateInput(
            metrics=candidate,
            baseline=BASELINE,
            limits=limits or Limits(min_episodes=20),
            robot=robot,
            sim_status=sim_status,
            required_episodes=required,
            extra={"proposal_id": "prp_x", "revision": 1},
        )
    )


def test_gate_passes_a_clean_candidate_with_enough_evidence(robot_limits):
    # 800 episodes: the Wilson bound on a 99% success rate clears the 2% margin.
    candidate = metrics(n=800, success_rate=0.99, fall_rate=0.004, torque_ratio=0.35, vel_ratio=0.4)
    verdict = evaluate(candidate, robot=robot_limits, required=400)
    assert verdict.verdict == PASS, [c for c in verdict.checks if c["status"] != PASS]
    assert verdict.stats["torque_limit_nm"] > 0


def test_gate_reviews_instead_of_passing_on_thin_evidence(robot_limits):
    # Same point estimates, 200 episodes: non-inferiority cannot be certified.
    verdict = evaluate(metrics(), robot=robot_limits, required=200)
    assert verdict.verdict == REVIEW
    review = [c for c in verdict.checks if c["status"] == REVIEW]
    assert any("Wilson lower bound" in c["detail"] for c in review)


def test_gate_rejects_a_fall_rate_over_the_limit(robot_limits):
    verdict = evaluate(metrics(fall_rate=0.08, success_rate=0.92), robot=robot_limits)
    assert verdict.verdict == REJECT
    assert any(c["name"] == "fall_rate_max" and c["status"] == REJECT for c in verdict.checks)


def test_gate_rejects_a_per_joint_torque_overload(robot_limits):
    verdict = evaluate(metrics(torque_ratio=1.4), robot=robot_limits)
    assert verdict.verdict == REJECT
    assert any(c["name"] == "torque_ratio" for c in verdict.checks)


def test_gate_rejects_a_per_joint_velocity_overload(robot_limits):
    verdict = evaluate(metrics(vel_ratio=2.0), robot=robot_limits)
    assert verdict.verdict == REJECT
    assert any(c["name"] == "vel_ratio" and c["status"] == REJECT for c in verdict.checks)


def test_gate_reviews_instead_of_certifying_without_per_joint_data(robot_limits):
    verdict = evaluate(metrics(torque_ratio=None, vel_ratio=None), robot=robot_limits)
    assert verdict.verdict == REVIEW
    assert any(c["name"] == "torque_peak_nm" and c["status"] == REVIEW for c in verdict.checks)


def test_gate_rejects_too_few_episodes(robot_limits):
    verdict = evaluate(metrics(n=5), robot=robot_limits, required=50)
    assert verdict.verdict == REJECT
    assert any(c["name"] == "episode_count" for c in verdict.checks)


def test_gate_rejects_a_failed_rehearsal(robot_limits):
    verdict = evaluate(metrics(), robot=robot_limits, sim_status="failed")
    assert verdict.verdict == REJECT
    assert verdict.checks[0]["name"] == "simulation"


def test_gate_rejects_a_success_regression(robot_limits):
    verdict = evaluate(metrics(success_rate=0.80, fall_rate=0.02), robot=robot_limits)
    assert verdict.verdict == REJECT
    assert any(c["name"] == "success_non_inferior" for c in verdict.checks)


def test_gate_rejects_a_tracking_regression(robot_limits):
    verdict = evaluate(metrics(tracking_error_rad=0.30), robot=robot_limits)
    assert verdict.verdict == REJECT
    assert any(c["name"] == "tracking_non_inferior" and c["status"] == REJECT for c in verdict.checks)


def test_gate_rejects_a_point_estimate_over_a_local_limit(robot_limits):
    limits = Limits(min_episodes=20, fall_rate_max=0.05)
    candidate = metrics(n=60, fall_rate=4 / 60, success_rate=56 / 60)
    verdict = evaluate(candidate, limits=limits, robot=robot_limits, required=60)
    statuses = {c["name"]: c["status"] for c in verdict.checks}
    assert statuses["fall_rate_max"] == REJECT


def test_absolute_limits_never_override_a_regression(robot_limits):
    """A looser absolute limit must not license a policy that got worse."""
    limits = Limits(min_episodes=20, fall_rate_max=0.10, success_drop_max=0.02)
    candidate = metrics(n=200, fall_rate=0.067, success_rate=0.97)
    verdict = evaluate(candidate, limits=limits, robot=robot_limits, required=100)
    statuses = {c["name"]: c["status"] for c in verdict.checks}
    assert statuses["fall_rate_max"] == PASS          # 6.7% is inside the 10% ceiling
    assert statuses["fall_rate_non_inferior"] == REJECT  # but worse than the 0.5% baseline
    assert verdict.verdict == REJECT


# --------------------------------------------------------------------------- policy contract


def test_policy_contract_accepts_the_shipped_configuration(real_repo):
    deploy = real_repo / "00_body_base/roboparty_deploy/src/inference/robots/rpo"
    urdf = load_urdf_limits(real_repo / "00_body_base/rpo_description/urdf/rpo.urdf")
    checks = check_policy_contract(
        urdf_joints=list(urdf.joints),
        policy_yaml=deploy / "configs/default.yaml",
        models_dir=deploy / "models",
        latents_dir=deploy / "latents",
        onnx_path=deploy / "models/policy.onnx",
        rpo_joint_order_file=real_repo / "04_training/roboparty_train/robolab/robolab/assets/robots/roboparty.py",
    )
    statuses = {check.name: check.status for check in checks}
    assert statuses["joint_num"] == POLICY_PASS
    assert statuses["usd2urdf"] == POLICY_PASS
    assert statuses["train_action_order"] == POLICY_PASS
    assert all(check.status != FAIL for check in checks), [c.to_dict() for c in checks if c.status == FAIL]


def test_policy_contract_flags_a_broken_permutation(real_repo, tmp_path):
    deploy = real_repo / "00_body_base/roboparty_deploy/src/inference/robots/rpo"
    tampered = tmp_path / "default.yaml"
    text = (deploy / "configs/default.yaml").read_text(encoding="utf-8")
    text = text.replace(
        "usd2urdf: [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]",
        "usd2urdf: [0, 0, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]",
    )
    tampered.write_text(text, encoding="utf-8")
    urdf = load_urdf_limits(real_repo / "00_body_base/rpo_description/urdf/rpo.urdf")
    checks = check_policy_contract(
        urdf_joints=list(urdf.joints), policy_yaml=tampered,
        models_dir=deploy / "models", latents_dir=deploy / "latents",
    )
    statuses = {check.name: check.status for check in checks}
    assert statuses["usd2urdf"] == FAIL


def test_policy_contract_flags_a_missing_model(real_repo, tmp_path):
    deploy = real_repo / "00_body_base/roboparty_deploy/src/inference/robots/rpo"
    urdf = load_urdf_limits(real_repo / "00_body_base/rpo_description/urdf/rpo.urdf")
    checks = check_policy_contract(
        urdf_joints=list(urdf.joints), policy_yaml=deploy / "configs/ufo.yaml",
        models_dir=tmp_path, latents_dir=tmp_path,
    )
    statuses = {check.name: check.status for check in checks}
    assert any(name.startswith("model:") and status == FAIL for name, status in statuses.items())
    assert any(name.startswith("latent:") and status == FAIL for name, status in statuses.items())


def test_policy_contract_skips_the_onnx_graph_without_runtime(real_repo):
    deploy = real_repo / "00_body_base/roboparty_deploy/src/inference/robots/rpo"
    urdf = load_urdf_limits(real_repo / "00_body_base/rpo_description/urdf/rpo.urdf")
    checks = check_policy_contract(
        urdf_joints=list(urdf.joints), policy_yaml=deploy / "configs/default.yaml",
        models_dir=deploy / "models", onnx_path=deploy / "models/policy.onnx",
    )
    graph = next(check for check in checks if check.name == "onnx_graph")
    assert graph.status in (POLICY_PASS, SKIP)


def test_gate_uses_per_joint_ratios_from_the_store(workspace):
    """A stored episode with a high ratio must reach the gate through Metrics."""
    cfg = workspace["cfg"]
    store = Store(cfg.store_path)
    store.add_episodes(make_episodes(30, falls=0, torque_ratio=0.4, vel_ratio=0.4))
    metrics = store.metrics_for("locomotion")
    assert metrics.torque_ratio == pytest.approx(0.4)
    store.close()
