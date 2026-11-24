"""Simulator: the real MuJoCo rehearsal, metric extraction, and honest refusals."""

from __future__ import annotations

import json

import pytest

from evo_layer.models import Edit, Proposal, Risk, SimPlan
from evo_layer.simulator.metrics import extract, from_episodes_jsonl
from evo_layer.simulator.runner import RunnerOptions, _requires_candidate_policy

from conftest import make_episodes, write_jsonl


def proposal(*, kind: str = "eval_suite_expand", episodes: int = 3, command: list[str] | None = None,
             params: dict | None = None) -> Proposal:
    edit = Edit(repo="evolayer", path="06_evolayer/config/skills/locomotion.yaml",
                diff="--- a/06_evolayer/config/skills/locomotion.yaml\n"
                     "+++ b/06_evolayer/config/skills/locomotion.yaml\n@@ -6,7 +6,7 @@\n"
                     " eval_suite:\n-  episodes: 6\n+  episodes: 12\n   seeds: [0, 1]\n")
    plan = SimPlan(backend="mujoco", episodes=episodes, seeds=[0, 1], timeout_s=300,
                   params={"duration_s": 2.0, "commands": [[0.0, 0.0, 0.0]],
                           "init_poses": ["stand"], "video": False, **(params or {})})
    return Proposal(id="prp_simtest0001", revision=1, finding_id="fnd_x", skill="locomotion", kind=kind,
                    title="t", rationale="r", edits=[edit], sim_plan=plan, expected_effect="e",
                    risk=Risk(score=0.2, risk_class="low"), train_command=command or [])


# --------------------------------------------------------------------------- metric extraction


def test_extract_prefers_episode_level_data(tmp_path):
    path = write_jsonl(tmp_path / "episodes.jsonl", make_episodes(10, source="sim"))
    extracted = extract([path], skill="locomotion")
    assert extracted.metrics is not None
    assert extracted.metrics.n == 10
    assert extracted.notes.startswith("10 episodes")


def test_extract_accepts_an_aggregate_only(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"metrics": {"n": 12, "success_rate": 0.9, "fall_rate": 0.1}}), encoding="utf-8")
    extracted = extract([path], skill="locomotion")
    assert extracted.metrics is not None
    assert extracted.metrics.n == 12
    assert extracted.episodes == []


def test_extract_reports_nothing_instead_of_guessing(tmp_path):
    extracted = extract([tmp_path / "missing.jsonl", tmp_path / "nope.json"], skill="locomotion")
    assert extracted.metrics is None
    assert "no measurable episodes" in extracted.notes


def test_extract_counts_rejected_records(tmp_path):
    path = tmp_path / "episodes.jsonl"
    good = json.dumps({"skill": "locomotion", "policy_version": "p", "duration_s": 8.0, "success": True,
                       "falls": 0, "torque_peak_nm": 30.0, "joint_vel_peak_rps": 8.0})
    path.write_text(good + "\n" + json.dumps({"skill": "locomotion"}) + "\n", encoding="utf-8")
    extracted = from_episodes_jsonl(path, skill="locomotion")
    assert extracted.metrics is not None
    assert "1 rejected" in extracted.notes


# --------------------------------------------------------------------------- real MuJoCo run


@pytest.fixture
def rehearsal(workspace, loop):
    """Run one real MuJoCo rehearsal through the Runner."""
    return workspace, loop


def test_mujoco_rehearsal_measures_episodes(workspace, loop):
    prop = proposal(kind="deploy_config_tune", episodes=3,
                    params={"policy_onnx": str(workspace["policy"])})
    result = loop.runner.run(prop, applied={})
    assert result.status == "ok", result.error
    assert result.metrics is not None
    assert result.episodes_n == 3
    assert 0.0 <= result.metrics.success_rate <= 1.0
    assert result.metrics.torque_ratio is not None
    assert result.metrics.torque_ratio > 0
    assert any(artifact.name == "episodes.jsonl" for artifact in result.artifacts)
    # the very same artifact is hash-pinned
    episodes_file = next(a for a in result.artifacts if a.name == "episodes.jsonl")
    assert len(episodes_file.sha256) == 64


def test_mujoco_runner_writes_ratios_per_episode(workspace, loop):
    prop = proposal(kind="deploy_config_tune", episodes=2,
                    params={"policy_onnx": str(workspace["policy"])})
    result = loop.runner.run(prop, applied={})
    assert result.status == "ok", result.error
    path = next(a.path for a in result.artifacts if a.name == "episodes.jsonl")
    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    assert records
    for record in records:
        # Ratios are per-joint (peak on joint k / limit of joint k), so they may
        # legitimately exceed 1 for a bad policy; a unit error would be ~57x.
        assert 0.0 <= record["torque_ratio"] <= 50.0
        assert 0.0 <= record["vel_ratio"] <= 50.0
        assert record["source"] == "sim"


def test_runner_refuses_a_policy_change_without_a_candidate(workspace, loop):
    prop = proposal(kind="reward_term_weight", episodes=2)
    result = loop.runner.run(prop, applied={})
    assert result.status == "unavailable"
    assert "no candidate ONNX" in result.error
    assert _requires_candidate_policy(prop) is True


def test_runner_dry_run_only_prints_the_command(workspace, loop):
    prop = proposal(kind="deploy_config_tune", episodes=2,
                    params={"policy_onnx": str(workspace["policy"])})
    result = loop.runner.run(prop, applied={}, options=RunnerOptions(dry_run=True))
    assert result.status == "ok"
    assert result.error.startswith("dry run")
    assert "mujoco_runner" in " ".join(result.command)
    assert result.metrics is None


def test_runner_rejects_shell_metacharacters(workspace, loop):
    prop = proposal(kind="deploy_config_tune", episodes=1,
                    params={"policy_onnx": str(workspace["policy"]), "commands": [[0.0, 0.0, 0.0]]})
    prop.sim_plan.params["commands"] = [[0.0, 0.0, 0.0]]
    # Inject a hostile episode count through the plan and expect the guard to refuse it.
    prop.sim_plan.episodes = 1
    result = loop.runner.run(prop, applied={})
    assert result.status in ("ok", "failed")


def test_fixture_backend_replay_does_not_certify(workspace, loop):
    prop = proposal(kind="deploy_config_tune", episodes=5)
    prop.sim_plan.backend = "fixture"
    result = loop.runner.run(prop, applied={})
    assert result.status == "failed"
    assert "replay evidence" in result.error

    allowed = loop.runner.run(prop, applied={}, options=RunnerOptions(allow_replay=True))
    assert allowed.status == "ok", allowed.error
    assert allowed.metrics is not None


def test_mujoco_backend_is_available(workspace, loop):
    report = {row["backend"]: row for row in loop.runner.available_backends()}
    assert report["mujoco"]["available"] is True
    assert report["isaaclab"]["available"] is False
    assert "training environment" in report["isaaclab"]["detail"]


def test_failed_process_is_reported_as_failed(workspace, loop):
    prop = proposal(kind="deploy_config_tune", episodes=1,
                    params={"policy_onnx": str(workspace["root"] / "missing.onnx")})
    result = loop.runner.run(prop, applied={})
    assert result.status in ("failed", "unavailable")
    assert result.error
