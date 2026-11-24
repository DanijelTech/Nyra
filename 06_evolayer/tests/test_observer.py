"""Observer: strict schema, unit-error rejection, episode assembly, CSV, discovery."""

from __future__ import annotations

import json

import pytest

from evo_layer.observer.ingest import (
    EpisodeAssembler,
    TelemetryError,
    discover_inputs,
    episode_from_record,
    parse_csv,
    parse_jsonl,
    sample_from_record,
)


def valid_record(**overrides):
    record = {
        "schema_version": 1,
        "skill": "locomotion",
        "policy_version": "amp-1",
        "started_at": 1.0,
        "duration_s": 8.0,
        "success": True,
        "falls": 0,
        "torque_peak_nm": 42.0,
        "joint_vel_peak_rps": 9.5,
        "collisions": 0,
        "battery_min_v": 48.2,
        "tracking_error_rad": 0.12,
    }
    record.update(overrides)
    return record


def test_valid_record_is_accepted():
    episode = episode_from_record(valid_record())
    assert episode.skill == "locomotion"
    assert episode.success is True
    assert episode.id.startswith("ep_")


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"duration_s": 0.0}, "duration"),
        ({"torque_peak_nm": 5000.0}, "plausible range"),
        ({"joint_vel_peak_rps": 900.0}, "plausible range"),
        ({"battery_min_v": 3.3}, "plausible range"),
        ({"success": "yes"}, "boolean"),
        ({"source": "teleport"}, "source"),
        ({"falls": -1}, "falls"),
        ({"schema_version": 99}, "schema_version"),
    ],
)
def test_bad_records_are_rejected_with_a_reason(overrides, reason):
    with pytest.raises(TelemetryError) as excinfo:
        episode_from_record(valid_record(**overrides))
    assert reason in str(excinfo.value)


def test_missing_field_is_named():
    record = valid_record()
    del record["torque_peak_nm"]
    with pytest.raises(TelemetryError) as excinfo:
        episode_from_record(record)
    assert "torque_peak_nm" in str(excinfo.value)


def test_parse_jsonl_separates_good_from_bad(tmp_path):
    path = tmp_path / "log.jsonl"
    lines = [
        json.dumps(valid_record()),
        "{not json",
        json.dumps(valid_record(duration_s=-1)),
        json.dumps(valid_record()),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report, assembler = parse_jsonl(path, skill="locomotion")
    assert report.accepted_n == 2
    assert len(report.rejected) == 2
    assert assembler is None
    summary = report.summary()
    assert summary["accepted"] == 2 and summary["rejected"] == 2


def test_samples_assemble_into_episodes_with_fall_detection(tmp_path):
    path = tmp_path / "samples.jsonl"
    rows = []
    # 2 s of upright samples, then 1 s of a tilted base -> one fall.
    for index in range(40):
        rows.append({"record": "sample", "t": index * 0.05, "base_height": 0.74,
                     "base_roll": 0.02, "base_pitch": 0.03, "torque_nm": 30.0 + index,
                     "joint_vel": 4.0, "battery_v": 48.0})
    for index in range(40, 60):
        rows.append({"record": "sample", "t": index * 0.05, "base_height": 0.20,
                     "base_roll": 1.2, "base_pitch": 0.1, "torque_nm": 60.0,
                     "joint_vel": 12.0, "battery_v": 47.0})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    report, assembler = parse_jsonl(path, skill="locomotion", policy_version="amp-1")
    assert assembler is not None
    assert report.accepted_n >= 1
    episode = report.accepted[0]
    assert episode.falls == 1
    assert episode.success is False
    assert episode.torque_peak_nm >= 60.0


def test_assembler_splits_on_idle_gaps():
    from evo_layer.observer.ingest import sample_from_record as _s

    # 0.1 s sampling inside a group, a 5 s gap between groups.
    samples = [_s({"t": 0.0, "base_height": 0.74}), _s({"t": 0.1, "base_height": 0.74}),
               _s({"t": 0.2, "base_height": 0.74}), _s({"t": 0.3, "base_height": 0.74}),
               _s({"t": 0.4, "base_height": 0.74}), _s({"t": 0.5, "base_height": 0.74})]
    samples += [_s({"t": 6.0, "base_height": 0.74}), _s({"t": 6.1, "base_height": 0.74}),
                _s({"t": 6.2, "base_height": 0.74}), _s({"t": 6.3, "base_height": 0.74}),
                _s({"t": 6.4, "base_height": 0.74}), _s({"t": 6.5, "base_height": 0.74})]
    assembler = EpisodeAssembler(skill="locomotion", policy_version="p", min_duration_s=0.5, idle_gap_s=0.5)
    episodes = assembler.assemble(samples)
    assert len(episodes) == 2
    assert [e.falls for e in episodes] == [0, 0]


def test_assembler_drops_too_short_groups():
    from evo_layer.observer.ingest import sample_from_record as _s

    samples = [_s({"t": 0.0, "base_height": 0.74}), _s({"t": 0.1, "base_height": 0.74})]
    assembler = EpisodeAssembler(skill="s", policy_version="p", min_duration_s=1.0, idle_gap_s=1.0)
    assert assembler.assemble(samples) == []


def test_csv_ingest(tmp_path):
    path = tmp_path / "episodes.csv"
    path.write_text(
        "skill,policy_version,duration_s,success,falls,torque_peak_nm,joint_vel_peak_rps\n"
        "locomotion,amp-1,8.0,true,0,41.5,9.1\n"
        "locomotion,amp-1,7.5,false,1,55.0,11.0\n",
        encoding="utf-8",
    )
    report = parse_csv(path, skill="locomotion")
    assert report.accepted_n == 2
    assert report.accepted[1].falls == 1


def test_sample_rejects_malformed_command():
    with pytest.raises(TelemetryError):
        sample_from_record({"t": 0.0, "cmd_vel": [1.0, 0.0]})


def test_discover_inputs_expands_directories(tmp_path):
    (tmp_path / "a.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/b.csv").write_text("x\n", encoding="utf-8")
    found = discover_inputs([tmp_path])
    assert {path.name for path in found} == {"a.jsonl", "b.csv"}
    with pytest.raises(TelemetryError):
        discover_inputs([tmp_path / "missing"])
