"""Strict telemetry schema and episode assembly.

Two ingestion paths exist and both end in :class:`Episode` records:

* **Recorded logs** — JSONL/CSV files produced by the robot, a simulation run or
  a simulator adapter. Each line is either a finished episode or a *sample*
  (joint state at a timestamp), which is assembled here.
* **Live ROS 2** — :mod:`evo_layer.observer.ros2_node` subscribes to the deploy
  topics and writes the same JSONL through the same code path.

Validation is strict on purpose: a silent unit error (degrees vs radians, N·m vs
scaled torque) would poison every downstream number, so malformed input is
rejected with a precise reason and counted, never coerced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from ..ids import content_id
from ..models import Episode, ModelError

TELEMETRY_SCHEMA_VERSION = 1

REQUIRED_EPISODE_FIELDS = (
    "skill",
    "policy_version",
    "duration_s",
    "success",
    "torque_peak_nm",
    "joint_vel_peak_rps",
)

OPTIONAL_EPISODE_FIELDS = (
    "episode_id",
    "started_at",
    "falls",
    "collisions",
    "battery_min_v",
    "tracking_error_rad",
    "torque_ratio",
    "vel_ratio",
    "notes",
    "source",
    "schema_version",
)

#: Physical plausibility envelope for the 23-DoF RPO. Values outside are treated
#: as a unit error, not as an interesting robot state.
SANE_TORQUE_NM = (0.0, 500.0)
SANE_JOINT_VEL = (0.0, 200.0)
SANE_BATTERY_V = (10.0, 60.0)
SANE_DURATION_S = (0.05, 600.0)


class TelemetryError(ValueError):
    """A telemetry record violates the schema or the plausibility envelope."""


@dataclass
class Reject:
    line_no: int
    reason: str
    raw: str = ""


@dataclass
class IngestReport:
    accepted: list[Episode]
    rejected: list[Reject]
    samples: int = 0

    @property
    def accepted_n(self) -> int:
        return len(self.accepted)

    def summary(self) -> dict[str, Any]:
        return {
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            "samples": self.samples,
            "reasons": sorted({r.reason.split(":")[0] for r in self.rejected}),
        }


def _require_number(record: dict[str, Any], key: str) -> float:
    if key not in record:
        raise TelemetryError(f"missing field: {key}")
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TelemetryError(f"{key} must be a number, got {type(value).__name__}")
    return float(value)


def _check_range(name: str, value: float, bounds: tuple[float, float]) -> None:
    low, high = bounds
    if not low <= value <= high:
        raise TelemetryError(f"{name}={value} outside plausible range [{low}, {high}] (unit error?)")


def episode_from_record(record: dict[str, Any], *, skill_hint: str | None = None) -> Episode:
    """Validate one finished-episode record and build an :class:`Episode`."""
    if not isinstance(record, dict):
        raise TelemetryError("record must be a JSON object")
    schema_version = record.get("schema_version", TELEMETRY_SCHEMA_VERSION)
    if int(schema_version) != TELEMETRY_SCHEMA_VERSION:
        raise TelemetryError(f"unsupported schema_version {schema_version}")

    missing = [field for field in REQUIRED_EPISODE_FIELDS if field not in record]
    if missing:
        raise TelemetryError(f"missing field: {', '.join(missing)}")

    skill = str(record["skill"] or skill_hint or "")
    if not skill:
        raise TelemetryError("skill is required")

    success = record["success"]
    if not isinstance(success, bool):
        raise TelemetryError("success must be a boolean")

    duration = _require_number(record, "duration_s")
    _check_range("duration_s", duration, SANE_DURATION_S)
    torque = _require_number(record, "torque_peak_nm")
    _check_range("torque_peak_nm", torque, SANE_TORQUE_NM)
    vel = _require_number(record, "joint_vel_peak_rps")
    _check_range("joint_vel_peak_rps", vel, SANE_JOINT_VEL)

    falls = int(record.get("falls", 0))
    collisions = int(record.get("collisions", 0))
    if falls < 0 or collisions < 0:
        raise TelemetryError("falls/collisions must be >= 0")

    battery = record.get("battery_min_v")
    if battery is not None:
        battery = float(battery)
        _check_range("battery_min_v", battery, SANE_BATTERY_V)

    tracking = record.get("tracking_error_rad")
    if tracking is not None:
        tracking = float(tracking)
        if tracking < 0:
            raise TelemetryError("tracking_error_rad must be >= 0")

    def optional_ratio(key: str) -> float | None:
        value = record.get(key)
        if value is None:
            return None
        value = float(value)
        if value < 0:
            raise TelemetryError(f"{key} must be >= 0")
        if value > 10.0:
            raise TelemetryError(f"{key}={value} is implausible (unit error?)")
        return value

    torque_ratio = optional_ratio("torque_ratio")
    vel_ratio = optional_ratio("vel_ratio")

    started_at = float(record.get("started_at", 0.0))
    source = str(record.get("source", "real"))
    if source not in ("sim", "real", "replay"):
        raise TelemetryError(f"source must be sim|real|replay, got {source!r}")

    identity = {
        "skill": skill,
        "policy_version": str(record["policy_version"]),
        "started_at": started_at,
        "duration_s": duration,
        "success": success,
        "falls": falls,
        "torque_peak_nm": torque,
        "joint_vel_peak_rps": vel,
        "collisions": collisions,
        "episode_id": record.get("episode_id"),
    }
    episode_id = str(record.get("episode_id") or content_id("episode", identity))

    try:
        return Episode(
            id=episode_id,
            skill=skill,
            source=source,
            policy_version=str(record["policy_version"]),
            started_at=started_at,
            duration_s=duration,
            success=success,
            falls=falls,
            torque_peak_nm=torque,
            joint_vel_peak_rps=vel,
            collisions=collisions,
            battery_min_v=battery,
            tracking_error_rad=tracking,
            torque_ratio=torque_ratio,
            vel_ratio=vel_ratio,
            notes=str(record.get("notes", "")),
        )
    except ModelError as exc:
        raise TelemetryError(str(exc)) from exc


# --------------------------------------------------------------------------- sample assembly


@dataclass
class Sample:
    """A single joint/base state sample from the robot or a simulator."""

    t: float
    base_height: float | None = None
    base_roll: float | None = None
    base_pitch: float | None = None
    torque_nm: float | None = None
    joint_vel: float | None = None
    battery_v: float | None = None
    tracking_error: float | None = None
    command: tuple[float, float, float] | None = None


class EpisodeAssembler:
    """Turn a sample stream into episodes.

    A fall is declared when the base pitch/roll exceeds ``tilt_rad`` for at least
    ``sustain_s`` seconds, or when the base height drops below ``height_floor`` —
    both measured, both configurable, no heuristics about intent.
    """

    def __init__(
        self,
        *,
        skill: str,
        policy_version: str,
        source: str = "real",
        tilt_rad: float = 0.7,
        sustain_s: float = 0.25,
        height_floor: float = 0.25,
        min_duration_s: float = 0.5,
        idle_gap_s: float = 0.5,
    ):
        self.skill = skill
        self.policy_version = policy_version
        self.source = source
        self.tilt_rad = tilt_rad
        self.sustain_s = sustain_s
        self.height_floor = height_floor
        self.min_duration_s = min_duration_s
        self.idle_gap_s = idle_gap_s

    def assemble(self, samples: Sequence[Sample], *, require_success: bool = True) -> list[Episode]:
        if not samples:
            return []
        ordered = sorted(samples, key=lambda s: s.t)
        groups: list[list[Sample]] = []
        current: list[Sample] = []
        for sample in ordered:
            if current and sample.t - current[-1].t > self.idle_gap_s:
                groups.append(current)
                current = []
            current.append(sample)
        if current:
            groups.append(current)

        episodes: list[Episode] = []
        for group in groups:
            episode = self._finish(group, require_success=require_success)
            if episode is not None:
                episodes.append(episode)
        return episodes

    def _finish(self, group: list[Sample], *, require_success: bool) -> Episode | None:
        duration = group[-1].t - group[0].t
        if duration < self.min_duration_s:
            return None
        tilt_start: float | None = None
        fall_time: float | None = None
        for sample in group:
            tilt = max(abs(sample.base_roll or 0.0), abs(sample.base_pitch or 0.0))
            low = sample.base_height is not None and sample.base_height < self.height_floor
            is_bad = tilt > self.tilt_rad or low
            if is_bad and tilt_start is None:
                tilt_start = sample.t
            elif not is_bad:
                tilt_start = None
            if tilt_start is not None and sample.t - tilt_start >= self.sustain_s:
                fall_time = tilt_start
                break

        torques = [s.torque_nm for s in group if s.torque_nm is not None]
        vels = [s.joint_vel for s in group if s.joint_vel is not None]
        batteries = [s.battery_v for s in group if s.battery_v is not None]
        tracking = [s.tracking_error for s in group if s.tracking_error is not None]
        falls = 0 if fall_time is None else 1
        survived = float(duration)
        if fall_time is not None:
            survived = max(0.0, fall_time - group[0].t)
        success = falls == 0
        if require_success and not success:
            success = False

        identity = {
            "skill": self.skill,
            "policy_version": self.policy_version,
            "started_at": group[0].t,
            "duration_s": duration,
            "falls": falls,
            "source": self.source,
        }
        try:
            return Episode(
                id=content_id("episode", identity),
                skill=self.skill,
                source=self.source,
                policy_version=self.policy_version,
                started_at=group[0].t,
                duration_s=round(max(duration, 1e-3), 6),
                success=success,
                falls=falls,
                torque_peak_nm=max(torques) if torques else 0.0,
                joint_vel_peak_rps=max(vels) if vels else 0.0,
                collisions=0,
                battery_min_v=min(batteries) if batteries else None,
                tracking_error_rad=(sum(tracking) / len(tracking)) if tracking else None,
                notes=f"assembled from {len(group)} samples; survived={survived:.2f}s",
            )
        except ModelError:
            return None


def sample_from_record(record: dict[str, Any]) -> Sample:
    if not isinstance(record, dict):
        raise TelemetryError("sample must be a JSON object")
    if "t" not in record:
        raise TelemetryError("sample requires a 't' timestamp")
    t = float(record["t"])
    if t < 0:
        raise TelemetryError("sample.t must be >= 0")

    def opt(key: str, bounds: tuple[float, float] | None = None) -> float | None:
        value = record.get(key)
        if value is None:
            return None
        value = float(value)
        if bounds:
            _check_range(key, value, bounds)
        return value

    command = record.get("cmd_vel")
    command_tuple = None
    if command is not None:
        if not isinstance(command, (list, tuple)) or len(command) != 3:
            raise TelemetryError("cmd_vel must be [vx, vy, wz]")
        command_tuple = (float(command[0]), float(command[1]), float(command[2]))

    return Sample(
        t=t,
        base_height=opt("base_height"),
        base_roll=opt("base_roll"),
        base_pitch=opt("base_pitch"),
        torque_nm=opt("torque_nm", SANE_TORQUE_NM),
        joint_vel=opt("joint_vel", SANE_JOINT_VEL),
        battery_v=opt("battery_v", SANE_BATTERY_V),
        tracking_error=opt("tracking_error"),
        command=command_tuple,
    )


def parse_jsonl(
    path: Path,
    *,
    skill: str | None = None,
    policy_version: str | None = None,
    source: str = "real",
) -> tuple[IngestReport, EpisodeAssembler | None]:
    """Parse a JSONL file of episodes and/or samples. Returns the report and the assembler used."""
    accepted: list[Episode] = []
    rejected: list[Reject] = []
    samples: list[Sample] = []
    assembler: EpisodeAssembler | None = None

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                rejected.append(Reject(line_no, f"invalid JSON: {exc.msg}", line[:120]))
                continue
            try:
                kind = record.get("record", "episode") if isinstance(record, dict) else "episode"
                if kind == "sample":
                    samples.append(sample_from_record(record))
                else:
                    accepted.append(episode_from_record(record, skill_hint=skill))
            except (TelemetryError, KeyError, TypeError, ValueError) as exc:
                rejected.append(Reject(line_no, str(exc), line[:120]))

    if samples:
        resolved_skill = skill or (accepted[0].skill if accepted else "")
        if not resolved_skill:
            raise TelemetryError("sample stream needs --skill")
        version = policy_version or (accepted[0].policy_version if accepted else "unknown")
        assembler = EpisodeAssembler(skill=resolved_skill, policy_version=version, source=source)
        accepted.extend(assembler.assemble(samples))

    return IngestReport(accepted=accepted, rejected=rejected, samples=len(samples)), assembler


def parse_csv(path: Path, *, skill: str) -> IngestReport:
    """Parse a flat CSV of finished episodes (header required)."""
    import csv

    accepted: list[Episode] = []
    rejected: list[Reject] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise TelemetryError("CSV requires a header row")
        for line_no, row in enumerate(reader, start=2):
            record: dict[str, Any] = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                if key in ("success",):
                    record[key] = str(value).strip().lower() in ("1", "true", "yes")
                elif key in ("falls", "collisions"):
                    record[key] = int(float(value))
                elif key in ("skill", "policy_version", "notes", "source", "episode_id"):
                    record[key] = value
                else:
                    try:
                        record[key] = float(value)
                    except ValueError:
                        record[key] = value
            record.setdefault("skill", skill)
            try:
                accepted.append(episode_from_record(record, skill_hint=skill))
            except (TelemetryError, ValueError) as exc:
                rejected.append(Reject(line_no, str(exc), str(row)[:120]))
    return IngestReport(accepted=accepted, rejected=rejected)


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            yield json.loads(line)


def discover_inputs(paths: Iterable[str | Path]) -> list[Path]:
    """Expand files and directories into a sorted list of supported inputs."""
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.suffix.lower() in (".jsonl", ".csv", ".ndjson"):
                    found.append(candidate)
        elif path.is_file():
            found.append(path)
        else:
            raise TelemetryError(f"input not found: {path}")
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        if path.resolve() not in seen:
            seen.add(path.resolve())
            unique.append(path)
    return unique
