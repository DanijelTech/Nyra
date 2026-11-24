"""Domain model for EvoLayer.

Everything that crosses a module boundary is one of these dataclasses. They are
frozen where identity matters, JSON-serialisable, and validated on construction
so that a malformed artifact can never enter the store.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Any

SEVERITIES = ("info", "low", "medium", "high", "critical")
RISK_CLASSES = ("low", "medium", "high")

FINDING_KINDS = ("regression", "weak_spot", "instability", "opportunity")
PROPOSAL_STATUSES = (
    "proposed",
    "validated",
    "simulated",
    "gated",
    "awaiting_hermes",
    "approved",
    "rejected",
    "failed",
    "deployed",
    "superseded",
)
VERDICTS = ("pass", "review", "reject")
DECISIONS = ("approve", "reject", "modify")


class ModelError(ValueError):
    """Raised when a domain object violates its invariants."""


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ModelError(f"{name} must be finite, got {value!r}")
    return value


def _fraction(value: float, name: str) -> float:
    value = _finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ModelError(f"{name} must be within [0, 1], got {value}")
    return value


@dataclass(frozen=True)
class Episode:
    """One rollout: either from the real robot or from a simulator."""

    id: str
    skill: str
    source: str
    policy_version: str
    started_at: float
    duration_s: float
    success: bool
    falls: int
    torque_peak_nm: float
    joint_vel_peak_rps: float
    collisions: int
    battery_min_v: float | None = None
    tracking_error_rad: float | None = None
    #: max over joints of |torque| / urdf_effort_limit, and |velocity| / urdf_velocity_limit.
    #: Unit-free, per-joint correct; preferred by the safety gate when present.
    torque_ratio: float | None = None
    vel_ratio: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.skill:
            raise ModelError("episode.skill is required")
        if self.source not in ("sim", "real", "replay"):
            raise ModelError(f"episode.source must be sim|real|replay, got {self.source!r}")
        if self.falls < 0 or self.collisions < 0:
            raise ModelError("episode.falls/collisions must be >= 0")
        if self.duration_s <= 0:
            raise ModelError("episode.duration_s must be > 0")
        _finite(self.torque_peak_nm, "episode.torque_peak_nm")
        _finite(self.joint_vel_peak_rps, "episode.joint_vel_peak_rps")
        if self.torque_peak_nm < 0 or self.joint_vel_peak_rps < 0:
            raise ModelError("episode peaks must be >= 0")
        for name, value in (("torque_ratio", self.torque_ratio), ("vel_ratio", self.vel_ratio)):
            if value is not None and float(value) < 0:
                raise ModelError(f"episode.{name} must be >= 0")


@dataclass
class Metrics:
    """Aggregated behaviour of one population of episodes."""

    n: int = 0
    success_rate: float = 0.0
    fall_rate: float = 0.0
    torque_peak_nm: float = 0.0
    joint_vel_peak_rps: float = 0.0
    collision_rate: float = 0.0
    battery_min_v: float | None = None
    tracking_error_rad: float | None = None
    duration_s: float = 0.0
    torque_ratio: float | None = None
    vel_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ModelError("metrics.n must be >= 0")
        _fraction(self.success_rate, "metrics.success_rate")
        _fraction(self.fall_rate, "metrics.fall_rate")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Metrics:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_episodes(cls, episodes: list[Episode]) -> Metrics:
        """Aggregate measured episodes. Pure arithmetic, no modelling."""
        if not episodes:
            return cls()
        n = len(episodes)
        tracking = [e.tracking_error_rad for e in episodes if e.tracking_error_rad is not None]
        batteries = [e.battery_min_v for e in episodes if e.battery_min_v is not None]
        return cls(
            n=n,
            success_rate=sum(1 for e in episodes if e.success) / n,
            fall_rate=sum(1 for e in episodes if e.falls > 0) / n,
            torque_peak_nm=max(e.torque_peak_nm for e in episodes),
            joint_vel_peak_rps=max(e.joint_vel_peak_rps for e in episodes),
            collision_rate=sum(e.collisions for e in episodes) / n,
            battery_min_v=min(batteries) if batteries else None,
            tracking_error_rad=(sum(tracking) / len(tracking)) if tracking else None,
            duration_s=sum(e.duration_s for e in episodes) / n,
            torque_ratio=_max_or_none(e.torque_ratio for e in episodes),
            vel_ratio=_max_or_none(e.vel_ratio for e in episodes),
        )


@dataclass
class Finding:
    """A weakness the critic believes is worth acting on."""

    id: str
    skill: str
    kind: str
    severity: str
    title: str
    summary: str
    metric: str
    baseline: float | None
    current: float | None
    delta_rel: float | None
    p_value: float | None
    recommendation: str
    params: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    window_start: float | None = None
    window_end: float | None = None
    created_at: float = 0.0
    status: str = "open"

    def __post_init__(self) -> None:
        if self.kind not in FINDING_KINDS:
            raise ModelError(f"finding.kind must be one of {FINDING_KINDS}, got {self.kind!r}")
        if self.severity not in SEVERITIES:
            raise ModelError(f"finding.severity must be one of {SEVERITIES}, got {self.severity!r}")
        if self.p_value is not None and not 0.0 <= self.p_value <= 1.0:
            raise ModelError("finding.p_value must be within [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Edit:
    """A single file change, carried as a unified diff against one repo."""

    repo: str
    path: str
    diff: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.repo or not self.path:
            raise ModelError("edit.repo and edit.path are required")
        if not self.diff.strip():
            raise ModelError("edit.diff must not be empty")
        if ".." in self.path.split("/"):
            raise ModelError(f"edit.path must not traverse: {self.path!r}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def added_lines(self) -> int:
        return sum(1 for line in self.diff.splitlines() if line.startswith("+") and not line.startswith("+++"))

    @property
    def removed_lines(self) -> int:
        return sum(1 for line in self.diff.splitlines() if line.startswith("-") and not line.startswith("---"))


@dataclass
class Risk:
    score: float
    risk_class: str
    factors: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _fraction(self.score, "risk.score")
        if self.risk_class not in RISK_CLASSES:
            raise ModelError(f"risk.risk_class must be one of {RISK_CLASSES}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class SimPlan:
    backend: str
    episodes: int
    seeds: list[int] = field(default_factory=list)
    motions: list[int] = field(default_factory=list)
    timeout_s: int = 3600
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ModelError("sim_plan.episodes must be > 0")
        if self.timeout_s <= 0:
            raise ModelError("sim_plan.timeout_s must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Proposal:
    id: str
    revision: int
    finding_id: str
    skill: str
    kind: str
    title: str
    rationale: str
    edits: list[Edit]
    sim_plan: SimPlan
    expected_effect: str
    risk: Risk
    config_overlay: dict[str, Any] = field(default_factory=dict)
    train_command: list[str] = field(default_factory=list)
    created_by: str = "template"
    created_at: float = 0.0
    status: str = "proposed"
    validation: dict[str, Any] | None = None
    supersedes: str | None = None

    def __post_init__(self) -> None:
        if self.status not in PROPOSAL_STATUSES:
            raise ModelError(f"proposal.status must be one of {PROPOSAL_STATUSES}, got {self.status!r}")
        if self.revision < 1:
            raise ModelError("proposal.revision must be >= 1")
        if not self.edits and not self.config_overlay and not self.train_command:
            raise ModelError("a proposal must carry at least one edit, config overlay or train command")

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        return data

    @property
    def touched_paths(self) -> list[tuple[str, str]]:
        return [(e.repo, e.path) for e in self.edits]


@dataclass
class Artifact:
    name: str
    path: str
    sha256: str
    kind: str = "file"
    size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class SimulationResult:
    proposal_id: str
    revision: int
    backend: str
    status: str
    metrics: Metrics | None = None
    artifacts: list[Artifact] = field(default_factory=list)
    command: list[str] = field(default_factory=list)
    log_path: str = ""
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    episodes_n: int = 0

    def __post_init__(self) -> None:
        if self.status not in ("ok", "failed", "timeout", "unavailable"):
            raise ModelError(f"simulation.status invalid: {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["metrics"] = self.metrics.to_dict() if self.metrics else None
        data["artifacts"] = [a.to_dict() for a in self.artifacts]
        return data


@dataclass
class Verdict:
    proposal_id: str
    revision: int
    verdict: str
    checks: list[dict[str, Any]]
    stats: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ModelError(f"verdict must be one of {VERDICTS}, got {self.verdict!r}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def blocking(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if c.get("status") == "reject"]


@dataclass
class Decision:
    proposal_id: str
    revision: int
    verdict: str
    actor: str
    reason: str
    nonce: str
    created_at: float = 0.0
    #: Structured changes requested with a ``modify`` decision, e.g.
    #: ``{"template_params": {"factor": 1.25}}``.
    modifications: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.verdict not in DECISIONS:
            raise ModelError(f"decision must be one of {DECISIONS}, got {self.verdict!r}")
        if not self.actor.strip():
            raise ModelError("decision.actor is required")
        if len(self.reason.strip()) < 3:
            raise ModelError("decision.reason must explain the call (>= 3 characters)")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Deployment:
    id: str
    proposal_id: str
    revision: int
    tag: str
    snapshot_id: str
    artifacts: list[Artifact] = field(default_factory=list)
    created_at: float = 0.0
    note: str = ""
    rolled_back: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["artifacts"] = [a.to_dict() for a in self.artifacts]
        return data


def _max_or_none(values: Any) -> float | None:
    present = [float(v) for v in values if v is not None]
    return max(present) if present else None


def canonical_json(data: Any) -> str:
    """Deterministic JSON for hashing and signing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
