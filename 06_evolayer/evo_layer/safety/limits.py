"""Hardware limits, read from the robot description itself.

The gate must not use numbers copied into a YAML file a year ago. Effort and
velocity limits come from ``rpo_description/urdf/rpo.urdf`` — the same file the
trainer consumes — so a hardware revision propagates to the safety gate with a
submodule bump and nothing else.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


class LimitsError(RuntimeError):
    """The robot description is missing or inconsistent."""


@dataclass
class JointLimit:
    name: str
    effort_nm: float
    velocity_rps: float
    lower: float
    upper: float


@dataclass
class RobotLimits:
    source: str
    joints: dict[str, JointLimit] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.joints)

    @property
    def max_effort_nm(self) -> float:
        return max((j.effort_nm for j in self.joints.values()), default=0.0)

    @property
    def max_velocity_rps(self) -> float:
        return max((j.velocity_rps for j in self.joints.values()), default=0.0)

    @property
    def weakest_effort_nm(self) -> float:
        """The most conservative effort limit — used for whole-body peak checks."""
        return min((j.effort_nm for j in self.joints.values()), default=0.0)

    @property
    def weakest_velocity_rps(self) -> float:
        return min((j.velocity_rps for j in self.joints.values()), default=0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "joints": self.count,
            "max_effort_nm": self.max_effort_nm,
            "weakest_effort_nm": self.weakest_effort_nm,
            "max_velocity_rps": self.max_velocity_rps,
            "weakest_velocity_rps": self.weakest_velocity_rps,
        }


def load_urdf_limits(path: str | Path) -> RobotLimits:
    urdf = Path(path)
    if not urdf.is_file():
        raise LimitsError(f"URDF not found: {urdf}")
    try:
        root = ET.parse(urdf).getroot()
    except ET.ParseError as exc:
        raise LimitsError(f"URDF is not valid XML: {exc}") from exc

    joints: dict[str, JointLimit] = {}
    for joint in root.findall("joint"):
        if joint.get("type") in (None, "fixed", "floating"):
            continue
        limit = joint.find("limit")
        if limit is None:
            raise LimitsError(f"joint {joint.get('name')!r} has no <limit> element")
        try:
            joints[joint.get("name", "")] = JointLimit(
                name=joint.get("name", ""),
                effort_nm=float(limit.get("effort", "0")),
                velocity_rps=float(limit.get("velocity", "0")),
                lower=float(limit.get("lower", "0")),
                upper=float(limit.get("upper", "0")),
            )
        except ValueError as exc:
            raise LimitsError(f"joint {joint.get('name')!r} has a non-numeric limit: {exc}") from exc

    if not joints:
        raise LimitsError(f"no actuated joints found in {urdf}")
    for joint in joints.values():
        if joint.effort_nm <= 0 or joint.velocity_rps <= 0:
            raise LimitsError(f"joint {joint.name!r} has a non-positive effort/velocity limit")
    return RobotLimits(source=str(urdf), joints=joints)
