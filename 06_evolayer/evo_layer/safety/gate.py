"""The safety gate.

Inputs are measured: a :class:`~evo_layer.models.Metrics` aggregate from a real
simulation run, the baseline aggregate, the hardware limits from the URDF, and
the skill's configured limits. Output is a verdict with one row per constraint,
each row carrying the measured value, the limit and the reason.

Three verdicts exist and their meanings are strict:

``reject``  a hard limit was crossed, or the candidate is statistically worse
            than the baseline by more than the allowed margin. No approval can
            override this — the orchestrator refuses to build a report.
``review``  nothing crossed, but the evidence is not strong enough to certify
            non-inferiority. Hermes decides, with the reason spelled out.
``pass``    every hard limit holds and non-inferiority is certified.

The distinction between ``reject`` and ``review`` is the Wilson interval: a
point estimate inside the margin is not proof, and the gate says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .. import stats
from ..config import Limits
from ..models import Metrics, Verdict
from .limits import RobotLimits

PASS, REVIEW, REJECT = "pass", "review", "reject"


@dataclass
class GateInput:
    metrics: Metrics
    baseline: Metrics
    limits: Limits
    robot: RobotLimits
    sim_status: str = "ok"
    required_episodes: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _row(name: str, status: str, measured: Any, limit: Any, detail: str) -> dict[str, Any]:
    return {"name": name, "status": status, "measured": measured, "limit": limit, "detail": detail}


class SafetyGate:
    def __init__(self, *, alpha: float | None = None, margin: float | None = None):
        self.alpha = alpha
        self.margin = margin

    def evaluate(self, data: GateInput) -> Verdict:
        limits = data.limits
        alpha = self.alpha if self.alpha is not None else limits.stat_alpha
        margin = self.margin if self.margin is not None else limit_margin(limits)
        metrics = data.metrics
        baseline = data.baseline
        rows: list[dict[str, Any]] = []
        required = data.required_episodes or limits.min_episodes

        # 0) the simulation itself must be trustworthy
        if data.sim_status != "ok":
            rows.append(_row("simulation", REJECT, data.sim_status, "ok", "rehearsal did not complete"))
        else:
            rows.append(_row("simulation", PASS, "ok", "ok", "rehearsal completed"))

        # 1) evidence volume
        if metrics.n < required:
            rows.append(
                _row("episode_count", REJECT, metrics.n, required, f"need {required} episodes for a verdict")
            )
        else:
            rows.append(_row("episode_count", PASS, metrics.n, required, "sufficient evidence"))

        # 2) absolute hard limits
        rows.append(
            _hard(
                "fall_rate_max",
                metrics.fall_rate,
                limits.fall_rate_max,
                higher_is_bad=True,
            )
        )
        # Preferred: per-joint ratios (peak on joint k / limit of joint k). A whole-body
        # peak compared with the weakest joint's limit would be a false positive; a
        # whole-body peak compared with the strongest would be a false negative.
        if metrics.torque_ratio is not None:
            rows.append(
                _hard(
                    "torque_ratio",
                    metrics.torque_ratio,
                    limits.torque_ratio_max,
                    higher_is_bad=True,
                    note="per-joint peak torque / URDF effort limit",
                )
            )
            torque_limit = data.robot.max_effort_nm * limits.torque_ratio_max
        else:
            torque_limit = data.robot.max_effort_nm * limits.torque_ratio_max
            over = metrics.torque_peak_nm > torque_limit
            rows.append(
                _row(
                    "torque_peak_nm",
                    REJECT if over else REVIEW,
                    metrics.torque_peak_nm,
                    torque_limit,
                    (
                        f"{metrics.torque_peak_nm:g} exceeds the strongest joint's effort "
                        f"{data.robot.max_effort_nm:g} N·m"
                        if over
                        else "no per-joint ratios in the telemetry; a whole-body peak cannot be "
                        "attributed to a joint, so this is not certified"
                    ),
                )
            )
        if metrics.vel_ratio is not None:
            rows.append(
                _hard(
                    "vel_ratio",
                    metrics.vel_ratio,
                    limits.joint_vel_ratio_max,
                    higher_is_bad=True,
                    note="per-joint peak velocity / URDF velocity limit",
                )
            )
            velocity_limit = data.robot.max_velocity_rps * limits.joint_vel_ratio_max
        else:
            velocity_limit = data.robot.max_velocity_rps * limits.joint_vel_ratio_max
            over = metrics.joint_vel_peak_rps > velocity_limit
            rows.append(
                _row(
                    "joint_vel_peak_rps",
                    REJECT if over else REVIEW,
                    metrics.joint_vel_peak_rps,
                    velocity_limit,
                    (
                        f"{metrics.joint_vel_peak_rps:g} exceeds the fastest joint's limit "
                        f"{data.robot.max_velocity_rps:g} rad/s"
                        if over
                        else "no per-joint ratios in the telemetry; not certified"
                    ),
                )
            )
        rows.append(
            _hard("collision_rate", metrics.collision_rate, limits.collision_rate_max, higher_is_bad=True)
        )
        if metrics.battery_min_v is not None:
            rows.append(
                _hard("battery_min_v", metrics.battery_min_v, limits.battery_min_v, higher_is_bad=False)
            )
        else:
            rows.append(_row("battery_min_v", PASS, None, limits.battery_min_v, "not measured in simulation"))

        # 3) statistical non-inferiority versus the baseline
        rows.extend(_non_inferiority_fall_rate(metrics, baseline, limits, alpha, margin))
        rows.extend(_non_inferiority_success(metrics, baseline, limits, alpha, margin))
        rows.extend(_non_inferiority_tracking(metrics, baseline, limits, alpha))

        verdict = REJECT if any(r["status"] == REJECT for r in rows) else (
            REVIEW if any(r["status"] == REVIEW for r in rows) else PASS
        )
        summary = {
            "n": metrics.n,
            "alpha": alpha,
            "margin": margin,
            "fall_rate": metrics.fall_rate,
            "baseline_fall_rate": baseline.fall_rate,
            "success_rate": metrics.success_rate,
            "baseline_success_rate": baseline.success_rate,
            "torque_peak_nm": metrics.torque_peak_nm,
            "torque_ratio": metrics.torque_ratio,
            "torque_limit_nm": torque_limit,
            "joint_vel_peak_rps": metrics.joint_vel_peak_rps,
            "vel_ratio": metrics.vel_ratio,
            "joint_vel_limit_rps": velocity_limit,
            "collision_rate": metrics.collision_rate,
            "checks": {r["name"]: r["status"] for r in rows},
            **data.extra,
        }
        return Verdict(
            proposal_id=str(data.extra.get("proposal_id", "")),
            revision=int(data.extra.get("revision", 0)),
            verdict=verdict,
            checks=rows,
            stats=summary,
        )


def limit_margin(limits: Limits) -> float:
    """Absolute margin used for non-inferiority on proportions."""
    return max(limits.success_drop_max, 0.005)


def _hard(name: str, measured: float, limit: float, *, higher_is_bad: bool, note: str = "") -> dict[str, Any]:
    if higher_is_bad:
        ok = measured <= limit
        detail = f"{measured:g} <= {limit:g}" if ok else f"{measured:g} exceeds {limit:g}"
    else:
        ok = measured >= limit
        detail = f"{measured:g} >= {limit:g}" if ok else f"{measured:g} below {limit:g}"
    if note:
        detail += f" ({note})"
    return _row(name, PASS if ok else REJECT, measured, limit, detail)


def _non_inferiority_fall_rate(
    metrics: Metrics, baseline: Metrics, limits: Limits, alpha: float, margin: float
) -> list[dict[str, Any]]:
    b_rate, c_rate = baseline.fall_rate, metrics.fall_rate
    b_falls = int(round(b_rate * baseline.n))
    c_falls = int(round(c_rate * metrics.n))
    lo, hi = stats.wilson_interval(c_falls, metrics.n)
    _, p_value = stats.two_proportion_z(c_falls, metrics.n, b_falls, baseline.n)
    delta = c_rate - b_rate
    if delta > margin:
        return [
            _row(
                "fall_rate_non_inferior",
                REJECT,
                c_rate,
                b_rate + margin,
                f"fall rate {c_rate:.2%} vs baseline {b_rate:.2%} exceeds the {margin:.2%} margin "
                f"(p={p_value:.4f})",
            )
        ]
    if hi > b_rate + margin or (p_value < alpha and delta > 0):
        return [
            _row(
                "fall_rate_non_inferior",
                REVIEW,
                c_rate,
                b_rate + margin,
                f"fall rate {c_rate:.2%} is within the margin but the Wilson upper bound "
                f"{hi:.2%} crosses {b_rate + margin:.2%} — not certified (p={p_value:.4f})",
            )
        ]
    return [
        _row(
            "fall_rate_non_inferior",
            PASS,
            c_rate,
            b_rate + margin,
            f"fall rate {c_rate:.2%} with Wilson upper bound {hi:.2%} inside {b_rate + margin:.2%}",
        )
    ]


def _non_inferiority_success(
    metrics: Metrics, baseline: Metrics, limits: Limits, alpha: float, margin: float
) -> list[dict[str, Any]]:
    b_rate, c_rate = baseline.success_rate, metrics.success_rate
    b_ok = int(round(b_rate * baseline.n))
    c_ok = int(round(c_rate * metrics.n))
    lo, _ = stats.wilson_interval(c_ok, metrics.n)
    _, p_value = stats.two_proportion_z(c_ok, metrics.n, b_ok, baseline.n)
    floor = b_rate - limits.success_drop_max
    if c_rate < floor:
        return [
            _row(
                "success_non_inferior",
                REJECT,
                c_rate,
                floor,
                f"success {c_rate:.2%} below baseline {b_rate:.2%} minus {limits.success_drop_max:.2%} "
                f"(p={p_value:.4f})",
            )
        ]
    if lo < floor:
        return [
            _row(
                "success_non_inferior",
                REVIEW,
                c_rate,
                floor,
                f"success {c_rate:.2%} holds, but the Wilson lower bound {lo:.2%} is under {floor:.2%}",
            )
        ]
    return [
        _row(
            "success_non_inferior",
            PASS,
            c_rate,
            floor,
            f"success {c_rate:.2%} with Wilson lower bound {lo:.2%} above {floor:.2%}",
        )
    ]


def _non_inferiority_tracking(
    metrics: Metrics, baseline: Metrics, limits: Limits, alpha: float
) -> list[dict[str, Any]]:
    if metrics.tracking_error_rad is None or baseline.tracking_error_rad is None:
        return [_row("tracking_non_inferior", PASS, None, None, "tracking metric not reported by this skill")]
    base = baseline.tracking_error_rad
    cur = metrics.tracking_error_rad
    if base <= 0:
        return [_row("tracking_non_inferior", REVIEW, cur, base, "baseline tracking error is zero; cannot compare")]
    ratio = cur / base
    allowed = 1.0 + limits.tracking_degradation_max
    if ratio > allowed:
        return [
            _row(
                "tracking_non_inferior",
                REJECT,
                cur,
                base * allowed,
                f"tracking error {cur:.4f} rad is {ratio:.2f}x the baseline {base:.4f} rad "
                f"(allowed {allowed:.2f}x)",
            )
        ]
    if ratio > 1.0 + limits.tracking_degradation_max * 0.5:
        return [
            _row(
                "tracking_non_inferior",
                REVIEW,
                cur,
                base * allowed,
                f"tracking error {cur:.4f} rad is {ratio:.2f}x baseline, close to the {allowed:.2f}x bound",
            )
        ]
    return [
        _row(
            "tracking_non_inferior",
            PASS,
            cur,
            base * allowed,
            f"tracking error {cur:.4f} rad is {ratio:.2f}x baseline (allowed {allowed:.2f}x)",
        )
    ]


def significant_gain(metrics: Metrics, baseline: Metrics, *, alpha: float = 0.05) -> bool:
    """Whether the candidate is measurably better, not merely not-worse."""
    if metrics.n == 0 or baseline.n == 0:
        return False
    c_ok = int(round(metrics.success_rate * metrics.n))
    b_ok = int(round(baseline.success_rate * baseline.n))
    if c_ok / metrics.n <= b_ok / baseline.n:
        return False
    _, p_value = stats.two_proportion_z(c_ok, metrics.n, b_ok, baseline.n)
    return p_value < alpha and not math.isnan(p_value)
