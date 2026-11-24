"""The critic: turns stored episodes into defensible findings.

Each rule answers one question with one statistic, and records the statistic as
evidence. There is no scoring function that hides its inputs — a finding without
a p-value or an interval is not emitted.

Rules
-----
1. ``fall_rate_regression``   — recent fall rate worse than baseline (Wilson bound + z-test)
2. ``success_rate_regression``— recent success rate worse (two-proportion z-test)
3. ``tracking_regression``    — tracking error worse (median + bootstrap CI)
4. ``torque_instability``     — sustained torque increase (CUSUM changepoint + p95)
5. ``recovery_weak_spot``     — absolute fall rate above the skill's hard limit
6. ``headroom_opportunity``   — stable, high success: room to raise the command envelope

Findings are content-addressed per (skill, kind, metric, hour) so re-running the
critic on the same data does not duplicate them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..config import EvoConfig
from ..ids import content_id
from ..models import Episode, Finding, Metrics
from ..store import Store
from .. import stats
from .baseline import InsufficientData, compute_baseline, split_recent


@dataclass
class RuleOutcome:
    finding: Finding


def _severity_from_ratio(ratio: float, *, medium: float = 1.25, high: float = 1.75, critical: float = 3.0) -> str:
    if ratio >= critical:
        return "critical"
    if ratio >= high:
        return "high"
    if ratio >= medium:
        return "medium"
    return "low"


def _finding(
    *,
    skill: str,
    kind: str,
    title: str,
    summary: str,
    metric: str,
    baseline_value: float | None,
    current_value: float | None,
    p_value: float | None,
    severity: str,
    recommendation: str,
    params: dict[str, Any],
    evidence: dict[str, Any],
    window: tuple[float, float],
    now: float,
) -> Finding:
    delta = (
        None
        if baseline_value in (None, 0) or current_value is None
        else (current_value - baseline_value) / abs(baseline_value)
    )
    bucket = int(now // 3600)
    finding_id = content_id(
        "finding",
        {
            "skill": skill,
            "kind": kind,
            "metric": metric,
            "hour": bucket,
            "baseline": round(baseline_value, 6) if baseline_value is not None else None,
            "current": round(current_value, 6) if current_value is not None else None,
        },
    )
    return Finding(
        id=finding_id,
        skill=skill,
        kind=kind,
        severity=severity,
        title=title,
        summary=summary,
        metric=metric,
        baseline=baseline_value,
        current=current_value,
        delta_rel=delta,
        p_value=p_value,
        recommendation=recommendation,
        params=params,
        evidence=evidence,
        window_start=window[0],
        window_end=window[1],
        created_at=now,
    )


class Critic:
    def __init__(self, cfg: EvoConfig, store: Store):
        self.cfg = cfg
        self.store = store

    # -- public ----------------------------------------------------------
    def analyze(self, skill: str, *, now: float | None = None) -> list[Finding]:
        spec = self.cfg.skill(skill)
        now = time.time() if now is None else float(now)
        limits = spec.limits

        episodes = self.store.episodes(skill)
        if len(episodes) < limits.min_episodes:
            raise InsufficientData(
                f"skill {skill!r}: {len(episodes)} episodes stored, need {limits.min_episodes}"
            )

        try:
            # Anchor to the newest stored episode, not wall clock: analysing last
            # week's log must not produce an empty window.
            baseline = compute_baseline(self.store, skill, min_episodes=limits.min_episodes)
            older, recent = split_recent(baseline.episodes)
        except InsufficientData:
            older, recent = [], []

        findings: list[Finding] = []
        if older and recent:
            findings.extend(self._regressions(skill, older, recent, now))
            findings.extend(self._instability(skill, older, recent, now))
        findings.extend(self._absolute_limits(skill, episodes, now))
        findings.extend(self._opportunity(skill, episodes, now))

        for finding in findings:
            self.store.add_finding(finding)
        return findings

    # -- rules -----------------------------------------------------------
    def _regressions(
        self, skill: str, older: list[Episode], recent: list[Episode], now: float
    ) -> list[Finding]:
        spec = self.cfg.skill(skill)
        limits = spec.limits
        out: list[Finding] = []
        base = Metrics.from_episodes(older)
        cur = Metrics.from_episodes(recent)
        window = (older[0].started_at, recent[-1].started_at)

        # 1) fall-rate regression, guarded by a Wilson upper bound
        base_falls = sum(1 for e in older if e.falls > 0)
        cur_falls = sum(1 for e in recent if e.falls > 0)
        _, p_fall = stats.two_proportion_z(cur_falls, len(recent), base_falls, len(older))
        cur_hi = stats.wilson_interval(cur_falls, len(recent))[1]
        if cur_hi > base.fall_rate + limits.success_drop_max and p_fall < limits.stat_alpha:
            ratio = (cur.fall_rate + 1e-9) / (base.fall_rate + 1e-9)
            out.append(
                _finding(
                    skill=skill,
                    kind="regression",
                    title=f"Fall rate regressed for {skill}",
                    summary=(
                        f"Recent fall rate {cur.fall_rate:.1%} ({cur_falls}/{len(recent)}) vs baseline "
                        f"{base.fall_rate:.1%} ({base_falls}/{len(older)}); Wilson upper bound "
                        f"{cur_hi:.1%}. p={p_fall:.4f}."
                    ),
                    metric="fall_rate",
                    baseline_value=base.fall_rate,
                    current_value=cur.fall_rate,
                    p_value=p_fall,
                    severity=_severity_from_ratio(ratio),
                    recommendation="reward_term_weight",
                    params={
                        "term": "ang_vel_xy_l2",
                        "direction": "increase_penalty",
                        "reason": "stabilise base rotation to reduce falls",
                    },
                    evidence={
                        "baseline_falls": base_falls,
                        "baseline_n": len(older),
                        "current_falls": cur_falls,
                        "current_n": len(recent),
                        "wilson_upper": cur_hi,
                        "z_p_value": p_fall,
                    },
                    window=window,
                    now=now,
                )
            )

        # 2) success-rate regression
        base_ok = sum(1 for e in older if e.success)
        cur_ok = sum(1 for e in recent if e.success)
        _, p_succ = stats.two_proportion_z(cur_ok, len(recent), base_ok, len(older))
        drop = base.success_rate - cur.success_rate
        if drop > limits.success_drop_max and p_succ < limits.stat_alpha:
            out.append(
                _finding(
                    skill=skill,
                    kind="regression",
                    title=f"Success rate regressed for {skill}",
                    summary=(
                        f"Success {cur.success_rate:.1%} ({cur_ok}/{len(recent)}) vs baseline "
                        f"{base.success_rate:.1%} ({base_ok}/{len(older)}), a drop of {drop:.1%}. "
                        f"p={p_succ:.4f}."
                    ),
                    metric="success_rate",
                    baseline_value=base.success_rate,
                    current_value=cur.success_rate,
                    p_value=p_succ,
                    severity=_severity_from_ratio(drop / max(limits.success_drop_max, 1e-9)),
                    recommendation="train_hyperparameter",
                    params={
                        "flag": "--max_iterations",
                        "scale": 1.5,
                        "reason": "extend training budget after a success regression",
                    },
                    evidence={
                        "baseline_success": base_ok,
                        "baseline_n": len(older),
                        "current_success": cur_ok,
                        "current_n": len(recent),
                        "drop": drop,
                        "z_p_value": p_succ,
                    },
                    window=window,
                    now=now,
                )
            )

        # 3) tracking-error regression
        base_track = [e.tracking_error_rad for e in older if e.tracking_error_rad is not None]
        cur_track = [e.tracking_error_rad for e in recent if e.tracking_error_rad is not None]
        if len(base_track) >= 8 and len(cur_track) >= 8:
            base_med, cur_med = stats.median(base_track), stats.median(cur_track)
            delta = stats.relative_delta(base_med, cur_med)
            lo, hi = stats.bootstrap_median_ci(cur_track)
            if delta is not None and delta > limits.tracking_degradation_max and lo > base_med:
                out.append(
                    _finding(
                        skill=skill,
                        kind="regression",
                        title=f"Tracking error regressed for {skill}",
                        summary=(
                            f"Median tracking error {cur_med:.4f} rad vs baseline {base_med:.4f} rad "
                            f"({delta:+.1%}); bootstrap 95% CI [{lo:.4f}, {hi:.4f}] excludes the baseline."
                        ),
                        metric="tracking_error_rad",
                        baseline_value=base_med,
                        current_value=cur_med,
                        p_value=None,
                        severity=_severity_from_ratio(1.0 + delta),
                        recommendation="train_hyperparameter",
                        params={
                            "flag": "--learning_rate",
                            "scale": 0.5,
                            "reason": "recover tracking accuracy with a smaller learning rate",
                        },
                        evidence={
                            "bootstrap_ci": [lo, hi],
                            "baseline_median": base_med,
                            "current_median": cur_med,
                            "relative_delta": delta,
                            "n_baseline": len(base_track),
                            "n_current": len(cur_track),
                        },
                        window=window,
                        now=now,
                    )
                )
        return out

    def _instability(
        self, skill: str, older: list[Episode], recent: list[Episode], now: float
    ) -> list[Finding]:
        torques = [e.torque_peak_nm for e in older] + [e.torque_peak_nm for e in recent]
        changepoint = stats.cusum_changepoint(torques)
        if changepoint is None:
            return []
        pre = torques[:changepoint]
        post = torques[changepoint:]
        if len(pre) < 8 or len(post) < 8:
            return []
        pre_p95, post_p95 = stats.percentile(pre, 95), stats.percentile(post, 95)
        ratio = (post_p95 + 1e-9) / (pre_p95 + 1e-9)
        if ratio < 1.10:
            return []
        window = (older[0].started_at, recent[-1].started_at)
        return [
            _finding(
                skill=skill,
                kind="instability",
                title=f"Torque peaks shifted upward for {skill}",
                summary=(
                    f"CUSUM flagged a change at sample {changepoint}: p95 torque {pre_p95:.1f} N·m -> "
                    f"{post_p95:.1f} N·m ({ratio:.2f}x). Fatigue and driver saturation risk."
                ),
                metric="torque_peak_nm",
                baseline_value=pre_p95,
                current_value=post_p95,
                p_value=None,
                severity=_severity_from_ratio(ratio, medium=1.10, high=1.30, critical=1.60),
                recommendation="reward_term_weight",
                params={
                    "term": "dof_torques_l2",
                    "direction": "increase_penalty",
                    "reason": "penalise torque spikes after a detected shift",
                },
                evidence={
                    "changepoint_index": changepoint,
                    "pre_p95": pre_p95,
                    "post_p95": post_p95,
                    "ratio": ratio,
                    "n_pre": len(pre),
                    "n_post": len(post),
                },
                window=window,
                now=now,
            )
        ]

    def _absolute_limits(self, skill: str, episodes: list[Episode], now: float) -> list[Finding]:
        spec = self.cfg.skill(skill)
        limits = spec.limits
        metrics = Metrics.from_episodes(episodes)
        window = (episodes[0].started_at, episodes[-1].started_at)
        if metrics.fall_rate <= limits.fall_rate_max:
            return []
        ratio = metrics.fall_rate / max(limits.fall_rate_max, 1e-9)
        return [
            _finding(
                skill=skill,
                kind="weak_spot",
                title=f"{skill} falls above its own limit",
                summary=(
                    f"Fall rate {metrics.fall_rate:.1%} exceeds the hard limit "
                    f"{limits.fall_rate_max:.1%} over {metrics.n} episodes — this is a standing "
                    "weakness, not a regression against a previous policy."
                ),
                metric="fall_rate",
                baseline_value=limits.fall_rate_max,
                current_value=metrics.fall_rate,
                p_value=None,
                severity=_severity_from_ratio(ratio, medium=1.0, high=2.0, critical=4.0),
                recommendation="reward_term_weight",
                params={
                    "term": "ang_vel_xy_l2",
                    "direction": "increase_penalty",
                    "reason": "absolute fall rate above the skill limit",
                },
                evidence={
                    "fall_rate": metrics.fall_rate,
                    "limit": limits.fall_rate_max,
                    "n": metrics.n,
                    "episodes_with_falls": sum(1 for e in episodes if e.falls > 0),
                },
                window=window,
                now=now,
            )
        ]

    def _opportunity(self, skill: str, episodes: list[Episode], now: float) -> list[Finding]:
        spec = self.cfg.skill(skill)
        limits = spec.limits
        metrics = Metrics.from_episodes(episodes)
        window = (episodes[0].started_at, episodes[-1].started_at)
        if not (metrics.fall_rate <= 0.002 and metrics.success_rate >= 0.995):
            return []
        if metrics.n < max(limits.min_episodes, 100):
            return []
        return [
            _finding(
                skill=skill,
                kind="opportunity",
                title=f"{skill} has headroom: raise the command envelope",
                summary=(
                    f"{metrics.success_rate:.2%} success and {metrics.fall_rate:.2%} falls over "
                    f"{metrics.n} episodes. A controlled widening of the command range is testable."
                ),
                metric="success_rate",
                baseline_value=metrics.success_rate,
                current_value=metrics.success_rate,
                p_value=None,
                severity="info",
                recommendation="train_hyperparameter",
                params={
                    "flag": "--num_envs",
                    "scale": 1.25,
                    "reason": "policy is stable; widen evaluation/training envelope carefully",
                },
                evidence={
                    "success_rate": metrics.success_rate,
                    "fall_rate": metrics.fall_rate,
                    "n": metrics.n,
                },
                window=window,
                now=now,
            )
        ]
