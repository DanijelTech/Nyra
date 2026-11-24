"""Baseline estimation.

The baseline is the reference the critic compares against and the safety gate
requires non-inferiority to. It is computed from *stored, measured* episodes of a
frozen policy version — never from a previous proposal's claimed improvement, so
a chain of small regressions cannot ratchet the reference downward.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..models import Episode, Metrics
from ..store import Store


class InsufficientData(RuntimeError):
    """Not enough episodes to make a claim."""


@dataclass
class Baseline:
    skill: str
    metrics: Metrics
    episodes: list[Episode]
    window: tuple[float, float]
    policy_version: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "episodes": len(self.episodes),
            "window": list(self.window),
            "policy_version": self.policy_version,
            "metrics": self.metrics.to_dict(),
        }


def compute_baseline(
    store: Store,
    skill: str,
    *,
    window_s: float = 7 * 24 * 3600,
    min_episodes: int = 30,
    policy_version: str | None = None,
    until: float | None = None,
) -> Baseline:
    """Episodes from ``until - window_s`` up to ``until``.

    ``until`` defaults to the newest stored episode rather than wall-clock time:
    analysing a log from last week must not produce an empty window, and the
    "last 7 days of data" is the meaningful reference offline. Pass ``until``
    explicitly to analyse an arbitrary point in time.
    """
    if until is None:
        latest = store.latest_started_at(skill)
        end = latest if latest is not None else time.time()
    else:
        end = float(until)
    start = end - float(window_s)
    episodes = store.episodes(skill, since=start, until=end, policy_version=policy_version)
    if len(episodes) < min_episodes:
        raise InsufficientData(
            f"skill {skill!r}: {len(episodes)} episodes in the last {window_s / 3600:.1f} h, "
            f"need {min_episodes} for a baseline"
        )
    return Baseline(
        skill=skill,
        metrics=Metrics.from_episodes(episodes),
        episodes=episodes,
        window=(start, end),
        policy_version=policy_version,
    )


def split_recent(
    episodes: list[Episode], *, recent_fraction: float = 0.25, min_recent: int = 10
) -> tuple[list[Episode], list[Episode]]:
    """Split a chronologically sorted list into (baseline, recent) populations."""
    if recent_fraction <= 0 or recent_fraction >= 1:
        raise ValueError("recent_fraction must be within (0, 1)")
    ordered = sorted(episodes, key=lambda e: e.started_at)
    n_recent = max(min_recent, int(round(len(ordered) * recent_fraction)))
    n_recent = min(n_recent, len(ordered) - 1)
    if n_recent <= 0:
        raise InsufficientData("not enough episodes to split into baseline/recent")
    return ordered[:-n_recent], ordered[-n_recent:]
