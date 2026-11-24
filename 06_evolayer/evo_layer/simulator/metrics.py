"""Metric extraction from simulator artifacts.

Backends either emit the canonical episode JSONL (our MuJoCo evaluator, the
fixture replay) or only upstream logs. This module turns whatever exists into
:class:`~evo_layer.models.Metrics`, and says plainly when it cannot. A backend
that produces no measurable episodes yields ``metrics=None``, which the gate
treats as a rejection — never as a pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import Episode, Metrics
from ..observer.ingest import IngestReport, parse_jsonl


@dataclass
class Extracted:
    metrics: Metrics | None
    episodes: list[Episode]
    source: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "episodes": len(self.episodes),
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "notes": self.notes,
        }


def from_episodes_jsonl(path: Path, *, skill: str | None = None) -> Extracted:
    if not path.is_file():
        return Extracted(None, [], str(path), "episodes.jsonl not produced")
    report: IngestReport = parse_jsonl(path, skill=skill)[0]
    if not report.accepted:
        return Extracted(None, [], str(path), f"no usable episodes ({len(report.rejected)} rejected)")
    metrics = Metrics.from_episodes(report.accepted)
    notes = f"{len(report.accepted)} episodes"
    if report.rejected:
        notes += f", {len(report.rejected)} rejected"
    return Extracted(metrics, report.accepted, str(path), notes)


def from_metrics_json(path: Path) -> Extracted:
    if not path.is_file():
        return Extracted(None, [], str(path), "metrics.json not produced")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return Extracted(None, [], str(path), f"unparseable metrics.json: {exc}")
    if isinstance(data, list):
        episodes = []
        for item in data:
            try:
                episodes.append(Episode(**item))
            except (TypeError, ValueError):
                continue
        if not episodes:
            return Extracted(None, [], str(path), "metrics.json contained no valid episodes")
        return Extracted(Metrics.from_episodes(episodes), episodes, str(path), f"{len(episodes)} episodes")
    if isinstance(data, dict):
        for key in ("metrics", "aggregate"):
            if isinstance(data.get(key), dict):
                return Extracted(Metrics.from_dict(data[key]), [], str(path), "aggregate only")
        if {"n", "success_rate", "fall_rate"} <= set(data):
            return Extracted(Metrics.from_dict(data), [], str(path), "aggregate only")
    return Extracted(None, [], str(path), "metrics.json has an unsupported shape")


def extract(paths: list[Path], *, skill: str | None = None) -> Extracted:
    """Prefer episode-level data, fall back to an aggregate, then give up honestly."""
    for path in paths:
        if path.name in ("episodes.jsonl", "episodes.ndjson"):
            result = from_episodes_jsonl(path, skill=skill)
            if result.metrics is not None:
                return result
    for path in paths:
        if path.name == "metrics.json":
            result = from_metrics_json(path)
            if result.metrics is not None:
                return result
    return Extracted(None, [], "", "backend produced no measurable episodes")
