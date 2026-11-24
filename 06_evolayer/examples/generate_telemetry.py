#!/usr/bin/env python3
"""Generate the deterministic telemetry fixtures used by the examples and tests.

Run from the repository root:

    python3 06_evolayer/examples/generate_telemetry.py

Two datasets are produced:

``real_locomotion.jsonl``  real-robot episodes for `locomotion`. The first 140
    episodes are the healthy baseline (~1% falls); the last 60 show a regression
    (~20% falls, higher torque). The critic must detect it with p < 0.05.

``sim_recovery.jsonl``  simulator episodes for `recovery` with a standing weakness
    (fall rate ~9% against a 5% limit). Used by the `fixture` replay backend.

Everything is derived from a fixed seed, so the fixtures are reproducible.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "telemetry"

JOINTS = 23


def episode(
    *,
    index: int,
    skill: str,
    source: str,
    falls: int,
    torque: float,
    vel: float,
    tracking: float | None,
    start: float,
    duration: float = 8.0,
    battery: float | None = 48.5,
    rng: random.Random | None = None,
    policy: str = "amp-2026.11",
) -> dict:
    return {
        "record": "episode",
        "schema_version": 1,
        "episode_id": f"ep_fixture_{skill}_{index:04d}",
        "skill": skill,
        "policy_version": policy,
        "source": source,
        "started_at": start,
        "duration_s": duration,
        "success": falls == 0,
        "falls": falls,
        "torque_peak_nm": round(torque, 3),
        "joint_vel_peak_rps": round(vel, 3),
        "collisions": 0,
        "battery_min_v": battery,
        "tracking_error_rad": round(tracking, 5) if tracking is not None else None,
        "notes": "generated fixture" + (f" rng={rng.random():.6f}" if rng else ""),
    }


def build_locomotion() -> list[dict]:
    rng = random.Random(20261101)
    rows: list[dict] = []
    start = 1_760_000_000.0
    for i in range(200):
        if i < 140:
            falls = 1 if rng.random() < 0.010 else 0
            torque = rng.gauss(46.0, 3.5)
            tracking = rng.gauss(0.115, 0.012)
            vel = rng.gauss(9.8, 0.9)
        else:
            falls = 1 if rng.random() < 0.200 else 0
            torque = rng.gauss(58.0, 5.0)
            tracking = rng.gauss(0.168, 0.020)
            vel = rng.gauss(12.4, 1.4)
        rows.append(
            episode(
                index=i,
                skill="locomotion",
                source="real",
                falls=falls,
                torque=max(1.0, torque),
                vel=max(0.1, vel),
                tracking=tracking,
                start=start + i * 120.0,
                rng=rng,
            )
        )
    return rows


def build_recovery() -> list[dict]:
    rng = random.Random(20261102)
    rows: list[dict] = []
    start = 1_760_100_000.0
    for i in range(120):
        falls = 1 if rng.random() < 0.09 else 0
        rows.append(
            episode(
                index=i,
                skill="recovery",
                source="sim",
                falls=falls,
                torque=max(1.0, rng.gauss(62.0, 6.0)),
                vel=max(0.1, rng.gauss(11.0, 1.2)),
                tracking=None,
                start=start + i * 90.0,
                duration=6.0,
                battery=None,
                rng=rng,
                policy="getup-2026.10",
            )
        )
    return rows


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"{path}: {len(rows)} episodes")


if __name__ == "__main__":
    write(OUT / "real_locomotion.jsonl", build_locomotion())
    write(OUT / "sim_recovery.jsonl", build_recovery())
