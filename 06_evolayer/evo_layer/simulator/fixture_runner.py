"""Replay backend used by the fixture simulator and by the loop's self-test.

It reads a recorded episode dataset, validates it with the same schema the live
observer uses, filters it to the skill, and writes ``episodes.jsonl`` in the
canonical form. Nothing is fabricated: if the dataset cannot cover the requested
episode count, the run fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..observer.ingest import parse_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--skill", required=True)
    parser.add_argument("--episodes", type=int, default=0, help="0 = all available")
    parser.add_argument("--source", default="replay")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    dataset = Path(args.dataset)
    if not dataset.is_file():
        print(f"dataset not found: {dataset}", file=sys.stderr)
        return 2

    report, _ = parse_jsonl(dataset, skill=args.skill, source=args.source)
    episodes = [ep for ep in report.accepted if ep.skill == args.skill]
    if not episodes:
        print(
            f"no episodes for skill {args.skill!r} in {dataset} "
            f"({len(report.accepted)} accepted, {len(report.rejected)} rejected)",
            file=sys.stderr,
        )
        return 3
    if args.episodes and len(episodes) < args.episodes:
        print(
            f"dataset only holds {len(episodes)} episodes for {args.skill!r}, "
            f"{args.episodes} requested",
            file=sys.stderr,
        )
        return 4

    selected = episodes[: args.episodes] if args.episodes else episodes
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for episode in selected:
            handle.write(
                json.dumps(
                    {
                        "record": "episode",
                        "schema_version": 1,
                        "episode_id": episode.id,
                        "skill": episode.skill,
                        "policy_version": episode.policy_version,
                        "source": args.source,
                        "started_at": episode.started_at,
                        "duration_s": episode.duration_s,
                        "success": episode.success,
                        "falls": episode.falls,
                        "torque_peak_nm": episode.torque_peak_nm,
                        "joint_vel_peak_rps": episode.joint_vel_peak_rps,
                        "collisions": episode.collisions,
                        "battery_min_v": episode.battery_min_v,
                        "tracking_error_rad": episode.tracking_error_rad,
                        "notes": f"replayed: {episode.notes}",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    print(f"replayed {len(selected)} episode(s) for {args.skill} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
