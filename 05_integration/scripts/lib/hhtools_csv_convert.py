#!/usr/bin/env python3
"""Convert an hhtools retarget CSV into the layouts other trainers expect.

hhtools writes:  time, root_x, root_y, root_z, root_qx, root_qy, root_qz, root_qw,
                 dof_<joint>[, ...]           (quaternion is xyzw)

Targets:
  ufo        time, root_pos_{x,y,z}, root_quat_{x,y,z,w}, <joint>[, ...]
             (UFO's robot_state_csv reader: root_* column names, joints by name)
  any4hdmi   no header, root xyz + quat xyzw + joints   (the lafan loader: 7 + N columns)
  roboparty  time + root xyz + quat xyzw + dof_<joint>  (drop-in for convert_motion.py, reordered)

Usage:
    hhtools_csv_convert.py --target ufo       -i walk.csv -o walk_ufo.csv
    hhtools_csv_convert.py --target any4hdmi  -i walk.csv -o walk.npz.csv --urdf rpo.urdf
    hhtools_csv_convert.py --target ufo       -i walk.csv -o out.csv --check-only
"""

from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT_POS = ["root_x", "root_y", "root_z"]
ROOT_QUAT = ["root_qx", "root_qy", "root_qz", "root_qw"]


def read_hhtools_csv(path: Path) -> tuple[list[str], list[list[str]], dict[str, str]]:
    meta: dict[str, str] = {}
    rows: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line in stream:
            if line.startswith("#"):
                key, sep, value = line[1:].partition(":")
                if sep:
                    meta[key.strip()] = value.strip()
            elif line.strip():
                rows.append(line)
    if not rows:
        raise SystemExit(f"{path}: no data rows")
    reader = csv.reader(rows)
    header = next(reader)
    if len(header) != len(set(header)):
        raise SystemExit(f"{path}: duplicate column names in header")
    missing = [c for c in ("time", *ROOT_POS, *ROOT_QUAT) if c not in header]
    if missing:
        raise SystemExit(
            f"{path}: missing {missing}. Re-export from hhtools with comments and column headers enabled."
        )
    return header, list(reader), meta


def urdf_joint_order(urdf: Path) -> list[str]:
    root = ET.parse(urdf).getroot()
    names = []
    for joint in root.findall("joint"):
        if joint.get("type") != "fixed":
            names.append(joint.get("name", ""))
    return names


def joint_columns(header: list[str]) -> list[str]:
    return [c for c in header if c.startswith("dof_")]


def convert(path: Path, target: str, out: Path, order: list[str] | None) -> tuple[list[str], list[str]]:
    header, rows, meta = read_hhtools_csv(path)
    dof_cols = joint_columns(header)
    source_joints = [c[len("dof_"):] for c in dof_cols]

    if order:
        missing = [j for j in order if j not in source_joints]
        extra = [j for j in source_joints if j not in order]
        if missing or extra:
            raise SystemExit(
                f"{path.name}: joint set mismatch with the target order.\n"
                f"  missing in CSV: {missing or '-'}\n  not in target: {extra or '-'}"
            )
        joints = order
    else:
        joints = source_joints

    idx = {name: i for i, name in enumerate(header)}
    pos_idx = [idx[c] for c in ROOT_POS]
    quat_idx = [idx[c] for c in ROOT_QUAT]
    time_idx = idx["time"]

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if target == "ufo":
            writer.writerow(
                ["time", "root_pos_x", "root_pos_y", "root_pos_z",
                 "root_quat_x", "root_quat_y", "root_quat_z", "root_quat_w", *joints]
            )
            for row in rows:
                writer.writerow([row[time_idx], *(row[i] for i in pos_idx), *(row[i] for i in quat_idx),
                                 *(row[idx[f"dof_{j}"]] for j in joints)])
        elif target == "any4hdmi":
            for row in rows:
                writer.writerow([*(row[i] for i in pos_idx), *(row[i] for i in quat_idx),
                                 *(row[idx[f"dof_{j}"]] for j in joints)])
        elif target == "roboparty":
            writer.writerow(["time", *ROOT_POS, *ROOT_QUAT, *(f"dof_{j}" for j in joints)])
            for row in rows:
                writer.writerow([row[time_idx], *(row[i] for i in pos_idx), *(row[i] for i in quat_idx),
                                 *(row[idx[f"dof_{j}"]] for j in joints)])
        else:
            raise SystemExit(f"unknown target: {target}")

    fps = meta.get("fps") or meta.get("sample_rate") or meta.get("fps_out")
    if fps:
        print(f"  hhtools metadata: fps={fps}")
    return source_joints, joints


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, choices=["ufo", "any4hdmi", "roboparty"])
    ap.add_argument("-i", "--in", dest="src", required=True, type=Path)
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--urdf", type=Path, help="order the joints like this URDF (non-fixed joints)")
    ap.add_argument("--joint-order", help="comma-separated joint order override")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    if not args.src.is_file():
        raise SystemExit(f"not a file: {args.src}")

    order: list[str] | None = None
    if args.joint_order:
        order = [j.strip() for j in args.joint_order.split(",") if j.strip()]
    elif args.urdf:
        order = urdf_joint_order(args.urdf)

    if args.check_only:
        header, rows, _ = read_hhtools_csv(args.src)
        joints = joint_columns(header)
        print(f"{args.src.name}: {len(rows)} frames, {len(joints)} joints")
        print("  " + " ".join(j[len('dof_'):] for j in joints))
        if order:
            same = order == [j[len("dof_"):] for j in joints]
            print(f"  order matches target: {same}")
        return 0

    if not args.out:
        raise SystemExit("-o/--out is required unless --check-only")

    source_joints, out_joints = convert(args.src, args.target, args.out, order)
    print(f"{args.src.name} -> {args.out} ({args.target}, {len(out_joints)} joints)")
    if source_joints != out_joints:
        print("  joint order was permuted to match the target")
    return 0


if __name__ == "__main__":
    sys.exit(main())
