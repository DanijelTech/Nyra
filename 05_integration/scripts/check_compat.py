#!/usr/bin/env python3
"""check_compat.py — verify that the Nyra modules still agree about the robot.

Checks the contract that train -> export -> deploy depends on:

  1  rpo_description/urdf/rpo.urdf: 23 revolute joints, known order
  2  rpo_description/mjcf/rpo.xml and UFO's rpo_23dof.xml: same joints, same order
  3  roboparty_train's rpo asset is the rpo_description submodule (byte-identical URDF)
  4  UFO's RPO meshes match rpo_description/meshes (sampled)
  5  robolab RPO_JOINT_ORDER is a permutation of the URDF joints, and roboparty_deploy's
     usd2urdf equals that permutation (policy action order -> URDF order)
  6  deploy policy YAMLs: joint_num 23, 23-entry arrays, referenced ONNX present
  7  deploy robot.yaml: 23 motors across 4 CAN buses
  8  MimicLite atom01 joint list equals the URDF joint list
  9  PartyMimic robot packages (g1mj, rp1) present; rpo reported as missing
 10  UFO configs/robots/rpo.yaml control_joints equals the URDF joint list

Exit code: 0 if no FAIL, 1 otherwise. Writes 06_assets/compat_report.json.

Usage: check_compat.py [--quiet] [--json PATH] [--no-write]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "00_body_base/roboparty_deploy"
DESC = ROOT / "00_body_base/rpo_description"
TRAIN = ROOT / "04_training/roboparty_train"
MOTION = ROOT / "02_motion_infra"
ADV = ROOT / "03_advanced"
ROBOT = "rpo"
ROBOT_DIR = DEPLOY / "src/inference/robots" / ROBOT
EXPECTED_DOF = 23

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
COLOR = {PASS: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m", SKIP: "\033[2m"}
RESET = "\033[0m"


@dataclass
class Check:
    id: str
    title: str
    status: str = SKIP
    detail: str = ""
    data: dict = field(default_factory=dict)


RESULTS: list[Check] = []


def record(check: Check) -> Check:
    RESULTS.append(check)
    return check


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def urdf_joints(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    return [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]


def mjcf_joints(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    names = []
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name and name != "floating_base_joint":
            names.append(name)
    return names


def load_yaml(path: Path) -> dict:
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        raise SystemExit("PyYAML required: pip install pyyaml") from None
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


# --------------------------------------------------------------------------- checks


def check_urdf() -> Check:
    c = Check("urdf", "URDF joint set")
    path = DESC / "urdf/rpo.urdf"
    if not path.is_file():
        c.status, c.detail = FAIL, f"missing {path.relative_to(ROOT)}"
        return c
    joints = urdf_joints(path)
    c.data = {"count": len(joints), "joints": joints}
    if len(joints) == EXPECTED_DOF:
        c.status, c.detail = PASS, f"{len(joints)} revolute joints"
    else:
        c.status, c.detail = FAIL, f"{len(joints)} joints, expected {EXPECTED_DOF}"
    return c


def check_mjcf_order(urdf: list[str]) -> Check:
    c = Check("mjcf", "MJCF joint order matches URDF")
    path = DESC / "mjcf/rpo.xml"
    if not path.is_file():
        c.status, c.detail = FAIL, f"missing {path.relative_to(ROOT)}"
        return c
    joints = mjcf_joints(path)
    if joints == urdf:
        c.status, c.detail = PASS, "mjcf/rpo.xml identical order"
    elif sorted(joints) == sorted(urdf):
        c.status, c.detail = FAIL, "same joints, different order"
    else:
        missing = sorted(set(urdf) - set(joints))
        c.status, c.detail = FAIL, f"joint set differs (missing: {missing or '-'})"
    return c


def check_ufo_mjcf(urdf: list[str]) -> Check:
    c = Check("ufo-mjcf", "UFO rpo_23dof.xml joint order")
    path = MOTION / "UFO/ufo/data/robots/rpo/rpo_23dof.xml"
    if not path.is_file():
        c.status, c.detail = WARN, f"missing {path.relative_to(ROOT)}"
        return c
    joints = mjcf_joints(path)
    c.status = PASS if joints == urdf else FAIL
    c.detail = "identical order" if joints == urdf else "order/set differs from URDF"
    return c


def check_train_asset() -> Check:
    c = Check("train-asset", "roboparty_train uses rpo_description")
    asset = TRAIN / "robolab/data/robots/roboparty/rpo/urdf/rpo.urdf"
    ref = DESC / "urdf/rpo.urdf"
    if not asset.is_file():
        c.status, c.detail = FAIL, f"missing {asset.relative_to(ROOT)} (submodule not initialised?)"
        return c
    if md5(asset) == md5(ref):
        c.status, c.detail = PASS, "URDF byte-identical to rpo_description"
    else:
        c.status, c.detail = FAIL, "URDF differs — the train asset is a stale copy"
    sub = TRAIN / "robolab/data/robots/roboparty/rpo"
    if (sub / ".git").exists():
        c.data["is_submodule"] = True
        c.detail += " (submodule)"
    return c


def check_ufo_meshes() -> Check:
    c = Check("ufo-meshes", "UFO RPO meshes match rpo_description")
    src = DESC / "meshes"
    dst = MOTION / "UFO/ufo/data/robots/rpo/meshes"
    if not src.is_dir() or not dst.is_dir():
        c.status, c.detail = WARN, "mesh directory missing"
        return c
    names = sorted(p.name for p in src.glob("*") if p.suffix.lower() in {".stl", ".obj", ".dae"})
    sample = names[:3] + names[-2:] if len(names) > 5 else names
    bad = [n for n in sample if not (dst / n).is_file() or md5(src / n) != md5(dst / n)]
    c.data = {"meshes": len(names), "sampled": sample}
    c.status = PASS if not bad else FAIL
    c.detail = f"{len(names)} meshes, sampled {len(sample)} identical" if not bad else f"differ: {bad}"
    return c


def parse_train_joint_order() -> list[str] | None:
    path = TRAIN / "robolab/robolab/assets/robots/roboparty.py"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    match = re.search(r"RPO_JOINT_ORDER\s*=\s*\((.*?)\)", text, re.S)
    if not match:
        return None
    return re.findall(r'"([^"]+_joint)"', match.group(1))


def check_action_order(urdf: list[str]) -> list[Check]:
    out: list[Check] = []
    order = parse_train_joint_order()
    c = Check("train-order", "robolab RPO_JOINT_ORDER")
    if order is None:
        c.status, c.detail = FAIL, "RPO_JOINT_ORDER not found in robolab/assets/robots/roboparty.py"
        return [c]
    if sorted(order) != sorted(urdf):
        c.status, c.detail = FAIL, "RPO_JOINT_ORDER is not a permutation of the URDF joints"
        return [c]
    c.status, c.detail = PASS, f"{len(order)} joints, interleaved L/R policy order"
    c.data = {"order": order}
    out.append(c)

    expected = [urdf.index(j) for j in order]
    c2 = Check("deploy-usd2urdf", "deploy usd2urdf matches the training action order")
    cfg = ROBOT_DIR / "configs/default.yaml"
    if not cfg.is_file():
        c2.status, c2.detail = FAIL, f"missing {cfg.relative_to(ROOT)}"
        return out + [c2]
    params = load_yaml(cfg)["inference_node"]["ros__parameters"]
    actual = params.get("usd2urdf")
    c2.data = {"expected": expected, "actual": actual}
    if actual == expected:
        c2.status, c2.detail = PASS, "policy action index -> URDF joint index verified"
    else:
        c2.status = FAIL
        c2.detail = "mismatch — exported actions would drive the wrong joints"
    return out + [c2]


def check_deploy_configs() -> list[Check]:
    out: list[Check] = []
    cfg_dir = ROBOT_DIR / "configs"
    models = ROBOT_DIR / "models"
    if not cfg_dir.is_dir():
        return [Check("deploy-configs", "deploy policy configs", FAIL, f"missing {cfg_dir.relative_to(ROOT)}")]

    for cfg in sorted(cfg_dir.glob("*.yaml")):
        c = Check(f"deploy-{cfg.stem}", f"configs/{cfg.name}")
        try:
            params = load_yaml(cfg)["inference_node"]["ros__parameters"]
        except Exception as exc:  # noqa: BLE001
            c.status, c.detail = FAIL, f"unreadable: {exc}"
            out.append(c)
            continue
        problems = []
        if params.get("joint_num") != EXPECTED_DOF:
            problems.append(f"joint_num={params.get('joint_num')}")
        for key in ("joint_default_angle", "kp", "kd", "usd2urdf"):
            value = params.get(key)
            if isinstance(value, list) and len(value) != EXPECTED_DOF:
                problems.append(f"{key} has {len(value)} entries")
        for name in params.get("model_names") or []:
            if not (models / name).is_file():
                problems.append(f"missing models/{name}")
        for name in params.get("latent_names") or []:
            if not (ROBOT_DIR / "latents" / name).is_file():
                problems.append(f"missing latents/{name}")
        c.status = PASS if not problems else FAIL
        c.detail = "23 DoF, arrays and artefacts consistent" if not problems else "; ".join(problems)
        out.append(c)
    return out


def check_robot_yaml() -> Check:
    c = Check("deploy-robot", "deploy robot.yaml hardware map")
    path = ROBOT_DIR / "robot.yaml"
    if not path.is_file():
        c.status, c.detail = FAIL, f"missing {path.relative_to(ROOT)}"
        return c
    data = load_yaml(path)
    motors = data.get("motors", {})
    ids = motors.get("motor_id") or []
    ifaces = motors.get("motor_interface") or []
    nums = motors.get("motor_num") or []
    urdf2motor = data.get("robot", {}).get("urdf2motor") or []
    problems = []
    if len(ids) != EXPECTED_DOF:
        problems.append(f"{len(ids)} motor ids")
    if len(urdf2motor) != EXPECTED_DOF:
        problems.append(f"urdf2motor has {len(urdf2motor)} entries")
    if len(ifaces) != len(nums):
        problems.append("motor_interface/motor_num length mismatch")
    if sum(nums) != EXPECTED_DOF:
        problems.append(f"motor_num sums to {sum(nums)}")
    c.data = {"can_buses": ifaces, "per_bus": nums}
    c.status = PASS if not problems else FAIL
    c.detail = (f"{sum(nums)} motors on {len(ifaces)} CAN buses" if not problems else "; ".join(problems))
    return c


def check_mimiclite(urdf: list[str]) -> Check:
    c = Check("mimiclite", "MimicLite atom profile joint list")
    path = MOTION / "MimicLite/mimic-lite/mimic_lite/assets/atom.py"
    if not path.is_file():
        c.status, c.detail = WARN, "MimicLite submodule not initialised"
        return c
    text = path.read_text(encoding="utf-8")
    match = re.search(r"ATOM01_JOINT_NAMES\s*=\s*\[(.*?)\]", text, re.S)
    if not match:
        c.status, c.detail = WARN, "ATOM01_JOINT_NAMES not found (upstream renamed it?)"
        return c
    joints = re.findall(r'"([^"]+)"', match.group(1))
    c.status = PASS if joints == urdf else FAIL
    c.detail = "same 23 joints, same order as rpo.urdf" if joints == urdf else "differs from rpo.urdf"
    return c


def check_partymimic() -> Check:
    c = Check("partymimic", "PartyMimic robot packages")
    base = MOTION / "PartyMimic/src/partymimic"
    if not base.is_dir():
        c.status, c.detail = WARN, "PartyMimic missing"
        return c
    packages = sorted(p.name for p in base.iterdir() if p.is_dir())
    if ROBOT in packages:
        c.status, c.detail = PASS, f"packages: {', '.join(packages)} (rpo present)"
    else:
        c.status = WARN
        c.detail = f"packages: {', '.join(packages)} — no 'rpo' package, RPO training needs one (README §6)"
    c.data = {"packages": packages}
    return c


def check_ufo_config(urdf: list[str]) -> Check:
    c = Check("ufo-config", "UFO configs/robots/rpo.yaml control joints")
    path = MOTION / "UFO/configs/robots/rpo.yaml"
    if not path.is_file():
        c.status, c.detail = WARN, "missing UFO RPO robot config"
        return c
    joints = load_yaml(path).get("control_joints") or []
    c.status = PASS if joints == urdf else FAIL
    c.detail = "same 23 joints, same order as rpo.urdf" if joints == urdf else "differs from rpo.urdf"
    return c


def check_pins() -> Check:
    c = Check("pins", "submodule pins recorded")
    import subprocess  # noqa: PLC0415

    out = subprocess.run(["git", "-C", str(ROOT), "submodule", "status"],
                         capture_output=True, text=True).stdout.splitlines()
    if not out:
        c.status, c.detail = WARN, "no submodules reported (is this a git checkout?)"
        return c
    uninit = [line.split()[1] for line in out if line.startswith("-")]
    c.data = {"modules": len(out), "uninitialised": uninit}
    if uninit:
        c.status, c.detail = FAIL, f"{len(uninit)} uninitialised: {', '.join(uninit[:5])}"
    else:
        c.status, c.detail = PASS, f"{len(out)} modules initialised"
    return c


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true", help="only failures and the summary")
    ap.add_argument("--json", type=Path, default=ROOT / "06_assets/compat_report.json")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    urdf_check = record(check_urdf())
    urdf = urdf_check.data.get("joints", [])

    if urdf:
        record(check_mjcf_order(urdf))
        record(check_ufo_mjcf(urdf))
        for c in check_action_order(urdf):
            record(c)
        record(check_mimiclite(urdf))
        record(check_ufo_config(urdf))
    record(check_train_asset())
    record(check_ufo_meshes())
    for c in check_deploy_configs():
        record(c)
    record(check_robot_yaml())
    record(check_partymimic())
    record(check_pins())

    for c in RESULTS:
        if args.quiet and c.status in (PASS, SKIP):
            continue
        print(f"{COLOR[c.status]}{c.status:4}{RESET} {c.id:22} {c.title}: {c.detail}")

    counts = {s: sum(1 for c in RESULTS if c.status == s) for s in (PASS, WARN, FAIL, SKIP)}
    print()
    print(f"{counts[PASS]} pass, {counts[WARN]} warn, {counts[FAIL]} fail, {counts[SKIP]} skipped")

    if not args.no_write:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "robot": ROBOT,
            "expected_dof": EXPECTED_DOF,
            "summary": counts,
            "checks": [
                {"id": c.id, "title": c.title, "status": c.status, "detail": c.detail, "data": c.data}
                for c in RESULTS
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"report: {args.json.relative_to(ROOT)}")

    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
