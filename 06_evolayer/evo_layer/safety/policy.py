"""Deploy-contract checks for a candidate policy.

A rehearsal in the wrong coordinate frame is worse than no rehearsal: it produces
confident, wrong numbers. These checks tie the ONNX artifact to the robot the way
``roboparty_deploy`` does at runtime:

* the URDF joint count matches the policy's action dimension;
* ``usd2urdf`` in the policy YAML is a permutation of ``0..N-1``;
* if ``RPO_JOINT_ORDER`` is present in the trainer, ``usd2urdf`` equals the
  permutation it implies (this is the link between training and deployment);
* every artifact referenced by the YAML exists and is non-empty;
* when ``onnxruntime`` is installed, the graph loads and its output width matches.

A missing optional dependency is reported as ``skip``, never as ``pass``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PASS, FAIL, SKIP = "pass", "fail", "skip"


@dataclass
class ContractCheck:
    name: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def parse_rpo_joint_order(path: str | Path) -> list[str] | None:
    file = Path(path)
    if not file.is_file():
        return None
    text = file.read_text(encoding="utf-8")
    match = re.search(r"RPO_JOINT_ORDER\s*=\s*\((.*?)\)", text, re.S)
    if not match:
        return None
    return re.findall(r'"([^"]+_joint)"', match.group(1))


def check_policy_contract(
    *,
    urdf_joints: list[str],
    policy_yaml: Path,
    models_dir: Path,
    latents_dir: Path | None = None,
    onnx_path: Path | None = None,
    rpo_joint_order_file: Path | None = None,
) -> list[ContractCheck]:
    checks: list[ContractCheck] = []
    n = len(urdf_joints)

    if not policy_yaml.is_file():
        return [ContractCheck("policy_yaml", FAIL, f"missing {policy_yaml}")]
    try:
        data = yaml.safe_load(policy_yaml.read_text(encoding="utf-8"))
        params = data["inference_node"]["ros__parameters"]
    except (yaml.YAMLError, KeyError, TypeError) as exc:
        return [ContractCheck("policy_yaml", FAIL, f"unreadable policy YAML: {exc}")]
    checks.append(ContractCheck("policy_yaml", PASS, str(policy_yaml)))

    joint_num = params.get("joint_num")
    checks.append(
        ContractCheck(
            "joint_num",
            PASS if joint_num == n else FAIL,
            f"joint_num={joint_num}, URDF has {n} actuated joints",
        )
    )

    mapping = params.get("usd2urdf")
    if not isinstance(mapping, list):
        checks.append(ContractCheck("usd2urdf", FAIL, "usd2urdf missing or not a list"))
    elif sorted(mapping) != list(range(n)):
        checks.append(ContractCheck("usd2urdf", FAIL, f"usd2urdf is not a permutation of 0..{n - 1}"))
    else:
        checks.append(ContractCheck("usd2urdf", PASS, f"valid permutation of {n} joints"))

    for key in ("joint_default_angle", "kp", "kd", "action_scale"):
        value = params.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            checks.append(ContractCheck(key, FAIL, f"{key} must be a list"))
        elif len(value) not in (n, 1):
            checks.append(ContractCheck(key, FAIL, f"{key} has {len(value)} entries, expected {n} or 1"))
        else:
            checks.append(ContractCheck(key, PASS, f"{len(value)} entries"))

    order_file = Path(rpo_joint_order_file) if rpo_joint_order_file else None
    order = parse_rpo_joint_order(order_file) if order_file else None
    if order is None:
        checks.append(ContractCheck("train_action_order", SKIP, "RPO_JOINT_ORDER not found; skipped"))
    elif isinstance(mapping, list) and len(order) == n and set(order) == set(urdf_joints):
        expected = [urdf_joints.index(joint) for joint in order]
        checks.append(
            ContractCheck(
                "train_action_order",
                PASS if mapping == expected else FAIL,
                "usd2urdf matches the trainer's action order"
                if mapping == expected
                else "usd2urdf does not match RPO_JOINT_ORDER -> URDF permutation",
            )
        )
    else:
        checks.append(
            ContractCheck("train_action_order", FAIL, "RPO_JOINT_ORDER is not a permutation of the URDF joints")
        )

    for name in params.get("model_names") or []:
        candidate = Path(models_dir) / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            checks.append(ContractCheck(f"model:{name}", PASS, f"{candidate.stat().st_size} bytes"))
        else:
            checks.append(ContractCheck(f"model:{name}", FAIL, f"missing or empty: {candidate}"))

    for name in params.get("latent_names") or []:
        if latents_dir is None:
            checks.append(ContractCheck(f"latent:{name}", FAIL, "policy expects latents but none configured"))
            continue
        candidate = Path(latents_dir) / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            checks.append(ContractCheck(f"latent:{name}", PASS, f"{candidate.stat().st_size} bytes"))
        else:
            checks.append(ContractCheck(f"latent:{name}", FAIL, f"missing or empty: {candidate}"))

    if onnx_path is not None:
        if not onnx_path.is_file():
            checks.append(ContractCheck("onnx_graph", FAIL, f"missing {onnx_path}"))
        else:
            checks.append(_inspect_onnx(onnx_path, n))

    return checks


def _inspect_onnx(path: Path, expected_dim: int) -> ContractCheck:
    try:
        import onnxruntime  # noqa: PLC0415
    except ImportError:
        return ContractCheck("onnx_graph", SKIP, "onnxruntime not installed; graph not inspected")
    try:
        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        outputs = session.get_outputs()
        if not outputs:
            return ContractCheck("onnx_graph", FAIL, "graph has no outputs")
        shape = outputs[0].shape
        width = shape[-1] if shape else None
        if isinstance(width, int) and width != expected_dim:
            return ContractCheck(
                "onnx_graph", FAIL, f"output width {width} != {expected_dim} joints ({outputs[0].name})"
            )
        return ContractCheck("onnx_graph", PASS, f"loaded; output {outputs[0].name} shape {shape}")
    except Exception as exc:  # noqa: BLE001 - ort raises a wide variety of errors
        return ContractCheck("onnx_graph", FAIL, f"failed to load: {exc}")
