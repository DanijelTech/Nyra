#!/usr/bin/env python3
"""Print ONNX graph inputs/outputs and sanity-check them against the RPO deploy layout.

Usage:
    python3 onnx_info.py model.onnx [--dof 23] [--json]

Needs `onnx` (or `onnxruntime`). The caller may run it through
`uv run --with onnx python onnx_info.py ...` when the package is absent.
"""

from __future__ import annotations

import argparse
import json
import sys


def dims_of(value_info) -> list[str]:
    shape = value_info.type.tensor_type.shape
    out: list[str] = []
    for dim in shape.dim:
        if dim.HasField("dim_value"):
            out.append(str(dim.dim_value))
        elif dim.HasField("dim_param") and dim.dim_param:
            out.append(dim.dim_param)
        else:
            out.append("?")
    return out


def describe(path: str) -> dict:
    import onnx  # noqa: PLC0415

    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    return {
        "file": path,
        "opset": [f"{o.domain or 'ai.onnx'}:{o.version}" for o in model.opset_import],
        "ir_version": model.ir_version,
        "inputs": [{"name": v.name, "shape": dims_of(v)} for v in graph.input],
        "outputs": [{"name": v.name, "shape": dims_of(v)} for v in graph.output],
        "nodes": len(graph.node),
        "external_data": any(t.data_location == 1 for t in graph.initializer),
    }


def check(info: dict, dof: int) -> list[str]:
    """Cheap contract checks against what roboparty_deploy expects."""
    notes: list[str] = []
    for item in info["outputs"]:
        shape = item["shape"]
        if shape and shape[-1].isdigit() and int(shape[-1]) != dof:
            notes.append(
                f"output '{item['name']}' last dim {shape[-1]} != {dof} joints — "
                "check the robot profile / joint count"
            )
    if info["external_data"]:
        notes.append("weights are external: ship the .onnx.data sidecar with the .onnx file")
    return notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--dof", type=int, default=23, help="expected action dimension (RPO = 23)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        info = describe(args.model)
    except ImportError:
        print("onnx package not available (pip install onnx, or: uv run --with onnx ...)", file=sys.stderr)
        return 2

    notes = check(info, args.dof)
    if args.json:
        print(json.dumps({**info, "notes": notes}, indent=2))
        return 0

    print(f"{info['file']}")
    print(f"  opset {', '.join(info['opset'])}  ir {info['ir_version']}  nodes {info['nodes']}")
    for label in ("inputs", "outputs"):
        for item in info[label]:
            print(f"  {label[:-1]:6} {item['name']:24} {item['shape']}")
    for note in notes:
        print(f"  note   {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
