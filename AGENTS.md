# AGENTS.md — repository memory

Notes for anyone (human or agent) working in this workspace. Read the top-level `README.md` first;
this file records what is easy to get wrong.

## What this repository is

An integration monorepo for the Nyra robot (ROBOTO_ORIGIN / RPO software lineage). 21 upstream
repositories are git submodules pinned to tested commits. First-party code lives in exactly two
places: `05_integration/` (install / launch / export / verify) and `06_evolayer/` (the evolutionary
loop). `00_body_base/roboto_origin` is a ~4 GB **read-only snapshot** of the same modules — never
edit it, never treat it as the source of truth.

`06_evolayer/` is the only component allowed to *change* the deployed stack, and it does so only
after a rehearsal, a safety gate and a recorded Hermes approval. Its invariants (no self-approval,
no shell, allowlisted paths, verified snapshots, hash-chained audit) are covered by
`06_evolayer/tests/` — run `cd 06_evolayer && python3 -m pytest -q` after touching it.

## Hard rules

1. **Do not fork upstream code in place.** A fix belongs in the module's own repository (branch +
   PR), then the pin is bumped here. Local experiments stay on a branch inside the submodule.
2. **Never modify `rpo_description` copies.** `roboparty_train/robolab/data/robots/roboparty/rpo`
   is a submodule of the same repo; UFO keeps its own copy of MJCF + meshes. Editing a copy
   silently desynchronises train and deploy. `check_compat.py` detects it.
3. **Run `05_integration/scripts/check_compat.py` after any submodule bump.** It is the definition
   of "still compatible".
4. **Do not commit generated artefacts**: `06_assets/*`, `04_training/IsaacLab-3.0.0`, `.venv*`.
   They are git-ignored; keep it that way.

## Facts that matter (verified, do not "clean up")

- The robot has **23 revolute joints**. URDF order: left leg 6, right leg 6, torso, left arm 5, right
  arm 5.
- `RPO_JOINT_ORDER` in `roboparty_train/robolab/robolab/assets/robots/roboparty.py` is a *different*
  (interleaved L/R) order. Deploy converts with `usd2urdf` in
  `roboparty_deploy/src/inference/robots/rpo/configs/default.yaml`. UFO policies use URDF order
  (identity mapping in `configs/ufo.yaml`).
- MimicLite's `atom` profile == Atom01 == the RPO lineage: same 23 joints, same order.
- PartyMimic has **no RPO robot package** (`g1`, `g1mj`, `rp1` only) — RPO needs one added under
  `src/partymimic/rpo/`.
- any4hdmi's converters hardcode a 29-joint G1 order (`src/any4hdmi/core/model.py`), so RPO motion
  through any4hdmi needs a one-constant patch plus
  `--mjcf-repo elijahgalahad/atom01 --mjcf-path mjcf/atom01-capsule.xml`.
- MimicLite's submodules are declared over SSH; the workspace maps them to HTTPS in
  `.git/modules/02_motion_infra/MimicLite/config`. MimicLite also expects
  `active-adaptation/projects/mimic-lite`, which is a symlink here (`setup_all.sh` creates it).
- Licences are mixed, and three modules are **CC BY-NC 4.0 (non-commercial)**: UFO, I-BFM-Code,
  P-BFM. Deploy is GPL-3.0. See README §11 before proposing anything commercial.

## Commands

```bash
05_integration/scripts/check_compat.py                    # geometry / joint order / ONNX contract
05_integration/scripts/setup_all.sh --check               # toolchain + env report
05_integration/scripts/setup_all.sh --motion --dry-run    # inspect before installing
05_integration/scripts/launch_base.sh --preflight-only    # robot-ready checks
05_integration/scripts/export_policy.sh --source train --run <dir> --task RPO-AMP-Play --name amp --verify
05_integration/scripts/retarget_and_train.sh --input <motion> --target ufo --dry-run
05_integration/scripts/update_all.sh --check              # pin drift, changes nothing
```

Style: `bash -n` + `shellcheck -S warning -x` must be clean; scripts support `--dry-run` and
`--help`; failures are collected and summarised rather than aborting the whole run.

## Environment notes (as of the last verified checkout)

- Dev boxes here have no ROS 2, no CUDA, no `uv`-synced environments: `--check` output showing
  warnings is normal, and `launch_base.sh --preflight-only` legitimately exits 1 off-robot.
- `pyyaml` is needed by `check_compat.py` (`pip install pyyaml`); `onnx` is optional and only used
  by `export_policy.sh --verify`.
- Nested submodule depth: deploy (5), train (robolab + rsl_rl + 2 robot assets), MimicLite (3),
  firmware (3), inference (1), navigation (2).
- EvoLayer rehearsal needs `mujoco`, `numpy`, `onnxruntime` (installed here); `onnx` is used by the
  fixture-policy builder in tests; `imageio` + `MUJOCO_GL` are needed for video artifacts.
- EvoLayer state lives in `06_assets/evolayer/` (git-ignored): `state.db`, `audit.jsonl`,
  `snapshots/`, `artifacts/`, `sim/`. Wipe it for a clean loop.
