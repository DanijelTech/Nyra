# Troubleshooting

Symptoms are grouped by where they bite. Every section starts with the cheapest diagnostic.

## Nothing builds / modules look empty

```bash
git submodule status --recursive | grep '^-'      # uninitialised
05_integration/scripts/setup_all.sh --check
```

- `00_body_base/roboparty_deploy/src/inference` empty → `git -C 00_body_base/roboparty_deploy submodule update --init --recursive`
- `04_training/roboparty_train/robolab` empty → same, inside `roboparty_train`
- `02_motion_infra/MimicLite/mimic-lite` empty → the nested submodules use SSH URLs:

  ```bash
  git -C 02_motion_infra/MimicLite config url."https://github.com/".insteadOf "git@github.com:"
  git -C 02_motion_infra/MimicLite submodule update --init --recursive
  ```

- `git -C <module> submodule sync` after changing remotes, never `submodule sync` *after* setting a
  URL override in `.git/config` (sync overwrites it from `.gitmodules`).

## colcon build fails in roboparty_deploy

- `fmt/spdlog/eigen` not found → `sudo apt install libfmt-dev libspdlog-dev libeigen3-dev`
- Wrong ROS: `start_robot.sh` supports Humble and Jazzy only; check `echo $ROS_DISTRO`
- Mixed Python: the ROS install and the shell's `python3` must share a minor version (3.10 for
  Humble on Ubuntu 22.04). A stray conda env is the usual culprit — `which python3` in a fresh shell.
- Build only the inference package (what `start_robot.sh` does):
  `colcon build --symlink-install --base-paths src --packages-up-to roboparty_inference`

## Robot starts but does not move / behaves violently

Stop immediately (`launch_base.sh --stop`) and work down this list:

1. **Zeroing** — was `/init_motors` + `/set_zeros` run, and the motors power-cycled afterwards?
   Wrong zero points look like "one joint is always offset".
2. **`motor_zero_offset`** — the waist value (`2.093`) must match the calibration method used
   (limit block vs 3D-printed fixture → `0.0`). See `docs/rpo_deploy.md`.
3. **Action order** — a policy trained with `roboparty_train` needs `usd2urdf` interleaved
   (`configs/default.yaml`); a UFO policy needs the identity mapping (`configs/ufo.yaml`). The
   symptom is joints moving in the wrong axis/limb. `check_compat.py` catches the default profile.
4. **Observation layout** — `obs_layouts` must match what the policy saw during training
   (history length, `frame_stacks`, scales). Wrong scale looks like "it moves but drifts".
5. **Parkour without depth** — `parkour` needs the camera node (`--camera`), otherwise `/depth_obs`
   is empty and the policy outputs garbage.
6. **CAN mapping** — `can0` left leg, `can1` right leg + waist, `can2` left hand, `can3` right hand.
   Swapped buses look like "half the robot twitches".

## CAN / IMU

```bash
ip -br link | grep can                                  # interfaces present?
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can0 txqueuelen 1000
ls -l /dev/ttyUSB0                                      # IMU
```

- Interface numbering follows USB enumeration unless udev rules pin them. Install
  `assets/99-auto-up-devs-orangepi.rules` (or `-sunrise.rules` for RDK X5) and reboot.
- No interfaces at all → `roboto_usb2can` firmware/kernel module missing
  (`00_body_base/roboparty_firmware/roboto_usb2can`).
- `ulimit -r` below 98 → real-time limits not applied (edit `/etc/security/limits.conf`, reboot).

## uv / Python environments

- `uv: command not found` → `python3 -m pip install --user uv`, then `export PATH="$HOME/.local/bin:$PATH"`.
- `uv sync --locked` fails with a resolution error → the lockfile targets a specific Python
  (3.10/3.12). Install that minor version; do not let uv pick the system one silently.
- `No module named isaaclab` → `04_training/IsaacLab-3.0.0` is missing or not linked; `pyproject.toml`
  resolves `../IsaacLab-3.0.0` from `roboparty_train`.
- PhysX and MJWarp do not share a venv. `.venv` = MJWarp, `.venv-physx` = PhysX; activate the right
  one and add `physics=isaacsim_physx` for the latter.
- MimicLite: `aa-discover-projects` finds nothing → the project must live at
  `active-adaptation/projects/mimic-lite`. `setup_all.sh` creates that symlink; do it by hand if you
  installed without the script:
  `ln -s ../../mimic-lite 02_motion_infra/MimicLite/active-adaptation/projects/mimic-lite`.

## Training

- `isaaclab list_envs --keyword RP` prints nothing → the extension is not installed in the active
  venv, or you are in the wrong venv.
- Task not found (`RPO-AMP-Play`, `RPO-BeyondMimic`, `RPO-Parkour-Play`, `RPO-Getup-Mimic`,
  `RP1-*`) → copy the exact id from `list_envs`.
- Out of VRAM: lower `--num_envs`; PhysX is known to slow down abnormally in some scenes — that is
  upstream behaviour, not your config.
- UFO: `rpo` data download fails → `HF_TOKEN` missing or the licence not accepted on Hugging Face.
- UFO: `--robot` and `--data-manifest` are mutually exclusive; choose one.
- P-BFM / I-BFM want 8 GPUs and hundreds of GB of replay; a single-GPU box will OOM at startup, not
  later. Use `--smoke` where upstream provides it.

## Export → deploy

```bash
05_integration/scripts/export_policy.sh --source <src> --run <run> --name <n> --verify
```

- No ONNX produced: `play.py` writes to **`<run>/exported/`** next to the checkpoint. If you passed a
  checkpoint from another directory, look there.
- Parkour: `0-depth_encoder.onnx` and `actor.onnx` each have a sibling `.onnx.data`. Copy both, or
  the ONNX fails to load with a cryptic protobuf error.
- `--verify` reports "last dim != 23": the policy was exported for another robot (RP1 is 24/28 DoF).
- Model loads but the robot is still: check `model_names` in the policy YAML and that the file is in
  `src/inference/robots/<robot>/models/`, not in `configs/`.
- UFO: keep `.onnx`, `.meta.json` and `.deploy.yaml` together; the metadata is bound to the ONNX hash
  and deploy needs it to build observations.

## Motion data

- hhtools `retarget run` needs a calibration reference for SMPL-family inputs
  (`--calibration-reference smpl`); the first run in the WebUI can calibrate and save
  `retarget_calibration_<ref>.yaml` next to the URDF.
- `convert_motion.py` says "Export CSV from HHTools with comments and column headers enabled" — the
  CSV was written without a header, so joint names are unknown.
- any4hdmi rejects the CSV width: its converters hardcode a 29-joint G1 order. For RPO, patch
  `src/any4hdmi/core/model.py` (`G1_JOINT_ORDER`) to the 23 RPO joints, then pass
  `--mjcf-repo elijahgalahad/atom01 --mjcf-path mjcf/atom01-capsule.xml`.
- PartyMimic: `--robot rpo` fails because no such package exists; `retarget_and_train.sh` prints the
  scaffold contract (MJCF, motion dir, actuators, default pose, tracked bodies).
- Datasets are licensed separately (LAFAN1 CC BY-NC-ND, BONES-SEED gated, MotionDecode
  non-commercial). Training on them constrains what you may redistribute.

## Git / submodules

- After `update_all.sh`, `git status` shows modified submodules: that is the point — commit the new
  SHAs.
- A submodule shows `+` in `git submodule status`: its HEAD differs from the recorded pin (local
  commits or a manual checkout). Commit inside the module and bump, or reset it.
- Never `git commit` inside this repo with a dirty submodule working tree unless you intend to
  release that pin; `update_all.sh --check` lists dirty modules.
