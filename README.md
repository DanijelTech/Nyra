# Nyra — ROBOTO ORIGIN / RPO software monorepo

One workspace for everything between the metal and the agents: hardware description, ROS 2
deployment, RL training, motion retargeting, motion tracking, behavioural foundation models and
world models.

Nyra is our robot; its software lineage is **ROBOTO_ORIGIN / RPO** (RoboParty, 23 DoF, `rpo` in
every config). All upstream code is vendored here as **git submodules pinned to a tested commit**,
plus an integration layer (`05_integration/`) that installs, verifies, launches and exports across
module boundaries. Nothing upstream is forked — local work goes on branches inside the submodule
that owns it.

---

## Kratek povzetek (SL)

- **Eno okolje, štiri plasti:** `00_body_base` (hardver + ROS 2 deploy) → `02_motion_infra`
  (retargeting + motion tracking) → `04_training` (Isaac Lab RL) → `03_advanced` (BFM, world model,
  representation learning). `01_roadmap/party-os` je referenčni roadmap celotnega stacka.
- **Vrstni red:** najprej `00_body_base` (deploy mora stati in hoditi), šele nato motion infra in
  training. Vsaka plast predpostavlja prejšnjo.
- **Ključna povezava:** `rpo_description` (URDF/MJCF, 23 sklepov) je *isti* asset, ki ga berejo
  trening (`roboparty_train` ga ima kot submodul), UFO in deploy. To preverja
  `05_integration/scripts/check_compat.py`.
- **Skripti:** `setup_all.sh`, `launch_base.sh`, `export_policy.sh`, `retarget_and_train.sh`,
  `update_all.sh` v `05_integration/scripts/`.
- **Opomba:** UFO, I-BFM in P-BFM so licencirani **CC BY-NC 4.0** (samo nekomercialno). Preberi
  razdelek [Licence](#11-licenses), preden karkoli od tega uporabiš v produktu.

---

## 1. What this repository is

| | |
|---|---|
| **Purpose** | Single checkout where hardware, deployment, training and advanced policies can be developed and tested together |
| **Robot** | ROBOTO_ORIGIN / RPO, 23 DoF (legs 12, waist 1, arms 10), `base_link` root |
| **Robot runtime** | ROS 2 Humble/Jazzy on Orange Pi 5 Plus or RDK X5, C++17 inference node, ONNX policies |
| **Training runtime** | Isaac Lab 3.0 + RSL-RL (PhysX or MJWarp), or mjlab / MuJoCo Warp for the model-based trackers |
| **Integration layer** | `05_integration/` — scripts, compatibility checks, docs |
| **Upstream** | 21 repositories as pinned submodules (see `.gitmodules`) |

Two rules keep this workable:

1. **Submodules are read-only.** Change code inside the owning repository (branch + PR there), then
   bump the pin here. `05_integration/scripts/update_all.sh` does the bumping.
2. **Compatibility is verified, not assumed.** `check_compat.py` checks the joint order, the URDF
   and the ONNX contract that train → export → deploy depend on.

---

## 2. Repository map

```
Nyra/
├── 00_body_base/                    # Tier 0 — body, firmware, base control
│   ├── roboto_origin/               # snapshot of all modules (read-only reference, ~4 GB)
│   ├── rpo_hardware/                # CAD, PCB, BOM, manufacturing (CERN-OHL-W-2.0)
│   ├── rpo_description/             # URDF + MJCF + meshes — the single source of robot geometry
│   ├── roboparty_deploy/            # ROS 2 deployment: motors, IMU, inference, camera
│   ├── roboparty_firmware/          # USB2CAN firmware, OrangePi / RDK X5 image builds
│   ├── roboparty_inference/         # ONNX inference core (submodule of deploy)
│   ├── roboparty_camera/            # RealSense + depth encoder ROS 2 package
│   ├── roboparty_navigation/        # Nav2 + localization
│   └── roboparty_xr_teleop/         # PICO VR teleoperation (arms)
├── 01_roadmap/
│   └── party-os/                    # the official stack roadmap this layout follows
├── 02_motion_infra/                 # Tier 1 — motion data and tracking
│   ├── human-humanoid-tools/        # hhtools: retargeting, datasets, WebUI/CLI/MCP
│   ├── any4hdmi/                    # qpos motion format + CSV converters (MimicLite data)
│   ├── MimicLite/                   # supervised tracking, PPO + PPO-ROA (model-free)
│   ├── PartyMimic/                  # supervised tracking, model-based (minutes on one GPU)
│   └── UFO/                         # unsupervised RL, FB / TeCH
├── 03_advanced/                     # Tier 2 — interaction, perception, world models
│   ├── PredActor/                   # action diffusion, steerable onboard control
│   ├── I-BFM-Code/                  # interaction BFM: carry / push / kick
│   ├── P-BFM/                       # perceptive BFM: terrain-adaptive
│   ├── INTACT-JEPA/                 # world model (research preview)
│   └── SADL/                        # representation learning loss
├── 04_training/
│   └── roboparty_train/             # Isaac Lab workspace: robolab + rsl_rl submodules
├── 05_integration/
│   ├── scripts/                     # setup_all, launch_base, export_policy, retarget_and_train, update_all, check_compat
│   └── docs/                        # compatibility contract, troubleshooting
├── 06_evolayer/                     # EvoLayer (Hermes-Evo): observe → propose → rehearse → gate → Hermes → deploy
├── tools/build/                     # nyra-build: one command that builds the workspace (npm package)
└── 06_assets/                       # generated reports, EvoLayer state, build manifests — not tracked
```

Layer names match [Party OS](01_roadmap/party-os/README.md), so upstream docs and this layout use
the same vocabulary.

---

## 3. Requirements

### 3.1 Robot side (real hardware)

| Item | Value |
|---|---|
| Controller | Orange Pi 5 Plus (Ubuntu 22.04, 5.10 RT kernel) or RDK X5 (Ubuntu 22.04, 6.1.83) |
| Buses | 4× USB-CAN @ 1 Mbit/s: `can0` left leg, `can1` right leg + waist, `can2` left hand, `can3` right hand |
| IMU | HiPNUC over serial, 921600 baud, ≥200 Hz (default 500 Hz) |
| Motors | Damiao (DM) series, IDs per the RPO assembly manual |
| Real-time | `rtprio 98`, `memlock unlimited` in `/etc/security/limits.conf` |
| ROS 2 | Humble or Jazzy |

### 3.2 Workstation / training side

| Layer | Minimum | Comfortable |
|---|---|---|
| Motion infra (hhtools, PartyMimic, MimicLite, UFO) | 1× NVIDIA GPU, CUDA 12.8 driver, 24 GB VRAM | RTX 4090 / L40S |
| `roboparty_train` | 1× GPU, Isaac Lab 3.0 + Isaac Sim 6.2 **or** MJWarp | 8× GPU for full runs |
| Advanced BFMs (I-BFM, P-BFM) | 8× GPU, large host RAM + disk (I-BFM replay ≈ 210 GiB) | H20 / H100 node |
| Disk | ~12 GB for this checkout (with the `roboto_origin` snapshot), + tens of GB per motion dataset | |

### 3.3 Software stack by tier

| Tier | Software |
|---|---|
| 0 — base | Ubuntu 22.04, ROS 2 Humble/Jazzy, C++17, `ccache libfmt-dev libspdlog-dev libeigen3-dev screen zlib1g-dev`, `ros-<distro>-joy`, `python3-yaml python3-numpy` |
| 1 — motion | Python 3.10–3.12, [uv](https://docs.astral.sh/uv/), MuJoCo / MuJoCo Warp (mjlab), CUDA 12.8, `ffmpeg`, Hugging Face token (`HF_TOKEN`) for gated datasets |
| 1b — training | Python 3.12, Isaac Lab 3.0.0 checkout at `04_training/IsaacLab-3.0.0`, Isaac Sim 6.2 (PhysX) or Newton 1.6.1 (MJWarp) |
| 2 — advanced | 8× GPU + NCCL, Git LFS (I-BFM), W&B account (optional) |

Python versions differ per module on purpose: deploy/ROS is 3.10 (system), `roboparty_train` is 3.12,
hhtools/UFO/P-BFM use their own `uv` environments. Never share one virtualenv across tiers.

---

## 3b. Build everything with one command

```bash
npm install            # zero dependencies: nothing is downloaded
npm run build          # submodules → assets → deploy → contract → tests → manifest
npm run status         # what the last build did, stage by stage
npm run doctor         # toolchain, capabilities, modules, EvoLayer self-check
```

`npm run build` runs every stage this machine supports and **skips the rest with a
reason** (`deploy` without ROS 2, `training` without Isaac Lab).

One-time, so the rehearsal tests have MuJoCo — the builder remembers the interpreter:

```bash
python3 -m venv ~/.venvs/nyra
~/.venvs/nyra/bin/pip install pyyaml pytest mujoco numpy onnxruntime onnx
npx nyra-build use-python ~/.venvs/nyra/bin/python
```

Python environments are opt-in because they download ~25 GB:

```bash
npm run build:envs     # + motion (~15 GB) and training (~10 GB) environments
```

The builder checks free disk space first and refuses rather than filling the disk.
Every run writes `06_assets/build/manifest.json` (git versions, toolchain,
per-stage status and reason) and appends to `manifest-history.jsonl`. Details and
all flags: [`tools/build/README.md`](tools/build/README.md).

Without Node: `python3 tools/build/nyra_build.py build` does exactly the same.

## 4. Setup order

Do the tiers in order. Tier 0 must walk before Tier 1 trains anything.

### Step 0 — clone everything

```bash
git clone --recursive https://github.com/DanijelTech/Nyra.git
cd Nyra
git submodule update --init --recursive    # nested submodules (deploy, train, MimicLite, firmware)
```

Already cloned but submodules are empty? Same command. Nested submodule depth:

```
roboparty_deploy   → src/inference, src/motors, src/imu, src/camera, tools/create_ap
roboparty_train    → robolab (→ rpo_description, rp1_description), rsl_rl
MimicLite          → active-adaptation, mimic-lite, sim2real
roboparty_firmware → roboto_usb2can, orangepi-build, x5-rdk-gen
```

MimicLite declares its submodules over SSH (`git@github.com:…`). Without GitHub SSH keys, map them
to HTTPS first:

```bash
git -C 02_motion_infra/MimicLite config url."https://github.com/".insteadOf "git@github.com:"
git -C 02_motion_infra/MimicLite submodule update --init --recursive
```

### Host specifics (Ubuntu 24.04, NVIDIA 50-series)

Ubuntu 24.04 pairs with **ROS 2 Jazzy** (22.04 uses Humble); Blackwell GPUs
(sm_120) need the CUDA 12.8 wheels that the modules' `uv.lock` files already pin.
See [`tools/build/README.md`](tools/build/README.md#host-notes-verified-on-ubuntu-2404--rtx-50-series).

### Step 1 — body + base control (Tier 0)

On the robot controller (or a dev box for a build-only check):

```bash
sudo apt update && sudo apt install -y ccache libfmt-dev libspdlog-dev libeigen3-dev screen zlib1g-dev
sudo apt install -y ros-humble-joy python3-yaml python3-numpy        # Jazzy: ros-jazzy-joy

cd 00_body_base/roboparty_deploy
colcon build --symlink-install
```

Then follow `00_body_base/roboparty_deploy/docs/rpo_deploy.md` for real-time limits, udev rules,
CAN mapping, IMU setup, motor zeroing and the safety guide. **Read the safety guide before the
first power-on.** Zeroing has two paths:

```bash
# A) with the robot software running
ros2 service call /init_motors std_srvs/srv/Trigger
ros2 service call /set_zeros    std_srvs/srv/Trigger

# B) per motor, interactive, before first deployment
python3 scripts/set_zero.py --config scripts/config/set_zero_rpo.yaml
```

Power-cycle the motors after zeroing so the new zero points take effect.

### Step 2 — robot description (Tier 0)

`00_body_base/rpo_description` is the single source of geometry:

| Consumer | How it gets the description |
|---|---|
| `roboparty_train` | submodule `robolab/data/robots/roboparty/rpo` → **same repository**, same commit |
| `roboparty_deploy` | 23-DoF joint order in `src/inference/robots/rpo/robot.yaml` + `configs/*.yaml` |
| `UFO` | copy of the MJCF + meshes in `ufo/data/robots/rpo/` |
| `MimicLite` | `atom` robot profile (Atom01 = the RPO lineage), 23 joints in URDF order |
| `hhtools` | `roboto_origin` robot preset, or `hhtools robot add 00_body_base/rpo_description/urdf/rpo.urdf` |

Never edit a copy. Edit `rpo_description`, then bump the pin in `roboparty_train` and re-run
`check_compat.py`.

### Step 3 — motion infra (Tier 1)

```bash
cd 05_integration/scripts
./setup_all.sh --motion          # uv sync for hhtools / any4hdmi / PartyMimic / UFO / MimicLite
```

What that does, per module:

| Module | Command |
|---|---|
| hhtools | `uv sync --locked --extra all` + `uv run python scripts/install_builtin_robots.py` |
| any4hdmi | `uv sync --locked` |
| PartyMimic | `uv sync` + `tools/download_data.sh` (G1/RP1 descriptions + LAFAN clips) |
| UFO | `uv sync --locked` + `bash scripts/download_data.sh rpo_lafan` (needs `HF_TOKEN`) |
| MimicLite | `uv sync --project venv/mjlab` and `uv sync --project venv/isaaclab`, then `aa-discover-projects` |

MimicLite's upstream layout expects `mimic-lite` checked out at
`active-adaptation/projects/mimic-lite`; this workspace keeps them as siblings, so `setup_all.sh`
creates that symlink. If you installed by hand:

```bash
mkdir -p 02_motion_infra/MimicLite/active-adaptation/projects
ln -s ../../mimic-lite 02_motion_infra/MimicLite/active-adaptation/projects/mimic-lite
```

### Step 4 — training (Tier 1b)

Isaac Lab 3.0 must sit **next to** `roboparty_train` — `pyproject.toml` resolves
`isaaclab-dev = { path = "../IsaacLab-3.0.0" }`:

```bash
git clone --branch v3.0.0 --recursive https://github.com/isaac-sim/IsaacLab.git 04_training/IsaacLab-3.0.0

cd 04_training/roboparty_train
uv sync --locked && source .venv/bin/activate          # MJWarp (default)
isaaclab list_envs --keyword RP                        # expect RPO-* and RP1-* tasks
```

PhysX variant:
`UV_PROJECT_ENVIRONMENT=.venv-physx uv sync --locked --no-default-groups --extra physx --no-install-package usd-exchange`.

`04_training/IsaacLab-3.0.0` is git-ignored here on purpose: it is a toolchain checkout, not a
module of this project.

### Step 5 — advanced layers (Tier 2)

```bash
./setup_all.sh --advanced        # PredActor, P-BFM, I-BFM (+ Git LFS), SADL, INTACT-JEPA
```

INTACT-JEPA is a **research preview**: documentation only, no training code yet. SADL is a
representation-learning loss evaluated on control benchmarks, not on the robot.

---

## 5. Run the robot: walk / run

From `00_body_base/roboparty_deploy`:

```bash
./tools/start_robot.sh --robot rpo --policy default      # or: ./tools/start_robot.sh rpo default
```

`start_robot.sh` builds with `colcon build --symlink-install` and starts two `screen` sessions:
`inference_session` (policy) and `joy_session` (gamepad). Attach with `screen -r inference_session`,
stop with `screen -S inference_session -X quit`.

Shipped policies (`src/inference/robots/rpo/configs/`):

| Policy | What it is |
|---|---|
| `default` | baseline locomotion (walk) |
| `amp` | AMP locomotion (walk / run, stylised gait) |
| `beyondmimic` | motion tracking from a reference clip |
| `getup` | recovery from prone/supine |
| `interrupt` | interruptible / safety behaviour |
| `parkour` | depth-based terrain traversal — **needs the camera node** |
| `ufo` | unsupervised FB policy exported from `02_motion_infra/UFO` |

Camera-dependent policy:

```bash
./tools/start_camera.sh --policy parkour     # second terminal
```

Or with the integration wrapper, which runs preflight checks first:

```bash
05_integration/scripts/launch_base.sh --policy amp
05_integration/scripts/launch_base.sh --policy parkour --camera
05_integration/scripts/launch_base.sh --preflight-only      # checks only, no start
05_integration/scripts/launch_base.sh --stop                # quit screen sessions
```

Simulation-only (no robot): use `04_training/roboparty_train` playback (§7) or UFO inference (§6).

---

## 6. MimicLite / PartyMimic / UFO on ROBOTO ORIGIN

Honest status per tool — the 23-DoF RPO is not the robot every upstream release was trained on.

### UFO — native RPO support ✅

`02_motion_infra/UFO` ships `ufo/data/robots/rpo/rpo_23dof.xml` (joint order identical to
`rpo.urdf`) and a released RPO data set (`rpo_lafan`: 56 motions, 1,168 clips, 50 Hz).

```bash
cd 02_motion_infra/UFO
uv sync --locked
export HF_TOKEN=<token>
bash scripts/download_data.sh rpo_lafan

# train FB on RPO
./run_train.sh --robot rpo --agent fb --gpu-ids all --work-dir runs/rpo_fb

# evaluate in simulation
./run_inference.sh tracking --model-folder runs/rpo_fb --motion-list 0 --save-mp4

# export ONNX + .meta.json + .deploy.yaml next to the checkpoint
./run_inference.sh tracking --model-folder runs/rpo_fb --motion-list 0 --export-onnx
```

Deployment on the robot goes through `roboparty_deploy` policy `ufo`
(`src/inference/robots/rpo/configs/ufo.yaml`, `model_names: ["policy_ufo.onnx"]`, 256-d latent,
`latents/*.npz`). Copy the exported trio into `src/inference/robots/rpo/models/` and
`.../latents/` — `export_policy.sh --install` does it.

RPO **TeCH is not released**; the validated RPO baseline is FB.

### MimicLite — `atom` profile ✅ (data pipeline required)

MimicLite's `task/robot=atom` profile is the Atom01 lineage: `base_link` root, `torso_link`,
`*_thigh_*`, `*_knee_*`, `*_ankle_*`, `*_arm_*`, `*_elbow_*` — the same 23 joints, same names, same
order as `rpo_description/urdf/rpo.urdf`. Released checkpoints cover G1 / H2 / A3 / RP1; for RPO you
train from retargeted RPO motion.

```bash
cd 02_motion_infra/MimicLite/active-adaptation
uv --project venv/mjlab run aa-list-tasks | grep atom       # profile is discoverable

# train (mjlab backend, single-stage PPO)
bash scripts/launch_ddp.sh 0 projects/mimic-lite/scripts/train.py venv/mjlab \
  task=tracking-base task/robot=atom backend=mjlab +exp=ppo/train wandb.mode=disabled

# play + export ONNX
uv run --project venv/mjlab projects/mimic-lite/scripts/play.py \
  task=tracking-base task/robot=atom checkpoint_path=<run or hf ref> \
  export_policy=true ++export_only=true
```

The atom motion config points at `../any4hdmi/output/atom/sonic`; override with
`task.command.motion_cfgs.sonic.path=/abs/path` when your dataset lives elsewhere.

### PartyMimic — needs a robot package for RPO ⚠️

PartyMimic selects a robot by Python package: `src/partymimic/{g1mj,rp1}`. There is no `rpo`
package, so RPO is **not** trainable out of the box. Two paths:

1. **Train on RP1 (28 DoF) or G1** and use it for those robots:
   ```bash
   cd 02_motion_infra/PartyMimic
   uv sync && tools/download_data.sh
   bash scripts/train.sh logs/rp1_s0 --robot rp1 --motion_dir data/motions/rp1_mix --seed 0
   uv run python tools/export_deploy_onnx.py logs/rp1_s0/ckpt_012000.pt deploy/rp1.onnx
   ```
2. **Add an `rpo` package** (one file, mirroring `rp1`): `MJCF_PATH` → an RPO MJCF,
   `MOTION_DIR` → your RPO clips, `ACTUATORS` kp/kd and `DEFAULT_JOINT_POS` from
   `roboparty_deploy/src/inference/robots/rpo/configs/default.yaml`, `TRACKED_BODIES` from
   `rpo_description`. Then `--robot rpo`. Motion data: any4hdmi `qpos` npz →
   `tools/data/prep_motions.py --format qpos`. `retarget_and_train.sh --target partymimic --robot rpo`
   prints the exact scaffold list.

Why bother: PartyMimic trains one general tracker in 7–30 min on a single RTX 4090 — the fastest
iteration loop for new Nyra motions (see `01_roadmap/party-os/docs/mimiclite-vs-partymimic.md`).

---

## 7. Train → export → deploy

```
            rpo_description (URDF/MJCF, 23 DoF)
                       │
      ┌────────────────┼─────────────────┐
      ▼                ▼                 ▼
 roboparty_train   UFO / PartyMimic   roboparty_deploy
 (Isaac Lab RL)    (mjlab trackers)   (ROS 2, ONNX runtime)
      │                │                 ▲
      └── exported/*.onnx ───────────────┘
                       ▲
   hhtools retarget → convert_motion / any4hdmi → training data
```

Train and export with `roboparty_train`:

```bash
cd 04_training/roboparty_train && source .venv/bin/activate

# walk/run baseline (RPO-Flat is the RPO velocity task; RP1-Flat for the 28-DoF RP1)
python robolab/scripts/rsl_rl/train.py --task=RPO-Flat --viz=none --logger=tensorboard --num_envs=4096
python robolab/scripts/rsl_rl/play.py  --task=RPO-AMP-Play --num_envs=1 --checkpoint=latest --viz=newton_gl

# motion tracking / parkour
python robolab/scripts/rsl_rl/play.py  --task=RPO-BeyondMimic  --checkpoint=latest --viz=newton_gl
python robolab/scripts/rsl_rl/play.py  --task=RPO-Parkour-Play --checkpoint=latest --viz=newton_gl
```

Playback writes `<run>/exported/policy.onnx` (Parkour also `0-depth_encoder.onnx`, `actor.onnx` and
their `.onnx.data`). Install it into deploy:

```bash
05_integration/scripts/export_policy.sh --source train --run logs/rsl_rl/<exp>/<run> \
    --task RPO-AMP-Play --name amp --install
05_integration/scripts/export_policy.sh --source ufo --run 02_motion_infra/UFO/runs/rpo_fb \
    --name ufo --install
05_integration/scripts/export_policy.sh --source partymimic \
    --run 02_motion_infra/PartyMimic/logs/rp1_s0/ckpt_012000.pt --name rp1_mimic
```

`--install` copies into `00_body_base/roboparty_deploy/src/inference/robots/<robot>/models/` as
`policy_<name>.onnx` (plus latents for UFO) and prints which YAML must reference it. Add
`--verify` to print the ONNX input/output shapes and compare them with the deploy observation
layout (`dof_pos:23, dof_vel:23, …`).

**The ONNX contract is not just shapes.** Joint order must match `usd2urdf` in the policy YAML:

| Producer | Action order | Deploy mapping |
|---|---|---|
| `roboparty_train` | `RPO_JOINT_ORDER` (interleaved L/R) | `usd2urdf: [0,6,12,1,7,13,18,…]` in `configs/default.yaml` |
| UFO | URDF order | `usd2urdf: 0…22` in `configs/ufo.yaml` |

Mixing them silently swaps joints. `check_compat.py` verifies the train-side permutation.

---

## 8. Integration scripts

All in `05_integration/scripts/`, all support `--help` and `--dry-run`.

| Script | Purpose |
|---|---|
| `setup_all.sh` | Install system deps, init submodules, `uv sync` per module. Phases: `--base --motion --train --advanced --all`, plus `--check` |
| `launch_base.sh` | Preflight (ROS, CAN, IMU, model files, rtprio) then start `roboparty_deploy`; `--camera`, `--stop`, `--preflight-only` |
| `export_policy.sh` | Export ONNX from `train` / `ufo` / `partymimic` / `mimiclite`, optionally install into deploy, optionally verify shapes |
| `retarget_and_train.sh` | hhtools retarget → dataset conversion → training for `train` / `ufo` / `mimiclite` / `partymimic` |
| `update_all.sh` | `git submodule sync/update --remote` with a pin report; `--check` shows drift without touching anything |
| `check_compat.py` | URDF ↔ MJCF ↔ UFO ↔ deploy ↔ train joint-order and asset verification; writes `06_assets/compat_report.json` |

A separate, self-contained layer sits above all of this:

| Component | Purpose |
|---|---|
| `06_evolayer/` | **EvoLayer** — a sandboxed evolutionary loop: observe → critique → propose → rehearse → safety gate → **Hermes decides** → deploy → rollback |
| `06_evolayer/docs/HERMES_CONTRACT.md` | the fork your local Hermes attaches to (signed review requests, signed/bearer decisions) |

EvoLayer never changes the robot on its own: a proposal must survive a real
MuJoCo rehearsal, hard safety limits and an explicit approval recorded in a
hash-chained audit log. See `06_evolayer/README.md`.

Typical first run:

```bash
05_integration/scripts/check_compat.py            # what works right now
05_integration/scripts/setup_all.sh --check       # what is installed / missing
05_integration/scripts/setup_all.sh --motion      # install the motion tier
```

---

## 9. Compatibility contract (verified)

`check_compat.py` asserts, on the actual checkout:

1. `rpo.urdf` has 23 revolute joints in a known order.
2. `rpo_description/mjcf/rpo.xml` and `ufo/data/robots/rpo/rpo_23dof.xml` use the same joint names
   in the same order.
3. `roboparty_train/robolab/data/robots/roboparty/rpo` is the `rpo_description` submodule and its
   `urdf/rpo.urdf` is byte-identical to `00_body_base/rpo_description/urdf/rpo.urdf`.
4. UFO's RPO meshes are byte-identical to `rpo_description/meshes` (sampled).
5. `RPO_JOINT_ORDER` in `robolab/assets/robots/roboparty.py` is a permutation of the URDF joints,
   and `roboparty_deploy/.../configs/default.yaml` `usd2urdf` equals that permutation applied to
   the URDF order.
6. Every deploy policy YAML has `joint_num: 23` and 23-entry `joint_default_angle`, `kp`, `kd`,
   `usd2urdf`, and every referenced `models/*.onnx` exists.
7. `roboparty_deploy/src/inference/robots/rpo/robot.yaml` declares 23 motors across 4 CAN buses.
8. MimicLite `atom01` joint list equals the URDF joint list.
9. PartyMimic robot packages present (`g1mj`, `rp1`); warns that `rpo` is missing.

Run it after every `update_all.sh`. Details and failure modes: `05_integration/docs/COMPATIBILITY.md`.

---

## 10. Keeping everything up to date

```bash
05_integration/scripts/update_all.sh --check     # report drift only
05_integration/scripts/update_all.sh             # fast-forward every submodule to its branch
05_integration/scripts/update_all.sh --nested    # include nested submodules (deploy/train/MimicLite)
05_integration/scripts/check_compat.py           # re-verify the contract
git add .gitmodules 00_body_base 02_motion_infra 03_advanced 04_training
git commit -m "chore: bump module pins"          # commit the new SHAs
```

Submodules track branches (`main`, or `master` for `PredActor`, `roboparty_inference`,
`roboparty_navigation`), but the **commit** recorded here is what everyone else gets. Bumping a pin
is a deliberate act: run `check_compat.py` first.

---

## 11. Licenses

| License | Modules |
|---|---|
| GPL-3.0 | `roboto_origin`, `roboparty_deploy`, `roboparty_inference`, `roboparty_camera`, `roboparty_firmware`, `roboparty_navigation`, `roboparty_xr_teleop`, `party-os`, `MimicLite` |
| CERN-OHL-W-2.0 | `rpo_hardware`, `rpo_description` |
| BSD-3-Clause | `roboparty_train` |
| Apache-2.0 | `human-humanoid-tools` |
| MIT | `PartyMimic`, `PredActor`, `INTACT-JEPA`, `SADL` |
| **CC BY-NC 4.0 (non-commercial)** | **`UFO`, `I-BFM-Code`, `P-BFM`** |
| not stated | `any4hdmi` |

Practical consequences:

- The deploy stack is **GPL-3.0**: linking it into a closed product triggers copyleft. Treat
  `roboparty_deploy` + `roboparty_inference` as GPL components.
- The **advanced BFMs and UFO are research-only**. A commercial Nyra may not ship them without a
  separate license from the authors.
- Hardware files are weakly reciprocal (CERN-OHL-W): modifications to the design files must be
  shared.
- Motion datasets carry their own terms (LAFAN1 CC BY-NC-ND, BONES-SEED gated, MotionDecode
  non-commercial). Check before redistributing checkpoints trained on them.

---

## 11b. Uninstalling (freeing the disk)

```bash
./uninstall.sh                      # DRY RUN: prints what it would delete, with sizes
./uninstall.sh --yes --all          # checkout + build outputs + python envs + npm + uv cache
./uninstall.sh --yes --envs --build # keep the code, remove the ~15 GB of environments
./uninstall.sh --yes --all --apt    # also remove the ROS 2 packages we told you to install
```

It is deliberately conservative:

* **dry run by default**; nothing is deleted without `--yes`;
* each category is opt-in (`--build`, `--envs`, `--npm`, `--uv-cache`, `--apt`, `--repo`);
* only paths on a whitelist can be deleted — it refuses `/`, `$HOME` and any directory
  without `.gitmodules`, and re-runs itself from a copy so deleting the checkout
  cannot truncate the script;
* `--apt` never runs `apt autoremove`, and generic packages (`ccache`, `screen`, …)
  need `--apt-generic` as well;
* it never touches your git config, the python/node/uv binaries, `~/.local/bin`, or
  other projects; it writes a log of every removal.

Tests (fake checkout + fake `HOME`, real deletions): `python3 -m pytest tools/build/tests/test_uninstall.py -q`.

## 12. Troubleshooting

See `05_integration/docs/TROUBLESHOOTING.md`. Fast checks:

```bash
05_integration/scripts/check_compat.py                    # geometry / joint-order / ONNX contract
05_integration/scripts/setup_all.sh --check               # missing toolchain, uv envs, models
05_integration/scripts/launch_base.sh --preflight-only    # ROS, CAN, IMU, model, rtprio
ip -br link | grep can                                    # CAN buses up?
ulimit -r                                                 # expect 98
git submodule status --recursive | grep '^-'              # uninitialised submodules
```

Common causes: uninitialised nested submodules (empty `src/inference`, `robolab`), ROS sourced from a
different Python minor version than the venv, ONNX copied without its `.onnx.data` (Parkour), wrong
policy YAML for the exported action order, and `HF_TOKEN` missing for gated motion datasets.
