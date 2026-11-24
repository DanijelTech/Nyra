# Compatibility contract

The modules in this workspace were written by different teams for different robots. What glues
them together is a small number of agreements about the RPO's 23 joints. This page records those
agreements, how they are verified, and what breaks when one of them changes.

Run the check at any time:

```bash
05_integration/scripts/check_compat.py            # human readable
05_integration/scripts/check_compat.py --quiet    # failures only, used by setup_all.sh --check
```

It writes a machine-readable copy to `06_assets/compat_report.json` (git-ignored) and exits
non-zero when any check fails.

## Current state

Verified on the pinned commits of this checkout:

```
PASS urdf                   URDF joint set: 23 revolute joints
PASS mjcf                   MJCF joint order matches URDF: mjcf/rpo.xml identical order
PASS ufo-mjcf               UFO rpo_23dof.xml joint order: identical order
PASS train-order            robolab RPO_JOINT_ORDER: 23 joints, interleaved L/R policy order
PASS deploy-usd2urdf        deploy usd2urdf matches the training action order
PASS mimiclite              MimicLite atom profile joint list: same 23 joints, same order as rpo.urdf
PASS ufo-config             UFO configs/robots/rpo.yaml control joints: same 23 joints, same order
PASS train-asset            roboparty_train uses rpo_description: URDF byte-identical (submodule)
PASS ufo-meshes             UFO RPO meshes match rpo_description: 24 meshes, sampled 5 identical
PASS deploy-amp             configs/amp.yaml: 23 DoF, arrays and artefacts consistent
PASS deploy-beyondmimic     configs/beyondmimic.yaml: 23 DoF, arrays and artefacts consistent
PASS deploy-default         configs/default.yaml: 23 DoF, arrays and artefacts consistent
PASS deploy-getup           configs/getup.yaml: 23 DoF, arrays and artefacts consistent
PASS deploy-interrupt       configs/interrupt.yaml: 23 DoF, arrays and artefacts consistent
PASS deploy-parkour         configs/parkour.yaml: 23 DoF, arrays and artefacts consistent
PASS deploy-ufo             configs/ufo.yaml: 23 DoF, arrays and artefacts consistent
PASS deploy-robot           deploy robot.yaml hardware map: 23 motors on 4 CAN buses
WARN partymimic             PartyMimic robot packages: g1, g1mj, rp1 — no 'rpo' package
PASS pins                   submodule pins recorded: 21 modules initialised

18 pass, 1 warn, 0 fail
```

The single warning is expected: PartyMimic has no RPO robot package upstream (see README §6).

## The three joint orders

The RPO has 23 actuated joints. Three different orderings circulate, and conflating them is the
most likely way to break the robot (a policy that "works" in simulation but drives the arms when
asked to move the knees).

| Order | Where | Content |
|---|---|---|
| **URDF order** | `rpo_description/urdf/rpo.urdf`, `mjcf/rpo.xml`, UFO's `rpo_23dof.xml`, `configs/robots/rpo.yaml`, MimicLite `atom01` | left leg 6, right leg 6, torso, left arm 5, right arm 5 |
| **Policy order** | `robolab/assets/robots/roboparty.py` → `RPO_JOINT_ORDER`, and every ONNX trained there | interleaved: `left_thigh_yaw, right_thigh_yaw, torso, left_thigh_roll, …` |
| **Mapping** | `roboparty_deploy/src/inference/robots/rpo/configs/default.yaml` → `usd2urdf` | `usd2urdf[i] = urdf_index_of(RPO_JOINT_ORDER[i])` |

`usd2urdf` is the contract between the two: the ONNX emits actions in *policy order*, deploy maps
them to *URDF order* before writing to the motors. The name is historical (`usd` came from an Isaac
Sim rewrite), the content is a permutation.

`check_compat.py` recomputes the permutation from the URDF and from `RPO_JOINT_ORDER` and compares
it with the YAML. If you retrain with a different joint order, this check fails *before* anything
reaches the hardware.

UFO is the exception: its policies are trained in URDF order, which is why `configs/ufo.yaml` has
the identity mapping `0…22`. Mixing a UFO ONNX into `default.yaml` swaps joints.

## What breaks when something changes

| Change | Breaks | Detection |
|---|---|---|
| Editing `rpo_description/urdf/rpo.urdf` | everything downstream (train asset, UFO copy, deploy joint count) | `urdf`, `train-asset`, `ufo-mjcf`, `ufo-meshes` |
| Bumping `rpo_description` pin without bumping `roboparty_train`'s `robolab` submodule | train uses stale geometry | `train-asset` (byte comparison) |
| Reordering `RPO_JOINT_ORDER` | deploy `usd2urdf` | `deploy-usd2urdf` |
| Adding/removing a policy YAML entry | policy silently mis-scaled | per-config array length checks |
| Swapping robot profiles in a policy YAML | wrong gains, wrong action scale | `deploy-<policy>` checks `joint_num` and array lengths |
| Replacing `models/policy_*.onnx` with a different action order | wrong joints move | `deploy-usd2urdf` covers the default profile; verify other profiles by hand when you add them |
| Upstream renaming a joint (e.g. `atom01` → `roboto_origin`) | MimicLite / UFO RPO profiles | `mimiclite`, `ufo-config` |

## Deploy artefact requirements

Per policy YAML in `src/inference/robots/rpo/configs/`:

- `joint_num: 23`
- `joint_default_angle`, `kp`, `kd`, `usd2urdf` — 23 entries each
- `model_names: [...]` — every file must exist under `models/`
- `latent_names: [...]` — every file must exist under `latents/` (UFO policies only)

`roboparty_deploy` resolves models relative to the robot directory, so an installed policy is
`models/policy_<name>.onnx` plus its sidecars (`.onnx.data` for Parkour, `.meta.json` and
`.deploy.yaml` for UFO). `export_policy.sh --install` copies them and prints the YAML it must be
referenced from.

## When a check fails

1. Read the detail column — it names the file and the mismatch.
2. If the module is upstream: do **not** patch it locally. Open an issue / PR upstream, or hold the
   pin at the last good commit (`update_all.sh --check` shows the drift).
3. If the mismatch is a local experiment: keep it on a branch inside the module and record it in the
   commit message that bumps the pin.
4. Re-run `check_compat.py` after every `update_all.sh`; a green report plus a walked robot is the
   definition of "compatible" here.
