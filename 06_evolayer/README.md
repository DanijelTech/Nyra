# EvoLayer (Hermes-Evo)

A sandboxed evolutionary layer above the Nyra stack. The robot observes itself,
finds weaknesses, proposes changes, rehearses them in physics, gates them on hard
safety limits — and **stops**. A human or agent (Hermes) decides. Nothing reaches
the robot without an explicit, auditable approval.

```
      ┌──────────────────────────────┐
      │   Hermes (your local agent)  │        ← EvoLayer only provides the fork:
      └───────────▲──────────────────┘          signed review requests out,
                  │ approval / reject / modify  signed decisions in, HTTP + client
      ┌───────────┴──────────────────┐
      │          EvoLayer            │
      │  observer → critic → proposer│
      │  → simulator → safety gate   │
      │  → versioning → deploy       │
      └───────────▲──────────────────┘
                  │ measurements / artifacts
      ┌───────────┴──────────────────┐
      │  roboparty_deploy + rpo_desc │
      │  roboparty_train / UFO       │
      └──────────────────────────────┘
```

## Why it is safe by construction

| Guarantee | How it is enforced |
|---|---|
| The robot cannot change its own policy | only `deploy/installer.py` writes into the deploy tree, and it refuses unless a passing gate, a successful rehearsal and an `approve` decision exist |
| Proposals are concrete, reviewable diffs | templates read the real file and emit a `difflib` patch; the validator proves it applies and parses before anything runs |
| No shell, no interpolation | every command is an argv list through `execution.run`, allowlisted binaries, sanitised environment, hard timeout |
| Numbers are measured, not claimed | the gate computes Wilson intervals, two-proportion tests and per-joint torque/velocity ratios from episode data |
| Hard limits cannot be overridden | a `reject` verdict cannot be approved — `HermesInterface.decide` refuses |
| Every action is auditable | append-only, hash-chained `audit.jsonl`; tampering breaks verification |
| Rollback is always possible | deploy takes a verified content-addressed snapshot **and** a git tag before writing |

## Quick start

```bash
cd 06_evolayer
python3 -m evo_layer doctor                  # configuration, backends, audit chain
python3 -m evo_layer backends                # what can rehearse here

# 1. observe
python3 -m evo_layer observe --skill locomotion \
    --input examples/telemetry/real_locomotion.jsonl

# 2. critique (real statistics on the stored episodes)
python3 -m evo_layer critique --skill locomotion

# 3. propose + validate (a diff against the real training config)
python3 -m evo_layer propose --finding <fnd_id>

# 4. rehearse in MuJoCo with a candidate ONNX
python3 -m evo_layer simulate --proposal <prp_id> --policy /path/to/policy.onnx

# 5. hard safety gate
python3 -m evo_layer gate --proposal <prp_id>

# 6. hand the report to Hermes (see docs/HERMES_CONTRACT.md)
python3 -m evo_layer serve                   # the fork your Hermes attaches to
python3 -m evo_layer review --proposal <prp_id>   # or decide locally

# 7. install (approved only) and roll back if needed
python3 -m evo_layer deploy   --proposal <prp_id>
python3 -m evo_layer rollback --deployment <dep_id> --actor hermes
```

`tick` runs steps 1–6 as far as possible without a human:

```bash
python3 -m evo_layer tick --skill locomotion
```

It stops at `awaiting_hermes`. There is no auto-approve flag, by design.

## What each module does

| Module | Responsibility | Key guarantee |
|---|---|---|
| `observer/` | ingest telemetry (JSONL/CSV/RECORD samples), assemble episodes, live ROS 2 node | strict schema + plausibility envelope; unit errors are rejected, never coerced |
| `critic/` | baseline, regressions, weak spots, instability, headroom; optional LLM prose | every finding carries its statistic and raw evidence |
| `proposer/` | templates → real diffs, risk scoring, validation | patch must apply to the current bytes, parse, stay on the allowlist, and contain no process/network primitives |
| `simulator/` | MuJoCo rehearsal, roboparty_train/UFO/PartyMimic adapters, metric extraction | a backend that cannot measure yields `unavailable`/`failed`, never a fake pass |
| `safety/` | URDF limits, hard gates, non-inferiority, deploy contract | per-joint torque/velocity ratios; `reject` is final |
| `hermes_interface/` | report, tokens, HTTP API, bridge, client, contract | one-time tokens, HMAC signatures, no approval logic inside EvoLayer |
| `versioning/` | git guard rails, content-addressed snapshots | never pushes, never commits on `main`, restore verifies hashes |
| `deploy/` | installer + rollback | snapshot + tag before the first write; auto-rollback on a failed contract check |
| `orchestrator/` | the resumable state machine | no path to deployment without a decision |

Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Hermes integration

EvoLayer ships **only the fork**, not an approval agent:

* `docs/HERMES_CONTRACT.md` — endpoints, JSON schemas, signing, error codes;
* `evo_layer/hermes_interface/contract.py` — the versioned constants and schemas;
* `evo_layer/hermes_interface/client.py` — a stdlib client to vendor into your Hermes;
* `examples/hermes_adapter.py` — a complete 60-line reference adapter.

Two delivery modes: `webhook` (EvoLayer POSTs a signed review request) or `pull`
(your Hermes polls `/api/pending`).

## Configuration

`config/evolayer.yaml` holds limits, skill definitions, the simulator backends and
the Hermes wiring. Everything that gates a deployment is validated on load. Secrets
are never in the file:

| Environment variable | Purpose |
|---|---|
| `EVOLAYER_HERMES_TOKEN` | bearer token your Hermes presents (≥16 chars) |
| `EVOLAYER_HERMES_SECRET` | HMAC key for one-time decision tokens (≥16 chars) |
| `EVOLAYER_HERMES_OUTBOUND_SECRET` | HMAC key for webhook/callback signing |

```bash
export EVOLAYER_HERMES_TOKEN="$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')"
export EVOLAYER_HERMES_SECRET="$(python3 -c 'import secrets;print(secrets.token_urlsafe(48))')"
export EVOLAYER_HERMES_OUTBOUND_SECRET="$(python3 -c 'import secrets;print(secrets.token_urlsafe(48))')"
```

## Skills

| Skill | Deploy policy | Rehearsal | Notes |
|---|---|---|---|
| `locomotion` | `amp` | MuJoCo | walk / run, fall-rate regressions |
| `recovery` | `getup` | MuJoCo | prone/supine/side init poses |
| `parkour` | `parkour` | MuJoCo | needs the camera node on the robot |
| `tracking` | `beyondmimic` | roboparty_train → MuJoCo | rehearsal needs the Isaac Lab env |

A skill is a YAML file in `config/skills/`: eval suite, limits, deploy target and
train task. Adding a capability means adding a file, not editing code.

## Requirements

* Python 3.10+ (the integration layer is stdlib + PyYAML).
* `mujoco`, `numpy`, `onnxruntime` for GPU-free rehearsal (`pip install mujoco numpy onnxruntime`).
  The rehearsal tests **skip** (never fail) when these are missing; the rest of the
  suite runs everywhere. Install them into the interpreter that runs the builder —
  the simulator backend now uses `sys.executable` for that reason.
* Optional: `imageio` + a GL backend (`MUJOCO_GL=egl` or `osmesa`) for video artifacts.
* Optional: `onnx` for the deeper ONNX checks in `evolayer doctor`/`export` paths.
* For training-style proposals: the `roboparty_train` environment (see the workspace README).

## Tests

```bash
cd 06_evolayer && python3 -m pytest -q      # 150 tests, ~30 s (includes real MuJoCo runs)
```

The suite runs real git, real SQLite, real HTTP servers, real HMAC verification
and real MuJoCo episodes in throw-away workspaces. There are no mocks of the code
under test.

## Limits, honestly

* The MuJoCo rehearsal uses `rpo_description`'s MJCF with the deploy PD gains and
  observation layout. Contact fidelity, actuator lag and sensor noise are **not**
  the real robot's — the report says so on every review.
* A passing gate certifies non-inferiority on the measured metrics, not overall safety.
* An LLM may add prose to a finding; it can never compute, adjust or veto a number
  (`llm.enabled: false` by default).
* Proposals that change the policy need a candidate ONNX. Without one the rehearsal
  is refused rather than measured against the wrong policy.
