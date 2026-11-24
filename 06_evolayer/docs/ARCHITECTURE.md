# Architecture

## Data flow

```
telemetry (ROS 2 / logs / simulator)
      │  observer/ingest.py  — strict schema, unit-error rejection, episode assembly
      ▼
   episodes (SQLite)  ──────────────┐
      │                            │
      │ critic/baseline.py          │  baseline = last N days anchored to the newest episode
      │ critic/analyze.py          │  6 rules, each with a statistic
      ▼                            │
   findings  ──────────────────────┤  (stored, deduplicated per hour)
      │                            │
      │ proposer/templates.py      │  reads the real file, emits a difflib patch
      │ proposer/validate.py       │  sandbox apply + parse + allowlist + forbidden primitives
      │ proposer/generator.py      │  risk score, rehearsal plan sized by risk
      ▼                            │
   proposal (validated)            │
      │                            │
      │ simulator/runner.py        │  materialises the candidate, runs the backend
      │   backends.py              │  mujoco | isaaclab→mujoco | ufo | partymimic | fixture
      ▼                            │
   simulation + artifacts (sha256)  │
      │                            │
      │ safety/gate.py             │  hard limits + Wilson/z/bootstrap non-inferiority
      │ safety/policy.py           │  deploy contract (joint_num, usd2urdf, ONNX)
      ▼                            │
   verdict (pass | review | reject) │
      │                            │
      │ hermes_interface/          │  report + review request (signed) → your Hermes
      ▼                            │
   decision (approve | reject | modify)
      │
      │ deploy/installer.py        │  snapshot → tag → patch → copy artifacts → contract check → commit
      ▼
   deployment (rollback point recorded)
```

## Module contracts

### `observer`

* `episode_from_record(record) -> Episode` — raises `TelemetryError` with the field
  name for anything malformed; plausibility envelope catches unit errors.
* `EpisodeAssembler` — groups samples by `idle_gap_s`; a fall is `|roll|` or
  `|pitch| > tilt_rad` sustained for `sustain_s`, or base height `< height_floor`.
* `ros2_node.py` — subscribes to `/joint_states`, `/imu/data`, `/odom`, `/cmd_vel`,
  `/diagnostics`; writes the same JSONL. Never commands the robot.

### `critic`

* `analyze(skill) -> list[Finding]`. The window is anchored to the newest stored
  episode, so offline logs behave like live data.
* Rules: fall-rate regression (Wilson bound + two-proportion z), success-rate
  regression, tracking regression (median + bootstrap CI), torque instability
  (split-z changepoint + p95), absolute weak spot, and headroom opportunity.
* Findings are content-addressed per `(skill, kind, metric, hour)`, so re-running the
  critic is idempotent.

### `proposer`

* A template gets the `Finding` and the config, reads the target file, computes the
  new content and emits `difflib.unified_diff`. If the anchor is missing it raises
  `TemplateUnavailable` — it never guesses.
* `validate.validate(proposal)` proves, in a temp tree: the repo is writable, the
  path is allowlisted, the patch applies with `git apply --check`, the result parses
  (`compile` / `yaml.safe_load`), added Python lines contain no
  `subprocess|os.system|eval|exec|socket|…`, the patch is under the size caps, and
  any training command uses allowed binaries and flags.
* Risk (documented, linear):
  `target_repo + 0.05·(files-1) + min(lines/200, 0.20) + kind + severity + controller`,
  clamped to `[0,1]`, classes `low < 0.35 ≤ medium < 0.65 ≤ high`.

### `simulator`

* `Plan` carries `argv`, `cwd`, `timeout_s`, expected `artifacts`, `emits_metrics`
  and `evidence_class` (`rehearsal | replay | producer`).
* `MujocoBackend` drives the RPO MJCF with the deploy PD gains, the deploy
  observation layout and per-joint URDF limits; it writes `episodes.jsonl` with
  `torque_ratio` and `vel_ratio` (peak on joint *k* / limit of joint *k*).
* `IsaacLabBackend` runs the training command and **chains** a MuJoCo rehearsal of
  `exported/policy.onnx`, so the measurements come from a real rollout.
* `FixtureBackend` replays recorded episodes and is refused as certification
  evidence unless the caller explicitly asks for a rehearsal-of-the-loop.
* A backend that cannot measure returns `failed`/`unavailable`; the gate then
  rejects. There is no path from "no data" to "pass".

### `safety`

* `load_urdf_limits` reads `effort`/`velocity` from `rpo_description` — the same file
  the trainer uses.
* The gate emits one row per constraint with `measured`, `limit` and a sentence.
  Verdict aggregation: any `reject` → `reject`; else any `review` → `review`; else `pass`.
* Absolute limits never override regression: a looser ceiling does not license a
  worse policy.

### `hermes_interface`

* `interface.build_report` produces the Markdown + JSON; the report includes the
  diffs, the gate table, artifact hashes and an explicit "what this does not prove"
  section.
* `interface.decide` refuses: unknown proposal, gate `reject`, no successful
  rehearsal, replay evidence, duplicate decision, token/actor/verdict mismatch.
* `api` is a stdlib `ThreadingHTTPServer`; bearer or HMAC, bounded bodies, no path
  traversal, every mutation audited.
* `bridge` pushes signed review requests (`webhook`) or leaves them for polling.
* `client` is the connector to vendor into your Hermes.

### `versioning`

* `GitRepo` refuses: protected branches, unprefixed branches, paths outside the
  repository, unknown refs, and has no push method.
* `Snapshotter` archives `path → sha256` manifests; `restore` verifies every hash
  before writing and refuses to write outside `target_root`.

### `deploy`

Ordering is the safety property: snapshot → branch → apply → install → contract
checks → commit → tag. Any failure restores the snapshot, marks the proposal
`failed` and raises `DeployError`.

### `orchestrator`

Resumable state machine; every step persists before returning:

```
proposed → validated → simulated → gated → awaiting_hermes → approved → deployed
                    ↘ failed                    ↘ rejected
                                                 ↘ superseded (modify)
```

`tick()` walks a skill as far as it can without a decision and stops.

## Storage schema (SQLite, WAL)

| Table | Purpose | Notable constraint |
|---|---|---|
| `episodes` | telemetry | `id` is content-addressed; re-ingest is a no-op |
| `findings` | critic output | dedup by id |
| `proposals` | generated changes | `UNIQUE(finding_id, revision)` |
| `simulations` | rehearsals + artifacts | FK → proposals |
| `verdicts` | gate results | FK → proposals |
| `decisions` | Hermes decisions | **PK (proposal_id, revision)**, nonce unique |
| `deployments` | installs + rollback points | FK → proposals |
| `kv` | schema version, nonces, LLM cache | `nonce:<n>` unique |

The connection is wrapped in a re-entrant lock (`check_same_thread=False`) because
the API is threaded; `transaction()` holds the lock for the whole block.

## Extension points

* **New skill** — add `config/skills/<id>.yaml` (eval suite, limits, deploy target).
* **New proposal kind** — add a function to `proposer/templates.py`, register it in
  `TEMPLATES`, add its risk weight in `generator.KIND_WEIGHTS`, and cover it in the
  validator tests.
* **New simulator** — implement `availability()` and `plan()` in
  `simulator/backends.py`, declare `evidence_class`, and add a parser if it emits a
  new artifact shape.
* **New decision channel** — implement the contract in
  `docs/HERMES_CONTRACT.md`; EvoLayer needs no change if you use the API.
