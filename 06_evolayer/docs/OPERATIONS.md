# Operations

## Where it runs

| Host | What runs | Why |
|---|---|---|
| Workstation / training box | observation of uploaded logs, propose, rehearse, gate, review API | has the GPU and the training checkout |
| Robot controller (Orange Pi / RDK X5) | **only** `observer/ros2_node.py` | it writes telemetry and nothing else |

Deployment writes into `00_body_base/roboparty_deploy`, so run `deploy` on the host
that holds the deploy checkout you intend to ship. Never run two `deploy`
invocations concurrently against the same checkout.

## Install

```bash
cd 06_evolayer
python3 -m pip install pyyaml                 # core only
python3 -m pip install mujoco numpy onnxruntime   # rehearsal
python3 -m pip install imageio                # optional: video artifacts
python3 -m pip install -e .                    # optional: the `evolayer` entry point
```

Without `pip install -e .`, use `python3 -m evo_layer` from `06_evolayer/`, or the
wrapper `scripts/evolayer`.

## Daily loop

```bash
# on the robot: capture telemetry (writes JSONL, never commands the robot)
python3 -m evo_layer.observer.ros2_node --skill locomotion \
    --out /var/log/evolayer/locomotion-$(date +%F).jsonl --duration 3600

# on the workstation
evolayer observe  --skill locomotion --input /var/log/evolayer/locomotion-2026-11-24.jsonl
evolayer critique --skill locomotion
evolayer tick     --skill locomotion          # propose → rehearse → gate → await Hermes
evolayer status   --skill locomotion
```

Cron example (workstation, after a log sync):

```cron
17 3 * * * cd /opt/nyra/06_evolayer && ./scripts/evolayer tick --skill locomotion >> /var/log/evolayer/tick.log 2>&1
```

`tick` is idempotent: findings are deduplicated per hour and a finding with an
existing proposal is not re-proposed.

## Running the Hermes fork

```bash
export EVOLAYER_HERMES_TOKEN=...          # your Hermes presents this
export EVOLAYER_HERMES_SECRET=...         # decision tokens
export EVOLAYER_HERMES_OUTBOUND_SECRET=...  # webhook/callback signing
evolayer serve                            # 127.0.0.1:8765 by default
```

Your local Hermes then either polls (`GET /api/pending`) or receives
`POST <hermes.outbound.url>`. See `docs/HERMES_CONTRACT.md` and
`examples/hermes_adapter.py`.

`GET /healthz` is public and reports the audit chain status — use it as a liveness
probe:

```json
{"status": "ok", "version": "1.0.0",
 "audit": {"entries": 1421, "chain_ok": true, "broken_at": null, "reason": "chain intact"}}
```

## Secrets

```bash
python3 -c 'import secrets;print(secrets.token_urlsafe(48))'   # generate
read -rs EVOLAYER_HERMES_SECRET && export EVOLAYER_HERMES_SECRET   # avoid history
```

* Rotate the token and secrets together; old tokens stop verifying immediately.
* `evolayer doctor` prints only `set` / `MISSING`.
* If `EVOLAYER_HERMES_SECRET` is missing, `serve` refuses to start (a decision token
  is the alternative auth path; without a secret it would be unusable).

## Backups

Back up `06_assets/evolayer/`:

| Path | Content | Restore value |
|---|---|---|
| `state.db` | episodes, findings, proposals, verdicts, decisions, deployments | the loop's memory |
| `audit.jsonl` | hash-chained event log | accountability |
| `snapshots/*.tar.gz` | pre-deploy deployment state | rollback |
| `artifacts/<proposal>/` | reports + review requests | review history |
| `sim/<run>/` | rehearsal logs and episodes | post-mortems |

`state.db` is SQLite in WAL mode: copy `state.db`, `state.db-wal`, `state.db-shm`
together, or use `sqlite3 state.db ".backup out.db"`.

## Rollback

```bash
evolayer status                                   # find the deployment id
evolayer snapshot verify <snp_id>
evolayer rollback --deployment <dep_id> --actor hermes --reason "drift on hardware"
```

The rollback restores the snapshot (verified hashes) and marks the deployment as
rolled back. The git tag created at deploy time remains as the record of what was
installed; the deploy branch keeps the commit history.

## Failure playbook

| Symptom | Likely cause | Action |
|---|---|---|
| `simulate` → `unavailable: backend …` | environment missing (Isaac Lab, UFO venv) | `setup_all.sh --train` / `--motion`, or use `--backend mujoco` |
| `simulate` → `unavailable: no candidate ONNX` | policy-changing proposal without a trained candidate | train first (backend `isaaclab` chains it), or pass `--policy` |
| gate `reject` on `episode_count` | rehearsal produced fewer episodes than planned | raise `simulator.max_episodes` or lower `eval_suite.episodes` |
| gate `review` on `success_non_inferior` | the interval is too wide to certify | increase episodes; this is intentional friction |
| `deploy` → `contract checks failed` | YAML/ONNX mismatch (joint count, usd2urdf, missing model) | fix the artifact or the config; the install already rolled back |
| `approval refused: replay evidence` | rehearsal used the `fixture` backend | rehearse with `mujoco` |
| `token expired` / `already used` | TTL passed or replay | mint a new token |
| `verify-audit` fails | the log was edited/truncated | treat as an incident; keep the file, investigate, restore from backup |
| `subprocess … imageio … MUJOCO_GL` | headless rendering | `export MUJOCO_GL=egl` (or `osmesa`) or disable video (`sim_plan.params.video=false`) |

## Tuning that is safe vs tuning that is not

| Change | Risk | Note |
|---|---|---|
| `eval_suite.episodes`, `seeds` | none | more evidence only |
| `simulator.max_episodes` | none | caps rehearsal cost |
| `limits.fall_rate_max` **upward** | **high** | weakens every gate; requires review and a commit message |
| `limits.torque_ratio_max` upward | **high** | same |
| `hermes.require_auth: false` | **critical** | only allowed on a loopback bind, and only for local experiments |
| `repos.*.allow` widened | **high** | lets proposals touch more files |
| `llm.enabled: true` | low | prose only |

Any change to limits or allowlists should be a commit with a reason in the message,
and the audit trail of the next deployment will reference the new values.
