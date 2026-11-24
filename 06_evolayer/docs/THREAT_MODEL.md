# Threat model

EvoLayer is a system that can change how a 30 kg biped moves. Its threat model is
therefore about *preventing* action, not only about detecting mistakes.

## Assets

| Asset | Why it matters |
|---|---|
| The deployed policy and its YAML | wrong bytes here move the robot incorrectly |
| The safety limits (`config/evolayer.yaml`, URDF) | lowering them silently weakens every future gate |
| The audit chain | the record of who approved what |
| The decision channel and its secrets | whoever holds them can approve |
| Rollback snapshots/tags | the only way back |

## Adversaries and abuse cases

### A1. A buggy or malicious proposal reaches the robot

**Vector:** generated code with side effects, a patch outside the intended file, a
command with shell metacharacters, a diff that applies to a stale file.

**Controls**
* templates derive diffs from the current file; the validator re-applies them in a
  sandbox (`git apply --check`) and re-parses the result;
* path allowlists per repo, deny globs, no `..`;
* added Python lines are rejected if they contain `subprocess`, `os.system`,
  `os.popen`, `eval(`, `exec(`, `__import__`, `socket`, `urllib`, `ctypes`,
  `shutil.rmtree`;
* size caps (200 changed lines / 128 KiB);
* commands are argv-only through `execution.run`: allowlisted binaries, no shell,
  sanitised environment, mandatory timeout, shell fragments rejected.

Residual risk: a semantic change inside an allowlisted numeric literal is exactly
what the proposer is for; it is caught by rehearsal + gate, not by static analysis.

### A2. The gate is talked into a pass

**Vector:** thin evidence, a wide interval, a metrics parser that silently drops
failures, a backend that never really ran.

**Controls**
* metrics come only from artifacts of a process with exit code 0; a run with no
  measurable episodes is `failed`, never `ok`;
* `episode_count` must reach the plan's episode count;
* non-inferiority uses Wilson bounds, so a lucky point estimate does not certify;
* absolute limits are independent of the baseline;
* per-joint ratios, not whole-body peaks;
* replay evidence cannot certify a policy change (`evidence_class`);
* a `reject` verdict cannot be approved — enforced in `HermesInterface.decide`, not
  in the UI.

### A3. Approval is forged, replayed or leaked

**Vector:** unauthenticated POST, replayed body, stolen token, expired token reused,
an approval for a different revision.

**Controls**
* bearer token compared in constant time; loopback bind by default; config refuses
  to disable auth on a non-loopback bind;
* webhook/callback bodies are HMAC-signed over `timestamp.body` with a ±300 s window;
* decision tokens bind proposal + revision + verdict + actor + nonce, expire
  (`hermes.token_ttl_s`) and are single-use (`store.consume_nonce`);
* `decisions` has a primary key on `(proposal_id, revision)`: the second decision on
  a revision raises;
* a superseded revision cannot be deployed (`installer.preflight` checks the latest
  revision).

### A4. Self-approval by the robot's own agent

**Vector:** the proposer (or an LLM inside it) approves its own change.

**Controls**
* there is no code path from `proposer` to `decide`;
* the delivery hook only *publishes*; `decide` requires an actor string that is
  recorded, and the token/actor binding means a token minted for one actor cannot be
  used by another;
* the LLM client can only add prose to a finding; it cannot create a proposal, alter
  a metric, or decide;
* `tick()` has no approve branch.

### A5. Rollback is impossible or restores the wrong thing

**Vector:** deploy writes without a snapshot, snapshot tampered, rollback writes
outside the workspace.

**Controls**
* `installer.deploy` takes a snapshot **before** the first write and fails closed if
  it cannot;
* snapshots are content-addressed; `verify` re-hashes every member;
* `restore` refuses any path that escapes `target_root` and re-checks hashes while
  writing;
* a failed contract check triggers automatic restore inside the same call.

### A6. Audit log tampering

**Vector:** delete a decision, edit a verdict, reorder entries.

**Controls:** each record commits to the previous hash; `verify()` reports the first
broken link and the count. `evolayer verify-audit` and `GET /healthz` expose it.
A break does not invalidate past deployments, but it invalidates the *claim* that
the record is complete — treat it as an incident.

### A7. Secret handling

**Vector:** secrets in config files, logs, or the review request.

**Controls**
* secrets are only read from the environment; the config holds the *names*;
* the review request contains no secret — it is signed with a secret, not containing it;
* `doctor` prints only `set`/`MISSING`;
* the audit log records decision metadata (actor, reason, nonce), never credentials.

## Known limitations (accepted, documented)

1. **Simulation fidelity.** The MuJoCo rehearsal uses the shipped MJCF and gains.
   Contact softness, actuator lag, sensor noise and thermal effects are not modelled.
   A passing gate is evidence, not proof; the report says so on every review.
2. **Statistical power.** Non-inferiority at 2% margins needs hundreds of episodes.
   The gate returns `review` rather than `pass` when the interval is too wide — that
   is deliberate friction.
3. **Local trust boundary.** The API is loopback-only by default. If you expose it,
   put it behind TLS and an authenticating proxy; the built-in server is not hardened
   for the public internet.
4. **LLM usage.** Disabled by default. If enabled, prose only; responses are schema-
   validated and cached, and calls are budgeted per run.
5. **Dataset licences.** Checkpoint provenance does not change dataset terms
   (LAFAN1 CC BY-NC-ND, BONES-SEED gated, MotionDecode non-commercial).

## Operator checklist before the first deploy

- [ ] `evolayer doctor` is clean; audit chain intact.
- [ ] Secrets are ≥16 chars and not in shell history (`read -s`, or a secret manager).
- [ ] `hermes.require_auth: true`, `bind: 127.0.0.1`.
- [ ] Limits match the physical robot (`fall_rate_max`, `battery_min_v`, torque ratio).
- [ ] `config/skills/<skill>.yaml` deploy target and eval suite reviewed.
- [ ] A rehearsal with a *known-good* policy passes the gate (`pass`/`review`, not `reject`).
- [ ] A rehearsal with a *known-bad* policy is rejected — if not, the gate is misconfigured.
- [ ] Rollback rehearsed once (`evolayer rollback --deployment <id>` on a test install).
