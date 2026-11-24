# Hermes integration contract (v1)

EvoLayer implements **no approval logic**. It publishes a review request and
accepts a decision. This page is the complete fork specification; implement it in
your local Hermes and the two systems are connected.

```
EvoLayer                                your local Hermes
   │  POST review request (signed)  ────────▶  endpoint you configure
   │                                    (or)  GET /api/pending  (poll)
   │  ◀────────  POST decision (signed or bearer)  ──────────│
   │  ─────────▶  200 {"decision": {...}}                    │
```

## 1. Transport

| Mode | How to enable | When to use |
|---|---|---|
| `pull` | `hermes.outbound.mode: pull` (default) | robot host cannot reach your workstation |
| `webhook` | `hermes.outbound.mode: webhook` + `url` | your Hermes listens; EvoLayer pushes |

The API binds `127.0.0.1:8765` by default. Never expose it without TLS and auth.

## 2. Authentication

Three independent mechanisms:

| Purpose | Credential | Header |
|---|---|---|
| Your Hermes → EvoLayer | `EVOLAYER_HERMES_TOKEN` | `Authorization: Bearer <token>` |
| EvoLayer → your Hermes | `EVOLAYER_HERMES_OUTBOUND_SECRET` | `X-EvoLayer-Signature` |
| Decision tokens (single use) | `EVOLAYER_HERMES_SECRET` | `"token"` field in the decision body |

Signature scheme (both directions):

```
X-EvoLayer-Timestamp: <unix seconds>
X-EvoLayer-Signature: sha256=hex(HMAC_SHA256(secret, timestamp + "." + raw_body))
```

`raw_body` is the exact bytes sent. Timestamps outside a ±300 s window are
rejected, so a captured request cannot be replayed later. Decision tokens carry a
nonce that the store burns on first use, so even a replayed *body* fails.

Comparison is constant-time (`hmac.compare_digest`).

## 3. `GET /api/contract`

Public, no auth. Returns the versioned schemas so your Hermes can validate itself
against this deployment:

```json
{
  "contract_version": "1",
  "headers": {"signature": "X-EvoLayer-Signature", "timestamp": "X-EvoLayer-Timestamp",
               "contract": "X-EvoLayer-Contract", "event": "X-EvoLayer-Event"},
  "events": ["review.requested", "review.updated"],
  "decisions": ["approve", "reject", "modify"],
  "review_request": { "$schema": "...", ... },
  "decision": { "$schema": "...", ... }
}
```

## 4. Review request (EvoLayer → Hermes)

`POST <hermes.outbound.url>`, `Content-Type: application/json`, signed.

```json
{
  "contract": "1",
  "event": "review.requested",
  "created_at": 1795539000.12,
  "proposal": {"id": "prp_09b09815924e", "revision": 1, "skill": "locomotion",
                "skill_title": "Locomotion (walk / run)", "kind": "reward_term_weight"},
  "risk": {"class": "medium", "score": 0.36},
  "gate": {"verdict": "pass", "checks": [{"name": "fall_rate_max", "status": "pass",
                                          "measured": 0.0, "limit": 0.02, "detail": "0 <= 0.02"}]},
  "rehearsal": {
    "summary": {"backend": "mujoco", "status": "ok", "episodes": 12, "evidence_class": "rehearsal"},
    "metrics": {"n": 12, "success_rate": 1.0, "fall_rate": 0.0, "torque_ratio": 0.41, ...},
    "baseline": {"n": 60, "success_rate": 0.98, "fall_rate": 0.01, ...}
  },
  "changes": [{"repo": "training", "path": "robolab/.../amp_env_cfg.py",
                "diff": "--- a/...\n+++ b/...\n@@ ...", "description": "..."}],
  "caveats": ["Rehearsal uses a reduced physics model ..."],
  "links": {
    "report": "http://127.0.0.1:8765/api/proposals/prp_.../report",
    "decide": "http://127.0.0.1:8765/api/proposals/prp_.../decide",
    "artifacts": [{"name": "episodes.jsonl", "kind": "metrics", "size": 4096,
                   "sha256": "...", "url": "http://.../artifacts/episodes.jsonl"}]
  },
  "rollback": "evolayer rollback --deployment <id> --actor <you>"
}
```

Delivery failures are retried (`retries`, `backoff_s`), audited, and the payload is
persisted as `review-request-r<rev>.json` next to the report so nothing is lost.

## 5. Read endpoints (auth required)

| Endpoint | Returns |
|---|---|
| `GET /api/pending` | proposals in `awaiting_hermes` with their gate verdict |
| `GET /api/proposals/<id>` | proposal + latest verdict + latest rehearsal |
| `GET /api/proposals/<id>/report` | full report JSON (same content as the Markdown) |
| `GET /api/proposals/<id>/artifacts/<name>` | an artifact (video, episodes, logs) |
| `GET /api/backends` | simulator availability |
| `GET /api/audit?since=<seq>` | audit entries after `seq` |
| `GET /healthz` | public: liveness + audit chain status |

Artifact names are validated (no `/`, no `..`) and resolved inside the artifact
root only.

## 6. Decision (Hermes → EvoLayer)

`POST /api/proposals/<id>/decide`

```json
{"verdict": "approve", "actor": "hermes", "reason": "evidence is solid",
 "token": "<optional: single-use token>", "modifications": {}}
```

* `verdict` ∈ `approve | reject | modify`
* `actor` (non-empty) and `reason` (≥3 chars) are mandatory and are recorded in the
  hash-chained audit log with your client address.
* `modifications` is only meaningful for `modify`:
  `{"template_params": {"factor": 1.1}}` regenerates the proposal as revision N+1
  with those template parameters, revalidated.

Responses:

| Status | Meaning |
|---|---|
| `200` | accepted: `{"decision": {...}, "proposal_id": "..."}` |
| `400` | malformed body / unknown verdict / missing actor or reason |
| `401` | no valid bearer token or signature |
| `403` | a decision token was required and missing/invalid/expired/forged |
| `404` | unknown proposal |
| `409` | the decision is refused: gate `reject`, no successful rehearsal, replay evidence, or this revision was already decided |

**A rejection by the safety gate cannot be approved.** That is a hard refusal, not
a policy setting.

## 7. Decision tokens

For a workflow where your Hermes holds a token instead of the bearer secret:

```bash
evolayer issue-token --proposal prp_... --verdict approve --actor hermes
# → base64(payload).base64(hmac)  (TTL: hermes.token_ttl_s, default 900 s)
```

The payload binds proposal, revision, verdict, actor, nonce and expiry. The token
is consumed once (`store.consume_nonce`), and a token for `reject` cannot be used
to `approve`.

## 8. Reference adapter

`examples/hermes_adapter.py` is a complete, runnable local Hermes:

```bash
export EVOLAYER_HERMES_TOKEN=...        # to read/decide
export EVOLAYER_HERMES_OUTBOUND_SECRET=...   # if you also receive webhooks
python3 06_evolayer/examples/hermes_adapter.py --base http://127.0.0.1:8765 --poll 5
```

It prints the pending reviews with their gate table, saves artifacts to a directory,
and either decides interactively or applies a policy (auto-reject on `gate: reject`,
auto-approve only on `pass` + `risk.low`, everything else waits for a human).

## 9. Client library

```python
from evo_layer.hermes_interface.client import HermesClient

client = HermesClient("http://127.0.0.1:8765", token=os.environ["EVOLAYER_HERMES_TOKEN"])
for pending in client.pending():
    client.save_artifact(pending["proposal_id"], "episodes.jsonl", "/tmp/review")
    client.decide(pending["proposal_id"], "reject", actor="hermes", reason="fall rate too high")
```

`ClientError.status` carries the HTTP code so your adapter can branch on `403`
(token) versus `409` (refused decision).

## 10. Versioning

`contract_version` changes only on a breaking change to these shapes. Additive
fields are allowed at any time; consumers must ignore unknown keys. Pin the version
your adapter implements and log it — mismatches are the most common integration bug.
