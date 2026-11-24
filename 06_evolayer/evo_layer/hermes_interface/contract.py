"""The Hermes integration contract.

This module is the *only* place that defines what crosses the boundary between
EvoLayer and a Hermes deployment. EvoLayer contains no approval logic of its own:
it publishes a review request (signed), accepts a decision (signed or
bearer-authenticated) and nothing else.

Versioning rule: ``CONTRACT_VERSION`` changes only on a breaking change to the
JSON shapes below. Additive fields do not bump it, and consumers must ignore
unknown keys.

    EvoLayer ──ReviewRequest (signed webhook)──▶ Hermes
    EvoLayer ◀──Decision (signed callback)────── Hermes
              ◀──GET /api/proposals/<id>───────  (poll instead of push)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

CONTRACT_VERSION = "1"

#: Webhook event names.
EVENT_REVIEW_REQUESTED = "review.requested"
EVENT_REVIEW_UPDATED = "review.updated"

#: Decision values accepted on the callback.
DECISIONS = ("approve", "reject", "modify")

#: Headers EvoLayer sends and expects.
HEADER_SIGNATURE = "X-EvoLayer-Signature"
HEADER_TIMESTAMP = "X-EvoLayer-Timestamp"
HEADER_CONTRACT = "X-EvoLayer-Contract"
HEADER_EVENT = "X-EvoLayer-Event"

#: Reject a callback whose timestamp is further than this from now (replay window).
MAX_CLOCK_SKEW_S = 300


class ContractError(RuntimeError):
    """A payload violates the contract, or its signature cannot be trusted."""


def canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign(secret: str, timestamp: str, body: bytes) -> str:
    """``sha256=<hex>`` over ``timestamp + "." + body`` (GitHub-style scheme)."""
    mac = hmac.new(secret.encode("utf-8"), timestamp.encode("ascii") + b"." + body, hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


def verify(secret: str, signature: str, timestamp: str, body: bytes, *, now: float | None = None) -> None:
    if not signature or not signature.startswith("sha256="):
        raise ContractError("missing or malformed signature")
    if not timestamp.isdigit():
        raise ContractError("missing or malformed timestamp header")
    current = time.time() if now is None else now
    if abs(current - int(timestamp)) > MAX_CLOCK_SKEW_S:
        raise ContractError("timestamp outside the accepted clock skew window")
    expected = sign(secret, timestamp, body)
    if not hmac.compare_digest(signature, expected):
        raise ContractError("signature mismatch")


# --------------------------------------------------------------------------- payloads


@dataclass
class ReviewRequest:
    """What EvoLayer pushes to Hermes. ``to_wire`` is the authoritative shape."""

    contract: str
    event: str
    proposal_id: str
    revision: int
    skill: str
    skill_title: str
    kind: str
    risk_class: str
    risk_score: float
    gate: str
    gate_checks: list[dict[str, Any]]
    sim_summary: dict[str, Any]
    baseline: dict[str, Any] | None
    metrics: dict[str, Any] | None
    diff_files: list[dict[str, Any]]
    report_url: str
    artifacts: list[dict[str, Any]]
    decide_url: str
    rollback_hint: str
    caveats: list[str] = field(default_factory=list)
    created_at: float = 0.0

    def to_wire(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "event": self.event,
            "created_at": self.created_at or time.time(),
            "proposal": {
                "id": self.proposal_id,
                "revision": self.revision,
                "skill": self.skill,
                "skill_title": self.skill_title,
                "kind": self.kind,
            },
            "risk": {"class": self.risk_class, "score": self.risk_score},
            "gate": {"verdict": self.gate, "checks": self.gate_checks},
            "rehearsal": {"summary": self.sim_summary, "metrics": self.metrics, "baseline": self.baseline},
            "changes": self.diff_files,
            "caveats": self.caveats,
            "links": {
                "report": self.report_url,
                "decide": self.decide_url,
                "artifacts": self.artifacts,
            },
            "rollback": self.rollback_hint,
        }


@dataclass
class CallbackDecision:
    verdict: str
    actor: str
    reason: str
    token: str | None = None
    modifications: dict[str, Any] = field(default_factory=dict)

    def to_wire(self, proposal_id: str, revision: int) -> dict[str, Any]:
        payload = {
            "contract": CONTRACT_VERSION,
            "proposal_id": proposal_id,
            "revision": revision,
            "verdict": self.verdict,
            "actor": self.actor,
            "reason": self.reason,
            "modifications": self.modifications,
            "sent_at": time.time(),
        }
        if self.token:
            payload["token"] = self.token
        return payload

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "CallbackDecision":
        verdict = str(payload.get("verdict", "")).strip().lower()
        if verdict not in DECISIONS:
            raise ContractError(f"verdict must be one of {DECISIONS}, got {verdict!r}")
        actor = str(payload.get("actor", "")).strip()
        reason = str(payload.get("reason", "")).strip()
        if not actor:
            raise ContractError("actor is required")
        if len(reason) < 3:
            raise ContractError("reason must explain the decision (>= 3 characters)")
        modifications = payload.get("modifications") or {}
        if not isinstance(modifications, dict):
            raise ContractError("modifications must be an object")
        token = payload.get("token")
        return cls(
            verdict=verdict,
            actor=actor,
            reason=reason,
            token=str(token) if token else None,
            modifications=dict(modifications),
        )


REVIEW_REQUEST_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "EvoLayer ReviewRequest",
    "type": "object",
    "required": ["contract", "event", "proposal", "risk", "gate", "links"],
    "properties": {
        "contract": {"type": "string"},
        "event": {"type": "string", "enum": [EVENT_REVIEW_REQUESTED, EVENT_REVIEW_UPDATED]},
        "created_at": {"type": "number"},
        "proposal": {
            "type": "object",
            "required": ["id", "revision", "skill", "kind"],
            "properties": {
                "id": {"type": "string"},
                "revision": {"type": "integer", "minimum": 1},
                "skill": {"type": "string"},
                "skill_title": {"type": "string"},
                "kind": {"type": "string"},
            },
        },
        "risk": {
            "type": "object",
            "required": ["class", "score"],
            "properties": {
                "class": {"type": "string", "enum": ["low", "medium", "high"]},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "gate": {
            "type": "object",
            "required": ["verdict"],
            "properties": {
                "verdict": {"type": "string", "enum": ["pass", "review", "reject"]},
                "checks": {"type": "array"},
            },
        },
        "rehearsal": {"type": "object"},
        "changes": {"type": "array"},
        "caveats": {"type": "array", "items": {"type": "string"}},
        "links": {
            "type": "object",
            "required": ["report", "decide", "artifacts"],
            "properties": {
                "report": {"type": "string"},
                "decide": {"type": "string"},
                "artifacts": {"type": "array"},
            },
        },
        "rollback": {"type": "string"},
    },
}

DECISION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "EvoLayer Decision",
    "type": "object",
    "required": ["proposal_id", "revision", "verdict", "actor", "reason"],
    "properties": {
        "contract": {"type": "string"},
        "proposal_id": {"type": "string"},
        "revision": {"type": "integer", "minimum": 1},
        "verdict": {"type": "string", "enum": list(DECISIONS)},
        "actor": {"type": "string", "minLength": 1},
        "reason": {"type": "string", "minLength": 3},
        "token": {"type": "string"},
        "modifications": {"type": "object"},
        "sent_at": {"type": "number"},
    },
}


def schemas() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "headers": {
            "signature": HEADER_SIGNATURE,
            "timestamp": HEADER_TIMESTAMP,
            "contract": HEADER_CONTRACT,
            "event": HEADER_EVENT,
        },
        "events": [EVENT_REVIEW_REQUESTED, EVENT_REVIEW_UPDATED],
        "decisions": list(DECISIONS),
        "review_request": REVIEW_REQUEST_SCHEMA,
        "decision": DECISION_SCHEMA,
    }
