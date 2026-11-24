"""Outbound bridge: EvoLayer → your local Hermes.

Two delivery modes, both supported by the same request object:

``webhook``
    EvoLayer POSTs a signed :class:`ReviewRequest` to the URL your Hermes exposes
    (``hermes.outbound.url``). Retries with exponential backoff; every attempt is
    audited. A delivery failure never blocks the loop — it is recorded and the
    request stays retrievable through ``GET /api/pending``.

``pull``
    Nothing is pushed. Your Hermes polls ``GET /api/pending`` and
    ``GET /api/proposals/<id>``. Useful when the robot host cannot reach your
    workstation.

The bridge never sends the decision token or any secret: it only publishes what
Hermes needs to decide, signed with ``hermes.outbound.secret_env`` so your
endpoint can verify authenticity.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ..audit import AuditLog
from ..config import EvoConfig
from .contract import (
    CONTRACT_VERSION,
    EVENT_REVIEW_REQUESTED,
    HEADER_CONTRACT,
    HEADER_EVENT,
    HEADER_SIGNATURE,
    HEADER_TIMESTAMP,
    ReviewRequest,
    sign,
)


class BridgeError(RuntimeError):
    """The request could not be delivered (kept for retry / pull)."""


@dataclass
class Delivery:
    ok: bool
    attempts: int
    status: int
    detail: str
    mode: str
    url: str = ""
    sent_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "attempts": self.attempts,
            "status": self.status,
            "detail": self.detail,
            "mode": self.mode,
            "url": self.url,
            "sent_at": self.sent_at,
        }


@dataclass
class BridgeOptions:
    mode: str = "pull"
    url: str = ""
    secret: str = ""
    timeout_s: float = 10.0
    retries: int = 3
    backoff_s: float = 1.5
    verify_tls: bool = True
    extra_headers: dict[str, str] = field(default_factory=dict)


class HermesBridge:
    def __init__(self, cfg: EvoConfig, audit: AuditLog, *, env: dict[str, str] | None = None):
        self.cfg = cfg
        self.audit = audit
        self.env = dict(env or {})
        outbound = (cfg.hermes.outbound or {}) if hasattr(cfg.hermes, "outbound") else {}
        self.options = BridgeOptions(
            mode=str(outbound.get("mode", "pull")),
            url=str(outbound.get("url", "")),
            secret=self.env.get(str(outbound.get("secret_env", "EVOLAYER_HERMES_OUTBOUND_SECRET")), ""),
            timeout_s=float(outbound.get("timeout_s", 10.0)),
            retries=int(outbound.get("retries", 3)),
            backoff_s=float(outbound.get("backoff_s", 1.5)),
            verify_tls=bool(outbound.get("verify_tls", True)),
            extra_headers=dict(outbound.get("headers") or {}),
        )

    # -- request construction -------------------------------------------
    def review_request(
        self,
        *,
        proposal,
        report,
        verdict,
        sim,
        baseline,
        artifacts: list[dict[str, Any]] | None = None,
        decide_url: str | None = None,
        report_url: str | None = None,
    ) -> ReviewRequest:
        payload = report.payload
        base = self._public_base()
        return ReviewRequest(
            contract=CONTRACT_VERSION,
            event=EVENT_REVIEW_REQUESTED,
            proposal_id=proposal.id,
            revision=proposal.revision,
            skill=proposal.skill,
            skill_title=payload.get("skill_title", proposal.skill),
            kind=proposal.kind,
            risk_class=proposal.risk.risk_class,
            risk_score=proposal.risk.score,
            gate=verdict.verdict,
            gate_checks=verdict.checks,
            sim_summary={
                "backend": sim.backend if sim else None,
                "status": sim.status if sim else None,
                "episodes": sim.episodes_n if sim else 0,
                "evidence_class": payload.get("evidence_class"),
            },
            baseline=baseline.to_dict() if baseline else None,
            metrics=sim.metrics.to_dict() if sim and sim.metrics else None,
            diff_files=[edit.to_dict() for edit in proposal.edits],
            report_url=report_url or f"{base}/api/proposals/{proposal.id}/report",
            artifacts=artifacts
            or [
                {
                    "name": artifact.name,
                    "kind": artifact.kind,
                    "size": artifact.size,
                    "sha256": artifact.sha256,
                    "url": f"{base}/api/proposals/{proposal.id}/artifacts/{artifact.name}",
                }
                for artifact in (sim.artifacts if sim else [])
            ],
            decide_url=decide_url or f"{base}/api/proposals/{proposal.id}/decide",
            rollback_hint="evolayer rollback --deployment <id> --actor <you>",
            caveats=list(payload.get("caveats") or []),
        )

    def _public_base(self) -> str:
        port = self.cfg.hermes.port
        host = "127.0.0.1" if self.cfg.hermes.bind in ("0.0.0.0", "::") else self.cfg.hermes.bind
        return f"http://{host}:{port}"

    # -- delivery --------------------------------------------------------
    def notify(self, request: ReviewRequest) -> Delivery:
        body = request.to_wire()
        started = time.time()
        if self.options.mode == "pull":
            self.audit.append(
                "bridge.review_available",
                subject=request.proposal_id,
                actor="evolayer",
                data={"mode": "pull", "payload": body},
            )
            return Delivery(ok=True, attempts=0, status=0, detail="pull mode: available via GET /api/pending",
                            mode="pull", sent_at=started)

        if self.options.mode != "webhook":
            raise BridgeError(f"unknown bridge mode {self.options.mode!r} (webhook|pull)")
        if not self.options.url:
            raise BridgeError("hermes.outbound.url is required for webhook mode")
        if not self.options.secret:
            raise BridgeError(
                "hermes.outbound secret is not set; refusing to POST an unsigned review request"
            )

        raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        last = Delivery(ok=False, attempts=0, status=0, detail="not attempted", mode="webhook",
                        url=self.options.url, sent_at=started)
        for attempt in range(1, max(1, self.options.retries) + 1):
            stamp = str(int(time.time()))
            headers = {
                "Content-Type": "application/json",
                HEADER_SIGNATURE: sign(self.options.secret, stamp, raw),
                HEADER_TIMESTAMP: stamp,
                HEADER_CONTRACT: CONTRACT_VERSION,
                HEADER_EVENT: request.event,
                "User-Agent": "EvoLayer-HermesBridge/1.0",
                **self.options.extra_headers,
            }
            http_request = urllib.request.Request(self.options.url, data=raw, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(http_request, timeout=self.options.timeout_s) as response:
                    status = int(response.status)
                    last = Delivery(ok=200 <= status < 300, attempts=attempt, status=status,
                                    detail=f"delivered ({status})", mode="webhook",
                                    url=self.options.url, sent_at=time.time())
                    break
            except urllib.error.HTTPError as exc:
                last = Delivery(ok=False, attempts=attempt, status=int(exc.code),
                                detail=f"HTTP {exc.code}", mode="webhook", url=self.options.url,
                                sent_at=time.time())
                if 400 <= int(exc.code) < 500:
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = Delivery(ok=False, attempts=attempt, status=0, detail=f"{type(exc).__name__}: {exc}",
                                mode="webhook", url=self.options.url, sent_at=time.time())
            if attempt < max(1, self.options.retries):
                time.sleep(self.options.backoff_s * attempt)

        self.audit.append(
            "bridge.notify",
            subject=request.proposal_id,
            actor="evolayer",
            data={"delivery": last.to_dict(), "contract": CONTRACT_VERSION},
        )
        if not last.ok:
            self._persist(request, body)
        return last

    def _persist(self, request: ReviewRequest, body: dict[str, Any]) -> None:
        """Undelivered requests stay retrievable: written next to the report."""
        directory = self.cfg.artifacts_dir / request.proposal_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"review-request-r{request.revision}.json"
        path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")

    def pending_payload(self, proposal_id: str, revision: int) -> dict[str, Any] | None:
        path = self.cfg.artifacts_dir / proposal_id / f"review-request-r{revision}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def self_test(self) -> Delivery:
        """Send a signed, empty review request to prove the wiring (used by `evolayer bridge-test`)."""
        probe = ReviewRequest(
            contract=CONTRACT_VERSION,
            event="review.requested",
            proposal_id="prp_selftest00",
            revision=1,
            skill="locomotion",
            skill_title="self test",
            kind="eval_suite_expand",
            risk_class="low",
            risk_score=0.15,
            gate="pass",
            gate_checks=[],
            sim_summary={"backend": "none", "status": "ok", "episodes": 0, "evidence_class": "rehearsal"},
            baseline=None,
            metrics=None,
            diff_files=[],
            report_url=f"{self._public_base()}/api/proposals/prp_selftest00/report",
            artifacts=[],
            decide_url=f"{self._public_base()}/api/proposals/prp_selftest00/decide",
            rollback_hint="n/a",
            caveats=["self test payload; do not act on it"],
        )
        return self.notify(probe)
