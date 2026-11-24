"""The Hermes interface: the human/agent veto.

Three pieces, all auditable:

* a **report** — one Markdown document plus a JSON payload containing the diff,
  the measured metrics, the gate table, the artifact hashes and what the evidence
  does *not* prove;
* a **decision token** — an HMAC over ``(proposal, revision, verdict, actor,
  nonce, expiry)``. Tokens are single-use (the nonce is burned in the store), so
  a captured request cannot be replayed;
* the **decision intake** — the only code path that can move a proposal to
  ``approved``, and it refuses approval when the gate said ``reject``.

Approval of a ``review`` verdict is allowed and is the whole point of review; it
is recorded with the actor and reason so the human accountable for it is named in
the audit chain.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..audit import AuditLog
from ..config import EvoConfig
from ..ids import nonce as new_nonce
from ..models import Decision, Finding, Metrics, Proposal, SimulationResult, Verdict, canonical_json
from ..store import DuplicateDecision, Store
from ..versioning.snapshot import Snapshotter


class HermesError(RuntimeError):
    """The decision cannot be accepted."""


class TokenError(HermesError):
    """The decision token is missing, malformed, expired, forged or already used."""


@dataclass
class Report:
    proposal_id: str
    revision: int
    markdown: str
    payload: dict[str, Any]
    #: Filled in by the orchestrator when the review request is pushed to Hermes.
    delivery: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "revision": self.revision,
            "markdown": self.markdown,
            "delivery": self.delivery,
            **self.payload,
        }


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


class HermesInterface:
    def __init__(
        self,
        cfg: EvoConfig,
        store: Store,
        audit: AuditLog,
        snapshots: Snapshotter,
        *,
        secret: str | None = None,
        env: dict[str, str] | None = None,
    ):
        self.cfg = cfg
        self.store = store
        self.audit = audit
        self.snapshots = snapshots
        self._secret = secret
        self._env = env

    # -- secrets ---------------------------------------------------------
    @property
    def secret(self) -> str:
        if self._secret:
            return self._secret
        return self.cfg.hermes_secret(self._env)

    # -- token -----------------------------------------------------------
    def issue_token(
        self,
        *,
        proposal_id: str,
        revision: int,
        verdict: str,
        actor: str,
        ttl_s: int | None = None,
    ) -> str:
        if verdict not in ("approve", "reject", "modify"):
            raise TokenError(f"cannot issue a token for verdict {verdict!r}")
        if not actor.strip():
            raise TokenError("actor is required to issue a token")
        now = int(time.time())
        payload = {
            "p": proposal_id,
            "r": int(revision),
            "v": verdict,
            "actor": actor,
            "nonce": new_nonce(12),
            "iat": now,
            "exp": now + int(ttl_s or self.cfg.hermes.token_ttl_s),
        }
        body = _b64(canonical_json(payload).encode("utf-8"))
        signature = hmac.new(self.secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
        return f"{body}.{_b64(signature)}"

    def verify_token(self, token: str, *, consume: bool = True) -> dict[str, Any]:
        if not token or token.count(".") != 1:
            raise TokenError("malformed token")
        body, signature = token.split(".", 1)
        expected = hmac.new(self.secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
        try:
            provided = _unb64(signature)
        except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
            raise TokenError("malformed signature") from exc
        if not hmac.compare_digest(provided, expected):
            raise TokenError("signature mismatch")
        try:
            payload = json.loads(_unb64(body).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise TokenError("malformed payload") from exc
        if int(time.time()) >= int(payload.get("exp", 0)):
            raise TokenError("token expired")
        if consume:
            if not self.store.consume_nonce(str(payload.get("nonce", ""))):
                raise TokenError("token already used (nonce replay)")
        return payload

    # -- report ----------------------------------------------------------
    def build_report(
        self,
        proposal: Proposal,
        finding: Finding,
        sim: SimulationResult | None,
        verdict: Verdict | None,
        baseline: Metrics | None = None,
        *,
        extra_caveats: list[str] | None = None,
    ) -> Report:
        skill = self.cfg.skill(proposal.skill)
        lines: list[str] = []
        lines.append(f"# EvoLayer review — {skill.title}")
        lines.append("")
        lines.append(f"**Proposal** `{proposal.id}` revision {proposal.revision}  ")
        lines.append(f"**Skill** `{skill.id}` · **kind** `{proposal.kind}` · "
                     f"**risk** {proposal.risk.risk_class.upper()} ({proposal.risk.score:.2f})  ")
        lines.append(f"**Created** {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime(proposal.created_at or time.time()))} "
                     f"by {proposal.created_by}  ")
        lines.append(f"**Finding** `{finding.id}` — {finding.title}")
        lines.append("")

        lines.append("## Why")
        lines.append("")
        lines.append(finding.summary or "(no summary)")
        lines.append("")
        evidence = finding.evidence or {}
        if evidence:
            lines.append("| evidence | value |")
            lines.append("|---|---|")
            for key in sorted(evidence):
                value = evidence[key]
                if isinstance(value, float):
                    value = f"{value:.4g}"
                lines.append(f"| `{key}` | {value} |")
            lines.append("")

        lines.append("## What changes")
        lines.append("")
        lines.append(proposal.rationale)
        lines.append("")
        if proposal.edits:
            for edit in proposal.edits:
                lines.append(f"### `{edit.repo}:{edit.path}` — {edit.description}")
                lines.append("")
                lines.append("```diff")
                lines.append(edit.diff.rstrip("\n"))
                lines.append("```")
                lines.append("")
        else:
            lines.append("_No file is modified._")
            lines.append("")
        if proposal.train_command:
            lines.append("Training command:")
            lines.append("")
            lines.append("```")
            lines.append(" ".join(proposal.train_command))
            lines.append("```")
            lines.append("")
        if proposal.config_overlay:
            lines.append("Config overlay:")
            lines.append("")
            lines.append("```yaml")
            lines.append(json.dumps(proposal.config_overlay, indent=2, sort_keys=True))
            lines.append("```")
            lines.append("")

        lines.append("## Rehearsal")
        lines.append("")
        if sim is None:
            lines.append("_Not rehearsed yet._")
        else:
            lines.append(f"- backend: `{sim.backend}` · status **{sim.status}** · "
                         f"{sim.episodes_n} episodes · {sim.finished_at - sim.started_at:.1f}s")
            lines.append(f"- command: `{' '.join(sim.command)}`")
            if sim.metrics:
                m = sim.metrics
                lines.append("")
                lines.append("| metric | candidate | baseline |")
                lines.append("|---|---|---|")
                rows = [
                    ("episodes", m.n, baseline.n if baseline else "-"),
                    ("success_rate", f"{m.success_rate:.2%}", f"{baseline.success_rate:.2%}" if baseline else "-"),
                    ("fall_rate", f"{m.fall_rate:.2%}", f"{baseline.fall_rate:.2%}" if baseline else "-"),
                    ("torque_peak_nm", f"{m.torque_peak_nm:.1f}", f"{baseline.torque_peak_nm:.1f}" if baseline else "-"),
                    ("joint_vel_peak_rps", f"{m.joint_vel_peak_rps:.1f}",
                     f"{baseline.joint_vel_peak_rps:.1f}" if baseline else "-"),
                    ("collision_rate", f"{m.collision_rate:.2f}", f"{baseline.collision_rate:.2f}" if baseline else "-"),
                ]
                for row in rows:
                    lines.append(f"| {row[0]} | {row[1]} | {row[2]} |")
            if sim.error:
                lines.append("")
                lines.append(f"**error:** `{sim.error[:400]}`")
            if sim.artifacts:
                lines.append("")
                lines.append("Artifacts (sha256 pinned):")
                lines.append("")
                for artifact in sim.artifacts:
                    lines.append(f"- `{artifact.kind}` **{artifact.name}** — {artifact.size} bytes — "
                                 f"`{artifact.sha256[:16]}…`")
        lines.append("")

        lines.append("## Safety gate")
        lines.append("")
        if verdict is None:
            lines.append("_Gate not evaluated._")
        else:
            lines.append(f"**Verdict: {verdict.verdict.upper()}**")
            lines.append("")
            lines.append("| check | status | measured | limit | detail |")
            lines.append("|---|---|---|---|---|")
            for check in verdict.checks:
                lines.append(
                    f"| `{check['name']}` | {check['status']} | {check['measured']} | {check['limit']} | "
                    f"{check['detail']} |"
                )
        lines.append("")

        lines.append("## What this does not prove")
        lines.append("")
        caveats = [
            "Rehearsal uses a reduced physics model with deploy-equivalent PD gains; contact "
            "fidelity and sensor noise are not the real robot's.",
            "Statistics describe the measured population only; rare failures need more episodes.",
            "A passing gate certifies non-inferiority on the measured metrics, not overall safety.",
        ]
        caveats.extend(extra_caveats or [])
        for caveat in caveats:
            lines.append(f"- {caveat}")
        lines.append("")

        lines.append("## Rollback")
        lines.append("")
        lines.append("Deployment takes a verified snapshot and a git tag before writing anything. "
                     "Roll back with:")
        lines.append("")
        lines.append("```bash")
        lines.append("evolayer rollback --deployment <id>")
        lines.append("```")
        lines.append("")

        lines.append("## Decide")
        lines.append("")
        lines.append("```bash")
        lines.append(f"evolayer approve --proposal {proposal.id} --actor <you> --reason 'why'")
        lines.append(f"evolayer reject  --proposal {proposal.id} --actor <you> --reason 'why'")
        lines.append(f"evolayer modify  --proposal {proposal.id} --actor <you> --reason 'why' \\")
        lines.append("        --set factor=1.25")
        lines.append("```")
        lines.append("")
        if verdict is not None and verdict.verdict == "reject":
            lines.append("> **The gate rejected this proposal. Approval is not possible; "
                         "reject or modify it.**")
            lines.append("")

        markdown = "\n".join(lines)
        payload = {
            "skill": skill.id,
            "skill_title": skill.title,
            "kind": proposal.kind,
            "risk": proposal.risk.to_dict(),
            "status": proposal.status,
            "edits": [edit.to_dict() for edit in proposal.edits],
            "train_command": proposal.train_command,
            "config_overlay": proposal.config_overlay,
            "sim_plan": proposal.sim_plan.to_dict(),
            "validation": proposal.validation,
            "finding": finding.to_dict(),
            "baseline": baseline.to_dict() if baseline else None,
            "simulation": sim.to_dict() if sim else None,
            "verdict": verdict.to_dict() if verdict else None,
            "caveats": caveats,
            "evidence_class": getattr(sim, "backend", None) and self._evidence_class(sim.backend) if sim else None,
            "generated_at": time.time(),
        }
        return Report(proposal_id=proposal.id, revision=proposal.revision, markdown=markdown, payload=payload)

    def write_report(self, report: Report) -> Path:
        directory = self.cfg.artifacts_dir / report.proposal_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"report-r{report.revision}.md"
        path.write_text(report.markdown, encoding="utf-8")
        json_path = directory / f"report-r{report.revision}.json"
        json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        self.audit.append(
            "hermes.report_written",
            subject=report.proposal_id,
            actor="evolayer",
            data={"revision": report.revision, "markdown": str(path), "json": str(json_path)},
        )
        return path

    # -- decision --------------------------------------------------------
    def decide(
        self,
        proposal: Proposal,
        *,
        verdict: str,
        actor: str,
        reason: str,
        token: str | None = None,
        gate: Verdict | None = None,
        sim: SimulationResult | None = None,
        modifications: dict[str, Any] | None = None,
        require_token: bool | None = None,
    ) -> Decision:
        # A decision is only meaningful for a proposal the store knows: the FK below
        # would fail, and a clearer error is better than an integrity error.
        try:
            self.store.get_proposal(proposal.id)
        except Exception as exc:  # noqa: BLE001 - NotFound
            raise HermesError(f"unknown proposal {proposal.id!r}: record it before deciding") from exc
        require = self.cfg.hermes.require_decision_token if require_token is None else require_token
        if verdict == "approve":
            if gate is None or gate.verdict == "reject":
                raise HermesError(
                    "approval refused: the safety gate rejected this proposal (or never ran)"
                )
            if sim is None or sim.status != "ok" or sim.metrics is None:
                raise HermesError("approval refused: no successful rehearsal is on record")
            if sim.backend and self._evidence_class(sim.backend) != "rehearsal":
                raise HermesError(
                    f"approval refused: backend {sim.backend!r} provides replay evidence, which cannot "
                    "certify a change"
                )
        nonce = new_nonce(12)
        if require:
            if not token:
                raise TokenError("a decision token is required (use the Hermes API or `evolayer issue-token`)")
            payload = self.verify_token(token)
            if payload.get("p") != proposal.id or int(payload.get("r", -1)) != proposal.revision:
                raise TokenError("token was issued for a different proposal revision")
            if payload.get("v") != verdict:
                raise TokenError(f"token authorises {payload.get('v')!r}, not {verdict!r}")
            if payload.get("actor") != actor:
                raise TokenError("token actor does not match the decision actor")
            nonce = str(payload["nonce"])

        decision = Decision(
            proposal_id=proposal.id,
            revision=proposal.revision,
            verdict=verdict,
            actor=actor,
            reason=reason,
            nonce=nonce,
            created_at=time.time(),
            modifications=dict(modifications or {}),
        )
        try:
            self.store.add_decision(decision)
        except DuplicateDecision as exc:
            raise HermesError(str(exc)) from exc
        self.audit.append(
            f"hermes.{verdict}",
            subject=proposal.id,
            actor=actor,
            data={
                "revision": proposal.revision,
                "reason": reason,
                "nonce": nonce,
                "gate": gate.verdict if gate else None,
                "sim_status": sim.status if sim else None,
                "modifications": decision.modifications,
            },
        )
        return decision

    def _evidence_class(self, backend: str) -> str:
        from ..simulator.backends import BACKENDS

        instance = BACKENDS.get(backend)
        if instance is None:
            return "unknown"
        return getattr(instance, "evidence_class", "producer")
