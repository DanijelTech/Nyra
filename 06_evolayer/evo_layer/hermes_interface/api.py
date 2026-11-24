"""Hermes HTTP API — the remote approval surface.

A single-file, standard-library server. The security properties that matter:

* **bind loopback by default** (``127.0.0.1``); the config refuses to disable auth
  on a non-loopback bind;
* **bearer auth** with a constant-time comparison; every mutating request is
  refused without it;
* **bounded request bodies** (default 64 KiB) and a JSON-only decision endpoint;
* **no path traversal** in artifact serving: the requested name is resolved and
  must stay inside the proposal's artifact directory;
* **every mutation is audited** with the actor and the client address.

Endpoints::

    GET  /healthz
    GET  /api/contract               (the integration contract: schemas + headers)
    GET  /api/pending
    GET  /api/proposals/<id>
    GET  /api/proposals/<id>/report
    GET  /api/proposals/<id>/artifacts/<name>
    POST /api/proposals/<id>/decide     {"verdict","actor","reason","token"?}
    GET  /api/audit?since=<seq>
    GET  /api/backends
"""

from __future__ import annotations

import hmac
import json
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

from ..audit import AuditLog
from ..config import EvoConfig
from ..hermes_interface.interface import HermesError, HermesInterface, TokenError
from ..store import NotFound, Store


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class ApiContext:
    cfg: EvoConfig
    store: Store
    audit: AuditLog
    hermes: HermesInterface
    token: str
    #: Returns the JSON body to send back. Must not contain secrets.
    decide_hook: Callable[[str, str, str, str, str | None], dict[str, Any]] | None = None
    artifacts_provider: Callable[[str], list[Path]] | None = None
    outbound_secret: str = ""
    #: Roots an artifact may be served from (reports and simulator workspaces).
    artifact_roots: list[Path] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class _Handler(BaseHTTPRequestHandler):
    server_version = "EvoLayerHermes/1.0"
    protocol_version = "HTTP/1.1"

    context: ApiContext  # injected by the factory

    # -- plumbing --------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib signature
        if self.context.cfg.hermes.require_auth:
            self.context.audit.append(
                "api.request",
                subject=self.path,
                actor=self.client_address[0],
                data={"method": self.command},
            )

    def _authorised(self) -> bool:
        if not self.context.cfg.hermes.require_auth:
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Bearer "):
            provided = header[len("Bearer ") :].strip()
            return hmac.compare_digest(provided.encode("utf-8"), self.context.token.encode("utf-8"))
        # A local Hermes that holds the outbound signing secret may authenticate a
        # callback with an HMAC instead of a bearer token.
        return self._signature_ok(b"", allow_empty_body=True)

    def _has_signature_headers(self) -> bool:
        from .contract import HEADER_SIGNATURE, HEADER_TIMESTAMP

        return bool(self.headers.get(HEADER_SIGNATURE) and self.headers.get(HEADER_TIMESTAMP))

    def _signature_ok(self, body: bytes, *, allow_empty_body: bool = False) -> bool:
        from .contract import HEADER_SIGNATURE, HEADER_TIMESTAMP, ContractError, verify

        secret = self.context.outbound_secret
        if not secret:
            return False
        signature = self.headers.get(HEADER_SIGNATURE, "")
        timestamp = self.headers.get(HEADER_TIMESTAMP, "")
        if not signature and not allow_empty_body:
            return False
        try:
            verify(secret, signature, timestamp, body)
        except ContractError:
            return False
        return True

    def _send(self, status: int, payload: dict[str, Any] | bytes, *, content_type: str = "application/json") -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send(status, {"error": message})

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ApiError(HTTPStatus.BAD_REQUEST, "request body required")
        if length > self.context.cfg.hermes.max_body_bytes:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body too large")
        return self.rfile.read(length)

    # -- routes ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        try:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path == "/healthz":
                ok, count, broken, reason = self.context.audit.verify()
                self._send(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "audit": {"entries": count, "chain_ok": ok, "broken_at": broken, "reason": reason},
                        "version": "1.0.0",
                    },
                )
                return
            if path == "/api/contract":
                # The contract is the spec your Hermes implements; it holds no state.
                from .contract import schemas

                self._send(HTTPStatus.OK, schemas())
                return
            if not self._authorised():
                self._error(HTTPStatus.UNAUTHORIZED, "bearer token required")
                return
            if path == "/api/pending":
                self._send(HTTPStatus.OK, {"pending": self._pending()})
                return
            if path == "/api/backends":
                from ..simulator.backends import availability_report

                self._send(HTTPStatus.OK, {"backends": availability_report(self.context.cfg)})
                return
            if path == "/api/audit":
                since = int(parse_qs(parsed.query).get("since", ["0"])[0])
                rows = [entry.to_dict() for entry in self.context.audit.entries() if entry.seq > since]
                self._send(HTTPStatus.OK, {"entries": rows})
                return
            if path.startswith("/api/proposals/"):
                self._proposal_routes(path)
                return
            self._error(HTTPStatus.NOT_FOUND, "no such endpoint")
        except ApiError as exc:
            self._error(exc.status, exc.message)
        except Exception as exc:  # noqa: BLE001 - never leak a stack trace to the wire
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        try:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            # A signed callback carries its credential in headers over the body, which
            # is only readable inside the handler; so presence is enough here and the
            # signature is verified against the exact bytes in `_decide`.
            if not self._authorised() and not self._has_signature_headers():
                self._error(HTTPStatus.UNAUTHORIZED, "bearer token or signature required")
                return
            if path.endswith("/decide") and path.startswith("/api/proposals/"):
                self._decide(path)
                return
            self._error(HTTPStatus.NOT_FOUND, "no such endpoint")
        except ApiError as exc:
            self._error(exc.status, exc.message)
        except Exception as exc:  # noqa: BLE001
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    # -- helpers ---------------------------------------------------------
    def _pending(self) -> list[dict[str, Any]]:
        rows = []
        for proposal in self.context.store.proposals(status="awaiting_hermes"):
            verdict = self.context.store.latest_verdict(proposal.id, proposal.revision)
            rows.append(
                {
                    "proposal_id": proposal.id,
                    "revision": proposal.revision,
                    "skill": proposal.skill,
                    "title": proposal.title,
                    "risk_class": proposal.risk.risk_class,
                    "risk_score": proposal.risk.score,
                    "gate": verdict.verdict if verdict else None,
                    "created_at": proposal.created_at,
                }
            )
        return rows

    def _proposal_routes(self, path: str) -> None:
        parts = [segment for segment in path.split("/") if segment]
        if len(parts) < 3:
            raise ApiError(HTTPStatus.NOT_FOUND, "no such endpoint")
        proposal_id = parts[2]
        try:
            proposal = self.context.store.get_proposal(proposal_id)
        except NotFound as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, str(exc)) from exc

        if len(parts) == 3:
            verdict = self.context.store.latest_verdict(proposal.id, proposal.revision)
            sim = self.context.store.latest_simulation(proposal.id, proposal.revision)
            self._send(
                HTTPStatus.OK,
                {
                    "proposal": proposal.to_dict(),
                    "verdict": verdict.to_dict() if verdict else None,
                    "simulation": sim.to_dict() if sim else None,
                },
            )
            return

        if parts[3] == "report":
            candidates = sorted((self.context.cfg.artifacts_dir / proposal_id).glob("report-r*.json"))
            if not candidates:
                raise ApiError(HTTPStatus.NOT_FOUND, "report not generated yet")
            self._send(HTTPStatus.OK, json.loads(candidates[-1].read_text(encoding="utf-8")))
            return

        if parts[3] == "artifacts" and len(parts) >= 5:
            name = parts[4]
            if "/" in name or name.startswith(".") or ".." in name:
                raise ApiError(HTTPStatus.BAD_REQUEST, "invalid artifact name")
            paths = self.context.artifacts_provider(proposal_id) if self.context.artifacts_provider else []
            roots = [root.resolve() for root in (self.context.artifact_roots or [self.context.cfg.artifacts_dir])]
            for candidate in paths:
                if candidate.name != name:
                    continue
                resolved = candidate.resolve()
                if not any(_is_within(resolved, root) for root in roots):
                    raise ApiError(HTTPStatus.FORBIDDEN, "artifact outside the configured roots")
                content_type = "video/mp4" if resolved.suffix == ".mp4" else "application/octet-stream"
                self._send(HTTPStatus.OK, resolved.read_bytes(), content_type=content_type)
                return
            raise ApiError(HTTPStatus.NOT_FOUND, f"artifact not found: {name}")

        raise ApiError(HTTPStatus.NOT_FOUND, "no such endpoint")

    def _decide(self, path: str) -> None:
        parts = [segment for segment in path.split("/") if segment]
        proposal_id = parts[2]
        raw = self._read_body()
        if not self._authorised() and not self._signature_ok(raw):
            raise ApiError(HTTPStatus.UNAUTHORIZED, "bearer token or a valid signature is required")
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "JSON object expected")
        verdict = str(body.get("verdict", "")).strip().lower()
        actor = str(body.get("actor", "")).strip()
        reason = str(body.get("reason", "")).strip()
        token = body.get("token")
        modifications = body.get("modifications") or {}
        if verdict not in ("approve", "reject", "modify"):
            raise ApiError(HTTPStatus.BAD_REQUEST, "verdict must be approve|reject|modify")
        if not actor or len(reason) < 3:
            raise ApiError(HTTPStatus.BAD_REQUEST, "actor and a reason (>= 3 chars) are required")
        if not isinstance(modifications, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "modifications must be an object")

        if self.context.decide_hook is None:
            raise ApiError(HTTPStatus.NOT_IMPLEMENTED, "no decision hook configured")
        try:
            result = self.context.decide_hook(proposal_id, verdict, actor, reason, token)
        except TokenError as exc:
            raise ApiError(HTTPStatus.FORBIDDEN, str(exc)) from exc
        except HermesError as exc:
            raise ApiError(HTTPStatus.CONFLICT, str(exc)) from exc
        except NotFound as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, str(exc)) from exc
        self.context.audit.append(
            "api.decision",
            subject=proposal_id,
            actor=actor,
            data={"verdict": verdict, "client": self.client_address[0]},
        )
        # The hook owns the response body (it is the same envelope the CLI prints).
        self._send(HTTPStatus.OK, result)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def make_server(context: ApiContext) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (_Handler,), {"context": context})
    server = ThreadingHTTPServer((context.cfg.hermes.bind, context.cfg.hermes.port), handler)
    server.daemon_threads = True
    return server


def serve(context: ApiContext) -> None:  # pragma: no cover - long running
    server = make_server(context)
    context.audit.append(
        "api.started",
        subject=f"{context.cfg.hermes.bind}:{server.server_address[1]}",
        actor="evolayer",
        data={"require_auth": context.cfg.hermes.require_auth},
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        context.audit.append("api.stopped", subject="evolayer", actor="evolayer", data={})
        server.server_close()
