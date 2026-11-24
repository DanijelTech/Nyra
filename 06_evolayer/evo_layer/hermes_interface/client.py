"""Client for a local Hermes to talk to EvoLayer.

This is the counterpart of :mod:`evo_layer.hermes_interface.bridge`: it is the
minimal, dependency-free connector your Hermes imports (or copies) to

* poll for reviews,
* fetch a report and its artifacts (video, logs),
* submit a decision (bearer token and/or signed callback),
* confirm the audit chain is intact before acting.

Everything is standard library, so it can be vendored into any Hermes runtime.

Example::

    from evo_layer.hermes_interface.client import HermesClient

    client = HermesClient("http://127.0.0.1:8765", token=os.environ["EVOLAYER_HERMES_TOKEN"])
    for pending in client.pending():
        report = client.report(pending["proposal_id"])
        client.save_artifact(pending["proposal_id"], "episodes.jsonl", "/tmp")
        client.decide(pending["proposal_id"], "approve", actor="hermes", reason="looks good")
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .contract import (
    CONTRACT_VERSION,
    HEADER_CONTRACT,
    HEADER_SIGNATURE,
    HEADER_TIMESTAMP,
    sign,
)


class ClientError(RuntimeError):
    def __init__(self, message: str, status: int = 0, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class HermesClient:
    base_url: str
    token: str = ""
    timeout_s: float = 15.0
    signing_secret: str = ""
    contract: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    # -- plumbing --------------------------------------------------------
    def _request(self, method: str, path: str, *, body: dict[str, Any] | None = None) -> tuple[int, bytes]:
        raw = b""
        headers = {"Accept": "application/json", "User-Agent": "EvoLayer-HermesClient/1.0"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if body is not None:
            raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
            if self.signing_secret:
                import time

                stamp = str(int(time.time()))
                headers[HEADER_SIGNATURE] = sign(self.signing_secret, stamp, raw)
                headers[HEADER_TIMESTAMP] = stamp
                headers[HEADER_CONTRACT] = self.contract
        request = urllib.request.Request(f"{self.base_url}{path}", data=raw or None, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return int(response.status), response.read()
        except urllib.error.HTTPError as exc:
            raise ClientError(f"{method} {path} failed", int(exc.code), exc.read().decode("utf-8", "replace")) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ClientError(f"{method} {path} unreachable: {type(exc).__name__}: {exc}") from exc

    def _json(self, method: str, path: str, *, body: dict[str, Any] | None = None) -> dict[str, Any]:
        status, raw = self._request(method, path, body=body)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ClientError(f"{method} {path} returned non-JSON", status, raw[:400].decode("utf-8", "replace")) from exc

    # -- read ------------------------------------------------------------
    def healthz(self) -> dict[str, Any]:
        return self._json("GET", "/healthz")

    def schema(self) -> dict[str, Any]:
        return self._json("GET", "/api/contract")

    def backends(self) -> list[dict[str, Any]]:
        return list(self._json("GET", "/api/backends").get("backends", []))

    def pending(self) -> list[dict[str, Any]]:
        return list(self._json("GET", "/api/pending").get("pending", []))

    def proposal(self, proposal_id: str) -> dict[str, Any]:
        return self._json("GET", f"/api/proposals/{proposal_id}")

    def report(self, proposal_id: str) -> dict[str, Any]:
        return self._json("GET", f"/api/proposals/{proposal_id}/report")

    def artifact(self, proposal_id: str, name: str) -> bytes:
        _, raw = self._request("GET", f"/api/proposals/{proposal_id}/artifacts/{name}")
        return raw

    def save_artifact(self, proposal_id: str, name: str, directory: str | Path) -> Path:
        target = Path(directory) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.artifact(proposal_id, name))
        return target

    def audit(self, since: int = 0) -> list[dict[str, Any]]:
        return list(self._json("GET", f"/api/audit?since={int(since)}").get("entries", []))

    # -- write -----------------------------------------------------------
    def decide(
        self,
        proposal_id: str,
        verdict: str,
        *,
        actor: str,
        reason: str,
        token: str | None = None,
        modifications: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = {
            "verdict": verdict,
            "actor": actor,
            "reason": reason,
            "modifications": modifications or {},
        }
        if token:
            body["token"] = token
        return self._json("POST", f"/api/proposals/{proposal_id}/decide", body=body)

    # -- convenience -----------------------------------------------------
    def drain(self, decision: str | None = None, *, actor: str = "", reason: str = "") -> Iterable[tuple[str, Any]]:
        """Yield each pending proposal, optionally deciding all of them the same way."""
        for pending in self.pending():
            proposal_id = pending["proposal_id"]
            yield proposal_id, self.proposal(proposal_id)
            if decision:
                yield proposal_id, self.decide(proposal_id, decision, actor=actor, reason=reason)


def from_env(base_url: str = "http://127.0.0.1:8765", env: dict[str, str] | None = None) -> HermesClient:
    env = dict(os.environ if env is None else env)
    return HermesClient(
        base_url=base_url,
        token=env.get("EVOLAYER_HERMES_TOKEN", ""),
        signing_secret=env.get("EVOLAYER_HERMES_OUTBOUND_SECRET", ""),
    )
