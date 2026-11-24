"""Hermes: decision tokens, the integration contract, the HTTP fork, the bridge.

Everything runs against real HTTP servers and real HMAC verification.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from evo_layer.hermes_interface.api import ApiContext, make_server
from evo_layer.hermes_interface.bridge import HermesBridge
from evo_layer.hermes_interface.client import ClientError, HermesClient
from evo_layer.hermes_interface.contract import (
    CONTRACT_VERSION,
    HEADER_EVENT,
    HEADER_SIGNATURE,
    HEADER_TIMESTAMP,
    ContractError,
    CallbackDecision,
    ReviewRequest,
    schemas,
    sign,
    verify,
)
from evo_layer.hermes_interface.interface import HermesError, HermesInterface, TokenError
from evo_layer.models import Finding, Metrics, Proposal, Risk, SimPlan, Verdict
from evo_layer.versioning.snapshot import Snapshotter

SECRET = "test-secret-0123456789abcdef"


@pytest.fixture
def hermes(workspace):
    cfg = workspace["cfg"]
    from evo_layer.audit import AuditLog
    from evo_layer.store import Store

    store = Store(cfg.store_path)
    audit = AuditLog(cfg.audit_path)
    interface = HermesInterface(cfg, store, audit, Snapshotter(cfg.snapshots_dir), secret=SECRET, env={})
    store.add_finding(_finding())
    store.add_proposal(_proposal())
    return {"cfg": cfg, "store": store, "audit": audit, "hermes": interface}


def _finding() -> Finding:
    return Finding(id="fnd_h", skill="locomotion", kind="regression", severity="high", title="t",
                   summary="s", metric="fall_rate", baseline=0.01, current=0.2, delta_rel=19.0,
                   p_value=0.001, recommendation="reward_term_weight", params={"term": "x"}, created_at=1.0)


def _proposal() -> Proposal:
    from evo_layer.models import Edit

    return Proposal(
        id="prp_h", revision=1, finding_id="fnd_h", skill="locomotion", kind="reward_term_weight",
        title="t", rationale="r",
        edits=[Edit(repo="training", path="robolab/a.py",
                    diff="--- a/robolab/a.py\n+++ b/robolab/a.py\n@@ -1 +1 @@\n-x\n+y\n")],
        sim_plan=SimPlan(backend="mujoco", episodes=5), expected_effect="e",
        risk=Risk(score=0.4, risk_class="medium"), created_at=1.0,
    )


# --------------------------------------------------------------------------- tokens


def test_deciding_an_unknown_proposal_is_refused(hermes):
    import dataclasses

    unknown = dataclasses.replace(_proposal(), id="prp_unknown")
    with pytest.raises(HermesError) as excinfo:
        hermes["hermes"].decide(unknown, verdict="reject", actor="hermes", reason="nope",
                                require_token=False)
    assert "unknown proposal" in str(excinfo.value)


def test_token_roundtrip_and_single_use(hermes):
    interface = hermes["hermes"]
    token = interface.issue_token(proposal_id="prp_h", revision=1, verdict="approve", actor="hermes")
    payload = interface.verify_token(token)
    assert payload["p"] == "prp_h" and payload["v"] == "approve" and payload["actor"] == "hermes"
    with pytest.raises(TokenError) as excinfo:
        interface.verify_token(token)
    assert "already used" in str(excinfo.value)


def test_token_rejects_tampering(hermes):
    interface = hermes["hermes"]
    token = interface.issue_token(proposal_id="prp_h", revision=1, verdict="reject", actor="hermes")
    body, signature = token.split(".")
    forged = body[:-2] + ("AA" if body[-2:] != "AA" else "BB") + "." + signature
    with pytest.raises(TokenError):
        interface.verify_token(forged)
    with pytest.raises(TokenError):
        interface.verify_token("not-a-token")
    with pytest.raises(TokenError):
        interface.verify_token(token + "x")


def test_token_expires(hermes):
    interface = hermes["hermes"]
    token = interface.issue_token(proposal_id="prp_h", revision=1, verdict="approve", actor="hermes", ttl_s=1)
    time.sleep(1.1)
    with pytest.raises(TokenError) as excinfo:
        interface.verify_token(token)
    assert "expired" in str(excinfo.value)


# --------------------------------------------------------------------------- decisions


def _gate(verdict="pass") -> Verdict:
    return Verdict(proposal_id="prp_h", revision=1, verdict=verdict,
                   checks=[{"name": "fall_rate_max", "status": verdict, "measured": 0.0,
                            "limit": 0.02, "detail": "ok"}])


def _sim(status="ok", backend="mujoco") -> object:
    from evo_layer.models import SimulationResult

    return SimulationResult(proposal_id="prp_h", revision=1, backend=backend, status=status,
                            metrics=Metrics(n=100, success_rate=0.99, fall_rate=0.0),
                            started_at=0.0, finished_at=1.0, episodes_n=100)


def test_approve_requires_a_token_when_configured(hermes):
    interface = hermes["hermes"]
    hermes["cfg"].hermes.require_decision_token = True
    with pytest.raises(TokenError):
        interface.decide(_proposal(), verdict="approve", actor="hermes", reason="fine",
                         gate=_gate(), sim=_sim(), require_token=True)
    token = interface.issue_token(proposal_id="prp_h", revision=1, verdict="approve", actor="hermes")
    decision = interface.decide(_proposal(), verdict="approve", actor="hermes", reason="fine",
                                gate=_gate(), sim=_sim(), token=token, require_token=True)
    assert decision.verdict == "approve"
    hermes["cfg"].hermes.require_decision_token = False


def test_approval_is_impossible_after_a_gate_rejection(hermes):
    with pytest.raises(HermesError) as excinfo:
        hermes["hermes"].decide(_proposal(), verdict="approve", actor="hermes", reason="trust me",
                                gate=_gate("reject"), sim=_sim(), require_token=False)
    assert "gate rejected" in str(excinfo.value)


def test_approval_needs_a_successful_rehearsal(hermes):
    with pytest.raises(HermesError):
        hermes["hermes"].decide(_proposal(), verdict="approve", actor="hermes", reason="ok",
                                gate=_gate(), sim=_sim("failed"), require_token=False)


def test_replay_evidence_cannot_be_approved(hermes):
    with pytest.raises(HermesError) as excinfo:
        hermes["hermes"].decide(_proposal(), verdict="approve", actor="hermes", reason="ok",
                                gate=_gate(), sim=_sim(backend="fixture"), require_token=False)
    assert "replay evidence" in str(excinfo.value)


def test_decision_is_unique(hermes):
    interface = hermes["hermes"]
    interface.decide(_proposal(), verdict="reject", actor="hermes", reason="no thanks",
                     gate=_gate(), sim=_sim(), require_token=False)
    with pytest.raises(HermesError):
        interface.decide(_proposal(), verdict="reject", actor="hermes", reason="again",
                         gate=_gate(), sim=_sim(), require_token=False)


def test_token_must_match_the_decision(hermes):
    interface = hermes["hermes"]
    token = interface.issue_token(proposal_id="prp_h", revision=1, verdict="reject", actor="hermes")
    with pytest.raises(TokenError):
        interface.decide(_proposal(), verdict="approve", actor="hermes", reason="changed",
                         gate=_gate(), sim=_sim(), token=token, require_token=True)


# --------------------------------------------------------------------------- report


def test_report_contains_the_evidence_a_reviewer_needs(hermes):
    report = hermes["hermes"].build_report(_proposal(), _finding(), _sim(), _gate(), Metrics(n=100))
    assert "# EvoLayer review" in report.markdown
    assert "What this does not prove" in report.markdown
    assert "```diff" in report.markdown
    assert "fall_rate_max" in report.markdown
    assert report.payload["risk"]["risk_class"] == "medium"
    assert report.payload["verdict"]["verdict"] == "pass"
    assert report.payload["simulation"]["metrics"]["n"] == 100
    path = hermes["hermes"].write_report(report)
    assert path.is_file()
    assert json.loads(path.with_suffix(".json").read_text())["proposal_id"] == "prp_h"


def test_report_states_a_rejection_loudly(hermes):
    report = hermes["hermes"].build_report(_proposal(), _finding(), _sim(), _gate("reject"), None)
    assert "gate rejected this proposal" in report.markdown


# --------------------------------------------------------------------------- contract


def test_contract_signing_and_verification():
    body = json.dumps({"hello": "world"}, sort_keys=True).encode()
    stamp = str(int(time.time()))
    signature = sign(SECRET, stamp, body)
    verify(SECRET, signature, stamp, body)
    with pytest.raises(ContractError):
        verify("other-secret-0123456789", signature, stamp, body)
    with pytest.raises(ContractError):
        verify(SECRET, signature, stamp, body + b"x")
    old = str(int(time.time()) - 10_000)
    with pytest.raises(ContractError) as excinfo:
        verify(SECRET, sign(SECRET, old, body), old, body)
    assert "skew" in str(excinfo.value)


def test_contract_schemas_are_published():
    payload = schemas()
    assert payload["contract_version"] == CONTRACT_VERSION
    assert payload["decisions"] == ["approve", "reject", "modify"]
    assert set(payload["review_request"]["required"]) >= {"contract", "event", "proposal", "risk", "gate", "links"}


def test_callback_decision_validates_input():
    decision = CallbackDecision.from_wire({"verdict": "approve", "actor": "hermes", "reason": "signed off"})
    assert decision.verdict == "approve"
    wire = decision.to_wire("prp_h", 1)
    assert wire["proposal_id"] == "prp_h"
    with pytest.raises(ContractError):
        CallbackDecision.from_wire({"verdict": "maybe", "actor": "h", "reason": "ok"})
    with pytest.raises(ContractError):
        CallbackDecision.from_wire({"verdict": "approve", "actor": "", "reason": "ok"})
    with pytest.raises(ContractError):
        CallbackDecision.from_wire({"verdict": "approve", "actor": "h", "reason": "no"})


# --------------------------------------------------------------------------- HTTP API


@pytest.fixture
def api(hermes):
    cfg = hermes["cfg"]
    cfg.hermes.port = 0  # ephemeral
    cfg.hermes.require_auth = True
    token = "test-bearer-token-0123456789"

    def decide_hook(proposal_id, verdict, actor, reason, token_value):
        proposal = hermes["store"].get_proposal(proposal_id)
        gate = hermes["store"].latest_verdict(proposal.id, proposal.revision)
        sim = hermes["store"].latest_simulation(proposal.id, proposal.revision)
        decision = hermes["hermes"].decide(
            proposal, verdict=verdict, actor=actor, reason=reason, token=token_value,
            gate=gate, sim=sim, require_token=False,
        )
        return {"decision": decision.to_dict(), "proposal_id": proposal.id}

    # Seed the stored proposal with a passing gate and a rehearsal.
    store = hermes["store"]
    proposal = _proposal()
    store.add_verdict(_gate())
    store.add_simulation(_sim())
    store.set_proposal_status(proposal.id, "awaiting_hermes")

    context = ApiContext(cfg=cfg, store=store, audit=hermes["audit"], hermes=hermes["hermes"],
                         token=token, decide_hook=decide_hook, outbound_secret=SECRET,
                         artifacts_provider=lambda pid: [])
    server = make_server(context)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield {"base": base, "token": token, "store": store, "proposal": proposal}
    server.shutdown()
    server.server_close()


def _get(url: str, token: str | None = None) -> tuple[int, dict]:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def _post(url: str, payload: dict, token: str | None = None, headers: dict | None = None) -> tuple[int, dict]:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", **(headers or {})})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def test_api_health_and_contract_are_public(api):
    status, payload = _get(f"{api['base']}/healthz")
    assert status == 200 and payload["status"] == "ok"
    assert payload["audit"]["chain_ok"] is True
    status, payload = _get(f"{api['base']}/api/contract")
    assert status == 200 and payload["contract_version"] == CONTRACT_VERSION


def test_api_requires_auth_for_state(api):
    status, _ = _get(f"{api['base']}/api/pending")
    assert status == 401
    status, payload = _get(f"{api['base']}/api/pending", api["token"])
    assert status == 200
    assert payload["pending"][0]["proposal_id"] == api["proposal"].id
    assert payload["pending"][0]["gate"] == "pass"


def test_api_exposes_the_proposal_and_report(api):
    status, payload = _get(f"{api['base']}/api/proposals/{api['proposal'].id}", api["token"])
    assert status == 200
    assert payload["proposal"]["kind"] == "reward_term_weight"
    assert payload["verdict"]["verdict"] == "pass"
    status, _ = _get(f"{api['base']}/api/proposals/prp_missing0000", api["token"])
    assert status == 404


def test_api_decision_with_bearer_token(api):
    url = f"{api['base']}/api/proposals/{api['proposal'].id}/decide"
    status, payload = _post(url, {"verdict": "reject", "actor": "hermes", "reason": "not convinced"},
                            api["token"])
    assert status == 200, payload
    assert payload["decision"]["verdict"] == "reject"
    # A second decision on the same revision is refused.
    status, payload = _post(url, {"verdict": "reject", "actor": "hermes", "reason": "again"}, api["token"])
    assert status == 409


def test_api_decision_accepts_a_signed_callback(api):
    url = f"{api['base']}/api/proposals/{api['proposal'].id}/decide"
    body = json.dumps({"verdict": "reject", "actor": "local-hermes", "reason": "signed callback"},
                      separators=(",", ":")).encode()
    stamp = str(int(time.time()))
    headers = {HEADER_SIGNATURE: sign(SECRET, stamp, body), HEADER_TIMESTAMP: stamp}
    request = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(request, timeout=5) as response:
        payload = json.loads(response.read())
    assert payload["decision"]["actor"] == "local-hermes"


def test_api_rejects_a_bad_signature(api):
    url = f"{api['base']}/api/proposals/{api['proposal'].id}/decide"
    body = json.dumps({"verdict": "reject", "actor": "x", "reason": "forged"}, separators=(",", ":")).encode()
    stamp = str(int(time.time()))
    headers = {HEADER_SIGNATURE: sign("wrong-secret-0123456789", stamp, body), HEADER_TIMESTAMP: stamp}
    request = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", **headers})
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=5)
    assert excinfo.value.code == 401


def test_api_validates_the_decision_body(api):
    url = f"{api['base']}/api/proposals/{api['proposal'].id}/decide"
    status, _ = _post(url, {"verdict": "maybe", "actor": "h", "reason": "why not"}, api["token"])
    assert status == 400
    status, _ = _post(url, {"verdict": "approve", "actor": "", "reason": ""}, api["token"])
    assert status == 400


def test_api_blocks_artifact_traversal(api):
    status, _ = _get(f"{api['base']}/api/proposals/{api['proposal'].id}/artifacts/..%2f..%2fstate.db",
                     api["token"])
    assert status in (400, 404)


# --------------------------------------------------------------------------- bridge


class _Receiver(BaseHTTPRequestHandler):
    received: list = []

    def do_POST(self):  # noqa: N802 - stdlib signature
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _Receiver.received.append(
            {
                "body": json.loads(body),
                "signature": self.headers.get(HEADER_SIGNATURE),
                "timestamp": self.headers.get(HEADER_TIMESTAMP),
                "event": self.headers.get(HEADER_EVENT),
                "raw": body,
            }
        )
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):  # silence
        return


@pytest.fixture
def receiver():
    _Receiver.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/evolayer/review"
    server.shutdown()
    server.server_close()


def test_bridge_posts_a_signed_review_request(workspace, receiver):
    cfg = workspace["cfg"]
    cfg.hermes.outbound = {"mode": "webhook", "url": receiver,
                           "secret_env": "EVOLAYER_HERMES_OUTBOUND_SECRET"}
    bridge = HermesBridge(cfg, workspace["cfg"] and __import__("evo_layer.audit", fromlist=["AuditLog"]).AuditLog(cfg.audit_path),
                          env={"EVOLAYER_HERMES_OUTBOUND_SECRET": SECRET})
    request = ReviewRequest(
        contract=CONTRACT_VERSION, event="review.requested", proposal_id="prp_h", revision=1,
        skill="locomotion", skill_title="Locomotion", kind="reward_term_weight",
        risk_class="medium", risk_score=0.4, gate="pass", gate_checks=[], sim_summary={},
        baseline=None, metrics=None, diff_files=[], report_url="http://x/report",
        artifacts=[], decide_url="http://x/decide", rollback_hint="evolayer rollback",
        caveats=["test"],
    )
    delivery = bridge.notify(request)
    assert delivery.ok, delivery.to_dict()
    assert len(_Receiver.received) == 1
    received = _Receiver.received[0]
    verify(SECRET, received["signature"], received["timestamp"], received["raw"])
    assert received["event"] == "review.requested"
    assert received["body"]["proposal"]["id"] == "prp_h"
    assert received["body"]["caveats"] == ["test"]


def test_bridge_pull_mode_records_availability(workspace):
    cfg = workspace["cfg"]
    cfg.hermes.outbound = {"mode": "pull"}
    from evo_layer.audit import AuditLog

    bridge = HermesBridge(cfg, AuditLog(cfg.audit_path), env={})
    request = ReviewRequest(
        contract=CONTRACT_VERSION, event="review.requested", proposal_id="prp_pull", revision=1,
        skill="locomotion", skill_title="L", kind="eval_suite_expand", risk_class="low", risk_score=0.1,
        gate="pass", gate_checks=[], sim_summary={}, baseline=None, metrics=None, diff_files=[],
        report_url="http://x", artifacts=[], decide_url="http://y", rollback_hint="z",
    )
    delivery = bridge.notify(request)
    assert delivery.ok and delivery.mode == "pull" and delivery.status == 0
    assert any(entry.event == "bridge.review_available" for entry in bridge.audit.entries())


def test_bridge_failure_is_recorded_and_persisted(workspace):
    cfg = workspace["cfg"]
    cfg.hermes.outbound = {"mode": "webhook", "url": "http://127.0.0.1:1/none",
                           "retries": 1, "timeout_s": 0.5}
    from evo_layer.audit import AuditLog

    bridge = HermesBridge(cfg, AuditLog(cfg.audit_path), env={"EVOLAYER_HERMES_OUTBOUND_SECRET": SECRET})
    request = ReviewRequest(
        contract=CONTRACT_VERSION, event="review.requested", proposal_id="prp_fail", revision=1,
        skill="locomotion", skill_title="L", kind="eval_suite_expand", risk_class="low", risk_score=0.1,
        gate="pass", gate_checks=[], sim_summary={}, baseline=None, metrics=None, diff_files=[],
        report_url="http://x", artifacts=[], decide_url="http://y", rollback_hint="z",
    )
    delivery = bridge.notify(request)
    assert delivery.ok is False
    assert delivery.attempts == 1
    persisted = bridge.pending_payload("prp_fail", 1)
    assert persisted is not None
    assert persisted["proposal"]["id"] == "prp_fail"


def test_bridge_refuses_to_send_unsigned(workspace, receiver):
    cfg = workspace["cfg"]
    cfg.hermes.outbound = {"mode": "webhook", "url": receiver, "secret_env": "MISSING_SECRET"}
    from evo_layer.audit import AuditLog

    bridge = HermesBridge(cfg, AuditLog(cfg.audit_path), env={})
    from evo_layer.hermes_interface.bridge import BridgeError

    with pytest.raises(BridgeError) as excinfo:
        bridge.self_test()
    assert "unsigned" in str(excinfo.value)


# --------------------------------------------------------------------------- client


def test_client_round_trip(api):
    client = HermesClient(api["base"], token=api["token"])
    assert client.healthz()["status"] == "ok"
    pending = client.pending()
    assert pending[0]["proposal_id"] == api["proposal"].id
    detail = client.proposal(api["proposal"].id)
    assert detail["proposal"]["id"] == api["proposal"].id
    schema = client.schema()
    assert schema["contract_version"] == CONTRACT_VERSION
    result = client.decide(api["proposal"].id, "reject", actor="hermes", reason="client says no")
    assert result["decision"]["verdict"] == "reject"


def test_client_reports_errors(api):
    client = HermesClient(api["base"], token="wrong-token-0123456789")
    with pytest.raises(ClientError) as excinfo:
        client.pending()
    assert excinfo.value.status == 401
