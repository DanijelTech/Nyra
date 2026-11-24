"""EvoLayer command line.

    evolayer doctor                       environment, backends, audit chain, config
    evolayer observe  --skill S --input path.jsonl|dir [--source real|sim|replay]
    evolayer critique --skill S [--json]
    evolayer propose  --finding ID [--template T]
    evolayer simulate --proposal ID [--backend B] [--dry-run] [--allow-replay]
    evolayer gate     --proposal ID
    evolayer review   --proposal ID            (interactive: approve/reject/modify)
    evolayer approve|reject|modify --proposal ID --actor A --reason R
    evolayer issue-token --proposal ID --verdict V --actor A
    evolayer deploy   --proposal ID [--actor A]
    evolayer rollback --deployment ID --actor A [--reason R]
    evolayer status   [--skill S] [--json]
    evolayer tick     [--skill S]
    evolayer serve    (Hermes HTTP API)
    evolayer verify-audit
    evolayer backends
    evolayer snapshot list | verify <id> | restore <id>

Exit codes: 0 ok, 1 refused/failed, 2 usage or configuration error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .audit import AuditError, AuditLog
from .config import ConfigError, EvoConfig
from .critic.baseline import InsufficientData
from .deploy.installer import DeployError
from .hermes_interface.interface import HermesError, TokenError
from .orchestrator.loop import EvoLoop
from .proposer.templates import TEMPLATES, TemplateUnavailable
from .simulator.backends import BACKENDS, availability_report
from .store import NotFound, Store
from .versioning.snapshot import SnapshotError, Snapshotter


def _out(data: Any) -> None:
    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2, sort_keys=True, default=str))
    else:
        print(data)


def _config(args: argparse.Namespace) -> EvoConfig:
    return EvoConfig.load(args.config, root=args.root)


def _loop(args: argparse.Namespace) -> EvoLoop:
    cfg = _config(args)
    return EvoLoop.from_config(cfg)


# --------------------------------------------------------------------------- commands


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = _config(args)
    cfg.ensure_dirs()
    rows: dict[str, Any] = {
        "version": __version__,
        "root": str(cfg.root),
        "config_sources": cfg.source_files,
        "state_dir": str(cfg.state_dir),
        "repos": {key: {"path": str(spec.path), "allow": spec.allow} for key, spec in cfg.repos.items()},
        "skills": {key: {"backend": spec.backend, "repo": spec.repo,
                         "deploy_policy": spec.deploy_policy,
                         "limits": spec.limits.__dict__} for key, spec in cfg.skills.items()},
        "backends": availability_report(cfg),
    }
    store = Store(cfg.store_path)
    rows["store"] = {"path": str(cfg.store_path), "episodes": {
        skill: store.episode_count(skill) for skill in cfg.skills
    }}
    store.close()
    audit = AuditLog(cfg.audit_path)
    ok, count, broken, reason = audit.verify()
    rows["audit"] = {"path": str(cfg.audit_path), "entries": count, "chain_ok": ok,
                     "broken_at": broken, "reason": reason}
    secrets = {}
    for name, env_key in (("hermes_token", cfg.hermes.token_env), ("hermes_secret", cfg.hermes.secret_env)):
        import os

        secrets[name] = f"{env_key}: {'set' if os.environ.get(env_key) else 'MISSING'}"
    rows["secrets"] = secrets
    _out(rows)
    return 0 if ok else 1


def cmd_backends(args: argparse.Namespace) -> int:
    cfg = _config(args)
    _out({"backends": availability_report(cfg)})
    return 0


def cmd_observe(args: argparse.Namespace) -> int:
    loop = _loop(args)
    result = loop.observe(args.skill, args.input, source=args.source)
    _out(result)
    return 0


def cmd_critique(args: argparse.Namespace) -> int:
    loop = _loop(args)
    try:
        findings = loop.critique(args.skill)
    except InsufficientData as exc:
        print(f"insufficient data: {exc}", file=sys.stderr)
        return 1
    if args.json:
        _out([finding.to_dict() for finding in findings])
    else:
        if not findings:
            print("no findings")
        for finding in findings:
            print(f"{finding.severity:8} {finding.kind:12} {finding.id}  {finding.title}")
            print(f"         {finding.summary}")
            print(f"         recommendation: {finding.recommendation}  params={finding.params}")
    return 0


def cmd_propose(args: argparse.Namespace) -> int:
    loop = _loop(args)
    try:
        proposal, validation = loop.propose_for_finding(args.finding, template=args.template)
    except (TemplateUnavailable, NotFound) as exc:
        print(f"proposal refused: {exc}", file=sys.stderr)
        return 1
    if args.json:
        _out({"proposal": proposal.to_dict(), "validation": validation})
        return 0 if validation["ok"] else 1
    print(f"proposal {proposal.id} r{proposal.revision} ({proposal.kind}, risk {proposal.risk.risk_class})")
    print(f"validation: {'ok' if validation['ok'] else 'FAILED'}")
    for check in validation["checks"]:
        if not check["ok"]:
            print(f"  - {check['name']}: {check['detail']}")
    return 0 if validation["ok"] else 1


def cmd_simulate(args: argparse.Namespace) -> int:
    loop = _loop(args)
    try:
        result = loop.simulate(
            args.proposal,
            backend=args.backend,
            dry_run=args.dry_run,
            allow_replay=args.allow_replay,
            policy_onnx=args.policy,
        )
    except HermesError as exc:
        print(f"simulation refused: {exc}", file=sys.stderr)
        return 1
    if args.json:
        _out(result.to_dict())
    else:
        print(f"status={result.status} backend={result.backend} episodes={result.episodes_n}")
        if result.metrics:
            _out({"metrics": result.metrics.to_dict()})
        if result.command:
            print("command:", " ".join(result.command))
    if result.error:
        print("error:", result.error[:600], file=sys.stderr)
    return 0 if result.status == "ok" or args.dry_run else 1


def cmd_gate(args: argparse.Namespace) -> int:
    loop = _loop(args)
    try:
        verdict = loop.gate_proposal(args.proposal)
    except HermesError as exc:
        print(f"gate refused: {exc}", file=sys.stderr)
        return 1
    print(f"verdict: {verdict.verdict.upper()}")
    for check in verdict.checks:
        print(f"  {check['status']:6} {check['name']:24} measured={check['measured']} limit={check['limit']}")
        print(f"         {check['detail']}")
    if args.json:
        _out(verdict.to_dict())
    return 0 if verdict.verdict != "reject" else 1


def cmd_review(args: argparse.Namespace) -> int:
    loop = _loop(args)
    from .store import NotFound as _NotFound

    try:
        proposal = loop.store.get_proposal(args.proposal)
    except _NotFound as exc:
        print(str(exc), file=sys.stderr)
        return 1
    report = loop.request_approval(proposal.id)
    print(report.markdown)
    print("\n" + "-" * 78)
    if args.yes is None:
        answer = input("Decision [a]pprove / [r]eject / [m]odify / [q]uit: ").strip().lower()
    else:
        answer = "a" if args.yes else "q"
    if answer.startswith("q") or not answer:
        print("no decision recorded")
        return 0
    mapping = {"a": "approve", "r": "reject", "m": "modify"}
    verdict = mapping.get(answer[0])
    if verdict is None:
        print("unknown choice", file=sys.stderr)
        return 2
    reason = args.reason or input("Reason (recorded in the audit log): ").strip()
    try:
        result = loop.decide(
            proposal.id, verdict=verdict, actor=args.actor, reason=reason, require_token=False
        )
    except (HermesError, TokenError) as exc:
        print(f"decision refused: {exc}", file=sys.stderr)
        return 1
    _out(result)
    return 0


def cmd_issue_token(args: argparse.Namespace) -> int:
    cfg = _config(args)
    from .hermes_interface.interface import HermesInterface

    store = Store(cfg.store_path)
    interface = HermesInterface(cfg, store, AuditLog(cfg.audit_path), Snapshotter(cfg.snapshots_dir))
    try:
        proposal = store.get_proposal(args.proposal)
        token = interface.issue_token(
            proposal_id=proposal.id,
            revision=proposal.revision,
            verdict=args.verdict,
            actor=args.actor,
        )
    except (HermesError, TokenError, ConfigError, NotFound) as exc:
        print(f"cannot issue a token: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()
    print(token)
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    loop = _loop(args)
    modifications = {}
    if args.set:
        params = {}
        for item in args.set:
            key, _, value = item.partition("=")
            if not key or not _:
                print(f"--set expects key=value, got {item!r}", file=sys.stderr)
                return 2
            try:
                params[key] = float(value) if "." in value or value.isdigit() else value
            except ValueError:
                params[key] = value
        modifications = {"template_params": params}
    try:
        result = loop.decide(
            args.proposal,
            verdict=args.verdict,
            actor=args.actor,
            reason=args.reason,
            token=args.token,
            modifications=modifications,
        )
    except (HermesError, TokenError) as exc:
        print(f"decision refused: {exc}", file=sys.stderr)
        return 1
    _out(result)
    return 0


def cmd_deploy(args: argparse.Namespace) -> int:
    loop = _loop(args)
    try:
        deployment = loop.deploy(args.proposal, actor=args.actor)
    except DeployError as exc:
        print(f"deployment refused: {exc}", file=sys.stderr)
        return 1
    _out(deployment.to_dict())
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    loop = _loop(args)
    try:
        deployment = loop.rollback(args.deployment, actor=args.actor, reason=args.reason)
    except (DeployError, SnapshotError, NotFound) as exc:
        print(f"rollback failed: {exc}", file=sys.stderr)
        return 1
    _out(deployment.to_dict())
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    loop = _loop(args)
    data = loop.status(args.skill)
    if args.json:
        _out(data)
        return 0
    for skill, info in data["skills"].items():
        print(f"{skill}: {info['episodes']}/{info['episodes_required']} episodes, "
              f"{info['findings_open']} open findings")
        for proposal in info["proposals"]:
            print(f"  {proposal['id']} r{proposal['revision']:>2} {proposal['status']:14} "
                  f"{proposal['kind']:22} risk={proposal['risk']:6} gate={proposal['gate']} sim={proposal['sim']}")
    if data["deployments"]:
        print("\nrecent deployments:")
        for deployment in data["deployments"]:
            print(f"  {deployment['id']} {deployment['tag']} rolled_back={deployment['rolled_back']}")
    return 0


def cmd_tick(args: argparse.Namespace) -> int:
    loop = _loop(args)
    results = loop.tick(args.skill)
    _out([result.to_dict() for result in results])
    return 0 if all(not result.errors for result in results) else 1


def cmd_serve(args: argparse.Namespace) -> int:
    cfg = _config(args)
    from .hermes_interface.api import ApiContext, serve

    loop = _loop(args)
    try:
        token = cfg.hermes_token()
    except ConfigError as exc:
        print(f"refusing to serve: {exc}", file=sys.stderr)
        return 2

    def decide_hook(proposal_id: str, verdict: str, actor: str, reason: str, token_value: str | None):
        return loop.decide(proposal_id, verdict=verdict, actor=actor, reason=reason, token=token_value)

    def artifacts_provider(proposal_id: str) -> list[Path]:
        proposal = loop.store.get_proposal(proposal_id)
        sim = loop.store.latest_simulation(proposal.id, proposal.revision)
        paths: list[Path] = []
        if sim:
            paths.extend(Path(artifact.path) for artifact in sim.artifacts)
        report_dir = cfg.artifacts_dir / proposal_id
        if report_dir.is_dir():
            paths.extend(sorted(report_dir.glob("*")))
        return paths

    context = ApiContext(
        cfg=cfg,
        store=loop.store,
        audit=loop.audit,
        hermes=loop.hermes,
        token=token,
        decide_hook=decide_hook,
        artifacts_provider=artifacts_provider,
        outbound_secret=loop.bridge.options.secret,
        artifact_roots=[cfg.artifacts_dir, cfg.sim_dir],
    )
    print(f"EvoLayer Hermes API on http://{cfg.hermes.bind}:{cfg.hermes.port} "
          f"(auth {'on' if cfg.hermes.require_auth else 'OFF'})")
    serve(context)
    return 0


def cmd_contract(args: argparse.Namespace) -> int:
    from .hermes_interface.contract import schemas

    _out(schemas())
    return 0


def cmd_bridge_test(args: argparse.Namespace) -> int:
    loop = _loop(args)
    delivery = loop.bridge.self_test()
    _out(delivery.to_dict())
    return 0 if delivery.ok else 1


def cmd_notify(args: argparse.Namespace) -> int:
    loop = _loop(args)
    try:
        report = loop.request_approval(args.proposal, notify=True)
    except HermesError as exc:
        print(f"cannot notify: {exc}", file=sys.stderr)
        return 1
    _out({"proposal": report.proposal_id, "revision": report.revision, "delivery": report.delivery})
    return 0 if (report.delivery or {}).get("ok") else 1


def cmd_verify_audit(args: argparse.Namespace) -> int:
    cfg = _config(args)
    audit = AuditLog(cfg.audit_path)
    ok, count, broken, reason = audit.verify()
    _out({"entries": count, "chain_ok": ok, "broken_at": broken, "reason": reason})
    return 0 if ok else 1


def cmd_snapshot(args: argparse.Namespace) -> int:
    cfg = _config(args)
    snapshotter = Snapshotter(cfg.snapshots_dir)
    if args.action == "list":
        _out([snapshot.to_dict() for snapshot in snapshotter.list()])
        return 0
    if args.action == "verify":
        ok, problems = snapshotter.verify(args.snapshot_id)
        _out({"snapshot": args.snapshot_id, "ok": ok, "problems": problems})
        return 0 if ok else 1
    if args.action == "restore":
        try:
            restored = snapshotter.restore(args.snapshot_id, target_root=cfg.root)
        except SnapshotError as exc:
            print(f"restore failed: {exc}", file=sys.stderr)
            return 1
        _out({"restored": restored})
        return 0
    print("unknown snapshot action", file=sys.stderr)
    return 2


def cmd_templates(args: argparse.Namespace) -> int:
    _out({"templates": sorted(TEMPLATES)})
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evolayer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="path to evolayer.yaml")
    parser.add_argument("--root", help="workspace root (default: discovered)")
    parser.add_argument("--version", action="version", version=f"evo-layer {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_text: str) -> argparse.ArgumentParser:
        sub_parser = sub.add_parser(name, help=help_text)
        sub_parser.set_defaults(func=func)
        return sub_parser

    add("doctor", cmd_doctor, "inspect configuration, backends, store and audit chain")
    add("backends", cmd_backends, "list simulator backends and availability")
    add("templates", cmd_templates, "list proposal templates")

    observe = add("observe", cmd_observe, "ingest telemetry")
    observe.add_argument("--skill", required=True)
    observe.add_argument("--input", required=True, nargs="+", help="file(s) or directory")
    observe.add_argument("--source", choices=["real", "sim", "replay"], default=None)

    critique = add("critique", cmd_critique, "analyse stored episodes")
    critique.add_argument("--skill", required=True)
    critique.add_argument("--json", action="store_true")

    propose = add("propose", cmd_propose, "generate and validate a proposal for a finding")
    propose.add_argument("--finding", required=True)
    propose.add_argument("--template", choices=sorted(TEMPLATES), default=None)
    propose.add_argument("--json", action="store_true")

    simulate = add("simulate", cmd_simulate, "rehearse a validated proposal")
    simulate.add_argument("--proposal", required=True)
    simulate.add_argument("--backend", choices=sorted(BACKENDS), default=None)
    simulate.add_argument("--dry-run", action="store_true")
    simulate.add_argument("--allow-replay", action="store_true")
    simulate.add_argument("--json", action="store_true")
    simulate.add_argument("--policy", default=None, help="candidate ONNX to rehearse")

    gate = add("gate", cmd_gate, "evaluate the safety gate")
    gate.add_argument("--proposal", required=True)
    gate.add_argument("--json", action="store_true")

    review = add("review", cmd_review, "local fallback: print the report and decide in this terminal")
    review.add_argument("--proposal", required=True)
    review.add_argument("--actor", default="hermes")
    review.add_argument("--reason", default="")
    review.add_argument("--yes", dest="yes", action="store_true", default=None)

    for verdict, text in (("approve", "approve a proposal"), ("reject", "reject a proposal"),
                          ("modify", "request a modified revision")):
        decide = add(verdict, cmd_decide, text)
        decide.set_defaults(verdict=verdict)
        decide.add_argument("--proposal", required=True)
        decide.add_argument("--actor", required=True)
        decide.add_argument("--reason", required=True)
        decide.add_argument("--token", default=None)
        decide.add_argument("--set", action="append", help="template param override (modify only), key=value")

    token = add("issue-token", cmd_issue_token, "mint a single-use decision token")
    token.add_argument("--proposal", required=True)
    token.add_argument("--verdict", required=True, choices=["approve", "reject", "modify"])
    token.add_argument("--actor", required=True)

    deploy = add("deploy", cmd_deploy, "install an approved proposal")
    deploy.add_argument("--proposal", required=True)
    deploy.add_argument("--actor", default=None)

    rollback = add("rollback", cmd_rollback, "restore the snapshot taken before a deployment")
    rollback.add_argument("--deployment", required=True)
    rollback.add_argument("--actor", required=True)
    rollback.add_argument("--reason", default="")

    status = add("status", cmd_status, "loop status")
    status.add_argument("--skill", default=None)
    status.add_argument("--json", action="store_true")

    tick = add("tick", cmd_tick, "advance the loop until Hermes must decide")
    tick.add_argument("--skill", default=None)

    add("serve", cmd_serve, "run the Hermes approval API (the fork your local Hermes attaches to)")
    add("contract", cmd_contract, "print the Hermes integration contract (schemas, headers, events)")
    add("bridge-test", cmd_bridge_test, "send a signed self-test review request to hermes.outbound.url")
    notify = add("notify", cmd_notify, "re-publish the review request for a proposal")
    notify.add_argument("--proposal", required=True)
    add("verify-audit", cmd_verify_audit, "verify the audit hash chain")

    snapshot = add("snapshot", cmd_snapshot, "inspect deployment snapshots")
    snapshot.add_argument("action", choices=["list", "verify", "restore"])
    snapshot.add_argument("snapshot_id", nargs="?")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except (HermesError, TokenError, DeployError, SnapshotError, AuditError, NotFound) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
