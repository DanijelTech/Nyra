#!/usr/bin/env python3
"""Reference local Hermes adapter.

This is the whole integration surface: poll EvoLayer, look at the evidence, decide.
Copy it into your Hermes and replace ``Policy.decide`` with your own logic (an LLM
reviewer, a Slack approval, a rules engine …).

Usage::

    export EVOLAYER_HERMES_TOKEN=...        # read + decide
    python3 examples/hermes_adapter.py --base http://127.0.0.1:8765 --poll 10
    python3 examples/hermes_adapter.py --once --auto-policy strict

Policies:

``manual``   print everything, ask on the terminal (default)
``strict``   auto-reject anything the gate did not pass or that is not low risk;
             everything else is left for a human
``guarded``  auto-approve only ``gate=pass`` + ``risk=low`` + ``rehearsal >= 100``
             episodes; auto-reject gate rejects; leave the rest for a human
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from evo_layer.hermes_interface.client import ClientError, HermesClient


@dataclass
class Decision:
    verdict: str
    reason: str


class Policy:
    def __init__(self, name: str, artifacts_dir: Path):
        self.name = name
        self.artifacts_dir = artifacts_dir

    def decide(self, review: dict) -> Decision | None:
        gate = (review.get("verdict") or {}).get("verdict", "reject")
        risk_class = risk_class_of(review)
        rehearsal = review.get("simulation") or {}
        episodes = rehearsal.get("episodes_n") or 0

        if gate == "reject":
            return Decision("reject", "safety gate rejected the proposal")
        if self.name == "strict":
            if risk_class != "low":
                return Decision("reject", f"risk class {risk_class!r} needs a human in strict mode")
            return None
        if self.name == "guarded":
            if gate == "pass" and risk_class == "low" and episodes >= 100:
                return Decision("approve", f"gate pass, low risk, {episodes} rehearsal episodes")
            return None
        return None


def risk_class_of(review: dict) -> str | None:
    proposal = review.get("proposal") or {}
    risk = proposal.get("risk") or {}
    return risk.get("risk_class") or proposal.get("risk_class")


def risk_score_of(review: dict) -> float | None:
    proposal = review.get("proposal") or {}
    risk = proposal.get("risk") or {}
    return risk.get("score", proposal.get("risk_score"))


def render(review: dict) -> str:
    proposal = review.get("proposal", {})
    verdict = review.get("verdict", {})
    sim = review.get("simulation") or {}
    lines = [
        "",
        "=" * 78,
        f"proposal  {proposal.get('id')} r{proposal.get('revision')}  "
        f"({proposal.get('skill')} / {proposal.get('kind')})",
        f"risk      {risk_class_of(review)} {risk_score_of(review)}",
        f"gate      {verdict.get('verdict')}",
        f"rehearsal {sim.get('backend')} {sim.get('status')} · {sim.get('episodes_n')} episodes",
        "=" * 78,
    ]
    for check in verdict.get("checks", []):
        lines.append(
            f"  {check.get('status'):6} {check.get('name'):24} measured={check.get('measured')} "
            f"limit={check.get('limit')}"
        )
    metrics = (sim.get("metrics") or {})
    if metrics:
        lines.append(
            f"  metrics: n={metrics.get('n')} success={metrics.get('success_rate')} "
            f"falls={metrics.get('fall_rate')} torque_ratio={metrics.get('torque_ratio')} "
            f"vel_ratio={metrics.get('vel_ratio')}"
        )
    for artifact in sim.get("artifacts", []):
        lines.append(f"  artifact {artifact.get('kind'):8} {artifact.get('name')} "
                     f"{artifact.get('size')} bytes sha256={str(artifact.get('sha256'))[:12]}…")
    for caveat in review.get("caveats", []):
        lines.append(f"  caveat: {caveat}")
    return "\n".join(lines)


def handle(client: HermesClient, pending: dict, policy: Policy, *, automatic: bool) -> None:
    proposal_id = pending["proposal_id"]
    if pending.get("gate") == "reject":
        # Do not even print an approval prompt: the gate is final.
        client.decide(proposal_id, "reject", actor="hermes-adapter",
                      reason="safety gate rejected this proposal")
        print(f"auto-rejected {proposal_id} (gate reject)")
        return

    review = client.proposal(proposal_id)
    print(render(review))
    for artifact in (review.get("simulation") or {}).get("artifacts", []):
        try:
            path = client.save_artifact(proposal_id, artifact["name"], policy.artifacts_dir / proposal_id)
            print(f"  saved {path}")
        except ClientError as exc:
            print(f"  could not fetch {artifact['name']}: {exc}", file=sys.stderr)

    decision = policy.decide(review)
    if decision is None and not automatic:
        answer = input("decide [a]pprove / [r]eject / [m]odify / [s]kip: ").strip().lower()
        if answer.startswith("a"):
            decision = Decision("approve", input("reason: ").strip() or "reviewed locally")
        elif answer.startswith("r"):
            decision = Decision("reject", input("reason: ").strip() or "reviewed locally")
        elif answer.startswith("m"):
            reason = input("reason: ").strip() or "adjust"
            decision = Decision("modify", reason)
    if decision is None:
        print(f"  left pending: {proposal_id}")
        return
    try:
        client.decide(proposal_id, decision.verdict, actor="hermes-adapter", reason=decision.reason)
        print(f"  {decision.verdict}: {proposal_id} — {decision.reason}")
    except ClientError as exc:
        print(f"  decision refused ({exc.status}): {exc.body[:200]}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8765")
    parser.add_argument("--token", default=os.environ.get("EVOLAYER_HERMES_TOKEN", ""))
    parser.add_argument("--signing-secret", default=os.environ.get("EVOLAYER_HERMES_OUTBOUND_SECRET", ""))
    parser.add_argument("--artifacts", default="/tmp/evolayer-review")
    parser.add_argument("--poll", type=float, default=0.0, help="seconds between polls (0 = once)")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--auto-policy", choices=["manual", "strict", "guarded"], default="manual")
    args = parser.parse_args(argv)

    client = HermesClient(args.base, token=args.token, signing_secret=args.signing_secret)
    policy = Policy(args.auto_policy, Path(args.artifacts))
    try:
        health = client.healthz()
    except ClientError as exc:
        print(f"EvoLayer unreachable at {args.base}: {exc}", file=sys.stderr)
        return 2
    print(f"EvoLayer {health.get('version')} · audit entries {health.get('audit', {}).get('entries')} · "
          f"chain_ok={health.get('audit', {}).get('chain_ok')} · contract {client.schema().get('contract_version')}")

    while True:
        try:
            pending = client.pending()
        except ClientError as exc:
            print(f"polling failed ({exc.status}): {exc.body[:200]}", file=sys.stderr)
            pending = []
        for item in pending:
            handle(client, item, policy, automatic=args.auto_policy != "manual")
        if args.once or args.poll <= 0:
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
