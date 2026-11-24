"""Durable state for the evolution loop (SQLite, WAL, schema-versioned).

The store is the single source of truth for the loop's progress. Two rules keep
it honest:

* A decision is unique per ``(proposal_id, revision)``. A second approval of the
  same revision is a hard error, not an overwrite — this is what makes replay
  attacks on the Hermes interface useless.
* Status changes go through dedicated setters so the orchestrator can assert the
  legal transition before writing. The store itself stays dumb but strict.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from .models import (
    Artifact,
    Decision,
    Deployment,
    Episode,
    Finding,
    Metrics,
    Proposal,
    SimulationResult,
    Verdict,
)

SCHEMA_VERSION = 2


class StoreError(RuntimeError):
    """Storage-level invariant violation."""


class DuplicateDecision(StoreError):
    """A decision already exists for this proposal revision."""


class NotFound(StoreError):
    """The requested object does not exist."""


SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skills (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    backend TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    skill TEXT NOT NULL,
    source TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    started_at REAL NOT NULL,
    duration_s REAL NOT NULL,
    success INTEGER NOT NULL,
    falls INTEGER NOT NULL,
    torque_peak_nm REAL NOT NULL,
    joint_vel_peak_rps REAL NOT NULL,
    collisions INTEGER NOT NULL,
    battery_min_v REAL,
    tracking_error_rad REAL,
    torque_ratio REAL,
    vel_ratio REAL,
    notes TEXT NOT NULL DEFAULT '',
    ingested_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_skill_time ON episodes (skill, started_at);

CREATE TABLE IF NOT EXISTS findings (
    id TEXT PRIMARY KEY,
    skill TEXT NOT NULL,
    kind TEXT NOT NULL,
    severity TEXT NOT NULL,
    title TEXT NOT NULL,
    metric TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'open'
);
CREATE INDEX IF NOT EXISTS idx_findings_skill ON findings (skill, created_at);

CREATE TABLE IF NOT EXISTS proposals (
    id TEXT PRIMARY KEY,
    finding_id TEXT NOT NULL,
    skill TEXT NOT NULL,
    revision INTEGER NOT NULL,
    title TEXT NOT NULL,
    kind TEXT NOT NULL,
    risk_score REAL NOT NULL,
    risk_class TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    supersedes TEXT,
    payload_json TEXT NOT NULL,
    UNIQUE (finding_id, revision),
    FOREIGN KEY (finding_id) REFERENCES findings (id)
);
CREATE INDEX IF NOT EXISTS idx_proposals_skill ON proposals (skill, created_at);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals (status);

CREATE TABLE IF NOT EXISTS simulations (
    id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    backend TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY (proposal_id) REFERENCES proposals (id)
);
CREATE INDEX IF NOT EXISTS idx_simulations_proposal ON simulations (proposal_id, revision);

CREATE TABLE IF NOT EXISTS verdicts (
    id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    verdict TEXT NOT NULL,
    created_at REAL NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY (proposal_id) REFERENCES proposals (id)
);
CREATE INDEX IF NOT EXISTS idx_verdicts_proposal ON verdicts (proposal_id, revision);

CREATE TABLE IF NOT EXISTS decisions (
    proposal_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    verdict TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT NOT NULL,
    nonce TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (proposal_id, revision),
    FOREIGN KEY (proposal_id) REFERENCES proposals (id)
);

CREATE TABLE IF NOT EXISTS deployments (
    id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    tag TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    rolled_back INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL,
    FOREIGN KEY (proposal_id) REFERENCES proposals (id)
);
CREATE INDEX IF NOT EXISTS idx_deployments_created ON deployments (created_at);
"""


class _LockedConnection:
    """Serialise access to one SQLite connection across the API's worker threads.

    SQLite objects are thread-bound; the Hermes API is threaded. A single
    re-entrant lock around every statement is the smallest correct fix, and
    ``transaction()`` holds it for the whole block so a transaction can never be
    interleaved with another thread's statements.
    """

    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock):
        self._conn = conn
        self._lock = lock

    def execute(self, *args, **kwargs):
        with self._lock:
            return self._conn.execute(*args, **kwargs)

    def executescript(self, *args, **kwargs):
        with self._lock:
            return self._conn.executescript(*args, **kwargs)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = _LockedConnection(
            sqlite3.connect(str(self.path), timeout=15.0, isolation_level=None, check_same_thread=False),
            self._lock,
        )
        self._conn._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._migrate()

    # -- plumbing --------------------------------------------------------
    def _migrate(self) -> None:
        self._conn.executescript(SCHEMA)
        with self.transaction():
            row = self._conn.execute("SELECT v FROM kv WHERE k = 'schema_version'").fetchone()
            current = int(row["v"]) if row else 0
            if current > SCHEMA_VERSION:
                raise StoreError(
                    f"database schema version {current} is newer than this build ({SCHEMA_VERSION})"
                )
            if current < 2:
                # v1 -> v2: per-joint safety ratios. Existing rows keep NULL.
                columns = {
                    row["name"]
                    for row in self._conn.execute("PRAGMA table_info(episodes)").fetchall()
                }
                for column in ("torque_ratio", "vel_ratio"):
                    if column not in columns:
                        self._conn.execute(f"ALTER TABLE episodes ADD COLUMN {column} REAL")
            self._set_kv("schema_version", str(SCHEMA_VERSION))

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- kv --------------------------------------------------------------
    def _set_kv(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO kv (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
            (key, value),
        )

    def kv_get(self, key: str, default: str | None = None) -> str | None:
        row = self._conn.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return row["v"] if row else default

    def kv_set(self, key: str, value: str) -> None:
        with self.transaction():
            self._set_kv(key, value)

    def consume_nonce(self, nonce: str) -> bool:
        """Atomically mark a nonce as used. ``False`` means it was already spent."""
        try:
            with self.transaction() as conn:
                conn.execute("INSERT INTO kv (k, v) VALUES (?, ?)", (f"nonce:{nonce}", str(time.time())))
        except sqlite3.IntegrityError:
            return False
        return True

    # -- skills ----------------------------------------------------------
    def upsert_skill(self, skill_id: str, title: str, backend: str, spec: dict[str, Any]) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO skills (id, title, backend, spec_json, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     title = excluded.title, backend = excluded.backend,
                     spec_json = excluded.spec_json, updated_at = excluded.updated_at""",
                (skill_id, title, backend, json.dumps(spec, sort_keys=True), time.time()),
            )

    def get_skill(self, skill_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()
        return dict(row) if row else None

    # -- episodes --------------------------------------------------------
    def add_episodes(self, episodes: Iterable[Episode]) -> int:
        now = time.time()
        inserted = 0
        with self.transaction() as conn:
            for ep in episodes:
                cur = conn.execute(
                    """INSERT OR IGNORE INTO episodes
                       (id, skill, source, policy_version, started_at, duration_s, success, falls,
                        torque_peak_nm, joint_vel_peak_rps, collisions, battery_min_v,
                        tracking_error_rad, torque_ratio, vel_ratio, notes, ingested_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        ep.id,
                        ep.skill,
                        ep.source,
                        ep.policy_version,
                        ep.started_at,
                        ep.duration_s,
                        1 if ep.success else 0,
                        ep.falls,
                        ep.torque_peak_nm,
                        ep.joint_vel_peak_rps,
                        ep.collisions,
                        ep.battery_min_v,
                        ep.tracking_error_rad,
                        ep.torque_ratio,
                        ep.vel_ratio,
                        ep.notes,
                        now,
                    ),
                )
                inserted += cur.rowcount
        return inserted

    def episodes(
        self,
        skill: str,
        *,
        since: float | None = None,
        until: float | None = None,
        source: str | None = None,
        policy_version: str | None = None,
        limit: int | None = None,
    ) -> list[Episode]:
        sql = "SELECT * FROM episodes WHERE skill = ?"
        params: list[Any] = [skill]
        if since is not None:
            sql += " AND started_at >= ?"
            params.append(since)
        if until is not None:
            sql += " AND started_at <= ?"
            params.append(until)
        if source:
            sql += " AND source = ?"
            params.append(source)
        if policy_version:
            sql += " AND policy_version = ?"
            params.append(policy_version)
        sql += " ORDER BY started_at ASC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = self._conn.execute(sql, params).fetchall()
        return [
            Episode(
                id=row["id"],
                skill=row["skill"],
                source=row["source"],
                policy_version=row["policy_version"],
                started_at=row["started_at"],
                duration_s=row["duration_s"],
                success=bool(row["success"]),
                falls=row["falls"],
                torque_peak_nm=row["torque_peak_nm"],
                joint_vel_peak_rps=row["joint_vel_peak_rps"],
                collisions=row["collisions"],
                battery_min_v=row["battery_min_v"],
                tracking_error_rad=row["tracking_error_rad"],
                torque_ratio=row["torque_ratio"],
                vel_ratio=row["vel_ratio"],
                notes=row["notes"],
            )
            for row in rows
        ]

    def latest_started_at(self, skill: str) -> float | None:
        row = self._conn.execute(
            "SELECT MAX(started_at) AS m FROM episodes WHERE skill = ?", (skill,)
        ).fetchone()
        return float(row["m"]) if row and row["m"] is not None else None

    def episode_count(self, skill: str) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS c FROM episodes WHERE skill = ?", (skill,)).fetchone()
        return int(row["c"])

    # -- findings --------------------------------------------------------
    def add_finding(self, finding: Finding) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO findings
                   (id, skill, kind, severity, title, metric, payload_json, created_at, status)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    finding.id,
                    finding.skill,
                    finding.kind,
                    finding.severity,
                    finding.title,
                    finding.metric,
                    json.dumps(finding.to_dict(), sort_keys=True),
                    finding.created_at or time.time(),
                    finding.status,
                ),
            )

    def findings(self, skill: str | None = None, status: str | None = None) -> list[Finding]:
        sql = "SELECT payload_json FROM findings WHERE 1=1"
        params: list[Any] = []
        if skill:
            sql += " AND skill = ?"
            params.append(skill)
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC"
        return [_finding_from(row["payload_json"]) for row in self._conn.execute(sql, params).fetchall()]

    def get_finding(self, finding_id: str) -> Finding:
        row = self._conn.execute("SELECT payload_json FROM findings WHERE id = ?", (finding_id,)).fetchone()
        if row is None:
            raise NotFound(f"finding {finding_id} not found")
        return _finding_from(row["payload_json"])

    def set_finding_status(self, finding_id: str, status: str) -> None:
        with self.transaction() as conn:
            row = conn.execute("SELECT payload_json FROM findings WHERE id = ?", (finding_id,)).fetchone()
            if row is None:
                raise NotFound(f"finding {finding_id} not found")
            payload = json.loads(row["payload_json"])
            payload["status"] = status
            conn.execute(
                "UPDATE findings SET status = ?, payload_json = ? WHERE id = ?",
                (status, json.dumps(payload, sort_keys=True), finding_id),
            )

    # -- proposals -------------------------------------------------------
    def add_proposal(self, proposal: Proposal) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO proposals
                   (id, finding_id, skill, revision, title, kind, risk_score, risk_class,
                    status, created_at, supersedes, payload_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    proposal.id,
                    proposal.finding_id,
                    proposal.skill,
                    proposal.revision,
                    proposal.title,
                    proposal.kind,
                    proposal.risk.score,
                    proposal.risk.risk_class,
                    proposal.status,
                    proposal.created_at or time.time(),
                    proposal.supersedes,
                    json.dumps(proposal.to_dict(), sort_keys=True),
                ),
            )

    def get_proposal(self, proposal_id: str) -> Proposal:
        row = self._conn.execute("SELECT payload_json FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
        if row is None:
            raise NotFound(f"proposal {proposal_id} not found")
        return _proposal_from(row["payload_json"])

    def proposals(self, skill: str | None = None, status: str | None = None) -> list[Proposal]:
        sql = "SELECT payload_json FROM proposals WHERE 1=1"
        params: list[Any] = []
        if skill:
            sql += " AND skill = ?"
            params.append(skill)
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC"
        return [_proposal_from(row["payload_json"]) for row in self._conn.execute(sql, params).fetchall()]

    def proposals_for_finding(self, finding_id: str) -> list[Proposal]:
        rows = self._conn.execute(
            "SELECT payload_json FROM proposals WHERE finding_id = ? ORDER BY revision ASC", (finding_id,)
        ).fetchall()
        return [_proposal_from(row["payload_json"]) for row in rows]

    def next_revision(self, finding_id: str) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(revision), 0) AS r FROM proposals WHERE finding_id = ?", (finding_id,)
        ).fetchone()
        return int(row["r"]) + 1

    def set_proposal_status(self, proposal_id: str, status: str) -> None:
        with self.transaction() as conn:
            row = conn.execute("SELECT payload_json FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
            if row is None:
                raise NotFound(f"proposal {proposal_id} not found")
            payload = json.loads(row["payload_json"])
            payload["status"] = status
            conn.execute(
                "UPDATE proposals SET status = ?, payload_json = ? WHERE id = ?",
                (status, json.dumps(payload, sort_keys=True), proposal_id),
            )

    def set_proposal_validation(self, proposal_id: str, validation: dict[str, Any]) -> None:
        with self.transaction() as conn:
            row = conn.execute("SELECT payload_json FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
            if row is None:
                raise NotFound(f"proposal {proposal_id} not found")
            payload = json.loads(row["payload_json"])
            payload["validation"] = validation
            conn.execute(
                "UPDATE proposals SET payload_json = ? WHERE id = ?",
                (json.dumps(payload, sort_keys=True), proposal_id),
            )

    # -- simulations -----------------------------------------------------
    def add_simulation(self, result: SimulationResult) -> None:
        from .ids import content_id

        sim_id = content_id(
            "simulation",
            {
                "proposal_id": result.proposal_id,
                "revision": result.revision,
                "backend": result.backend,
                "started_at": result.started_at,
            },
        )
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO simulations
                   (id, proposal_id, revision, backend, status, created_at, payload_json)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    sim_id,
                    result.proposal_id,
                    result.revision,
                    result.backend,
                    result.status,
                    result.finished_at or time.time(),
                    json.dumps(result.to_dict(), sort_keys=True),
                ),
            )

    def simulations(self, proposal_id: str, revision: int | None = None) -> list[SimulationResult]:
        sql = "SELECT payload_json FROM simulations WHERE proposal_id = ?"
        params: list[Any] = [proposal_id]
        if revision is not None:
            sql += " AND revision = ?"
            params.append(revision)
        sql += " ORDER BY created_at ASC"
        return [_simulation_from(row["payload_json"]) for row in self._conn.execute(sql, params).fetchall()]

    def latest_simulation(self, proposal_id: str, revision: int | None = None) -> SimulationResult | None:
        sims = self.simulations(proposal_id, revision)
        return sims[-1] if sims else None

    # -- verdicts --------------------------------------------------------
    def add_verdict(self, verdict: Verdict) -> None:
        from .ids import content_id

        verdict_id = content_id(
            "verdict",
            {
                "proposal_id": verdict.proposal_id,
                "revision": verdict.revision,
                "verdict": verdict.verdict,
                "checks": verdict.checks,
            },
        )
        with self.transaction() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO verdicts
                   (id, proposal_id, revision, verdict, created_at, payload_json)
                   VALUES (?,?,?,?,?,?)""",
                (
                    verdict_id,
                    verdict.proposal_id,
                    verdict.revision,
                    verdict.verdict,
                    verdict.created_at or time.time(),
                    json.dumps(verdict.to_dict(), sort_keys=True),
                ),
            )

    def latest_verdict(self, proposal_id: str, revision: int | None = None) -> Verdict | None:
        sql = "SELECT payload_json FROM verdicts WHERE proposal_id = ?"
        params: list[Any] = [proposal_id]
        if revision is not None:
            sql += " AND revision = ?"
            params.append(revision)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT 1"
        row = self._conn.execute(sql, params).fetchone()
        if row is None:
            return None
        data = json.loads(row["payload_json"])
        return Verdict(
            proposal_id=data["proposal_id"],
            revision=data["revision"],
            verdict=data["verdict"],
            checks=data["checks"],
            stats=data.get("stats", {}),
            created_at=data.get("created_at", 0.0),
        )

    # -- decisions -------------------------------------------------------
    def add_decision(self, decision: Decision) -> None:
        """Insert a decision. Raises :class:`DuplicateDecision` on replay."""
        try:
            with self.transaction() as conn:
                conn.execute(
                    """INSERT INTO decisions
                       (proposal_id, revision, verdict, actor, reason, nonce, created_at, payload_json)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        decision.proposal_id,
                        decision.revision,
                        decision.verdict,
                        decision.actor,
                        decision.reason,
                        decision.nonce,
                        decision.created_at or time.time(),
                        json.dumps(decision.to_dict(), sort_keys=True),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateDecision(
                f"a decision already exists for {decision.proposal_id} rev {decision.revision} "
                f"or the nonce was reused"
            ) from exc

    def get_decision(self, proposal_id: str, revision: int) -> Decision | None:
        row = self._conn.execute(
            "SELECT payload_json FROM decisions WHERE proposal_id = ? AND revision = ?",
            (proposal_id, revision),
        ).fetchone()
        if row is None:
            return None
        data = json.loads(row["payload_json"])
        return Decision(
            proposal_id=data["proposal_id"],
            revision=data["revision"],
            verdict=data["verdict"],
            actor=data["actor"],
            reason=data["reason"],
            nonce=data["nonce"],
            created_at=data.get("created_at", 0.0),
            modifications=dict(data.get("modifications") or {}),
        )

    # -- deployments -----------------------------------------------------
    def add_deployment(self, deployment: Deployment) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO deployments
                   (id, proposal_id, revision, tag, snapshot_id, created_at, rolled_back, payload_json)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    deployment.id,
                    deployment.proposal_id,
                    deployment.revision,
                    deployment.tag,
                    deployment.snapshot_id,
                    deployment.created_at or time.time(),
                    1 if deployment.rolled_back else 0,
                    json.dumps(deployment.to_dict(), sort_keys=True),
                ),
            )

    def get_deployment(self, deployment_id: str) -> Deployment:
        row = self._conn.execute("SELECT payload_json FROM deployments WHERE id = ?", (deployment_id,)).fetchone()
        if row is None:
            raise NotFound(f"deployment {deployment_id} not found")
        return _deployment_from(row["payload_json"])

    def deployments(self, limit: int = 50) -> list[Deployment]:
        rows = self._conn.execute(
            "SELECT payload_json FROM deployments ORDER BY created_at DESC LIMIT ?", (int(limit),)
        ).fetchall()
        return [_deployment_from(row["payload_json"]) for row in rows]

    def latest_deployment(self) -> Deployment | None:
        rows = self.deployments(limit=1)
        return rows[0] if rows else None

    def mark_rolled_back(self, deployment_id: str) -> None:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT payload_json FROM deployments WHERE id = ?", (deployment_id,)
            ).fetchone()
            if row is None:
                raise NotFound(f"deployment {deployment_id} not found")
            payload = json.loads(row["payload_json"])
            payload["rolled_back"] = True
            conn.execute(
                "UPDATE deployments SET rolled_back = 1, payload_json = ? WHERE id = ?",
                (json.dumps(payload, sort_keys=True), deployment_id),
            )

    # -- aggregates ------------------------------------------------------
    def metrics_for(self, skill: str, **filters: Any) -> Metrics:
        return Metrics.from_episodes(self.episodes(skill, **filters))


# --------------------------------------------------------------------------- deserialisers


def _finding_from(payload: str) -> Finding:
    data = json.loads(payload)
    return Finding(
        id=data["id"],
        skill=data["skill"],
        kind=data["kind"],
        severity=data["severity"],
        title=data["title"],
        summary=data.get("summary", ""),
        metric=data.get("metric", ""),
        baseline=data.get("baseline"),
        current=data.get("current"),
        delta_rel=data.get("delta_rel"),
        p_value=data.get("p_value"),
        recommendation=data.get("recommendation", ""),
        params=data.get("params", {}),
        evidence=data.get("evidence", {}),
        window_start=data.get("window_start"),
        window_end=data.get("window_end"),
        created_at=data.get("created_at", 0.0),
        status=data.get("status", "open"),
    )


def _proposal_from(payload: str) -> Proposal:
    from .models import Edit, Risk, SimPlan

    data = json.loads(payload)
    return Proposal(
        id=data["id"],
        revision=data["revision"],
        finding_id=data["finding_id"],
        skill=data["skill"],
        kind=data["kind"],
        title=data["title"],
        rationale=data["rationale"],
        edits=[Edit(**edit) for edit in data.get("edits", [])],
        sim_plan=SimPlan(**data["sim_plan"]),
        expected_effect=data.get("expected_effect", ""),
        risk=Risk(**data["risk"]),
        config_overlay=data.get("config_overlay", {}),
        train_command=list(data.get("train_command") or []),
        created_by=data.get("created_by", "template"),
        created_at=data.get("created_at", 0.0),
        status=data.get("status", "proposed"),
        validation=data.get("validation"),
        supersedes=data.get("supersedes"),
    )


def _simulation_from(payload: str) -> SimulationResult:
    data = json.loads(payload)
    metrics = Metrics.from_dict(data["metrics"]) if data.get("metrics") else None
    return SimulationResult(
        proposal_id=data["proposal_id"],
        revision=data["revision"],
        backend=data["backend"],
        status=data["status"],
        metrics=metrics,
        artifacts=[Artifact(**a) for a in data.get("artifacts", [])],
        command=list(data.get("command") or []),
        log_path=data.get("log_path", ""),
        error=data.get("error", ""),
        started_at=data.get("started_at", 0.0),
        finished_at=data.get("finished_at", 0.0),
        episodes_n=data.get("episodes_n", 0),
    )


def _deployment_from(payload: str) -> Deployment:
    data = json.loads(payload)
    return Deployment(
        id=data["id"],
        proposal_id=data["proposal_id"],
        revision=data["revision"],
        tag=data["tag"],
        snapshot_id=data["snapshot_id"],
        artifacts=[Artifact(**a) for a in data.get("artifacts", [])],
        created_at=data.get("created_at", 0.0),
        note=data.get("note", ""),
        rolled_back=bool(data.get("rolled_back", False)),
    )
