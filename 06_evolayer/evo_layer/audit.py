"""Append-only, hash-chained audit log.

Every state change in EvoLayer — an ingested batch, a finding, a decision, a
deployment, a rollback, an HTTP request that mutates state — is appended here.
Each record commits to the previous record's hash, so removing or editing a line
in the middle of the file is detectable: :meth:`AuditLog.verify` walks the chain
and reports the first broken link.

The file is written with ``O_APPEND`` under an exclusive ``flock`` and fsynced
before the call returns, which makes concurrent writers (CLI + HTTP server)
safe on a POSIX filesystem.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

try:  # POSIX only; the fallback keeps the module importable elsewhere.
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

GENESIS = "0" * 64


class AuditError(RuntimeError):
    """Raised when the audit log cannot be trusted or written."""


@dataclass(frozen=True)
class Entry:
    seq: int
    ts: float
    event: str
    subject: str
    actor: str
    data: dict[str, Any]
    prev_hash: str
    hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "event": self.event,
            "subject": self.subject,
            "actor": self.actor,
            "data": self.data,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }


def _hash(prev_hash: str, body: dict[str, Any]) -> str:
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256((prev_hash + payload).encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch(mode=0o600)

    # -- write -----------------------------------------------------------
    def append(self, event: str, *, subject: str = "", actor: str = "evolayer", data: dict[str, Any] | None = None) -> Entry:
        if not event:
            raise AuditError("audit event name is required")
        ts = time.time()
        with self.path.open("a", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                prev_seq, prev_hash = self._tail_state()
                body = {
                    "seq": prev_seq + 1,
                    "ts": ts,
                    "event": event,
                    "subject": subject,
                    "actor": actor,
                    "data": data or {},
                }
                entry_hash = _hash(prev_hash, body)
                record = {**body, "prev_hash": prev_hash, "hash": entry_hash}
                handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return Entry(
            seq=body["seq"], ts=ts, event=event, subject=subject, actor=actor,
            data=data or {}, prev_hash=prev_hash, hash=entry_hash,
        )

    def _tail_state(self) -> tuple[int, str]:
        """Read only the last record, so appending stays O(1)."""
        size = self.path.stat().st_size
        if size == 0:
            return (0, GENESIS)
        with self.path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            pos = handle.tell()
            buffer = b""
            while pos > 0:
                step = min(4096, pos)
                pos -= step
                handle.seek(pos)
                buffer = handle.read(step) + buffer
                lines = [line for line in buffer.split(b"\n") if line.strip()]
                if len(lines) >= 2 or pos == 0:
                    record = json.loads(lines[-1])
                    return (int(record["seq"]), str(record["hash"]))
        return (0, GENESIS)

    # -- read ------------------------------------------------------------
    def entries(self) -> Iterator[Entry]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    yield Entry(
                        seq=int(record["seq"]),
                        ts=float(record["ts"]),
                        event=str(record["event"]),
                        subject=str(record.get("subject", "")),
                        actor=str(record.get("actor", "")),
                        data=dict(record.get("data") or {}),
                        prev_hash=str(record["prev_hash"]),
                        hash=str(record["hash"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise AuditError(f"malformed audit record at line {line_no}: {exc}") from exc

    def count(self) -> int:
        return sum(1 for _ in self.entries())

    def tail(self, n: int = 20) -> list[Entry]:
        items = list(self.entries())
        return items[-n:]

    # -- integrity -------------------------------------------------------
    def verify(self) -> tuple[bool, int, int | None, str]:
        """Return ``(ok, count, broken_seq, reason)``."""
        prev_hash = GENESIS
        expected_seq = 1
        count = 0
        for entry in self.entries():
            count += 1
            if entry.seq != expected_seq:
                return (False, count, entry.seq, f"sequence gap: expected {expected_seq}, found {entry.seq}")
            if entry.prev_hash != prev_hash:
                return (False, count, entry.seq, "previous-hash mismatch (record removed or reordered)")
            body = {
                "seq": entry.seq,
                "ts": entry.ts,
                "event": entry.event,
                "subject": entry.subject,
                "actor": entry.actor,
                "data": entry.data,
            }
            if _hash(prev_hash, body) != entry.hash:
                return (False, count, entry.seq, "content hash mismatch (record edited)")
            prev_hash = entry.hash
            expected_seq += 1
        return (True, count, None, "chain intact")
