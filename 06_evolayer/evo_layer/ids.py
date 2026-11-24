"""Content-addressed identifiers and nonces.

Identifiers encode *what* the object is, not *when* it was created, so a
repeated proposal in identical circumstances produces the same id and the store
rejects the duplicate instead of silently accumulating near-copies.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Any

from .models import canonical_json

PREFIXES = {
    "episode": "ep",
    "finding": "fnd",
    "proposal": "prp",
    "simulation": "sim",
    "verdict": "vrd",
    "decision": "dec",
    "deployment": "dep",
    "snapshot": "snp",
    "report": "rpt",
    "episode_batch": "bat",
}

_DIGEST_LEN = 12


def digest(payload: Any) -> str:
    """Full hex sha256 of the canonical JSON form of ``payload``."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def content_id(kind: str, payload: Any) -> str:
    """Prefixed, truncated content hash, e.g. ``prp_3f9a1c0b7d2e``."""
    if kind not in PREFIXES:
        raise KeyError(f"unknown id kind: {kind!r}")
    return f"{PREFIXES[kind]}_{digest(payload)[:_DIGEST_LEN]}"


def is_id(kind: str, value: str) -> bool:
    prefix = PREFIXES.get(kind)
    if prefix is None:
        return False
    head, _, tail = value.partition("_")
    return head == prefix and len(tail) == _DIGEST_LEN and all(c in "0123456789abcdef" for c in tail)


def nonce(nbytes: int = 16) -> str:
    """Cryptographically strong, single-use token component."""
    return secrets.token_hex(nbytes)
