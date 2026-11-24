"""Content-addressed snapshots of deployment state.

A snapshot is a tar.gz plus a manifest of ``path -> sha256``. Restoring verifies
every hash *before* writing, so a corrupted or tampered archive cannot silently
reintroduce different bytes. ``snapshot_id`` is derived from the manifest, which
makes identical deployment states collide on purpose.

Snapshots are the rollback mechanism for artifacts (ONNX files, policy YAMLs);
git tags are the rollback mechanism for source. The installer takes both.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..ids import content_id


class SnapshotError(RuntimeError):
    """A snapshot is missing, corrupt, or refuses to overwrite unknown state."""


@dataclass
class ManifestEntry:
    path: str
    sha256: str
    size: int
    mode: int

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size, "mode": self.mode}


@dataclass
class Snapshot:
    id: str
    archive: Path
    manifest: list[ManifestEntry]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "archive": str(self.archive),
            "meta": self.meta,
            "files": [entry.to_dict() for entry in self.manifest],
        }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Snapshotter:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create(self, paths: Iterable[str | Path], *, root: str | Path, meta: dict[str, Any] | None = None) -> Snapshot:
        """Archive ``paths`` (files or directories) relative to ``root``."""
        root = Path(root).resolve()
        files: list[tuple[Path, str]] = []
        for raw in paths:
            candidate = Path(raw)
            candidate = candidate if candidate.is_absolute() else root / candidate
            if not candidate.exists():
                continue
            if candidate.is_dir():
                for child in sorted(candidate.rglob("*")):
                    if child.is_file():
                        files.append((child, str(child.relative_to(root))))
            elif candidate.is_file():
                files.append((candidate, str(candidate.relative_to(root))))
        if not files:
            raise SnapshotError("nothing to snapshot (all paths missing or empty)")

        manifest: list[ManifestEntry] = []
        blobs: list[tuple[str, bytes, int]] = []
        for path, relpath in sorted(files, key=lambda item: item[1]):
            data = path.read_bytes()
            if ".." in Path(relpath).parts:
                raise SnapshotError(f"refusing to snapshot a traversing path: {relpath}")
            manifest.append(
                ManifestEntry(path=relpath, sha256=_sha256(data), size=len(data), mode=path.stat().st_mode & 0o777)
            )
            blobs.append((relpath, data, path.stat().st_mode & 0o777))

        snapshot_id = content_id("snapshot", [entry.to_dict() for entry in manifest])
        archive = self.base_dir / f"{snapshot_id}.tar.gz"
        if not archive.exists():
            with tarfile.open(archive, "w:gz") as tar:
                for relpath, data, mode in blobs:
                    info = tarfile.TarInfo(relpath)
                    info.size = len(data)
                    info.mode = mode
                    info.mtime = 0
                    tar.addfile(info, io.BytesIO(data))
                payload = json.dumps(
                    {"id": snapshot_id, "meta": meta or {}, "files": [m.to_dict() for m in manifest]},
                    indent=2,
                    sort_keys=True,
                ).encode("utf-8")
                info = tarfile.TarInfo("manifest.json")
                info.size = len(payload)
                info.mtime = 0
                tar.addfile(info, io.BytesIO(payload))
            os.chmod(archive, 0o600)
        return Snapshot(id=snapshot_id, archive=archive, manifest=manifest, meta=meta or {})

    def load(self, snapshot_id: str) -> Snapshot:
        archive = self.base_dir / f"{snapshot_id}.tar.gz"
        if not archive.is_file():
            raise SnapshotError(f"snapshot archive not found: {archive}")
        with tarfile.open(archive, "r:gz") as tar:
            member = tar.getmember("manifest.json")
            handle = tar.extractfile(member)
            if handle is None:
                raise SnapshotError(f"snapshot {snapshot_id} has no readable manifest")
            data = json.loads(handle.read().decode("utf-8"))
        manifest = [ManifestEntry(**entry) for entry in data["files"]]
        return Snapshot(id=data.get("id", snapshot_id), archive=archive, manifest=manifest, meta=data.get("meta", {}))

    def verify(self, snapshot_id: str) -> tuple[bool, list[str]]:
        snapshot = self.load(snapshot_id)
        problems: list[str] = []
        with tarfile.open(snapshot.archive, "r:gz") as tar:
            names = set(tar.getnames())
            for entry in snapshot.manifest:
                if entry.path not in names:
                    problems.append(f"missing from archive: {entry.path}")
                    continue
                handle = tar.extractfile(entry.path)
                if handle is None:
                    problems.append(f"unreadable: {entry.path}")
                    continue
                if _sha256(handle.read()) != entry.sha256:
                    problems.append(f"hash mismatch: {entry.path}")
        return (not problems, problems)

    def restore(self, snapshot_id: str, *, target_root: str | Path, only: list[str] | None = None) -> list[str]:
        """Verify, then restore into ``target_root``. Never touches paths outside it."""
        ok, problems = self.verify(snapshot_id)
        if not ok:
            raise SnapshotError(f"snapshot {snapshot_id} failed verification: {'; '.join(problems[:5])}")
        snapshot = self.load(snapshot_id)
        root = Path(target_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        restored: list[str] = []
        with tarfile.open(snapshot.archive, "r:gz") as tar:
            for entry in snapshot.manifest:
                if only is not None and entry.path not in only:
                    continue
                handle = tar.extractfile(entry.path)
                if handle is None:
                    raise SnapshotError(f"cannot read {entry.path} from {snapshot.archive}")
                data = handle.read()
                if _sha256(data) != entry.sha256:
                    raise SnapshotError(f"hash changed while restoring {entry.path}")
                destination = (root / entry.path).resolve()
                try:
                    destination.relative_to(root)
                except ValueError as exc:
                    raise SnapshotError(f"refusing to write outside {root}: {entry.path}") from exc
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                os.chmod(destination, entry.mode)
                restored.append(entry.path)
        return restored

    def list(self) -> list[Snapshot]:
        out: list[Snapshot] = []
        for archive in sorted(self.base_dir.glob("snp_*.tar.gz")):
            try:
                out.append(self.load(archive.name.removesuffix(".tar.gz")))
            except (SnapshotError, KeyError, json.JSONDecodeError):
                continue
        return out


def snapshot_paths_for_deploy(cfg_root: Path, deploy_root: Path, robot: str) -> list[str]:
    """The deployment state worth snapshotting before a change."""
    return [
        str(Path("00_body_base/roboparty_deploy/src/inference/robots") / robot / "configs"),
        str(Path("00_body_base/roboparty_deploy/src/inference/robots") / robot / "models"),
        str(Path("00_body_base/roboparty_deploy/src/inference/robots") / robot / "latents"),
    ]


def temp_extract_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="evolayer-snapshot-"))
