"""Versioning: git guard rails and content-addressed snapshots.

Real repositories and real git in a temporary directory — no mocks. The point is
to prove the refusals, not only the happy path.
"""

from __future__ import annotations

import json
import subprocess
import tarfile
import pytest

from evo_layer.versioning.repo import GitError, GitRepo, sanitise_branch_component
from evo_layer.versioning.snapshot import SnapshotError, Snapshotter


@pytest.fixture
def repo(tmp_path) -> GitRepo:
    path = tmp_path / "repo"
    path.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True, capture_output=True)
    (path / "config").mkdir()
    (path / "config/app.yaml").write_text("value: 1\n", encoding="utf-8")
    (path / "README.md").write_text("hello\n", encoding="utf-8")
    for args in (["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@e", "commit", "-qm", "init"]):
        subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)
    return GitRepo(path)


# --------------------------------------------------------------------------- git


def test_branch_guards(repo):
    with pytest.raises(GitError):
        repo.ensure_branch("main")
    with pytest.raises(GitError):
        repo.ensure_branch("feature-x")  # must be prefixed
    branch = repo.ensure_branch("evolayer/locomotion/test")
    assert branch == "evolayer/locomotion/test"
    assert repo.branch == "evolayer/locomotion/test"


def test_commit_and_tag(repo):
    repo.ensure_branch("evolayer/locomotion/test")
    (repo.toplevel / "config/app.yaml").write_text("value: 2\n", encoding="utf-8")
    commit = repo.commit(["config/app.yaml"], "evolayer: raise value")
    assert commit.sha == repo.head
    assert commit.branch.startswith("evolayer/")
    tag = repo.tag("evolayer-locomotion-20260101-000000-prp_x", "rollback point")
    assert tag in repo.tags()
    with pytest.raises(GitError):
        repo.tag(tag, "duplicate")


def test_commit_refuses_main(repo):
    (repo.toplevel / "config/app.yaml").write_text("value: 3\n", encoding="utf-8")
    with pytest.raises(GitError) as excinfo:
        repo.commit(["config/app.yaml"], "should not happen")
    assert "main" in str(excinfo.value)


def test_paths_cannot_escape_the_repository(repo):
    repo.ensure_branch("evolayer/locomotion/test")
    with pytest.raises(GitError):
        repo.stage_paths(["../outside.txt"])


def test_rollback_restores_content(repo):
    repo.ensure_branch("evolayer/locomotion/test")
    before = (repo.toplevel / "config/app.yaml").read_text(encoding="utf-8")
    (repo.toplevel / "config/app.yaml").write_text("value: 99\n", encoding="utf-8")
    repo.commit(["config/app.yaml"], "change")
    repo.tag("evolayer-locomotion-20260101-000001-prp_y", "state before rollback")
    rolled = repo.restore_paths_from("HEAD~1", ["config/app.yaml"], message="rollback")
    assert (repo.toplevel / "config/app.yaml").read_text(encoding="utf-8") == before
    assert rolled.sha == repo.head


def test_rollback_refuses_unknown_ref(repo):
    repo.ensure_branch("evolayer/locomotion/test")
    with pytest.raises(GitError):
        repo.restore_paths_from("does-not-exist", ["config/app.yaml"], message="nope")


def test_dirty_detection(repo):
    assert repo.is_dirty() is False
    (repo.toplevel / "config/app.yaml").write_text("value: 5\n", encoding="utf-8")
    assert repo.is_dirty(["config/app.yaml"]) is True
    assert repo.is_dirty(["README.md"]) is False


def test_sanitise_branch_component():
    assert sanitise_branch_component("locomotion / bad:name") == "locomotion-bad-name"
    assert sanitise_branch_component("") == "change"


# --------------------------------------------------------------------------- snapshots


def test_snapshot_create_verify_restore(tmp_path):
    root = tmp_path / "ws"
    (root / "models").mkdir(parents=True)
    (root / "models/policy.onnx").write_bytes(b"onnx-bytes")
    (root / "config").mkdir()
    (root / "config/app.yaml").write_text("value: 1\n", encoding="utf-8")

    snapshots = Snapshotter(tmp_path / "snaps")
    snapshot = snapshots.create(["models", "config/app.yaml"], root=root, meta={"why": "test"})
    assert snapshot.id.startswith("snp_")
    ok, problems = snapshots.verify(snapshot.id)
    assert ok, problems

    (root / "models/policy.onnx").write_bytes(b"tampered")
    (root / "config/app.yaml").write_text("value: 999\n", encoding="utf-8")
    restored = snapshots.restore(snapshot.id, target_root=root)
    assert set(restored) == {"models/policy.onnx", "config/app.yaml"}
    assert (root / "models/policy.onnx").read_bytes() == b"onnx-bytes"
    assert (root / "config/app.yaml").read_text(encoding="utf-8") == "value: 1\n"


def test_snapshot_detects_a_corrupted_archive(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("original\n", encoding="utf-8")
    snapshots = Snapshotter(tmp_path / "snaps")
    snapshot = snapshots.create(["a.txt"], root=root)
    # Rewrite the archive with different bytes but the original manifest.
    import io

    with tarfile.open(snapshot.archive, "r:gz") as tar:
        manifest = tar.extractfile("manifest.json").read()
    with tarfile.open(snapshot.archive, "w:gz") as tar:
        payload = b"evil\n"
        info = tarfile.TarInfo("a.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
        info = tarfile.TarInfo("manifest.json")
        info.size = len(manifest)
        tar.addfile(info, io.BytesIO(manifest))
    ok, problems = snapshots.verify(snapshot.id)
    assert ok is False
    assert any("hash mismatch" in problem for problem in problems)
    with pytest.raises(SnapshotError):
        snapshots.restore(snapshot.id, target_root=root)


def test_snapshot_refuses_an_empty_set(tmp_path):
    snapshots = Snapshotter(tmp_path / "snaps")
    with pytest.raises(SnapshotError):
        snapshots.create(["missing"], root=tmp_path)


def test_snapshot_list_and_identical_states_collide(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("same\n", encoding="utf-8")
    snapshots = Snapshotter(tmp_path / "snaps")
    first = snapshots.create(["a.txt"], root=root)
    second = snapshots.create(["a.txt"], root=root)
    assert first.id == second.id
    assert [snap.id for snap in snapshots.list()] == [first.id]


def test_snapshot_manifest_is_json_readable(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("x\n", encoding="utf-8")
    snapshots = Snapshotter(tmp_path / "snaps")
    snapshot = snapshots.create(["a.txt"], root=root, meta={"proposal": "prp_1"})
    with tarfile.open(snapshot.archive, "r:gz") as tar:
        manifest = json.loads(tar.extractfile("manifest.json").read())
    assert manifest["meta"]["proposal"] == "prp_1"
    assert manifest["files"][0]["path"] == "a.txt"


def test_snapshot_blocks_traversal(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret\n", encoding="utf-8")
    (root / "link.txt").symlink_to(outside)
    snapshots = Snapshotter(tmp_path / "snaps")
    # Symlinks pointing outside are copied by content, but the manifest path stays
    # inside the root, so a restore can never write outside it.
    snapshot = snapshots.create(["link.txt"], root=root)
    assert snapshot.manifest[0].path == "link.txt"
