"""Video artifacts for Hermes review.

Hermes needs to *see* the rehearsal, not only read numbers. The MuJoCo evaluator
renders frames; this module turns them into an mp4 when ``ffmpeg`` exists and
otherwise keeps the frames, hashes everything, and reports exactly what is
available — a missing video is stated, never fabricated.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..execution import run
from ..models import Artifact


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_for(path: Path, *, kind: str = "file") -> Artifact:
    return Artifact(
        name=path.name,
        path=str(path),
        sha256=sha256_file(path),
        kind=kind,
        size=path.stat().st_size,
    )


@dataclass
class VideoResult:
    path: Path | None
    kind: str
    detail: str


def encode_frames(frames_dir: Path, out_path: Path, *, fps: int = 25) -> VideoResult:
    """Encode PNG frames to mp4 with ffmpeg, or keep the frames."""
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        return VideoResult(None, "none", "no frames rendered")
    if shutil.which("ffmpeg") is None:
        return VideoResult(frames[0], "frame", f"ffmpeg missing; kept {len(frames)} frames (first: {frames[0].name})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "%05d.png"),
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ],
        cwd=str(frames_dir),
        timeout=300,
    )
    if result.ok and out_path.is_file() and out_path.stat().st_size > 0:
        return VideoResult(out_path, "video", f"encoded {len(frames)} frames at {fps} fps")
    return VideoResult(frames[0], "frame", f"ffmpeg failed; kept {len(frames)} frames ({result.tail(3)})")


def probe(path: Path) -> dict[str, object]:
    """Minimal, dependency-free description of a media file."""
    if not path.is_file():
        return {"exists": False}
    info: dict[str, object] = {
        "exists": True,
        "size": path.stat().st_size,
        "suffix": path.suffix,
        "sha256": sha256_file(path),
    }
    if shutil.which("ffprobe"):
        result = run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
            cwd=str(path.parent),
            timeout=30,
        )
        if result.ok:
            try:
                info["duration_s"] = float(result.stdout.strip())
            except ValueError:
                pass
    return info
