"""Pinned, checksummed downloads for the ONNX models the core uses.

Each model is fetched once into ``~/.voicesmith/models`` and verified by SHA-256.
A download is written to a ``.part`` file and only renamed into place after the
hash matches, so a half-finished download can never be mistaken for a model.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from voicesmith import paths

_SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    sha256: str
    size: int
    licence: str
    purpose: str
    archive: bool = False  # .tar.bz2 that unpacks to a directory named ``name``

    @property
    def path(self) -> Path:
        return paths.models_dir() / (self.name if self.archive else Path(self.url).name)


MODELS: dict[str, ModelSpec] = {
    "vad": ModelSpec(
        name="silero-vad",
        url=f"{_SHERPA}/asr-models/silero_vad.onnx",
        sha256="9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
        size=643_854,
        licence="MIT (Silero)",
        purpose="finds speech in long recordings",
    ),
    "speaker": ModelSpec(
        name="titanet-small",
        url=f"{_SHERPA}/speaker-recongition-models/nemo_en_titanet_small.onnx",
        sha256="ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e",
        size=40_257_283,
        licence="CC-BY-4.0 (NVIDIA NeMo TitaNet)",
        purpose="speaker embeddings for filtering, consent matching and similarity scoring",
    ),
    "asr": ModelSpec(
        name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        url=f"{_SHERPA}/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
        sha256="5793d0fd397c5778d2cf2126994d58e9d56b1be7c04d13c7a15bb1b4eafb16bf",
        size=487_170_055,
        licence="CC-BY-4.0 (NVIDIA Parakeet TDT 0.6B v3)",
        purpose="transcripts with word timings in 25 European languages",
        archive=True,
    ),
    "tagger": ModelSpec(
        name="sherpa-onnx-ced-mini-audio-tagging-2024-04-19",
        url=f"{_SHERPA}/audio-tagging-models/sherpa-onnx-ced-mini-audio-tagging-2024-04-19.tar.bz2",
        sha256="bec8bfa0af2c20ec3a9e7c6dd6c92d0fb6e96d6b25370ebe7a48e7283a60a02c",
        size=47_658_399,
        licence="Apache-2.0 (CED, AudioSet labels)",
        purpose="spots music, laughter and applause so they never end up in a reference clip",
        archive=True,
    ),
}

Progress = Callable[[int, int], None]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def is_ready(key: str) -> bool:
    spec = MODELS[key]
    if spec.archive:
        return (spec.path / ".verified").exists()
    return spec.path.exists() and spec.path.with_suffix(spec.path.suffix + ".verified").exists()


def ensure(key: str, progress: Progress | None = None) -> Path:
    """Return the local path of a model, downloading and verifying it if needed."""
    spec = MODELS[key]
    if is_ready(key):
        return spec.path
    if os.environ.get("VOICESMITH_OFFLINE") == "1":
        raise RuntimeError(f"model {spec.name!r} is not downloaded and VOICESMITH_OFFLINE=1")
    dest_dir = paths.models_dir()
    with tempfile.TemporaryDirectory(dir=dest_dir, prefix=".dl-") as tmp:
        part = Path(tmp) / (Path(spec.url).name + ".part")
        _download(spec.url, part, spec.size, progress)
        got = sha256_file(part)
        if got != spec.sha256:
            raise RuntimeError(f"checksum mismatch for {spec.name}: expected {spec.sha256[:12]}..., got {got[:12]}...")
        if spec.archive:
            with tarfile.open(part, "r:bz2") as tar:
                _safe_extract(tar, Path(tmp))
            if spec.path.exists():
                shutil.rmtree(spec.path)
            (Path(tmp) / spec.name).replace(spec.path)
            (spec.path / ".verified").write_text(spec.sha256)
        else:
            part.replace(spec.path)
            spec.path.with_suffix(spec.path.suffix + ".verified").write_text(spec.sha256)
    return spec.path


def _download(url: str, dest: Path, expected: int, progress: Progress | None) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "voicesmith"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length") or expected)
        done = 0
        while block := resp.read(1 << 20):
            out.write(block)
            done += len(block)
            if progress:
                progress(done, total)


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    root = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(root)) or member.issym() or member.islnk():
            raise RuntimeError(f"refusing unsafe archive member {member.name!r}")
    tar.extractall(dest)


def status() -> list[dict[str, object]]:
    return [
        {
            "key": key,
            "name": spec.name,
            "ready": is_ready(key),
            "size_mb": round(spec.size / 1e6, 1),
            "licence": spec.licence,
            "purpose": spec.purpose,
        }
        for key, spec in MODELS.items()
    ]
