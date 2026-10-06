"""Get audio onto disk: local files, URLs (via yt-dlp), or the microphone."""

from __future__ import annotations

import hashlib
import shutil
import time
from collections.abc import Callable
from pathlib import Path

Log = Callable[[str], None]

MEDIA_EXT = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".webm", ".mp4", ".mkv", ".mov", ".wma", ".aiff", ".aif"}


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def from_path(src: str, dest: Path, log: Log) -> list[dict]:
    p = Path(src).expanduser()
    files = sorted(f for f in p.rglob("*") if f.suffix.lower() in MEDIA_EXT) if p.is_dir() else [p]
    out = []
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f)
        target = dest / f"{len(list(dest.iterdir())):03d}-{f.name}"
        shutil.copyfile(f, target)
        out.append({"kind": "file", "origin": str(f.resolve()), "path": target.name, "sha256": sha256(target)})
        log(f"added {f.name}")
    return out


def from_url(url: str, dest: Path, log: Log, max_items: int = 25) -> list[dict]:
    """Download audio only, keeping the original stream (no lossy re-encode)."""
    from yt_dlp import YoutubeDL

    before = set(dest.iterdir())
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dest / "%(extractor)s-%(id)s.%(ext)s"),
        "writeinfojson": True,
        "playlistend": max_items,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ignoreerrors": "only_download",
        "restrictfilenames": True,
        "retries": 5,
    }
    with YoutubeDL(opts) as ydl:
        ydl.download([url])
    out = []
    for f in sorted(set(dest.iterdir()) - before):
        if f.suffix.lower() == ".json" or f.name.endswith(".part"):
            continue
        info = f.with_suffix(".info.json")
        meta = {}
        if info.exists():
            import json

            d = json.loads(info.read_text(encoding="utf-8"))
            meta = {k: d.get(k) for k in ("title", "uploader", "webpage_url", "upload_date", "duration", "language")}
        out.append({"kind": "url", "origin": url, "path": f.name, "sha256": sha256(f), **meta})
        log(f"downloaded {meta.get('title') or f.name}")
    if not out:
        raise RuntimeError(f"nothing was downloaded from {url}")
    return out


def record(dest: Path, seconds: float, log: Log, sample_rate: int = 48_000) -> dict:
    """Record from the default microphone."""
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio missing on some Linux installs
        raise RuntimeError("microphone recording needs PortAudio (e.g. apt install libportaudio2)") from exc
    import numpy as np
    import soundfile as sf

    log(f"recording {seconds:.0f}s from the default microphone...")
    data = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    data = data.reshape(-1)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak < 0.01:
        raise RuntimeError("the recording is silent; check the microphone and its permissions")
    if peak > 0.99:
        log("warning: the recording clipped; move back from the microphone or lower the input gain")
    target = dest / f"mic-{time.strftime('%Y%m%d-%H%M%S')}.wav"
    sf.write(target, data, sample_rate, subtype="PCM_24")
    return {"kind": "mic", "origin": "microphone", "path": target.name, "sha256": sha256(target)}
