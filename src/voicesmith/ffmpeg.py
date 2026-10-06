"""Thin, strict wrappers around the ffmpeg binary.

voicesmith prefers a system ffmpeg and falls back to the static build that
ships with ``imageio-ffmpeg``, so a fresh install works with no extra steps.
"""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


class FFmpegError(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - only hit on broken installs
        raise FFmpegError("ffmpeg not found: install it or reinstall voicesmith (imageio-ffmpeg)") from exc


def run(args: list[str], *, input_bytes: bytes | None = None, timeout: float = 600) -> subprocess.CompletedProcess:
    cmd = [exe(), "-hide_banner", "-nostdin", *args]
    proc = subprocess.run(cmd, input=input_bytes, capture_output=True, timeout=timeout, creationflags=NO_WINDOW)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace")[-1500:]
        raise FFmpegError(f"ffmpeg failed ({proc.returncode}): {' '.join(args[:6])}...\n{tail}")
    return proc


@dataclass
class Probe:
    duration: float | None
    sample_rate: int | None
    channels: int | None
    codec: str | None


_DUR = re.compile(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)")
_AUD = re.compile(r"Audio: ([\w\-]+)[^\n]*?, (\d+) Hz, ([^,\n]+)")


def probe(path: str | Path) -> Probe:
    """Read basic stream facts from ffmpeg's banner (imageio-ffmpeg ships no ffprobe)."""
    proc = subprocess.run([exe(), "-hide_banner", "-nostdin", "-i", str(path)], capture_output=True, timeout=60,
                          creationflags=NO_WINDOW)
    text = proc.stderr.decode("utf-8", "replace")
    if "Audio:" not in text:
        raise FFmpegError(f"no audio stream found in {path}")
    dur = None
    if m := _DUR.search(text):
        h, mi, s = m.groups()
        dur = int(h) * 3600 + int(mi) * 60 + float(s)
    sr = ch = codec = None
    if m := _AUD.search(text):
        codec, sr_s, layout = m.groups()
        sr = int(sr_s)
        layout = layout.strip()
        ch = 1 if layout.startswith("mono") else 2 if layout.startswith("stereo") else None
        if ch is None and (n := re.match(r"(\d+) channels", layout)):
            ch = int(n.group(1))
    return Probe(duration=dur, sample_rate=sr, channels=ch, codec=codec)


def decode(path: str | Path, sample_rate: int, *, start: float | None = None, duration: float | None = None) -> np.ndarray:
    """Decode any audio or video file to mono float32 at ``sample_rate``.

    Decoding always goes through ffmpeg so behaviour does not depend on which
    codecs libsndfile or torchaudio happen to support on this machine.
    """
    args: list[str] = []
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(path)]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    args += ["-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    proc = run(args)
    audio = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32, copy=True)
    if audio.size == 0:
        raise FFmpegError(f"decoded no samples from {path}")
    return audio


@dataclass
class Loudness:
    integrated: float  # LUFS
    true_peak: float  # dBTP
    lra: float  # LU


_EBU_I = re.compile(r"I:\s+(-?[\d.]+|-inf) LUFS")
_EBU_LRA = re.compile(r"LRA:\s+(-?[\d.]+) LU")
_EBU_PEAK = re.compile(r"Peak:\s+(-?[\d.]+|-inf) dBFS")


def measure_loudness(path: str | Path) -> Loudness:
    """EBU R128 integrated loudness and true peak via ffmpeg's ebur128 filter."""
    proc = run(["-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"])
    text = proc.stderr.decode("utf-8", "replace")
    summary = text[text.rfind("Summary:") :]
    def grab(rx: re.Pattern[str]) -> float:
        m = rx.search(summary)
        if not m:
            raise FFmpegError(f"could not parse loudness summary for {path}")
        return float("-inf") if m.group(1) == "-inf" else float(m.group(1))
    return Loudness(integrated=grab(_EBU_I), true_peak=grab(_EBU_PEAK), lra=grab(_EBU_LRA))


CODECS = {
    "wav": ["-c:a", "pcm_s24le"],
    "flac": ["-c:a", "flac"],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "192k", "-id3v2_version", "3"],
    "m4a": ["-c:a", "aac", "-b:a", "192k", "-movflags", "+use_metadata_tags"],
    "ogg": ["-c:a", "libopus", "-b:a", "96k"],
}


def encode(src: str | Path, dst: str | Path, metadata: dict[str, str] | None = None) -> Path:
    dst = Path(dst)
    fmt = dst.suffix.lower().lstrip(".")
    if fmt not in CODECS:
        raise ValueError(f"unsupported output format {fmt!r}; choose one of {', '.join(CODECS)}")
    args = ["-y", "-i", str(src), "-map_metadata", "-1"]
    # Ogg/Opus keeps tags on the audio stream, every other container on the file.
    flag = "-metadata:s:a:0" if fmt == "ogg" else "-metadata"
    for key, value in (metadata or {}).items():
        args += [flag, f"{key}={value}"]
    args += [*CODECS[fmt], str(dst)]
    run(args)
    return dst


def read_metadata(path: str | Path) -> dict[str, str]:
    proc = run(["-i", str(path), "-f", "ffmetadata", "-"])
    text = proc.stdout.decode("utf-8", "replace")
    meta: dict[str, str] = {}
    if str(path).lower().endswith((".ogg", ".opus")):
        # Opus tags live on the stream, which ffmetadata cannot dump; read them from the banner.
        banner = run(["-i", str(path), "-f", "null", "-"]).stderr.decode("utf-8", "replace")
        for m in re.finditer(r"^\s{6,}(\w+)\s*:\s?(.*)$", banner, re.M):
            meta.setdefault(m.group(1).strip().lower(), m.group(2).strip())
    for line in text.splitlines():
        if not line or line.startswith((";", "[")) or "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.strip().lower()] = value.replace("\\=", "=").replace("\\;", ";").strip()
    return meta


def capabilities() -> dict[str, bool]:
    """Which encoders and filters this ffmpeg build actually has."""
    enc = run(["-encoders"]).stdout.decode("utf-8", "replace")
    flt = run(["-filters"]).stdout.decode("utf-8", "replace")
    return {
        "libmp3lame": " libmp3lame " in enc,
        "aac": " aac " in enc,
        "libopus": " libopus " in enc,
        "flac": " flac " in enc,
        "ebur128": " ebur128 " in flt,
    }


def json_dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
