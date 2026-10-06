"""Finish a take: trim, set loudness, watermark, encode, then verify what ships.

Loudness is a single measured linear gain, capped so the true peak stays under
the ceiling. ffmpeg's loudnorm filter silently switches to dynamic mode (an
AGC that flattens delivery) when its conditions are not met, so it is not used.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from voicesmith import audio, ffmpeg, provenance
from voicesmith.engines import client

TARGET_LUFS = -16.0
CEILING_DBTP = -1.5
WM_MIN_PROB = 0.5


class MasteringError(RuntimeError):
    pass


@dataclass
class Delivered:
    path: Path
    loudness_lufs: float
    true_peak_dbtp: float
    watermark: dict
    tags_ok: bool


@dataclass
class MasterResult:
    files: list[Delivered] = field(default_factory=list)
    gain_db: float = 0.0


def _write_tmp(dirpath: Path, name: str, wav: np.ndarray, sr: int) -> Path:
    return audio.save_wav(dirpath / name, wav, sr, subtype="FLOAT")


def master(wav: np.ndarray, sr: int, outputs: list[Path], tags: dict[str, str]) -> MasterResult:
    w = client.any_watermark_worker()
    if w is None:
        raise MasteringError("no engine environment is installed to apply the watermark")
    x = audio.trim(wav, sr, lead_ms=120, tail_ms=250)
    with tempfile.TemporaryDirectory(prefix="voicesmith-") as tmp:
        tmpd = Path(tmp)
        raw = _write_tmp(tmpd, "raw.wav", x, sr)
        loud = ffmpeg.measure_loudness(raw)
        if not np.isfinite(loud.integrated):
            raise MasteringError("take is silent")
        gain_db = min(TARGET_LUFS - loud.integrated, CEILING_DBTP - loud.true_peak)
        leveled = _write_tmp(tmpd, "leveled.wav", x * (10 ** (gain_db / 20)), sr)
        marked = tmpd / "marked.wav"
        wm = w.call("watermark", {"in_wav": str(leveled), "out_wav": str(marked)}, timeout=600)
        if wm.get("audioseal_prob", 0) < WM_MIN_PROB or not wm.get("audioseal_payload_ok"):
            raise MasteringError(f"watermark did not verify after embedding: {wm}")
        result = MasterResult(gain_db=round(gain_db, 2))
        for out in outputs:
            out.parent.mkdir(parents=True, exist_ok=True)
            part = out.with_name(out.stem + ".part" + out.suffix)
            ffmpeg.encode(marked, part, tags)
            # Verify the encoded bytes, not the WAV they came from.
            decoded = _write_tmp(tmpd, f"check-{out.suffix.lstrip('.')}.wav", ffmpeg.decode(part, sr), sr)
            det = w.call("detect", {"wav": str(decoded)}, timeout=600)
            meta = ffmpeg.read_metadata(part)
            lv = ffmpeg.measure_loudness(part)
            problems = []
            if det.get("audioseal_prob", 0) < WM_MIN_PROB or not det.get("audioseal_payload_ok"):
                problems.append(f"watermark lost in {out.suffix} ({det})")
            if not provenance.check_tags(meta):
                problems.append(f"disclosure tags missing from {out.suffix}")
            if lv.true_peak > CEILING_DBTP + 1.0:
                problems.append(f"true peak {lv.true_peak:.1f} dBTP after encoding")
            if problems:
                part.unlink(missing_ok=True)
                raise MasteringError("; ".join(problems))
            part.replace(out)
            result.files.append(
                Delivered(path=out, loudness_lufs=lv.integrated, true_peak_dbtp=lv.true_peak, watermark=det, tags_ok=True)
            )
    return result
