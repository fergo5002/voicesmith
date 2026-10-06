"""Small, dependable audio helpers built on numpy and scipy only."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from voicesmith import ffmpeg

ANALYSIS_SR = 16_000
REFERENCE_SR = 24_000


def load(path: str | Path, sample_rate: int = ANALYSIS_SR) -> np.ndarray:
    return ffmpeg.decode(path, sample_rate)


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int, subtype: str = "PCM_24") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    sf.write(tmp, np.clip(audio, -1.0, 1.0).astype(np.float32), sample_rate, subtype=subtype, format="WAV")
    tmp.replace(path)
    return path


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return audio.astype(np.float32, copy=False)
    g = gcd(sr_from, sr_to)
    return resample_poly(audio, sr_to // g, sr_from // g).astype(np.float32)


def db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-12)))


def rms_db(audio: np.ndarray) -> float:
    return db(float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0)


def peak_db(audio: np.ndarray) -> float:
    return db(float(np.max(np.abs(audio))) if audio.size else 0.0)


def clipping_ratio(audio: np.ndarray, threshold: float = 0.999) -> float:
    return float(np.mean(np.abs(audio) >= threshold)) if audio.size else 0.0


def frame_rms_db(audio: np.ndarray, sample_rate: int, frame_ms: float = 10.0) -> np.ndarray:
    hop = max(1, int(sample_rate * frame_ms / 1000))
    n = len(audio) // hop
    if n == 0:
        return np.array([rms_db(audio)])
    frames = audio[: n * hop].reshape(n, hop).astype(np.float64)
    return 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-12)


def active_bounds(
    audio: np.ndarray,
    sample_rate: int,
    *,
    floor_db: float = -45.0,
    relative_db: float = -40.0,
    sustained_ms: float = 30.0,
    frame_ms: float = 10.0,
) -> tuple[int, int]:
    """First and last sample of sustained sound.

    A frame counts as active when it is above both an absolute floor and a level
    relative to the loudest frame, and activity must last ``sustained_ms`` so a
    click or breath does not count as the start of speech.
    """
    levels = frame_rms_db(audio, sample_rate, frame_ms)
    thresh = max(floor_db, float(levels.max()) + relative_db)
    active = levels > thresh
    need = max(1, int(round(sustained_ms / frame_ms)))
    run = np.convolve(active.astype(int), np.ones(need, dtype=int), mode="valid") >= need
    idx = np.flatnonzero(run)
    hop = int(sample_rate * frame_ms / 1000)
    if idx.size == 0:
        return 0, len(audio)
    start = int(idx[0]) * hop
    end = min(len(audio), (int(idx[-1]) + need) * hop)
    return start, end


def fade(audio: np.ndarray, sample_rate: int, in_ms: float = 10.0, out_ms: float = 10.0) -> np.ndarray:
    out = audio.astype(np.float32, copy=True)
    n_in = min(len(out), int(sample_rate * in_ms / 1000))
    n_out = min(len(out), int(sample_rate * out_ms / 1000))
    if n_in:
        out[:n_in] *= np.sin(np.linspace(0, np.pi / 2, n_in)) ** 2
    if n_out:
        out[-n_out:] *= np.cos(np.linspace(0, np.pi / 2, n_out)) ** 2
    return out


def trim(audio: np.ndarray, sample_rate: int, *, lead_ms: float = 120.0, tail_ms: float = 200.0) -> np.ndarray:
    """Trim silence to a natural lead-in and tail, then fade the edges."""
    start, end = active_bounds(audio, sample_rate)
    start = max(0, start - int(sample_rate * lead_ms / 1000))
    end = min(len(audio), end + int(sample_rate * tail_ms / 1000))
    return fade(audio[start:end], sample_rate)


def spectral_rolloff_hz(audio: np.ndarray, sample_rate: int, fraction: float = 0.95) -> float:
    """Frequency below which ``fraction`` of the energy sits. Phone audio and low-bitrate rips sit low."""
    if audio.size < 2048:
        return 0.0
    n = 2048
    hop = 1024
    frames = np.lib.stride_tricks.sliding_window_view(audio, n)[::hop]
    spec = np.abs(np.fft.rfft(frames * np.hanning(n), axis=1)) ** 2
    energy = spec.sum(axis=0)
    cum = np.cumsum(energy)
    if cum[-1] <= 0:
        return 0.0
    k = int(np.searchsorted(cum, fraction * cum[-1]))
    return float(k * sample_rate / n)


def snr_db(audio: np.ndarray, sample_rate: int) -> float:
    """Rough SNR: loud-frame level minus quiet-frame level over 20 ms frames.

    Speech always contains short pauses, so the 10th percentile of frame energy
    is a fair estimate of the noise floor and the 95th of the speech level. It is
    only used to rank clips from the same speaker against each other.
    """
    levels = frame_rms_db(audio, sample_rate, frame_ms=20.0)
    if levels.size < 10:
        return 0.0
    return float(np.percentile(levels, 95) - np.percentile(levels, 10))
