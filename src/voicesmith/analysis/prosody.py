"""Pitch and pace measurements in plain numpy.

A clone can match a speaker's timbre and still sound wrong because it is flatter,
higher, or faster than them. These numbers let the scorer notice that.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

SR = 16_000


@dataclass
class Prosody:
    f0_median: float  # Hz, voiced frames only
    f0_range_st: float  # 5th to 95th percentile, in semitones
    voiced_ratio: float
    rate_wps: float  # words per second of speech, 0 if unknown

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def f0_track(audio: np.ndarray, sr: int = SR, fmin: float = 60.0, fmax: float = 420.0, hop_ms: float = 10.0) -> np.ndarray:
    """YIN pitch track (de Cheveigne and Kawahara, 2002). Unvoiced frames are 0."""
    win = int(sr * 0.04)
    hop = int(sr * hop_ms / 1000)
    if len(audio) < win + hop:
        return np.zeros(0)
    frames = np.lib.stride_tricks.sliding_window_view(audio.astype(np.float64), win)[::hop]
    tau_min, tau_max = max(2, int(sr / fmax)), min(win - 2, int(sr / fmin))
    n_fft = 1 << int(np.ceil(np.log2(2 * win)))
    spec = np.fft.rfft(frames, n_fft, axis=1)
    acf = np.fft.irfft(spec * np.conj(spec), n_fft, axis=1)[:, :win]
    energy = np.cumsum(frames**2, axis=1)
    e_head = energy[:, ::-1]  # sum of x[0 : win - tau]^2
    e_tail = energy[:, -1:] - np.concatenate([np.zeros((len(frames), 1)), energy[:, :-1]], axis=1)  # sum of x[tau:]^2
    d = e_head + e_tail - 2 * acf
    d[:, 0] = 0.0
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] * np.arange(1, win) / (np.cumsum(d[:, 1:], axis=1) + 1e-12)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    loud = rms > max(1e-4, 0.05 * float(np.percentile(rms, 95)))
    out = np.zeros(len(frames))
    for i in np.flatnonzero(loud):
        row = cmnd[i]
        below = np.flatnonzero(row[tau_min:tau_max] < 0.15)
        if below.size == 0:
            continue
        t = tau_min + int(below[0])
        while t + 1 < tau_max and row[t + 1] < row[t]:
            t += 1
        a, b, c = row[t - 1], row[t], row[t + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        out[i] = sr / (t + float(np.clip(shift, -1, 1)))
    return out


def measure(audio: np.ndarray, words: int = 0, speech_seconds: float | None = None) -> Prosody:
    f0 = f0_track(audio)
    voiced = f0[f0 > 0]
    if voiced.size < 10:
        return Prosody(0.0, 0.0, float(voiced.size / max(1, f0.size)), 0.0)
    lo, hi = np.percentile(voiced, [5, 95])
    secs = speech_seconds if speech_seconds else len(audio) / SR
    return Prosody(
        f0_median=float(np.median(voiced)),
        f0_range_st=float(12 * np.log2(hi / lo)),
        voiced_ratio=float(voiced.size / f0.size),
        rate_wps=float(words / secs) if words and secs > 0 else 0.0,
    )
