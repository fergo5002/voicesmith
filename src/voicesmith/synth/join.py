"""Join separately rendered chunks so the seams do not show.

The private predecessor joined paragraphs by adding fixed digital silence on
top of whatever silence the model had already left, giving 0.6 to 0.7 second
holes and audible resets. Here the pause is the total gap: model edge silence
is trimmed first, the gap is sized by punctuation, and it is filled with the
take's own noise floor rather than digital zero.
"""

from __future__ import annotations

import numpy as np

from voicesmith import audio

PAUSE = {".": 0.42, "!": 0.42, "?": 0.45, ";": 0.32, ":": 0.3, ",": 0.22}
DEFAULT_PAUSE = 0.3


def pause_after(chunk_text: str) -> float:
    t = chunk_text.rstrip().rstrip("\"')]")
    return PAUSE.get(t[-1:], DEFAULT_PAUSE) if t else DEFAULT_PAUSE


def _floor_rms(x: np.ndarray, sr: int) -> float:
    levels = audio.frame_rms_db(x, sr, 20.0)
    return float(10 ** (np.percentile(levels, 5) / 20)) if levels.size else 1e-4


def _room_tone(n: int, rms: float, rng: np.random.Generator) -> np.ndarray:
    # Gently low-passed noise at the take's own floor reads as room, not hiss.
    noise = rng.standard_normal(n + 64)
    kernel = np.hanning(9)
    kernel /= kernel.sum()
    tone = np.convolve(noise, kernel, mode="same")[32 : 32 + n]
    tone *= min(rms, 10 ** (-55 / 20)) / (np.sqrt(np.mean(tone**2)) + 1e-12)
    return tone.astype(np.float32)


def join(chunks: list[np.ndarray], texts: list[str], sr: int, seed: int = 0) -> np.ndarray:
    if len(chunks) == 1:
        return chunks[0]
    rng = np.random.default_rng(seed)
    trimmed = []
    for c in chunks:
        s, e = audio.active_bounds(c, sr)
        guard = int(0.025 * sr)
        trimmed.append(audio.fade(c[max(0, s - guard) : min(len(c), e + guard)], sr, 8, 12))
    floor = float(np.median([_floor_rms(c, sr) for c in chunks]))
    out = [trimmed[0]]
    for prev_text, nxt in zip(texts[:-1], trimmed[1:]):
        gap = int(pause_after(prev_text) * sr)
        out.append(_room_tone(gap, floor, rng))
        out.append(nxt)
    return np.concatenate(out).astype(np.float32)
