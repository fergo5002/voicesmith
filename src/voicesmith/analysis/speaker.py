"""Speaker embeddings and similarity (TitaNet-small via sherpa-onnx).

On 48 LibriSpeech utterances from 8 speakers, TitaNet-small separated same and
different speakers cleanly (mean cosine 0.81 vs 0.05) where two other encoders
we tried did not, so it is the one encoder the core relies on.
"""

from __future__ import annotations

import numpy as np

from voicesmith.analysis import runtime

SR = 16_000
MIN_SECONDS = 1.0


def embed(audio: np.ndarray) -> np.ndarray:
    if len(audio) < SR * MIN_SECONDS:
        audio = np.pad(audio, (0, int(SR * MIN_SECONDS) - len(audio)))
    ext = runtime.speaker_extractor()
    stream = ext.create_stream()
    stream.accept_waveform(SR, audio.astype(np.float32))
    stream.input_finished()
    vec = np.asarray(ext.compute(stream), dtype=np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


def centroid(vectors: list[np.ndarray] | np.ndarray) -> np.ndarray:
    mat = np.asarray(vectors, dtype=np.float32)
    c = mat.mean(axis=0)
    return c / (np.linalg.norm(c) + 1e-9)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def self_similarity(vectors: np.ndarray) -> float:
    """Median leave-one-out similarity of a speaker's own clips to their centroid.

    This is the ceiling a clone can realistically reach: a perfect copy should
    sound as much like the person as their own recordings do.
    """
    mat = np.asarray(vectors, dtype=np.float32)
    if len(mat) < 2:
        return 1.0
    total = mat.sum(axis=0)
    sims = []
    for v in mat:
        rest = total - v
        rest /= np.linalg.norm(rest) + 1e-9
        sims.append(float(np.dot(v, rest)))
    return float(np.median(sims))
