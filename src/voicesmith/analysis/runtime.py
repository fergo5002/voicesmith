"""Lazily constructed, process-wide model handles.

Loading Parakeet takes a few seconds, so each model is built once per process
and reused. Thread count defaults to the physical core count, capped at 8,
which measured as the sweet spot for these small ONNX graphs.
"""

from __future__ import annotations

import functools
import os

import psutil

from voicesmith import models


def threads() -> int:
    env = os.environ.get("VOICESMITH_ONNX_THREADS")
    if env:
        return max(1, int(env))
    cores = psutil.cpu_count(logical=False) or os.cpu_count() or 4
    return max(1, min(8, cores))


@functools.lru_cache(maxsize=1)
def recognizer():
    import sherpa_onnx

    root = models.ensure("asr")
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(root / "encoder.int8.onnx"),
        decoder=str(root / "decoder.int8.onnx"),
        joiner=str(root / "joiner.int8.onnx"),
        tokens=str(root / "tokens.txt"),
        model_type="nemo_transducer",
        num_threads=threads(),
    )


@functools.lru_cache(maxsize=1)
def speaker_extractor():
    import sherpa_onnx

    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(models.ensure("speaker")), num_threads=threads())
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


@functools.lru_cache(maxsize=1)
def tagger():
    import sherpa_onnx

    root = models.ensure("tagger")
    cfg = sherpa_onnx.AudioTaggingConfig(
        model=sherpa_onnx.AudioTaggingModelConfig(ced=str(root / "model.int8.onnx"), num_threads=threads()),
        labels=str(root / "class_labels_indices.csv"),
        top_k=5,
    )
    return sherpa_onnx.AudioTagging(cfg)


def vad_model_path() -> str:
    return str(models.ensure("vad"))
