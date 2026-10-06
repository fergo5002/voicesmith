"""Voice activity detection (Silero VAD via sherpa-onnx)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voicesmith.analysis import runtime

SR = 16_000
WINDOW = 512


@dataclass
class Region:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def speech_regions(
    audio: np.ndarray,
    *,
    threshold: float = 0.5,
    min_silence: float = 0.3,
    min_speech: float = 0.25,
    max_speech: float = 30.0,
) -> list[Region]:
    import sherpa_onnx

    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = runtime.vad_model_path()
    cfg.silero_vad.threshold = threshold
    cfg.silero_vad.min_silence_duration = min_silence
    cfg.silero_vad.min_speech_duration = min_speech
    cfg.silero_vad.max_speech_duration = max_speech
    cfg.silero_vad.window_size = WINDOW
    cfg.sample_rate = SR
    cfg.num_threads = 1
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=max(60, int(max_speech * 2)))
    regions: list[Region] = []

    def drain() -> None:
        while not vad.empty():
            seg = vad.front
            regions.append(Region(seg.start / SR, (seg.start + len(seg.samples)) / SR))
            vad.pop()

    audio = audio.astype(np.float32)
    for i in range(0, len(audio), WINDOW * 64):
        vad.accept_waveform(audio[i : i + WINDOW * 64])
        drain()
    vad.flush()
    drain()
    return regions
