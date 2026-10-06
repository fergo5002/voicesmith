"""Halo Research Sopro V2 Turbo (Apache-2.0), a 120M model built for CPU."""

from __future__ import annotations

import os

import numpy as np


class Adapter:
    def __init__(self, engine: str, device: str):
        from sopro.model import SoproTTS

        # int8 measured 2.4x faster than fp32 on CPU with the same similarity and WER
        # (docs/benchmarks.md); it is CPU-only in Sopro.
        quant = "int8" if device == "cpu" else None
        self.model = SoproTTS.from_pretrained(device=device, quantization=quant)
        self.sample_rate = int(self.model.sample_rate)
        self._ref_key = None
        self._ref = None

    def synthesize(self, text: str, ref_wav: str, ref_text: str | None, options: dict) -> np.ndarray:
        key = (ref_wav, os.path.getmtime(ref_wav))
        if key != self._ref_key:
            self._ref = self.model.prepare_reference(ref_audio_path=ref_wav)
            self._ref_key = key
        wav = self.model.synthesize(
            text,
            ref=self._ref,
            lang=options.get("language"),
            temperature=options.get("temperature"),
            top_p=options.get("top_p"),
        )
        return wav.squeeze().detach().cpu().numpy()
