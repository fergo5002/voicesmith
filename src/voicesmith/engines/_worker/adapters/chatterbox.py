"""Resemble AI Chatterbox family (MIT weights, PerTh watermark built in)."""

from __future__ import annotations

import os

import numpy as np


class Adapter:
    def __init__(self, engine: str, device: str):
        self.engine = engine
        self.device = device
        self._cond_key = None
        if engine in ("chatterbox-turbo", "chatterbox-nano"):
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            self.model = ChatterboxTurboTTS.from_pretrained(device=device, nano=engine == "chatterbox-nano")
            self.kind = "turbo"
        elif engine == "chatterbox-multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            self.model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
            self.kind = "mtl"
        else:
            raise ValueError(f"unknown chatterbox engine {engine!r}")
        self.sample_rate = int(self.model.sr)

    def _conditionals(self, ref_wav: str, exaggeration: float) -> None:
        key = (ref_wav, os.path.getmtime(ref_wav), exaggeration)
        if key != self._cond_key:
            if self.kind == "turbo":
                self.model.prepare_conditionals(ref_wav)
            else:
                self.model.prepare_conditionals(ref_wav, exaggeration=exaggeration)
            self._cond_key = key

    def synthesize(self, text: str, ref_wav: str, ref_text: str | None, options: dict) -> np.ndarray:
        exaggeration = float(options.get("exaggeration", 0.5))
        self._conditionals(ref_wav, exaggeration)
        if self.kind == "turbo":
            wav = self.model.generate(
                text,
                temperature=float(options.get("temperature", 0.8)),
                top_p=float(options.get("top_p", 0.95)),
                repetition_penalty=float(options.get("repetition_penalty", 1.2)),
            )
        else:
            wav = self.model.generate(
                text,
                language_id=options.get("language", "en"),
                exaggeration=exaggeration,
                cfg_weight=float(options.get("cfg_weight", 0.5)),
                temperature=float(options.get("temperature", 0.8)),
            )
        return wav.squeeze().detach().cpu().numpy()
