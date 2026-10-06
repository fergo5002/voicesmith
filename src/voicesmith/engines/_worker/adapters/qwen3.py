"""Alibaba Qwen3-TTS Base models (Apache-2.0).

In-context cloning needs the exact transcript of the reference clip, which
voicesmith always has. Without one it falls back to speaker-embedding mode.
"""

from __future__ import annotations

import os

import numpy as np

REPOS = {
    "qwen3-tts-0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "qwen3-tts-1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}
LANGS = {"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "de": "German", "fr": "French",
         "ru": "Russian", "pt": "Portuguese", "es": "Spanish", "it": "Italian"}
CODEC_HZ = 12.5


class Adapter:
    def __init__(self, engine: str, device: str):
        import torch
        from qwen_tts import Qwen3TTSModel

        if engine not in REPOS:
            raise ValueError(f"unknown qwen engine {engine!r}")
        dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
        device_map = "cuda:0" if device.startswith("cuda") else device
        self.model = Qwen3TTSModel.from_pretrained(REPOS[engine], device_map=device_map, dtype=dtype)
        self._prompt_key = None
        self._prompt = None
        self.sample_rate = 24_000

    def _prompt_for(self, ref_wav: str, ref_text: str | None):
        key = (ref_wav, os.path.getmtime(ref_wav), ref_text)
        if key != self._prompt_key:
            self._prompt = self.model.create_voice_clone_prompt(
                ref_audio=ref_wav, ref_text=ref_text, x_vector_only_mode=not ref_text
            )
            self._prompt_key = key
        return self._prompt

    def synthesize(self, text: str, ref_wav: str, ref_text: str | None, options: dict) -> np.ndarray:
        prompt = self._prompt_for(ref_wav, ref_text)
        words = max(1, len(text.split()))
        # Cap generation at about 2.5x a slow speaking rate so a sampling loop
        # cannot run for minutes, which happened in the private predecessor.
        max_tokens = int(CODEC_HZ * (words / 1.6) * 2.5) + 40
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=LANGS.get(options.get("language", "en"), "English"),
            voice_clone_prompt=prompt,
            max_new_tokens=int(options.get("max_new_tokens", max_tokens)),
            temperature=float(options.get("temperature", 0.9)),
            top_p=float(options.get("top_p", 1.0)),
            top_k=int(options.get("top_k", 50)),
        )
        self.sample_rate = int(sr)
        return np.asarray(wavs[0], dtype=np.float32)
