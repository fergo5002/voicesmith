"""Engine-agnostic watermarking with Meta's AudioSeal (MIT code and weights).

Every file voicesmith produces carries this mark, whatever engine made it.
Engines with their own mark (Chatterbox's PerTh) keep theirs as well.
The 16-bit payload spells "VS" so a detector can tell our mark apart.
"""

from __future__ import annotations

import functools

import numpy as np
import soundfile as sf

PAYLOAD = [int(b) for b in format(0x5653, "016b")]  # "VS"
WM_SR = 16_000


@functools.lru_cache(maxsize=1)
def _models():
    import torch
    from audioseal import AudioSeal

    gen = AudioSeal.load_generator("audioseal_wm_16bits").eval()
    det = AudioSeal.load_detector("audioseal_detector_16bits").eval()
    torch.set_grad_enabled(False)
    return gen, det


def _resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    import torch
    import torchaudio.functional as F

    if sr_from == sr_to:
        return x
    return F.resample(torch.from_numpy(x).float(), sr_from, sr_to).numpy()


def embed(in_wav: str, out_wav: str) -> dict:
    """Add the mark. AudioSeal works at 16 kHz, so the watermark signal is made
    at 16 kHz and upsampled back onto the original, leaving the rest untouched."""
    import torch

    gen, det = _models()
    audio, sr = sf.read(in_wav, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    x16 = _resample(audio, sr, WM_SR)
    msg = torch.tensor([PAYLOAD], dtype=torch.int32)
    wm16 = gen.get_watermark(torch.from_numpy(x16)[None, None, :], sample_rate=WM_SR, message=msg)[0, 0].numpy()
    wm = _resample(wm16, WM_SR, sr)[: len(audio)]
    if len(wm) < len(audio):
        wm = np.pad(wm, (0, len(audio) - len(wm)))
    out = np.clip(audio + wm, -1.0, 1.0).astype(np.float32)
    sf.write(out_wav, out, sr, subtype="FLOAT")
    return {"embedded": True, **detect(out_wav)}


def detect(wav: str) -> dict:
    import torch

    _, det = _models()
    audio, sr = sf.read(wav, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    x16 = _resample(audio, sr, WM_SR)
    prob, message = det.detect_watermark(torch.from_numpy(x16)[None, None, :], sample_rate=WM_SR)
    bits = [int(b) for b in message[0].tolist()]
    result = {"audioseal_prob": float(prob), "audioseal_payload_ok": bits == PAYLOAD}
    try:
        import perth

        result["perth_conf"] = float(perth.PerthImplicitWatermarker().get_watermark(audio, sample_rate=sr))
    except Exception:  # PerTh is optional: only engines that ship it have it installed
        pass
    return result
