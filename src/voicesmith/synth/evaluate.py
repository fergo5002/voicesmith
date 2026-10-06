"""Score a rendered take, cheapest checks first, and decide if it is fit to ship."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from voicesmith import audio, text, voices
from voicesmith.analysis import asr, prosody, speaker

SR = 16_000
# TitaNet cosine between unrelated speakers sits around 0.05 (measured on
# LibriSpeech); this floor maps "a stranger" to 0 on the normalised scale.
SIM_FLOOR = 0.15
DEFAULT_RATE_WPS = 2.6


@dataclass
class Thresholds:
    max_cer: float = 0.08
    min_similarity: float = 0.40
    max_pause: float = 1.6
    duration_ratio: tuple[float, float] = (0.5, 2.0)


@dataclass
class Score:
    ok: bool
    reason: str
    q: float
    cer: float
    wer: float
    transcript: str
    similarity: float
    sim_norm: float
    duration: float
    expected_duration: float
    longest_pause: float
    f0_ratio: float
    rate_ratio: float

    def as_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def expected_seconds(script: str, voice: voices.Voice) -> float:
    words = len(text.normalise(script).split())
    rate = voice.stats.get("rate_wps") or DEFAULT_RATE_WPS
    return words / max(1.2, float(rate))


def evaluate(wav: np.ndarray, sr: int, script: str, voice: voices.Voice, th: Thresholds | None = None) -> Score:
    th = th or Thresholds()
    x = audio.resample(wav.astype(np.float32), sr, SR)
    dur = len(x) / SR
    exp = expected_seconds(script, voice)
    blank = dict(q=0.0, cer=1.0, wer=1.0, transcript="", similarity=0.0, sim_norm=0.0, duration=dur,
                 expected_duration=exp, longest_pause=0.0, f0_ratio=1.0, rate_ratio=1.0)

    if not np.all(np.isfinite(x)) or dur < 0.3:
        return Score(ok=False, reason="empty or broken audio", **blank)
    if audio.clipping_ratio(x, 0.999) > 0.001:
        return Score(ok=False, reason="clipping", **blank)
    ratio = dur / max(0.5, exp)
    if not th.duration_ratio[0] <= ratio <= th.duration_ratio[1] + 1.5 / max(0.5, exp):
        return Score(ok=False, reason=f"implausible length ({dur:.1f}s for ~{exp:.1f}s of text)", **blank)

    tr = asr.transcribe(x)
    cer = text.cer(script, tr.text)
    wer = text.wer(script, tr.text)
    blank.update(cer=cer, wer=wer, transcript=tr.text)
    gaps = [b.start - a.end for a, b in zip(tr.words, tr.words[1:])]
    longest = max(gaps) if gaps else 0.0
    blank["longest_pause"] = longest

    reason = ""
    ref_words = text.normalise(script).split()
    hyp_words = text.normalise(tr.text).split()
    if ref_words and ref_words[-1] not in hyp_words[-3:] and text.cer(ref_words[-1], " ".join(hyp_words[-2:])) > 0.5:
        reason = "ending cut off or garbled"
    elif rep := text.repeated_ngram(script, tr.text):
        reason = f"repeated phrase: {rep!r}"
    elif cer > th.max_cer:
        reason = f"wording off (character error {cer:.0%})"
    elif longest > th.max_pause:
        reason = f"dead air ({longest:.1f}s pause)"

    emb = speaker.embed(x)
    target = voice.holdout_centroid()
    sim = speaker.cosine(emb, target) if target is not None else 0.0
    ceiling = float(voice.stats.get("self_similarity") or 0.8)
    sim_norm = float(np.clip((sim - SIM_FLOOR) / max(0.1, ceiling - SIM_FLOOR), 0.0, 1.2))
    if not reason and sim < th.min_similarity:
        reason = f"does not sound like the speaker (similarity {sim:.2f})"

    p = prosody.measure(x, words=len(hyp_words), speech_seconds=_speech_seconds(tr))
    f0_ref = float(voice.stats.get("f0_median") or 0)
    rate_ref = float(voice.stats.get("rate_wps") or 0)
    f0_ratio = p.f0_median / f0_ref if f0_ref and p.f0_median else 1.0
    rate_ratio = p.rate_wps / rate_ref if rate_ref and p.rate_wps else 1.0

    s_int = float(np.exp(-cer / 0.03))
    s_pros = float(np.exp(-abs(np.log(f0_ratio)) / 0.15) * np.exp(-abs(np.log(rate_ratio)) / 0.25))
    q = 0.5 * min(1.0, sim_norm) + 0.35 * s_int + 0.15 * s_pros
    blank.update(similarity=sim, sim_norm=sim_norm, f0_ratio=f0_ratio, rate_ratio=rate_ratio, q=q)
    return Score(ok=not reason, reason=reason, **blank)


def _speech_seconds(tr: asr.Transcript) -> float | None:
    if not tr.words:
        return None
    return sum(w.end - w.start for w in tr.words) + sum(
        min(0.25, max(0.0, b.start - a.end)) for a, b in zip(tr.words, tr.words[1:])
    )
