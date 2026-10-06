"""Transcription with word timings (Parakeet TDT 0.6B v3 via sherpa-onnx)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voicesmith.analysis import runtime

SR = 16_000
_PUNCT = set(",.?!;:…\"')")


@dataclass
class Word:
    text: str
    start: float
    end: float
    logprob: float  # mean token log-probability, a cheap confidence signal


@dataclass
class Transcript:
    text: str
    words: list[Word]

    @property
    def confidence(self) -> float:
        if not self.words:
            return 0.0
        return float(np.exp(np.mean([w.logprob for w in self.words])))


def transcribe(audio: np.ndarray, offset: float = 0.0) -> Transcript:
    """Transcribe 16 kHz mono audio. Long inputs are fine up to a few minutes."""
    rec = runtime.recognizer()
    stream = rec.create_stream()
    stream.accept_waveform(SR, audio.astype(np.float32))
    rec.decode_stream(stream)
    res = stream.result
    return Transcript(text=res.text.strip(), words=_words(res, offset))


def transcribe_many(chunks: list[np.ndarray]) -> list[Transcript]:
    rec = runtime.recognizer()
    streams = []
    for chunk in chunks:
        s = rec.create_stream()
        s.accept_waveform(SR, chunk.astype(np.float32))
        streams.append(s)
    if streams:
        rec.decode_streams(streams)
    return [Transcript(text=s.result.text.strip(), words=_words(s.result, 0.0)) for s in streams]


def _words(res, offset: float) -> list[Word]:
    words: list[Word] = []
    tokens = list(res.tokens)
    stamps = list(res.timestamps)
    durs = list(res.durations) if len(res.durations) == len(tokens) else [0.08] * len(tokens)
    logps = list(res.ys_log_probs) if len(res.ys_log_probs) == len(tokens) else [0.0] * len(tokens)
    cur: dict | None = None
    for tok, t0, d, lp in zip(tokens, stamps, durs, logps):
        starts_word = tok.startswith(" ") or cur is None
        piece = tok.strip()
        if not piece:
            continue
        if starts_word and not (piece and all(c in _PUNCT for c in piece)):
            if cur:
                words.append(_finish(cur))
            cur = {"text": piece, "start": t0 + offset, "end": t0 + d + offset, "lps": [lp]}
        else:
            assert cur is not None
            cur["text"] += piece
            cur["end"] = t0 + d + offset
            cur["lps"].append(lp)
    if cur:
        words.append(_finish(cur))
    return words


def _finish(cur: dict) -> Word:
    return Word(text=cur["text"], start=float(cur["start"]), end=float(cur["end"]), logprob=float(np.mean(cur["lps"])))
