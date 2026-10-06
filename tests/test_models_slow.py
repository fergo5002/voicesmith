"""Tests that need the real ONNX models (downloaded on first run, about 530 MB)."""

from pathlib import Path

import pytest

from voicesmith import audio, text
from voicesmith.analysis import asr, speaker, vad

pytestmark = pytest.mark.slow
PAIR = Path(__file__).parent / "fixtures" / "speech-pair.flac"


def test_vad_splits_at_the_real_pause():
    regions = vad.speech_regions(audio.load(PAIR))
    assert len(regions) == 2
    assert 9.8 < regions[0].end < 10.6 and 11.3 < regions[1].start < 12.1


def test_asr_transcribes_with_word_timings():
    tr = asr.transcribe(audio.load(PAIR))
    assert "dinner" in text.normalise(tr.text)
    assert tr.words and all(b.start >= a.start for a, b in zip(tr.words, tr.words[1:]))
    assert tr.confidence > 0.5


def test_same_speaker_scores_high():
    x = audio.load(PAIR)
    a, b = speaker.embed(x[: 16000 * 10]), speaker.embed(x[16000 * 12 :])
    assert speaker.cosine(a, b) > 0.6
