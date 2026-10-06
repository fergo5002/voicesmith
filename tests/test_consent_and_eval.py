import time

import numpy as np
import pytest

from voicesmith import audio, consent, text, voices
from voicesmith.analysis import asr
from voicesmith.synth import evaluate

from .conftest import tone


class FakeASR:
    def __init__(self, said: str, words=None):
        self.said = said
        self.words = words

    def __call__(self, audio_, offset=0.0):
        ws = self.words or [asr.Word(w, i * 0.4, i * 0.4 + 0.35, -0.05) for i, w in enumerate(self.said.split())]
        return asr.Transcript(text=self.said, words=ws)


@pytest.fixture
def voice():
    return voices.create("alice", "Alice Example")


@pytest.fixture
def recording(tmp_path):
    p = tmp_path / "consent.wav"
    audio.save_wav(p, tone(180, 8.0), 16_000)
    return p


@pytest.fixture
def fake_embed(monkeypatch):
    vec = np.ones(192, dtype=np.float32) / np.sqrt(192)
    monkeypatch.setattr("voicesmith.analysis.speaker.embed", lambda a: vec)
    return vec


def test_render_is_blocked_without_consent(voice):
    with pytest.raises(PermissionError, match="no usable consent"):
        consent.require(voice)


def test_spoken_consent_accepts_the_statement(voice, recording, monkeypatch, fake_embed):
    c = consent.request(voice)
    monkeypatch.setattr("voicesmith.analysis.asr.transcribe", FakeASR(c.statement))
    c = consent.verify_recording(voice, recording)
    assert c.status == "verified" and c.asr_cer == 0
    assert consent.require(voice).kind == "spoken"
    assert (voice.dir / "consent" / "anchor.npy").exists()


def test_spoken_consent_rejects_the_wrong_code(voice, recording, monkeypatch, fake_embed):
    c = consent.request(voice)
    wrong = c.statement.replace(c.code, "purple monkey dishwasher 12")
    monkeypatch.setattr("voicesmith.analysis.asr.transcribe", FakeASR(wrong))
    with pytest.raises(ValueError, match="code"):
        consent.verify_recording(voice, recording)
    assert consent.load(voice).status == "pending"


def test_spoken_consent_rejects_an_expired_code(voice, recording, monkeypatch, fake_embed):
    c = consent.request(voice)
    c.expires = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 60))
    consent.save(voice, c)
    monkeypatch.setattr("voicesmith.analysis.asr.transcribe", FakeASR(c.statement))
    with pytest.raises(ValueError, match="expired"):
        consent.verify_recording(voice, recording)


def test_consent_from_a_different_speaker_is_refused(voice, recording, monkeypatch):
    c = consent.request(voice)
    np.save(voice.dir / "centroid.npy", np.eye(192, dtype=np.float32)[0])
    monkeypatch.setattr("voicesmith.analysis.speaker.embed", lambda a: np.eye(192, dtype=np.float32)[1])
    monkeypatch.setattr("voicesmith.analysis.asr.transcribe", FakeASR(c.statement))
    with pytest.raises(ValueError, match="does not sound like"):
        consent.verify_recording(voice, recording)
    with pytest.raises(PermissionError):
        consent.require(voice)


def test_attestation_needs_evidence_and_revocation_blocks(voice):
    with pytest.raises(ValueError):
        consent.attest(voice, by="Op", evidence=" ")
    consent.attest(voice, by="Op Erator", evidence="Email from Alice, 2026-10-01")
    assert consent.require(voice).kind == "attested"
    consent.revoke(voice)
    with pytest.raises(PermissionError):
        consent.require(voice)


def test_consent_code_is_fresh_and_readable():
    codes = {consent.new_code() for _ in range(50)}
    assert len(codes) == 50
    assert all(len(text.normalise(c).split()) >= 4 for c in codes)


# ------------------------------------------------------------------ evaluator

SCRIPT = "The quick brown fox jumps over the lazy dog near the river bank"


def _eval(monkeypatch, said, sim_vec, seconds=4.6, words=None):
    v = voices.create("bob", "Bob Example")
    target = np.eye(192, dtype=np.float32)[0]
    np.save(v.dir / "holdout.npy", target)
    v.stats.update({"self_similarity": 0.8, "rate_wps": 2.8, "f0_median": 150.0})
    monkeypatch.setattr("voicesmith.analysis.asr.transcribe", FakeASR(said, words))
    monkeypatch.setattr("voicesmith.analysis.speaker.embed", lambda a: sim_vec)
    return evaluate.evaluate(tone(150, seconds), 16_000, SCRIPT, v)


def _vec(cos):
    v = np.zeros(192, dtype=np.float32)
    v[0], v[1] = cos, np.sqrt(1 - cos**2)
    return v


def test_eval_passes_a_good_take(monkeypatch):
    s = _eval(monkeypatch, SCRIPT, _vec(0.75))
    assert s.ok, s.reason
    assert s.q > 0.8


def test_eval_catches_a_truncated_ending(monkeypatch):
    s = _eval(monkeypatch, "The quick brown fox jumps over the lazy dog near the", _vec(0.75))
    assert not s.ok and "ending" in s.reason


def test_eval_catches_a_loop(monkeypatch):
    s = _eval(monkeypatch, SCRIPT + " near the river bank near the river bank", _vec(0.75), seconds=6.0)
    assert not s.ok and "repeated" in s.reason


def test_eval_catches_the_wrong_voice(monkeypatch):
    s = _eval(monkeypatch, SCRIPT, _vec(0.2))
    assert not s.ok and "sound like" in s.reason


def test_eval_catches_dead_air(monkeypatch):
    ws = [asr.Word(w, i * 0.3, i * 0.3 + 0.25, -0.05) for i, w in enumerate(SCRIPT.split())]
    for w in ws[6:]:
        w.start += 2.5
        w.end += 2.5
    s = _eval(monkeypatch, SCRIPT, _vec(0.75), seconds=7.0, words=ws)
    assert not s.ok and "dead air" in s.reason


def test_eval_rejects_implausible_length_before_running_asr(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("ASR should not run")
    v = voices.create("carl", "Carl")
    v.stats.update({"rate_wps": 2.8})
    monkeypatch.setattr("voicesmith.analysis.asr.transcribe", boom)
    s = evaluate.evaluate(tone(150, 30.0), 16_000, SCRIPT, v)
    assert not s.ok and "length" in s.reason
