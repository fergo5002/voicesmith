"""Best-of-N orchestration with a fake engine and scripted scores."""

import numpy as np
import pytest

from voicesmith import audio, consent, voices
from voicesmith.synth import evaluate, master, render


class FakeWorker:
    def __init__(self):
        self.calls = 0

    def call(self, method, params, timeout=0):
        assert method == "synthesize"
        self.calls += 1
        t = np.arange(24_000) / 24_000
        audio.save_wav(params["out_wav"], (0.2 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), 24_000, "FLOAT")
        return {"seconds": 2.0, "duration": 1.0, "sample_rate": 24_000}


def _score(ok, q, reason=""):
    return evaluate.Score(ok=ok, reason=reason, q=q, cer=0.0, wer=0.0, transcript="x", similarity=0.7, sim_norm=0.8,
                          duration=1.0, expected_duration=1.0, longest_pause=0.0, f0_ratio=1.0, rate_ratio=1.0)


@pytest.fixture
def setup(monkeypatch, tmp_path):
    v = voices.create("fay", "Fay Example")
    refp = v.dir / "refs" / "ref01.wav"
    audio.save_wav(refp, np.zeros(24_000, np.float32), 24_000)
    v.references = [voices.Reference("ref01", "refs/ref01.wav", "hello there", 1.0, "s", 0.0, 1.0, 0.8, 30, 120, 2.5)]
    v.save()
    consent.attest(v, by="Op", evidence="test")
    worker = FakeWorker()
    monkeypatch.setattr(render, "choose_engine", lambda voice, requested=None: "sopro")
    monkeypatch.setattr(render.client, "engine_worker", lambda eng, *a, **k: worker)
    delivered = []

    def fake_master(wav, sr, outputs, tags):
        delivered.append(outputs)
        return master.MasterResult(files=[])

    monkeypatch.setattr(render.master, "master", fake_master)
    monkeypatch.setattr(render.provenance, "write_manifest", lambda p, d: p)

    def script(scores):
        it = iter(scores)
        monkeypatch.setattr(render.evaluate, "evaluate", lambda *a, **k: next(it))

    return worker, script, delivered, tmp_path


def test_balanced_stops_at_the_first_good_enough_take(setup):
    worker, script, delivered, tmp = setup
    script([_score(True, 0.85), _score(True, 0.99)])
    r = render.render("fay", "Hello world.", outputs=[tmp / "o.wav"], quality="balanced")
    assert worker.calls == 1 and r.takes == 1 and delivered


def test_rejected_takes_are_retried(setup):
    worker, script, delivered, tmp = setup
    script([_score(False, 0.5, "ending cut off"), _score(False, 0.6, "dead air"), _score(True, 0.9)])
    r = render.render("fay", "Hello world.", outputs=[tmp / "o.wav"], quality="balanced")
    assert worker.calls == 3 and r.score["ok"] is True


def test_best_keeps_the_highest_scoring_take(setup):
    worker, script, delivered, tmp = setup
    script([_score(True, 0.80), _score(True, 0.91), _score(True, 0.85)] + [_score(True, 0.5)] * 10)
    r = render.render("fay", "Hello world.", outputs=[tmp / "o.wav"], quality="best")
    assert r.score["q"] == pytest.approx(0.91)


def test_fails_closed_when_no_take_passes(setup):
    worker, script, delivered, tmp = setup
    script([_score(False, 0.4, "wording off")] * 20)
    with pytest.raises(render.RenderError, match="never passed"):
        render.render("fay", "Hello world.", outputs=[tmp / "o.wav"], quality="fast")
    assert not delivered


def test_render_refuses_without_consent(setup):
    worker, script, delivered, tmp = setup
    consent.revoke(voices.load("fay"))
    with pytest.raises(PermissionError):
        render.render("fay", "Hello world.", outputs=[tmp / "o.wav"])
    assert worker.calls == 0
