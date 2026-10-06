"""Regression tests for the issues found in the pre-release code review."""

import asyncio
import sys
import time

import numpy as np
import pytest
from fastmcp import Client

from voicesmith import audio, consent, mcp_server, paths, voices
from voicesmith.analysis import asr
from voicesmith.engines import client
from voicesmith.ingest import pipeline
from voicesmith.synth import master, render

from .test_render_logic import FakeWorker, _score


def _ready_voice(name="gail"):
    v = voices.create(name, "Gail Example")
    audio.save_wav(v.dir / "refs" / "ref01.wav", np.zeros(24_000, np.float32), 24_000)
    v.references = [voices.Reference("ref01", "refs/ref01.wav", "hello", 1.0, "s", 0.0, 1.0, 0.8, 30, 120, 2.5)]
    v.save()
    consent.attest(v, by="Op", evidence="test")
    return v


@pytest.fixture
def fake_engine(monkeypatch):
    worker = FakeWorker()
    monkeypatch.setattr(render, "choose_engine", lambda voice, requested=None, quality="balanced": "sopro")
    monkeypatch.setattr(render.client, "engine_worker", lambda eng, *a, **k: worker)
    monkeypatch.setattr(render.evaluate, "evaluate", lambda *a, **k: _score(True, 0.95))
    return worker


# ---- HIGH 1: MCP output paths cannot touch consent or other files

def test_mcp_speak_refuses_paths_outside_the_outputs_folder(fake_engine):
    _ready_voice("gail")
    victim = _ready_voice("hana")
    target = victim.dir / "consent" / "consent.wav"

    async def go():
        async with Client(mcp_server.server) as c:
            return await c.call_tool("speak", {"voice": "gail", "text": "hi", "output_path": str(target)},
                                     raise_on_error=False)

    res = asyncio.run(go())
    assert res.is_error and "outputs" in str(res.content)
    assert consent.load(voices.load("hana")).status == "attested"


def test_manifest_never_replaces_an_existing_json(fake_engine, monkeypatch, tmp_path):
    _ready_voice("gail")
    settings = tmp_path / "settings.json"
    settings.write_text('{"keep": true}')
    monkeypatch.setattr(render.master, "master", lambda wav, sr, outs, tags, before_deliver=None: master.MasterResult())
    r = render.render("gail", "Hello there.", outputs=[tmp_path / "settings.wav"])
    assert settings.read_text() == '{"keep": true}'
    assert r.manifest.name == "settings.voicesmith.json"


def test_render_refuses_to_write_inside_the_voicesmith_voices_folder(fake_engine):
    v = _ready_voice("gail")
    with pytest.raises(render.RenderError, match="voices folder"):
        render.render("gail", "Hello there.", outputs=[v.dir / "refs" / "ref01.wav"])


# ---- HIGH 2: revocation during a render stops delivery

def test_revocation_mid_render_blocks_delivery(monkeypatch, tmp_path):
    _ready_voice("gail")

    class RevokingWorker(FakeWorker):
        def call(self, method, params, timeout=0):
            consent.revoke(voices.load("gail"))
            return super().call(method, params, timeout)

    w = RevokingWorker()
    monkeypatch.setattr(render, "choose_engine", lambda voice, requested=None, quality="balanced": "sopro")
    monkeypatch.setattr(render.client, "engine_worker", lambda eng, *a, **k: w)
    monkeypatch.setattr(render.evaluate, "evaluate", lambda *a, **k: _score(True, 0.95))
    delivered = []

    def fake_master(wav, sr, outs, tags, before_deliver=None):
        if before_deliver:
            before_deliver()
        delivered.append(outs)
        return master.MasterResult()

    monkeypatch.setattr(render.master, "master", fake_master)
    with pytest.raises(PermissionError):
        render.render("gail", "Hello there.", outputs=[tmp_path / "o.wav"])
    assert not delivered


# ---- MEDIUM 3: attestation cannot silently undo a revocation

def test_attest_over_a_revocation_needs_an_override_and_history_is_kept():
    v = voices.create("ivy", "Ivy Example")
    consent.attest(v, by="Op", evidence="email")
    consent.revoke(v)
    with pytest.raises(ValueError, match="revoked"):
        consent.attest(v, by="Op", evidence="email again")
    consent.attest(v, by="Op", evidence="new signed letter", override_revocation=True)
    history = (v.dir / "consent" / "history.jsonl").read_text().splitlines()
    assert [line.count('"revoked"') > 0 for line in history].count(True) >= 1
    assert len(history) == 3


# ---- MEDIUM 4: very short sources do not crash ingest

def test_one_or_two_regions_do_not_crash_target_finding():
    rng = np.random.default_rng(0)
    c = rng.standard_normal(192)
    c /= np.linalg.norm(c)
    one = [pipeline.Clip("a", 0, 8, "", 1.0, emb=c)]
    assert float(pipeline.find_target(one, None, lambda m: None) @ c) > 0.99
    other = rng.standard_normal(192)
    other /= np.linalg.norm(other)
    two_people = [pipeline.Clip("a", 0, 8, "", 1.0, emb=c), pipeline.Clip("a", 10, 18, "", 1.0, emb=other)]
    with pytest.raises(pipeline.AmbiguousSpeaker):
        pipeline.find_target(two_people, None, lambda m: None)


# ---- MEDIUM 5: a worker that fails its handshake is not left running

def test_failed_hello_kills_the_worker(monkeypatch, tmp_path):
    script = tmp_path / "bad_worker.py"
    script.write_text(
        "import json, sys\n"
        "print(json.dumps({'event': 'ready'}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    req = json.loads(line)\n"
        "    print(json.dumps({'id': req['id'], 'error': 'torch is broken'}), flush=True)\n"
    )
    monkeypatch.setattr(client.manager, "installed", lambda fam: True)
    monkeypatch.setattr(client.manager, "python_path", lambda fam: sys.executable)
    monkeypatch.setattr(client.manager, "worker_script", lambda: script)
    spawned = []
    real_init = client.Worker.__init__

    def tracking_init(self, *a, **k):
        real_init(self, *a, **k)
        spawned.append(self)

    monkeypatch.setattr(client.Worker, "__init__", tracking_init)
    with pytest.raises(client.WorkerError):
        client.worker("brokenfam")
    time.sleep(0.5)
    assert spawned and spawned[0].proc.poll() is not None


# ---- MEDIUM 6: nothing unverified is left behind when a check fails

def test_master_leaves_no_files_when_verification_fails(monkeypatch, tmp_path):
    class W:
        def call(self, method, params, timeout=0):
            if method == "watermark":
                import shutil
                shutil.copyfile(params["in_wav"], params["out_wav"])
                return {"audioseal_prob": 1.0, "audioseal_payload_ok": True}
            raise client.WorkerError("detect timed out")

    monkeypatch.setattr(master.client, "any_watermark_worker", lambda: W())
    t = np.arange(48_000) / 24_000
    wav = (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    outs = [tmp_path / "a.mp3", tmp_path / "b.wav"]
    with pytest.raises(client.WorkerError):
        master.master(wav, 24_000, outs, {"comment": "x"})
    assert sorted(p.name for p in tmp_path.iterdir()) == []


# ---- LOW 8: a leading punctuation token does not crash word grouping

def test_leading_punctuation_token_is_skipped():
    class R:
        tokens = [".", " hello", " world"]
        timestamps = [0.0, 0.1, 0.5]
        durations = [0.05, 0.3, 0.3]
        ys_log_probs = [0.0, -0.1, -0.1]

    words = asr._words(R(), 0.0)
    assert [w.text for w in words] == ["hello", "world"]


# ---- LOW 10: list_voices "ready" matches what render would allow

def test_ready_flag_respects_a_failed_speaker_match():
    v = _ready_voice("jo")
    c = consent.load(v)
    c.kind, c.speaker_match = "spoken", 0.1
    consent.save(v, c)
    rows = {r["name"]: r for r in mcp_server.list_voices()}
    assert rows["jo"]["ready"] is False


def test_outputs_dir_is_under_home():
    assert paths.sub("outputs").is_relative_to(paths.home())
