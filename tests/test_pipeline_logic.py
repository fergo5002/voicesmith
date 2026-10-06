import numpy as np

from voicesmith.analysis.asr import Word
from voicesmith.ingest import pipeline
from voicesmith.synth import join

from .conftest import tone


def words_from(spec):
    """spec: list of (text, start, end)."""
    return [Word(t, s, e, -0.05) for t, s, e in spec]


def test_candidate_clips_cut_only_in_pauses_and_respect_limits():
    spec = []
    t = 0.0
    for i in range(40):
        word = f"word{i}" + ("." if i % 8 == 7 else "")
        spec.append((word, t, t + 0.35))
        t += 0.35 + (0.4 if i % 8 == 7 else 0.05)  # long pause after each sentence
    clips = pipeline.candidate_clips(words_from(spec), "src.wav")
    assert clips
    for c in clips:
        assert pipeline.MIN_CLIP <= c.duration <= pipeline.MAX_CLIP
        assert c.text.endswith(".")  # sentence ends are preferred when they fit


def test_candidate_clips_skip_unbroken_monologue():
    spec = [(f"w{i}", i * 0.3, i * 0.3 + 0.29) for i in range(100)]  # no pause ever reaches MIN_GAP
    assert pipeline.candidate_clips(words_from(spec), "x") == []


def test_select_prefers_central_clean_clips_and_spreads_them_out():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(192)
    base /= np.linalg.norm(base)
    clips = []
    for i in range(12):
        e = base + 0.3 * rng.standard_normal(192)
        e /= np.linalg.norm(e)
        c = pipeline.Clip(source="a.wav", start=i * 70.0, end=i * 70.0 + 9.0, text="x", confidence=0.9, emb=e)
        c.sim = float(e @ base)
        c.metrics = {"snr_db": 30.0, "f0_median": 120.0, "rate_wps": 2.5}
        clips.append(c)
    # Two clips from the same minute: only one may be chosen.
    twin = pipeline.Clip(source="a.wav", start=10.0, end=19.0, text="x", confidence=0.9, emb=clips[0].emb)
    twin.sim, twin.metrics = clips[0].sim, dict(clips[0].metrics)
    chosen = pipeline.select(clips + [twin], 5, lambda m: None)
    assert len(chosen) == 5
    starts = sorted(c.start for c in chosen)
    assert all(b - a >= 60 for a, b in zip(starts, starts[1:]))


def test_join_sizes_gaps_by_punctuation_and_never_uses_digital_silence():
    sr = 24_000
    a = np.concatenate([np.zeros(int(0.5 * sr)), tone(150, 1.0, sr), np.zeros(int(0.6 * sr))]).astype(np.float32)
    b = np.concatenate([np.zeros(int(0.4 * sr)), tone(170, 1.0, sr), np.zeros(int(0.5 * sr))]).astype(np.float32)
    out = join.join([a, b], ["First sentence.", "Second one"], sr)
    # 1s + 1s of tone + one sentence pause (0.42s) + small guards, not the 2s of silence the inputs carried.
    assert abs(len(out) / sr - (2.0 + join.PAUSE["."])) < 0.15
    mid = out[int(1.06 * sr) : int(1.36 * sr)]
    assert np.any(mid != 0.0), "gap must be room tone, not digital zero"
    assert np.max(np.abs(mid)) < 0.01


def test_pause_after_reads_punctuation():
    assert join.pause_after("Really?") == join.PAUSE["?"]
    assert join.pause_after('He said "no."') == join.PAUSE["."]
    assert join.pause_after("and then") == join.DEFAULT_PAUSE
