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


def _speaker_clips(rng, centre, n, start=0.0, dur=8.0):
    out = []
    for i in range(n):
        e = centre + 0.04 * rng.standard_normal(centre.size)  # ~0.85 cosine to the centre, like real clips
        e /= np.linalg.norm(e)
        out.append(pipeline.Clip(source="pod.wav", start=start + i * 20, end=start + i * 20 + dur, text="x",
                                 confidence=0.9, emb=e))
    return out


def _centres(rng, k):
    out = []
    for _ in range(k):
        c = rng.standard_normal(192)
        out.append(c / np.linalg.norm(c))
    return out


def test_dominant_speaker_is_found_when_clear():
    rng = np.random.default_rng(1)
    a, b = _centres(rng, 2)
    clips = _speaker_clips(rng, a, 14) + _speaker_clips(rng, b, 5, start=1000)
    target = pipeline.find_target(clips, None, lambda m: None)
    assert float(target @ a) > 0.9 and float(target @ b) < 0.3


def test_two_equal_hosts_without_an_anchor_refuse_to_guess():
    rng = np.random.default_rng(2)
    a, b = _centres(rng, 2)
    clips = _speaker_clips(rng, a, 9) + _speaker_clips(rng, b, 10, start=1000)
    try:
        pipeline.find_target(clips, None, lambda m: None)
    except pipeline.AmbiguousSpeaker as exc:
        assert "--target" in str(exc) and "voice 2" in str(exc)
    else:
        raise AssertionError("must not guess between two equal speakers")


def test_an_anchor_picks_the_minority_speaker():
    rng = np.random.default_rng(3)
    a, b = _centres(rng, 2)
    clips = _speaker_clips(rng, a, 14) + _speaker_clips(rng, b, 5, start=1000)
    target = pipeline.find_target(clips, b, lambda m: None)
    assert float(target @ b) > 0.9


def test_pick_chooses_a_listed_voice():
    rng = np.random.default_rng(2)
    a, b = _centres(rng, 2)
    clips = _speaker_clips(rng, a, 9) + _speaker_clips(rng, b, 10, start=1000)
    picked = pipeline.find_target(clips, None, lambda m: None, pick=2)
    assert max(float(picked @ a), float(picked @ b)) > 0.9
