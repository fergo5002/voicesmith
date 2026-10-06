import numpy as np

from voicesmith import audio
from voicesmith.analysis import prosody

SR = 16_000


def tone(freq: float, seconds: float, sr: int = SR, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    # A few harmonics so it looks like a voiced sound rather than a pure sine.
    return (amp * (np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(4 * np.pi * freq * t) + 0.25 * np.sin(6 * np.pi * freq * t))).astype(np.float32)


def test_f0_tracks_a_known_pitch():
    for f in (95.0, 180.0, 260.0):
        track = prosody.f0_track(tone(f, 1.0))
        voiced = track[track > 0]
        assert voiced.size > 50
        assert abs(np.median(voiced) - f) / f < 0.02


def test_f0_ignores_silence():
    sig = np.concatenate([np.zeros(SR // 2, np.float32), tone(150, 0.5)])
    track = prosody.f0_track(sig)
    assert np.all(track[:40] == 0)


def test_range_in_semitones_for_a_glide():
    t = np.arange(SR * 2) / SR
    freq = np.linspace(100, 200, t.size)  # exactly one octave
    sig = (0.3 * np.sin(2 * np.pi * np.cumsum(freq) / SR)).astype(np.float32)
    p = prosody.measure(sig)
    assert 9.0 < p.f0_range_st < 12.5


def test_active_bounds_and_trim():
    sig = np.concatenate([np.zeros(SR), tone(150, 1.0), np.zeros(SR)]).astype(np.float32)
    start, end = audio.active_bounds(sig, SR)
    assert abs(start - SR) < SR * 0.03 and abs(end - 2 * SR) < SR * 0.05
    trimmed = audio.trim(sig, SR, lead_ms=100, tail_ms=100)
    assert abs(len(trimmed) / SR - 1.2) < 0.06


def test_snr_ranks_clean_above_noisy():
    rng = np.random.default_rng(0)
    speechy = np.concatenate([tone(150, 0.4), np.zeros(int(0.2 * SR)), tone(170, 0.4), np.zeros(int(0.2 * SR))] * 3)
    clean = speechy + 0.001 * rng.standard_normal(speechy.size)
    noisy = speechy + 0.05 * rng.standard_normal(speechy.size)
    assert audio.snr_db(clean.astype(np.float32), SR) > audio.snr_db(noisy.astype(np.float32), SR) + 15


def test_resample_round_trip_keeps_length():
    sig = tone(200, 1.0, sr=24_000)
    down = audio.resample(sig, 24_000, 16_000)
    assert len(down) == 16_000


def test_bandwidth_tells_wideband_from_phone_audio():
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(0)
    wide = (0.1 * rng.standard_normal(SR * 3)).astype(np.float32)
    phone = sosfilt(butter(8, 3400, "low", fs=SR, output="sos"), wide).astype(np.float32)
    assert audio.bandwidth_hz(wide, SR) > 7000
    assert audio.bandwidth_hz(phone, SR) < 6000  # the ingest gate threshold
