import numpy as np
import pytest

from voicesmith import audio, ffmpeg, provenance

TAGS = provenance.tags(voice="sam", speaker="Sam Example", consent_kind="spoken", engine="chatterbox-turbo")


@pytest.fixture
def src(tmp_path):
    t = np.arange(48_000) / 24_000
    return audio.save_wav(tmp_path / "src.wav", (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 24_000)


@pytest.mark.parametrize("fmt", ["wav", "flac", "mp3", "m4a", "ogg"])
def test_disclosure_survives_every_container(src, tmp_path, fmt):
    out = ffmpeg.encode(src, tmp_path / f"o.{fmt}", TAGS)
    assert provenance.check_tags(ffmpeg.read_metadata(out))


@pytest.mark.parametrize("fmt", ["wav", "m4a", "ogg"])
def test_untagged_files_fail_the_check(src, tmp_path, fmt):
    out = ffmpeg.encode(src, tmp_path / f"plain.{fmt}", {})
    assert not provenance.check_tags(ffmpeg.read_metadata(out))


def test_decode_and_loudness(src):
    x = ffmpeg.decode(src, 16_000)
    assert abs(len(x) - 32_000) < 50
    loud = ffmpeg.measure_loudness(src)
    assert -20 < loud.integrated < -10 and loud.true_peak < 0


def test_unknown_format_is_refused(src, tmp_path):
    with pytest.raises(ValueError):
        ffmpeg.encode(src, tmp_path / "o.xyz", TAGS)
