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


WINDOWS_BANNER = """Input #0, ogg, from 'o.ogg':
  Duration: 00:00:02.01, start: 0.000000, bitrate: 117 kb/s
  Stream #0:0: Audio: opus, 48000 Hz, mono, fltp
    Metadata:
      encoder         : Lavc62.28.102 libopus
      comment         : AI-generated voice made with voicesmith. Not a live recording.
"""

# Shaped like the CI failure on Linux and macOS: a tag straight after "Metadata:".
# The first parser folded this comment line into a bogus "metadata" key.
LINUX_BANNER = """Input #0, ogg, from 'o.ogg':
  Duration: 00:00:02.01, start: 0.000000, bitrate: 117 kb/s
  Stream #0:0: Audio: opus, 48000 Hz, mono, fltp
      Metadata:
        comment         : AI-generated voice made with voicesmith. Not a live recording.
        encoder         : Lavc60.31.102 libopus
        ITUNES-COMPILATION: 0
        empty_tag       :
Stream mapping:
  Stream #0:0 -> #0:0 (opus (native) -> pcm_s16le (native))
Output #0, null, to 'pipe:':
  Metadata:
    encoder         : Lavf60.16.100
"""


@pytest.mark.parametrize("banner", [WINDOWS_BANNER, LINUX_BANNER, WINDOWS_BANNER.replace("\n", "\r\n"),
                                    LINUX_BANNER.replace("\n", "\r\n")])
def test_banner_tags_read_every_layout(banner):
    tags = ffmpeg.banner_tags(banner)
    assert tags["comment"] == "AI-generated voice made with voicesmith. Not a live recording."
    assert tags["encoder"].startswith("Lavc")  # the input stream's encoder, not ffmpeg's output Lavf
    assert "metadata" not in tags


def test_banner_tags_keep_unusual_keys_empty_values_and_continuations():
    tags = ffmpeg.banner_tags(LINUX_BANNER)
    assert tags["itunes-compilation"] == "0"
    assert tags["empty_tag"] == ""
    multi = "    Metadata:\n      comment         : first line\n                      : second line\n"
    assert ffmpeg.banner_tags(multi)["comment"] == "first line\nsecond line"
