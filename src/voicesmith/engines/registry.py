"""What each engine is, what it needs, and what it is allowed to be used for."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Family:
    """One isolated Python environment that can host several engines."""

    name: str
    python: str
    packages: tuple[str, ...]
    overrides: tuple[str, ...] = ()
    torch_pin: str | None = None  # some engines hard-pin torch; others take the backend default


@dataclass(frozen=True)
class Engine:
    name: str
    family: str
    title: str
    weights_licence: str
    commercial: bool
    languages: tuple[str, ...]
    needs_ref_text: bool
    ref_window: tuple[float, float]  # seconds of reference the engine actually uses well
    max_chars: int  # longest text it handles reliably in one pass
    params_m: int
    devices: tuple[str, ...] = ("cpu", "cuda", "mps")
    builtin_watermark: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


_WATERMARK = ("audioseal>=0.2", "omegaconf")

FAMILIES: dict[str, Family] = {
    "chatterbox": Family(
        name="chatterbox",
        python="3.12",
        # Pinned to the commit that added Nano; PyPI 0.1.7 predates it.
        packages=(
            "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox@5de7a54aa4e5e2baadb0182dde554908b48b85c2",
            "setuptools<81",
            *_WATERMARK,
        ),
    ),
    "qwen3": Family(
        name="qwen3",
        python="3.12",
        packages=("qwen-tts==0.1.1", "torch", "torchaudio", *_WATERMARK),
    ),
    "sopro": Family(
        name="sopro",
        python="3.12",
        packages=("sopro==2.2.0", "torchaudio", *_WATERMARK),
    ),
}

ENGINES: dict[str, Engine] = {
    e.name: e
    for e in [
        Engine(
            name="chatterbox-turbo",
            family="chatterbox",
            title="Chatterbox Turbo (Resemble AI)",
            weights_licence="MIT",
            commercial=True,
            languages=("en",),
            needs_ref_text=False,
            ref_window=(7.0, 10.0),
            max_chars=300,
            params_m=350,
            builtin_watermark="perth",
            tags=("default-cpu", "paralinguistic-tags"),
        ),
        Engine(
            name="chatterbox-nano",
            family="chatterbox",
            title="Chatterbox Nano (Resemble AI)",
            weights_licence="MIT",
            commercial=True,
            languages=("en",),
            needs_ref_text=False,
            ref_window=(7.0, 10.0),
            max_chars=300,
            params_m=110,
            builtin_watermark="perth",
            tags=("fast",),
        ),
        Engine(
            name="chatterbox-multilingual",
            family="chatterbox",
            title="Chatterbox Multilingual V3 (Resemble AI)",
            weights_licence="MIT",
            commercial=True,
            languages=("ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl",
                       "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"),
            needs_ref_text=False,
            ref_window=(7.0, 10.0),
            max_chars=300,
            params_m=500,
            builtin_watermark="perth",
            tags=("multilingual",),
        ),
        Engine(
            name="qwen3-tts-0.6b",
            family="qwen3",
            title="Qwen3-TTS 0.6B Base (Alibaba)",
            weights_licence="Apache-2.0",
            commercial=True,
            languages=("en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"),
            needs_ref_text=True,
            ref_window=(6.0, 12.0),
            max_chars=600,
            params_m=600,
            tags=("identity", "long-form"),
        ),
        Engine(
            name="qwen3-tts-1.7b",
            family="qwen3",
            title="Qwen3-TTS 1.7B Base (Alibaba)",
            weights_licence="Apache-2.0",
            commercial=True,
            languages=("en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"),
            needs_ref_text=True,
            ref_window=(6.0, 12.0),
            max_chars=600,
            params_m=1700,
            tags=("identity", "long-form", "gpu-preferred"),
        ),
        Engine(
            name="sopro",
            family="sopro",
            title="Sopro V2 Turbo (Halo Research)",
            weights_licence="Apache-2.0",
            commercial=True,
            languages=("en", "pt", "fr", "de"),
            needs_ref_text=False,
            ref_window=(5.0, 15.0),
            max_chars=400,
            params_m=120,
            tags=("fast",),
        ),
    ]
}


def get(name: str) -> Engine:
    try:
        return ENGINES[name]
    except KeyError:
        raise KeyError(f"unknown engine {name!r}; known: {', '.join(ENGINES)}") from None


def for_language(lang: str) -> list[Engine]:
    return [e for e in ENGINES.values() if lang in e.languages]
