import os
from pathlib import Path

import numpy as np
import pytest

# Real models are shared across test runs so slow tests download them once.
_MODELS = os.environ.get("VOICESMITH_MODELS") or str(
    Path(os.environ.get("VOICESMITH_HOME") or Path.home() / ".voicesmith") / "models"
)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test gets its own empty voicesmith home (but shares downloaded models)."""
    monkeypatch.setenv("VOICESMITH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VOICESMITH_MODELS", _MODELS)
    yield tmp_path / "home"


def tone(freq: float, seconds: float, sr: int = 16_000, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * (np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(4 * np.pi * freq * t))).astype(np.float32)
