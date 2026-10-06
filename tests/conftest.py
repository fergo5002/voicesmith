import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test gets its own empty voicesmith home."""
    monkeypatch.setenv("VOICESMITH_HOME", str(tmp_path / "home"))
    yield tmp_path / "home"


def tone(freq: float, seconds: float, sr: int = 16_000, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * (np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(4 * np.pi * freq * t))).astype(np.float32)
