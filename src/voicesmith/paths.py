"""Where voicesmith keeps things on disk.

Everything lives under one home directory so it is easy to find, back up, and
delete: ``VOICESMITH_HOME`` if set, otherwise ``~/.voicesmith``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


def home() -> Path:
    root = os.environ.get("VOICESMITH_HOME")
    path = Path(root).expanduser() if root else Path.home() / ".voicesmith"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sub(*parts: str) -> Path:
    path = home().joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def voices_dir() -> Path:
    return sub("voices")


def envs_dir() -> Path:
    return sub("envs")


def models_dir() -> Path:
    """ONNX analysis models. ``VOICESMITH_MODELS`` lets several homes (or test runs) share one copy."""
    shared = os.environ.get("VOICESMITH_MODELS")
    if shared:
        path = Path(shared).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path
    return sub("models")


def cache_dir() -> Path:
    return sub("cache")


def logs_dir() -> Path:
    return sub("logs")


def jobs_dir() -> Path:
    return sub("jobs")


def check_name(name: str) -> str:
    """Voice names become directory names, so keep them boring and portable."""
    if not _NAME_RE.match(name):
        raise ValueError(
            f"invalid name {name!r}: use 1-63 lowercase letters, digits, '-' or '_', starting with a letter or digit"
        )
    return name
