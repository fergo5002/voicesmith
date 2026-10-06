"""Disclosure metadata and the manifest written next to every output."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from voicesmith import __version__

DISCLOSURE = "AI-generated voice made with voicesmith. Not a live recording."


def tags(*, voice: str, speaker: str, consent_kind: str, engine: str) -> dict[str, str]:
    """Tags written into every delivered file. Readable by any media player."""
    basis = "spoken consent" if consent_kind == "spoken" else "operator attestation"
    return {
        "comment": f"{DISCLOSURE} Synthetic voice of {speaker}, made with their {basis}.",
        "description": DISCLOSURE,
        "encoded_by": f"voicesmith {__version__}",
        "ai_generated": "true",
        "synthetic_voice": speaker,
        "voicesmith_voice": voice,
        "voicesmith_engine": engine,
        "voicesmith_consent": consent_kind,
        "voicesmith_watermark": "audioseal:VS",
    }


def check_tags(meta: dict[str, str]) -> bool:
    """The disclosure sentence must be readable. It lives in ``comment``, the one tag
    every container keeps (WAV's INFO chunk drops custom keys like ai_generated)."""
    return DISCLOSURE in meta.get("comment", "") + " " + meta.get("description", "")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(path: Path, data: dict) -> Path:
    data = {"generator": f"voicesmith {__version__}", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **data}
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path
