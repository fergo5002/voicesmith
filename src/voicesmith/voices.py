"""Voice profiles on disk.

A voice is a directory under ``~/.voicesmith/voices/<name>`` holding its
consent record, its source manifest, the selected reference clips, and the
settings that ``tune`` found work best for it.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from voicesmith import paths


@dataclass
class Reference:
    id: str
    path: str  # relative to the voice directory
    text: str
    duration: float
    source: str
    start: float
    score: float
    similarity: float
    snr_db: float
    f0_median: float
    rate_wps: float


@dataclass
class Tuning:
    engine: str
    reference: str
    options: dict
    score: float
    similarity: float
    cer: float
    rtf: float
    tuned_at: str
    tried: list[dict] = field(default_factory=list)


@dataclass
class Voice:
    name: str
    speaker: str
    language: str = "en"
    created: str = ""
    sources: list[dict] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    tuning: Tuning | None = None
    notes: str = ""

    @property
    def dir(self) -> Path:
        return paths.voices_dir() / self.name

    def ref_path(self, ref: Reference) -> Path:
        return self.dir / ref.path

    def reference(self, ref_id: str) -> Reference:
        for r in self.references:
            if r.id == ref_id:
                return r
        raise KeyError(f"voice {self.name!r} has no reference {ref_id!r}")

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        tmp = self.dir / "voice.json.part"
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.dir / "voice.json")

    def centroid(self) -> np.ndarray | None:
        p = self.dir / "centroid.npy"
        return np.load(p) if p.exists() else None

    def holdout_centroid(self) -> np.ndarray | None:
        """Centroid of clips that are NOT references, so scoring rewards sounding
        like the person rather than copying one clip's room and microphone."""
        p = self.dir / "holdout.npy"
        return np.load(p) if p.exists() else self.centroid()


def exists(name: str) -> bool:
    return (paths.voices_dir() / paths.check_name(name) / "voice.json").exists()


def create(name: str, speaker: str, language: str = "en") -> Voice:
    paths.check_name(name)
    if exists(name):
        raise FileExistsError(f"voice {name!r} already exists")
    v = Voice(name=name, speaker=speaker, language=language, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    v.save()
    return v


def load(name: str) -> Voice:
    paths.check_name(name)
    p = paths.voices_dir() / name / "voice.json"
    if not p.exists():
        raise FileNotFoundError(f"no voice called {name!r}; see: voicesmith voices")
    data = json.loads(p.read_text(encoding="utf-8"))
    refs = [Reference(**r) for r in data.pop("references", [])]
    tuning = data.pop("tuning", None)
    v = Voice(**data, references=refs)
    v.tuning = Tuning(**tuning) if tuning else None
    return v


def all_voices() -> list[Voice]:
    out = []
    for d in sorted(paths.voices_dir().iterdir()):
        if (d / "voice.json").exists():
            try:
                out.append(load(d.name))
            except Exception:
                continue
    return out


def delete(name: str) -> None:
    d = paths.voices_dir() / paths.check_name(name)
    if d.exists():
        shutil.rmtree(d)
