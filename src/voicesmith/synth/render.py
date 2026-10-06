"""Text in, verified audio out."""

from __future__ import annotations

import json
import secrets
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from voicesmith import audio, consent, hardware, paths, provenance, text, voices
from voicesmith.engines import client, manager, registry
from voicesmith.synth import evaluate, join, master

Progress = Callable[[str], None]


@dataclass(frozen=True)
class Quality:
    name: str
    max_takes: int
    good_enough: float


QUALITY = {
    "fast": Quality("fast", 2, 0.0),  # first take that passes the gates
    "balanced": Quality("balanced", 4, 0.80),
    "best": Quality("best", 8, 0.93),
}

DEFAULT_ORDER = {
    "cuda": ["qwen3-tts-1.7b", "chatterbox-turbo", "qwen3-tts-0.6b", "chatterbox-multilingual"],
    "mps": ["chatterbox-turbo", "qwen3-tts-0.6b", "chatterbox-multilingual", "chatterbox-nano"],
    "cpu": ["chatterbox-turbo", "qwen3-tts-0.6b", "chatterbox-nano", "sopro", "chatterbox-multilingual"],
}


class RenderError(RuntimeError):
    def __init__(self, message: str, attempts: list[dict] | None = None):
        super().__init__(message)
        self.attempts = attempts or []


@dataclass
class Result:
    files: list[Path]
    manifest: Path
    engine: str
    reference: str
    seconds: float
    duration: float
    takes: int
    score: dict
    attempts: list[dict] = field(default_factory=list)


def choose_engine(voice: voices.Voice, requested: str | None = None) -> str:
    if requested:
        spec = registry.get(requested)
        if not manager.installed(spec.family):
            raise RenderError(f"engine {requested!r} is not installed; run: voicesmith engines install {spec.family}")
        return requested
    if voice.tuning and manager.installed(registry.get(voice.tuning.engine).family):
        return voice.tuning.engine
    device = hardware.detect().preferred_device
    for name in DEFAULT_ORDER[device]:
        spec = registry.get(name)
        if voice.language in spec.languages and manager.installed(spec.family):
            return name
    raise RenderError("no installed engine supports this voice's language; run: voicesmith engines install chatterbox")


def choose_reference(voice: voices.Voice, engine: str, requested: str | None = None) -> voices.Reference:
    if not voice.references:
        raise RenderError(f"voice {voice.name!r} has no references yet; run: voicesmith ingest {voice.name} <audio>")
    if requested:
        return voice.reference(requested)
    if voice.tuning and voice.tuning.engine == engine:
        return voice.reference(voice.tuning.reference)
    lo, hi = registry.get(engine).ref_window
    fits = [r for r in voice.references if lo - 1.5 <= r.duration <= hi + 3.0]
    return max(fits or voice.references, key=lambda r: r.score)


def default_output(voice: voices.Voice, fmt: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return paths.sub("outputs") / f"{voice.name}-{stamp}.{fmt}"


def render(
    voice_name: str,
    script: str,
    *,
    outputs: list[Path] | None = None,
    engine: str | None = None,
    reference: str | None = None,
    quality: str = "balanced",
    seed: int | None = None,
    options: dict | None = None,
    allow_imperfect: bool = False,
    progress: Progress = lambda _m: None,
) -> Result:
    t_start = time.time()
    script = " ".join(script.split())
    if not script:
        raise RenderError("nothing to say")
    voice = voices.load(voice_name)
    grant = consent.require(voice)
    q = QUALITY[quality]
    eng = choose_engine(voice, engine)
    spec = registry.get(eng)
    ref = choose_reference(voice, eng, reference)
    opts = {"language": voice.language, **(voice.tuning.options if voice.tuning and voice.tuning.engine == eng else {}),
            **(options or {})}
    outputs = outputs or [default_output(voice, "wav")]
    base_seed = seed if seed is not None else secrets.randbelow(2**31)

    progress(f"loading {spec.title}")
    w = client.engine_worker(eng)
    chunks = text.chunk(script, spec.max_chars)
    progress(f"rendering {len(chunks)} part(s) with reference {ref.id}")
    attempts: list[dict] = []
    best_chunks: list[np.ndarray] = []
    chunk_scores: list[evaluate.Score] = []
    sr = 24_000
    with tempfile.TemporaryDirectory(prefix="voicesmith-") as tmp:
        for ci, chunk in enumerate(chunks):
            best: tuple[evaluate.Score, np.ndarray] | None = None
            for take in range(q.max_takes + (2 if quality != "fast" else 1)):
                s = base_seed + 1000 * ci + take
                out_wav = Path(tmp) / f"c{ci}-t{take}.wav"
                gen = w.call(
                    "synthesize",
                    {"text": chunk, "ref_wav": str(voice.ref_path(ref)), "ref_text": ref.text, "seed": s,
                     "options": opts, "out_wav": str(out_wav)},
                    timeout=max(300.0, 60.0 * len(chunk) / 10),
                )
                wav, sr = audio.read_wav(out_wav)
                score = evaluate.evaluate(wav, sr, chunk, voice)
                attempts.append({"part": ci, "seed": s, "rtf": round(gen["seconds"] / max(0.1, gen["duration"]), 2),
                                 **score.as_dict()})
                verdict = f"q={score.q:.2f}" if score.ok else f"rejected: {score.reason}"
                progress(f"part {ci + 1}/{len(chunks)} take {take + 1}: {verdict}")
                if score.ok and (best is None or not best[0].ok or score.q > best[0].q):
                    best = (score, wav)
                elif best is None or (not best[0].ok and score.q > best[0].q):
                    best = (score, wav)
                passed = [a for a in attempts if a["part"] == ci and a["ok"]]
                if best[0].ok and (best[0].q >= q.good_enough or len(passed) >= q.max_takes):
                    break
            assert best is not None
            if not best[0].ok and not allow_imperfect:
                raise RenderError(f"part {ci + 1} never passed the checks (last problem: {best[0].reason})", attempts)
            best_chunks.append(best[1])
            chunk_scores.append(best[0])

    wav = join.join(best_chunks, chunks, sr, seed=base_seed)
    final = _combine(chunk_scores, [len(c) for c in best_chunks])
    progress("mastering, watermarking and verifying")
    tags = provenance.tags(voice=voice.name, speaker=voice.speaker, consent_kind=grant.kind, engine=eng)
    mastered = master.master(wav, sr, outputs, tags)
    manifest_path = outputs[0].with_suffix(".json")
    provenance.write_manifest(
        manifest_path,
        {
            "voice": voice.name,
            "speaker": voice.speaker,
            "consent": {"kind": grant.kind, "status": grant.status, "created": grant.created},
            "text": script,
            "engine": eng,
            "reference": {"id": ref.id, "text": ref.text, "source": ref.source},
            "options": opts,
            "quality": quality,
            "seed": base_seed,
            "score": final.as_dict(),
            "attempts": attempts,
            "files": [
                {"path": str(d.path), "sha256": provenance.sha256(d.path), "loudness_lufs": d.loudness_lufs,
                 "true_peak_dbtp": d.true_peak_dbtp, "watermark": d.watermark}
                for d in mastered.files
            ],
            "seconds": round(time.time() - t_start, 1),
        },
    )
    return Result(
        files=[d.path for d in mastered.files],
        manifest=manifest_path,
        engine=eng,
        reference=ref.id,
        seconds=round(time.time() - t_start, 1),
        duration=round(len(wav) / sr, 2),
        takes=len(attempts),
        score=final.as_dict(),
        attempts=attempts,
    )


def _combine(scores: list[evaluate.Score], weights: list[int]) -> evaluate.Score:
    """Duration-weighted summary of the parts, so long scripts are not re-transcribed whole."""
    if len(scores) == 1:
        return scores[0]
    w = np.asarray(weights, dtype=float) / sum(weights)

    def avg(attr: str) -> float:
        return float(sum(wi * getattr(s, attr) for wi, s in zip(w, scores)))

    bad = [s.reason for s in scores if not s.ok]
    return evaluate.Score(
        ok=not bad, reason="; ".join(bad), q=avg("q"), cer=avg("cer"), wer=avg("wer"),
        transcript=" ".join(s.transcript for s in scores), similarity=avg("similarity"), sim_norm=avg("sim_norm"),
        duration=sum(s.duration for s in scores), expected_duration=sum(s.expected_duration for s in scores),
        longest_pause=max(s.longest_pause for s in scores), f0_ratio=avg("f0_ratio"), rate_ratio=avg("rate_ratio"),
    )


def result_json(r: Result) -> str:
    return json.dumps(
        {"files": [str(f) for f in r.files], "manifest": str(r.manifest), "engine": r.engine, "reference": r.reference,
         "seconds": r.seconds, "duration": r.duration, "takes": r.takes, "score": r.score},
        indent=2,
    )
