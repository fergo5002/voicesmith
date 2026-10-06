"""Find the engine, reference and settings that clone a particular voice best.

Run once per voice. Every installed engine that supports the voice's language
renders a few probe sentences from each of the strongest references, and the
combination with the best similarity and intelligibility wins. Rendering a
message afterwards reuses the winner, so the tournament cost is paid once.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from voicesmith import audio, consent, hardware, voices
from voicesmith.engines import client, manager, registry
from voicesmith.synth import evaluate

Log = Callable[[str], None]

PROBES = {
    "en": [
        "Honestly, I didn't expect it to work this well, but here we are.",
        "Could you send me the numbers by Thursday? I'd like to check them before the meeting at half ten.",
        "The light was fading over the harbour as the last boats came in.",
    ],
}


def probes_for(voice: voices.Voice, n: int) -> list[str]:
    base = PROBES.get(voice.language)
    if base:
        return base[:n]
    # Other languages: borrow the speaker's own sentences from references we will not use as prompts.
    texts = [r.text for r in sorted(voice.references, key=lambda r: r.score)][:n]
    return texts or PROBES["en"][:n]


def candidate_engines(voice: voices.Voice, requested: list[str] | None, device: str) -> list[str]:
    if requested:
        return [registry.get(e).name for e in requested]
    out = []
    for e in registry.ENGINES.values():
        if voice.language not in e.languages or not manager.installed(e.family):
            continue
        if "gpu-preferred" in e.tags and device == "cpu":
            continue
        if "multilingual" in e.tags and voice.language == "en":
            continue  # dedicated English engines exist; no need to download a multilingual model
        out.append(e.name)
    return out


def tune(
    voice_name: str,
    *,
    engines: list[str] | None = None,
    n_refs: int = 3,
    n_probes: int = 2,
    budget_minutes: float | None = None,
    log: Log = print,
) -> voices.Tuning:
    voice = voices.load(voice_name)
    consent.require(voice)
    if not voice.references:
        raise RuntimeError(f"voice {voice.name!r} has no references; run ingest first")
    device = hardware.detect().preferred_device
    names = candidate_engines(voice, engines, device)
    if not names:
        raise RuntimeError("no installed engine supports this voice; run: voicesmith engines install chatterbox")
    probes = probes_for(voice, n_probes)
    deadline = time.time() + budget_minutes * 60 if budget_minutes else None
    tried: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="voicesmith-tune-") as tmp:
        for eng in names:
            spec = registry.get(eng)
            lo, hi = spec.ref_window
            refs = sorted(voice.references, key=lambda r: (not (lo - 1.5 <= r.duration <= hi + 3.0), -r.score))[:n_refs]
            log(f"{eng}: loading")
            w = client.engine_worker(eng)
            for ref in refs:
                scores, rtfs = [], []
                for pi, probe in enumerate(probes):
                    if deadline and time.time() > deadline:
                        log("time budget reached")
                        break
                    out = Path(tmp) / f"{eng}-{ref.id}-{pi}.wav"
                    gen = w.call("synthesize", {"text": probe, "ref_wav": str(voice.ref_path(ref)), "ref_text": ref.text,
                                                "seed": 1234 + pi, "options": {"language": voice.language},
                                                "out_wav": str(out)}, timeout=900)
                    wav, sr = audio.read_wav(out)
                    s = evaluate.evaluate(wav, sr, probe, voice)
                    scores.append(s)
                    rtfs.append(gen["seconds"] / max(0.1, gen["duration"]))
                if not scores:
                    continue
                row = {
                    "engine": eng,
                    "reference": ref.id,
                    "pass_rate": float(np.mean([s.ok for s in scores])),
                    "similarity": float(np.mean([s.similarity for s in scores])),
                    "sim_norm": float(np.mean([min(1.0, s.sim_norm) for s in scores])),
                    "cer": float(np.mean([s.cer for s in scores])),
                    "q": float(np.mean([s.q for s in scores])),
                    "rtf": float(np.mean(rtfs)),
                }
                tried.append(row)
                log(f"{eng} {ref.id}: similarity {row['similarity']:.3f}, CER {row['cer']:.1%}, "
                    f"passed {row['pass_rate']:.0%}, {row['rtf']:.1f}x real time")
            if deadline and time.time() > deadline:
                break
    viable = [t for t in tried if t["pass_rate"] >= 0.5] or tried
    if not viable:
        raise RuntimeError("tuning produced no usable takes")
    best = max(viable, key=lambda t: (t["q"], t["similarity"]))
    voice.tuning = voices.Tuning(
        engine=best["engine"],
        reference=best["reference"],
        options={},
        score=round(best["q"], 4),
        similarity=round(best["similarity"], 4),
        cer=round(best["cer"], 4),
        rtf=round(best["rtf"], 2),
        tuned_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        tried=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()} for t in tried],
    )
    voice.save()
    log(f"best: {best['engine']} with {best['reference']} (similarity {best['similarity']:.3f}, CER {best['cer']:.1%})")
    return voice.tuning
