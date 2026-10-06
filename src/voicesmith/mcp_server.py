"""MCP server so coding agents can speak in consented voices.

Design rules:

* Agents can list voices, render speech, poll jobs and verify files.
* Agents can never create or change consent. ``consent_steps`` only tells the
  human what to do.
* Renders run as background jobs. ``speak`` waits up to ``wait_seconds`` (50 by
  default, under Codex's 60 second tool limit) and otherwise returns a job id.
* Results are file paths plus scores, never inline audio.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

from fastmcp import Context, FastMCP

from voicesmith import __version__, consent, jobs, paths, voices

OUTPUT_FORMATS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
IDLE_UNLOAD_SECONDS = 900

INSTRUCTIONS = """voicesmith renders speech in cloned voices that have recorded consent.
Call list_voices first. Use speak to render; if it returns status "running", call job_status with the job_id
until it is "done". Outputs are watermarked and labelled as AI-generated. Never present them as real recordings.
If a voice has no consent, call consent_steps and pass the steps to the human; you cannot grant consent."""

server = FastMCP("voicesmith", instructions=INSTRUCTIONS, version=__version__)


def _voice_row(v: voices.Voice) -> dict:
    c = consent.load(v)
    try:
        consent.require(v)
        allowed = True
    except PermissionError:
        allowed = False
    return {
        "name": v.name,
        "speaker": v.speaker,
        "language": v.language,
        "consent": c.status if c else "none",
        "ready": bool(allowed and v.references),
        "references": len(v.references),
        "tuned_engine": v.tuning.engine if v.tuning else None,
    }


@server.tool
def list_voices() -> list[dict]:
    """List every voice with its consent state and whether it can render."""
    return [_voice_row(v) for v in voices.all_voices()]


@server.tool
def voice_info(name: str) -> dict:
    """Details for one voice: references, stats and tuning."""
    v = voices.load(name)
    row = _voice_row(v)
    row["stats"] = v.stats
    row["references"] = [{"id": r.id, "duration": r.duration, "text": r.text, "similarity": r.similarity} for r in v.references]
    if v.tuning:
        row["tuning"] = {"engine": v.tuning.engine, "reference": v.tuning.reference, "similarity": v.tuning.similarity,
                         "cer": v.tuning.cer, "rtf": v.tuning.rtf}
    return row


@server.tool
def consent_steps(name: str) -> dict:
    """What the human must do before this voice can render. Agents cannot do these steps."""
    v = voices.load(name)
    c = consent.load(v)
    if c and c.usable:
        return {"consent": c.status, "steps": []}
    return {
        "consent": c.status if c else "none",
        "steps": [
            f"Run in a terminal: voicesmith consent request {v.name}",
            f"Have {v.speaker} read the printed statement aloud and save the recording",
            f"Run: voicesmith consent verify {v.name} <recording>",
            f"Or, if {v.speaker} authorised it another way: voicesmith consent attest {v.name} --by \"Your Name\" --evidence \"...\"",
        ],
    }


@server.tool
async def speak(
    voice: str,
    text: str,
    output_path: str | None = None,
    quality: str = "balanced",
    polish: bool = False,
    wait_seconds: float = 50.0,
    ctx: Context | None = None,
) -> dict:
    """Render ``text`` in ``voice``. Returns file paths and quality scores, or a job id if still running.

    output_path: a file name (or path) inside ~/.voicesmith/outputs ending in .wav, .mp3, .m4a, .ogg or
        .flac. Agents may only write new files there; existing files are never replaced.
    quality: "fast", "balanced" (default) or "best".
    polish: also try a voice-conversion pass per take, kept only when it scores higher (slower).
    """
    from voicesmith.synth import render

    v = voices.load(voice)
    consent.require(v)  # fail fast with the human-readable reason
    outs = [_safe_output(output_path)] if output_path else None

    def work(log) -> dict:
        r = render.render(voice, text, outputs=outs, quality=quality, polish=polish, progress=log)
        return {"files": [str(f) for f in r.files], "manifest": str(r.manifest), "duration_s": r.duration,
                "engine": r.engine, "takes": r.takes, "similarity": r.score["similarity"], "wer": r.score["wer"],
                "seconds": r.seconds}

    job = jobs.start("speak", work)
    deadline = asyncio.get_running_loop().time() + max(0.0, wait_seconds)
    seen = 0
    while job.status == "running" and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.5)
        if ctx is not None and len(job.log) > seen:
            seen = len(job.log)
            try:
                await ctx.report_progress(progress=seen, total=None, message=job.log[-1])
            except Exception:
                pass
    return job.view()


def _safe_output(output_path: str) -> Path:
    """Agents write only new audio files inside the outputs folder."""
    root = paths.sub("outputs").resolve()
    p = Path(output_path).expanduser()
    p = (p if p.is_absolute() else root / p).resolve()
    if not p.is_relative_to(root):
        raise ValueError(f"output_path must be inside {root} (the voicesmith outputs folder)")
    if p.suffix.lower() not in OUTPUT_FORMATS:
        raise ValueError(f"output_path must end in one of {', '.join(sorted(OUTPUT_FORMATS))}")
    if p.exists() or p.with_name(p.stem + ".voicesmith.json").exists():
        raise ValueError(f"{p.name} already exists; choose a new file name")
    return p


@server.tool
def job_status(job_id: str) -> dict:
    """Progress and result of a render started by speak."""
    job = jobs.get(job_id)
    if job is None:
        return {"job_id": job_id, "status": "unknown"}
    return job.view()


@server.tool
def verify_audio(path: str) -> dict:
    """Check a file for voicesmith's watermark and AI-disclosure tags."""
    import tempfile

    from voicesmith import audio, ffmpeg, provenance
    from voicesmith.engines import client

    p = Path(path).expanduser()
    meta = ffmpeg.read_metadata(p)
    w = client.any_watermark_worker()
    det = None
    if w is not None:
        with tempfile.TemporaryDirectory() as tmp:
            wav = audio.save_wav(Path(tmp) / "x.wav", ffmpeg.decode(p, 24_000), 24_000, subtype="FLOAT")
            # A render may be using this worker; do not block an agent's tool call for minutes.
            det = w.call("detect", {"wav": str(wav)}, lock_timeout=40)
    return {
        "disclosure_tags": provenance.check_tags(meta),
        "watermark": det,
        "voicesmith_watermark_found": bool(det and det.get("audioseal_prob", 0) >= 0.5 and det.get("audioseal_payload_ok")),
    }


@server.tool
def doctor() -> list[dict]:
    """Health check of this machine's voicesmith setup."""
    from voicesmith import doctor as doc

    checks, _ = doc.run(probe_engines=True)
    return [c.__dict__ for c in checks]


def _reaper() -> None:
    from voicesmith.engines import client

    while True:
        time.sleep(60)
        try:
            client.reap_idle(IDLE_UNLOAD_SECONDS)
        except Exception:
            pass


def serve() -> None:
    # Engines hold gigabytes of RAM or VRAM; unload ones an agent has not used for a while.
    threading.Thread(target=_reaper, daemon=True, name="voicesmith-reaper").start()
    server.run(show_banner=False)
