"""Check everything voicesmith depends on and say how to fix what is wrong."""

from __future__ import annotations

import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from voicesmith import __version__, ffmpeg, hardware, models, paths
from voicesmith.engines import client, manager, registry


@dataclass
class Check:
    name: str
    ok: bool | None  # None = informational
    detail: str
    fix: str = ""


def run(probe_engines: bool = True) -> tuple[list[Check], hardware.Machine]:
    checks: list[Check] = []
    m = hardware.detect()
    checks.append(Check("voicesmith", True, f"{__version__} on Python {sys.version.split()[0]} ({platform.system()} {m.arch})"))
    checks.append(Check("home", True, str(paths.home())))
    checks.append(
        Check("disk", m.disk_free_gb >= 10, f"{m.disk_free_gb:.0f} GB free",
              "engines and their models need 3 to 8 GB each; free some space or set VOICESMITH_HOME" if m.disk_free_gb < 10 else "")
    )
    gpu = ", ".join(f"{g.name}" + (f" ({g.vram_gb:.0f} GB)" if g.vram_gb else "") for g in m.gpus) or "none usable"
    checks.append(Check("hardware", None, f"{m.cpu}; {m.physical_cores} cores; {m.ram_gb:.0f} GB RAM; GPU: {gpu}; "
                                            f"using {m.preferred_device}"))
    busy = hardware.busy_percent(0.5)
    if busy > 50:
        checks.append(Check("load", None, f"CPU is {busy:.0f}% busy right now; speeds will be slower than usual"))

    try:
        caps = ffmpeg.capabilities()
        missing = [k for k, v in caps.items() if not v]
        checks.append(Check("ffmpeg", not missing, f"{ffmpeg.exe()}" + (f" (missing: {', '.join(missing)})" if missing else ""),
                            "install a full ffmpeg build and put it on PATH" if missing else ""))
    except Exception as exc:
        checks.append(Check("ffmpeg", False, str(exc), "install ffmpeg, or reinstall voicesmith"))

    for st in models.status():
        checks.append(Check(f"model:{st['key']}", True if st["ready"] else None,
                            f"{st['name']} ({st['size_mb']} MB) {'ready' if st['ready'] else 'downloads on first use'}"))

    any_engine = False
    for fam in registry.FAMILIES:
        if not manager.installed(fam):
            continue
        any_engine = True
        detail = "installed"
        ok: bool | None = True
        if probe_engines:
            try:
                info = client.worker(fam).info
                detail = (f"torch {info.get('torch')}, python {info.get('python')}"
                          + (f", CUDA {info.get('cuda_name')}" if info.get("cuda") else "")
                          + (", MPS" if info.get("mps") else ""))
                if m.nvidia and not info.get("cuda"):
                    ok, detail = False, detail + "; NVIDIA GPU present but this torch build cannot use it"
            except Exception as exc:
                ok, detail = False, f"worker failed to start: {exc}"
        checks.append(Check(f"engine:{fam}", ok, detail,
                            f"voicesmith engines install {fam} --force" if ok is False else ""))
    if not any_engine:
        checks.append(Check("engines", False, "no synthesis engine installed",
                            "voicesmith engines install chatterbox   (about 4 GB with models)"))

    try:
        import yt_dlp

        js = shutil.which("deno") or shutil.which("node") or shutil.which("bun")
        checks.append(Check("yt-dlp", None, f"{yt_dlp.version.__version__}; JavaScript runtime: {js or 'none'}"
                            + ("" if js else " (YouTube downloads need deno: https://deno.com)")))
    except Exception as exc:
        checks.append(Check("yt-dlp", False, str(exc), "reinstall voicesmith"))

    try:
        import sounddevice as sd

        dev = sd.query_devices(kind="input")
        checks.append(Check("microphone", None, f"default input: {dev['name']}"))
    except Exception as exc:
        checks.append(Check("microphone", None, f"unavailable ({type(exc).__name__}); recording consent from a file still works"))

    if os.name == "nt":
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.symlink(__file__, Path(tmp) / "probe")
                links = True
            except OSError:
                links = False
        checks.append(Check("symlinks", None, "available" if links else
                            "not available (Developer Mode off); model caches use plain copies, which works but uses more disk"))
        checks.append(Check("antivirus", None, "Windows Defender scans multi-GB model files on first load; "
                            "adding ~/.voicesmith and the Hugging Face cache to its exclusions speeds that up"))
    return checks, m
