"""Create and remove the isolated environment each engine family runs in.

Engines pin conflicting versions of torch and transformers, so each family
gets its own uv-managed virtual environment under ``~/.voicesmith/envs``. uv
hard-links packages from its cache, so shared wheels do not cost disk twice.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from voicesmith import __version__, hardware, paths
from voicesmith.engines import registry

Log = Callable[[str], None]


def uv_exe() -> str:
    try:
        from uv import find_uv_bin

        return find_uv_bin()
    except Exception:
        found = shutil.which("uv")
        if not found:
            raise RuntimeError("uv not found; reinstall voicesmith (it depends on the uv package)") from None
        return found


def env_dir(family: str) -> Path:
    return paths.envs_dir() / family


def python_path(family: str) -> Path:
    d = env_dir(family)
    return d / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def marker(family: str) -> Path:
    return env_dir(family) / "voicesmith-env.json"


def installed(family: str) -> bool:
    return python_path(family).exists() and marker(family).exists()


def info(family: str) -> dict | None:
    if not installed(family):
        return None
    return json.loads(marker(family).read_text(encoding="utf-8"))


def worker_env() -> dict[str, str]:
    """Environment for anything we run inside an engine env."""
    env = dict(os.environ)
    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            # A stale or broken saved Hugging Face login must never block public downloads.
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
            "PYTHONWARNINGS": "ignore",
            # AudioSeal's vendored moshi code calls torch.compile, which needs a C++
            # compiler most Windows machines do not have. Never require one.
            "NO_TORCH_COMPILE": "1",
            "TORCHDYNAMO_DISABLE": "1",
        }
    )
    env.pop("VIRTUAL_ENV", None)
    return env


def install(family: str, *, log: Log = print, machine: hardware.Machine | None = None, force: bool = False) -> Path:
    spec = registry.FAMILIES[family]
    if installed(family) and not force:
        log(f"{family}: already installed")
        return env_dir(family)
    machine = machine or hardware.detect()
    uv = uv_exe()
    target = env_dir(family)
    if target.exists():
        shutil.rmtree(target)
    t0 = time.time()
    log(f"{family}: creating Python {spec.python} environment")
    _run([uv, "venv", "--python", spec.python, "--quiet", str(target)], log)
    cmd = [uv, "pip", "install", "--python", str(python_path(family))]
    if machine.torch_backend:
        cmd += ["--torch-backend", machine.torch_backend]
    cmd += list(spec.packages)
    log(f"{family}: installing {len(spec.packages)} packages (torch backend: {machine.torch_backend or 'default'})")
    _run(cmd, log)
    marker(family).write_text(
        json.dumps(
            {
                "family": family,
                "packages": list(spec.packages),
                "torch_backend": machine.torch_backend,
                "voicesmith": __version__,
                "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "seconds": round(time.time() - t0, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"{family}: ready in {time.time() - t0:.0f}s")
    return target


def remove(family: str) -> None:
    if env_dir(family).exists():
        shutil.rmtree(env_dir(family))


def _run(cmd: list[str], log: Log) -> None:
    logfile = paths.logs_dir() / "install.log"
    with open(logfile, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", env=worker_env())
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
        code = proc.wait()
    if code != 0:
        tail = logfile.read_text(encoding="utf-8", errors="replace")[-2500:]
        raise RuntimeError(f"command failed ({code}): {' '.join(cmd[:4])} ...\n{tail}")


def worker_script() -> Path:
    return Path(__file__).parent / "_worker" / "main.py"


def self_python() -> str:
    return sys.executable
