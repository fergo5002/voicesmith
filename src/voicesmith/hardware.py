"""Find out what this machine can do, cheaply and without importing torch."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field

import psutil

from voicesmith import paths


@dataclass
class Gpu:
    vendor: str
    name: str
    vram_gb: float | None = None


@dataclass
class Machine:
    os: str
    arch: str
    cpu: str
    physical_cores: int
    logical_cores: int
    ram_gb: float
    ram_free_gb: float
    disk_free_gb: float
    gpus: list[Gpu] = field(default_factory=list)

    @property
    def nvidia(self) -> Gpu | None:
        return next((g for g in self.gpus if g.vendor == "nvidia"), None)

    @property
    def apple_silicon(self) -> bool:
        return self.os == "Darwin" and self.arch in ("arm64", "aarch64")

    @property
    def preferred_device(self) -> str:
        if self.nvidia:
            return "cuda"
        if self.apple_silicon:
            return "mps"
        return "cpu"

    @property
    def torch_backend(self) -> str | None:
        """Value for ``uv pip install --torch-backend``. None means "use the index default".

        uv's ``auto`` mode also picks Intel XPU wheels whenever any Intel display
        adapter exists, including older iGPUs PyTorch XPU does not support, so we
        only hand over to ``auto`` when there is an NVIDIA GPU to target.
        """
        if self.nvidia:
            return "auto"
        if self.os == "Darwin":
            return None
        return "cpu"

    def fingerprint(self) -> str:
        key = json.dumps([self.os, self.arch, self.cpu, self.physical_cores, round(self.ram_gb), [g.name for g in self.gpus]])
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["preferred_device"] = self.preferred_device
        d["fingerprint"] = self.fingerprint()
        return d


def _nvidia() -> list[Gpu]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                gpus.append(Gpu("nvidia", parts[0], round(float(parts[1]) / 1024, 1)))
            except ValueError:
                gpus.append(Gpu("nvidia", parts[0]))
    return gpus


def _cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    if platform.system() == "Darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout.strip()
        except OSError:
            pass
    if os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def detect() -> Machine:
    vm = psutil.virtual_memory()
    gpus = _nvidia()
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        gpus.append(Gpu("apple", "Apple Silicon GPU", round(vm.total / 1e9, 1)))
    return Machine(
        os=platform.system(),
        arch=platform.machine().lower(),
        cpu=_cpu_name(),
        physical_cores=psutil.cpu_count(logical=False) or os.cpu_count() or 1,
        logical_cores=psutil.cpu_count(logical=True) or os.cpu_count() or 1,
        ram_gb=round(vm.total / 1e9, 1),
        ram_free_gb=round(vm.available / 1e9, 1),
        disk_free_gb=round(shutil.disk_usage(paths.home()).free / 1e9, 1),
        gpus=gpus,
    )


def busy_percent(seconds: float = 1.0) -> float:
    """System-wide CPU use right now. A busy machine makes speed calibration lie."""
    return float(psutil.cpu_percent(interval=seconds))
