"""Talk to engine workers: spawn, call with timeouts, keep warm, clean up."""

from __future__ import annotations

import atexit
import itertools
import json
import queue
import subprocess
import threading
import time
from pathlib import Path

from voicesmith import paths
from voicesmith.engines import manager, registry


class WorkerError(RuntimeError):
    def __init__(self, message: str, trace: str | None = None):
        super().__init__(message)
        self.trace = trace


class Worker:
    def __init__(self, family: str, *, start_timeout: float = 120.0):
        if not manager.installed(family):
            raise WorkerError(f"engine family {family!r} is not installed; run: voicesmith engines install {family}")
        self.family = family
        self.log_path = paths.logs_dir() / f"worker-{family}.log"
        self._log = open(self.log_path, "a", encoding="utf-8", errors="replace")
        self._log.write(f"\n==== start {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")
        self.proc = subprocess.Popen(
            [str(manager.python_path(family)), "-u", str(manager.worker_script())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=manager.worker_env(),
            bufsize=1,
        )
        self._ids = itertools.count(1)
        self._replies: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self.loaded: tuple[str, str] | None = None
        self.info: dict = {}
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        ready = self._next(start_timeout)
        if ready.get("event") != "ready":
            raise WorkerError(f"worker did not start cleanly: {ready}")
        self.last_used = time.time()

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._replies.put(json.loads(line))
            except json.JSONDecodeError:
                self._log.write(f"[non-protocol stdout] {line}\n")
        self._replies.put({"event": "exit", "code": self.proc.wait()})

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self._log.write(line)
            self._log.flush()

    def _next(self, timeout: float) -> dict:
        try:
            msg = self._replies.get(timeout=timeout)
        except queue.Empty:
            self.kill()
            raise WorkerError(f"{self.family} worker timed out after {timeout:.0f}s; log: {self.log_path}") from None
        if msg.get("event") == "exit":
            raise WorkerError(f"{self.family} worker exited with code {msg.get('code')}; log: {self.log_path}")
        return msg

    def call(self, method: str, params: dict | None = None, timeout: float = 600.0) -> dict:
        with self._lock:
            if self.proc.poll() is not None:
                raise WorkerError(f"{self.family} worker is not running; log: {self.log_path}")
            rid = next(self._ids)
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps({"id": rid, "method": method, "params": params or {}}) + "\n")
            self.proc.stdin.flush()
            deadline = time.time() + timeout
            while True:
                msg = self._next(max(1.0, deadline - time.time()))
                if msg.get("id") == rid:
                    break
            self.last_used = time.time()
            if "error" in msg:
                raise WorkerError(f"{self.family}.{method}: {msg['error']}", msg.get("trace"))
            return msg["result"]

    def alive(self) -> bool:
        return self.proc.poll() is None

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.write(json.dumps({"id": 0, "method": "shutdown"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=15)
            except Exception:
                self.kill()
        self._log.close()


_POOL: dict[str, Worker] = {}
_POOL_LOCK = threading.Lock()


def worker(family: str) -> Worker:
    with _POOL_LOCK:
        w = _POOL.get(family)
        if w is None or not w.alive():
            w = Worker(family)
            w.info = w.call("hello", timeout=180)
            _POOL[family] = w
        return w


def engine_worker(engine: str, device: str | None = None, threads: int = 0, timeout: float = 1800.0) -> Worker:
    """A warm worker with ``engine`` loaded on the best available device.

    The first load of an engine may download its weights, hence the long timeout.
    """
    spec = registry.get(engine)
    w = worker(spec.family)
    device = resolve_device(w, engine, device)
    if w.loaded != (engine, device):
        w.call("load", {"engine": engine, "family": spec.family, "device": device, "threads": threads}, timeout=timeout)
        w.loaded = (engine, device)
    return w


def any_watermark_worker() -> Worker | None:
    """Every engine environment carries the watermarker; reuse a warm one before starting another."""
    with _POOL_LOCK:
        for w in _POOL.values():
            if w.alive():
                return w
    for family in registry.FAMILIES:
        if manager.installed(family):
            return worker(family)
    return None


def shutdown_all() -> None:
    with _POOL_LOCK:
        for w in _POOL.values():
            w.close()
        _POOL.clear()


def reap_idle(max_idle: float) -> None:
    with _POOL_LOCK:
        for family, w in list(_POOL.items()):
            if time.time() - w.last_used > max_idle:
                w.close()
                del _POOL[family]


atexit.register(shutdown_all)


def resolve_device(w: Worker, engine: str, requested: str | None = None) -> str:
    """Pick the best device the worker's torch build can actually use."""
    spec = registry.get(engine)
    if requested and requested != "auto":
        return requested
    if w.info.get("cuda") and "cuda" in spec.devices:
        return "cuda"
    if w.info.get("mps") and "mps" in spec.devices:
        return "mps"
    return "cpu"


def log_tail(family: str, lines: int = 40) -> str:
    p: Path = paths.logs_dir() / f"worker-{family}.log"
    if not p.exists():
        return ""
    return "\n".join(p.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
