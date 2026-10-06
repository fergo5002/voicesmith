"""In-process background jobs for the MCP server: long renders without long tool calls."""

from __future__ import annotations

import secrets
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Job:
    id: str
    kind: str
    status: str = "running"  # running | done | failed
    started: float = field(default_factory=time.time)
    finished: float | None = None
    log: list[str] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None

    def view(self) -> dict:
        return {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status,
            "elapsed_s": round((self.finished or time.time()) - self.started, 1),
            "progress": self.log[-6:],
            "result": self.result,
            "error": self.error,
        }


_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()
# One heavy job at a time: two renders on one CPU are slower than two in a row.
_RUN = threading.Semaphore(1)


def start(kind: str, fn: Callable[[Callable[[str], None]], dict]) -> Job:
    job = Job(id=secrets.token_hex(6), kind=kind)
    with _LOCK:
        _JOBS[job.id] = job

    def runner() -> None:
        job.log.append("queued")
        with _RUN:
            try:
                job.result = fn(lambda m: job.log.append(m))
                job.status = "done"
            except Exception as exc:
                job.error = f"{type(exc).__name__}: {exc}"
                job.log.append(traceback.format_exc(limit=3)[-800:])
                job.status = "failed"
            finally:
                job.finished = time.time()

    threading.Thread(target=runner, daemon=True, name=f"job-{job.id}").start()
    return job


def get(job_id: str) -> Job | None:
    return _JOBS.get(job_id)


def wait(job: Job, seconds: float) -> Job:
    deadline = time.time() + seconds
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.25)
    return job
