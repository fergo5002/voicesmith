"""Engine worker: runs inside an engine's own environment.

This file and its siblings must import nothing from ``voicesmith``. They are
executed with the engine environment's Python, which has torch and the engine
installed but not the voicesmith core.

Protocol: one JSON object per line. Requests ``{"id", "method", "params"}``;
replies ``{"id", "result"}`` or ``{"id", "error", "trace"}``. Audio moves as
file paths, never inline.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
import traceback

# Libraries love printing to stdout. Keep a private handle on the real stdout
# for the protocol and point fd 1 at stderr so stray prints cannot corrupt it.
_PROTO = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STATE: dict = {"adapter": None, "engine": None, "device": None}


def send(obj: dict) -> None:
    _PROTO.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _PROTO.flush()


def hello(params: dict) -> dict:
    import torch

    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_vram_gb": (torch.cuda.get_device_properties(0).total_memory / 1e9) if torch.cuda.is_available() else None,
        "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "threads": torch.get_num_threads(),
    }
    return info


def load(params: dict) -> dict:
    import torch

    engine = params["engine"]
    device = params.get("device", "cpu")
    threads = int(params.get("threads") or 0)
    if threads > 0:
        torch.set_num_threads(threads)
    if STATE["engine"] == engine and STATE["device"] == device:
        return {"loaded": engine, "cached": True}
    family = params["family"]
    # Built-in families live in adapters/; a dotted name loads a third-party adapter
    # module installed in the engine environment (the plugin route).
    mod = importlib.import_module(family if "." in family else f"adapters.{family}")
    t0 = time.time()
    STATE["adapter"] = mod.Adapter(engine, device)
    STATE["engine"], STATE["device"] = engine, device
    return {"loaded": engine, "seconds": time.time() - t0, "sample_rate": STATE["adapter"].sample_rate}


def synthesize(params: dict) -> dict:
    import numpy as np
    import soundfile as sf
    import torch

    adapter = STATE["adapter"]
    if adapter is None:
        raise RuntimeError("no engine loaded")
    seed = int(params.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    t0 = time.time()
    wav = adapter.synthesize(
        text=params["text"],
        ref_wav=params["ref_wav"],
        ref_text=params.get("ref_text"),
        options=params.get("options") or {},
    )
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(wav)):
        raise RuntimeError("engine produced non-finite samples")
    gen = time.time() - t0
    sf.write(params["out_wav"], wav, adapter.sample_rate, subtype="FLOAT")
    return {"seconds": gen, "duration": len(wav) / adapter.sample_rate, "sample_rate": adapter.sample_rate}


def watermark(params: dict) -> dict:
    import watermark as wm

    return wm.embed(params["in_wav"], params["out_wav"])


def detect(params: dict) -> dict:
    import watermark as wm

    return wm.detect(params["wav"])


METHODS = {"hello": hello, "load": load, "synthesize": synthesize, "watermark": watermark, "detect": detect}


def main() -> None:
    send({"event": "ready", "pid": os.getpid()})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            send({"id": None, "error": f"bad json: {exc}"})
            continue
        rid, method = req.get("id"), req.get("method")
        if method == "shutdown":
            send({"id": rid, "result": {"bye": True}})
            break
        fn = METHODS.get(method)
        if fn is None:
            send({"id": rid, "error": f"unknown method {method!r}"})
            continue
        try:
            send({"id": rid, "result": fn(req.get("params") or {})})
        except Exception as exc:  # report, never die on a bad request
            send({"id": rid, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-4000:]})


if __name__ == "__main__":
    main()
