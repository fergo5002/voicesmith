import json
import subprocess
import sys

from voicesmith import models
from voicesmith.engines import manager


def _talk(lines):
    proc = subprocess.run(
        [sys.executable, str(manager.worker_script())],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        timeout=60,
    )
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()], proc


def test_worker_handshake_errors_and_shutdown():
    replies, proc = _talk([
        "this is not json",
        json.dumps({"id": 1, "method": "nope"}),
        json.dumps({"id": 2, "method": "synthesize", "params": {}}),
        json.dumps({"id": 3, "method": "shutdown"}),
        json.dumps({"id": 4, "method": "nope"}),  # after shutdown: never answered
    ])
    assert replies[0]["event"] == "ready"
    assert "bad json" in replies[1]["error"]
    assert "unknown method" in replies[2]["error"]
    assert replies[3]["id"] == 2 and "error" in replies[3]  # no engine loaded, reported not crashed
    assert replies[4] == {"id": 3, "result": {"bye": True}}
    assert len(replies) == 5
    assert proc.returncode == 0


def test_stray_prints_cannot_corrupt_the_protocol(tmp_path):
    # A fake adapter that prints to stdout while loading, like many real libraries do.
    plugin = tmp_path / "noisy_plugin"
    plugin.mkdir()
    (plugin / "__init__.py").write_text("")
    (plugin / "adapter.py").write_text(
        "print('loading weights... 100%')\n"
        "class Adapter:\n"
        "    def __init__(self, engine, device):\n"
        "        print('hello from the engine'); self.sample_rate = 16000\n"
    )
    code = (
        "import sys, runpy; sys.argv=['main.py'];"
        f"sys.path.insert(0, {str(tmp_path)!r});"
        "import types; sys.modules['torch'] = types.SimpleNamespace(set_num_threads=lambda n: None);"
        f"runpy.run_path({str(manager.worker_script())!r}, run_name='__main__')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        input=json.dumps({"id": 1, "method": "load", "params": {"engine": "x", "family": "noisy_plugin.adapter"}}) + "\n",
        capture_output=True, text=True, timeout=60,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    parsed = [json.loads(ln) for ln in lines]  # would raise if a print leaked into stdout
    assert parsed[1]["result"]["loaded"] == "x"
    assert "hello from the engine" in proc.stderr


def test_model_checksum_mismatch_leaves_nothing_behind(monkeypatch, tmp_path):
    monkeypatch.setenv("VOICESMITH_MODELS", str(tmp_path / "models"))
    def fake_download(url, dest, expected, progress):
        dest.write_bytes(b"not the model")
    monkeypatch.setattr(models, "_download", fake_download)
    try:
        models.ensure("vad")
    except RuntimeError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("expected a checksum failure")
    assert not models.is_ready("vad")
    assert not models.MODELS["vad"].path.exists()
