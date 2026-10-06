# voicesmith: notes for coding agents

## Layout

- `src/voicesmith/` is the core. It never imports PyTorch.
  - `analysis/` ONNX speech analysis via sherpa-onnx: VAD, ASR with word timings, speaker embeddings, sound events, pitch.
  - `ingest/` acquisition (files, yt-dlp, microphone) and the reference-selection pipeline.
  - `engines/` the engine registry, uv environment manager and worker client.
  - `engines/_worker/` runs **inside** engine environments. It must import nothing from `voicesmith`; it has only the engine's own dependencies plus numpy, soundfile and torch.
  - `synth/` evaluation, best-of-N rendering, joins and mastering.
  - `consent.py`, `provenance.py`, `tune.py`, `cli.py`, `mcp_server.py`, `doctor.py`.
- `tests/` fast unit tests that need no models or engines. Mark anything heavier with `@pytest.mark.slow`.
- `docs/research/` why things are the way they are. `docs/benchmarks.md` measured numbers.

## Commands

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest -q
uv run ruff check src tests
```

## Rules for changes

- Keep the core free of torch. Anything needing torch goes in a worker adapter.
- Never add a way to skip consent, the watermark, or the disclosure tags.
- Decode media through `voicesmith.ffmpeg`, never through an engine's own loader.
- A new engine needs: a `Family` and `Engine` entry in `engines/registry.py`, an adapter in `engines/_worker/adapters/`, a licence check (weights, not just code), and measured numbers in `docs/benchmarks.md` from `scripts/bench_engines.py`.
- Claims in docs must say how they were measured and on what machine.
