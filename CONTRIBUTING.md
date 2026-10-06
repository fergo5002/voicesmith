# Contributing

Thanks for helping. A few things matter more than usual in a voice cloning project.

## Ground rules

- **Safeguards stay.** Pull requests that add a way around consent, the watermark or the disclosure tags will be closed. If a safeguard gets in the way of a legitimate use, open an issue so we can find a better design.
- **Measure, then claim.** Speed and quality claims in code comments and docs need numbers from `scripts/bench_engines.py` (or a described test), with the hardware they came from.
- **Keep the core light.** `src/voicesmith` must not import PyTorch. Anything that needs torch belongs in a worker adapter under `src/voicesmith/engines/_worker/`, which must not import `voicesmith`.

## Setup

```bash
git clone https://github.com/fergo5002/voicesmith && cd voicesmith
uv venv && uv pip install -e ".[dev]"
uv run pytest -q -m "not slow"     # fast unit tests, no downloads
uv run pytest -q -m slow           # needs the ONNX models (about 530 MB, fetched once)
uv run ruff check src tests scripts
```

## Adding an engine

1. Check the **weights** licence, not just the code licence. Non-commercial weights cannot be a default.
2. Add a `Family` (its environment) and an `Engine` entry in `src/voicesmith/engines/registry.py`.
3. Add an adapter in `src/voicesmith/engines/_worker/adapters/` with an `Adapter(engine, device)` class exposing `sample_rate` and `synthesize(text, ref_wav, ref_text, options)`. Cache any per-reference conditioning.
4. Run `scripts/bench_engines.py` with your engine and add the numbers to `docs/benchmarks.md`.

The worker can already load an adapter from any importable module path (a family name containing a dot), which is the hook for out-of-tree engines. Registering such engines from configuration, without editing `registry.py`, is not built yet.
