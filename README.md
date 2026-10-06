# voicesmith

Local voice cloning that asks for consent, picks its own best settings for each voice, checks every take before you hear it, and plugs straight into coding agents.

```bash
uv tool install git+https://github.com/fergo5002/voicesmith
voicesmith doctor
voicesmith engines install recommended
```

- **Runs on your machine.** CPU laptops, NVIDIA GPUs and Apple Silicon. Nothing is uploaded.
- **Hours of messy audio in, clean references out.** Podcasts, videos, voice notes, YouTube links. voicesmith finds the speech, transcribes it, keeps only the target speaker, throws out clips with music, laughter, clipping or noise, and picks a varied set of the best references.
- **Tuned per voice.** `voicesmith tune` tries every installed engine with the strongest references and keeps the combination that sounds most like the person. You pay that cost once.
- **Every take is checked.** Each render is transcribed and voice-matched. Takes that drop words, loop, trail off, leave dead air or drift to another voice are thrown away and re-rendered, before you hear anything.
- **Consent first.** A voice will not render until the speaker has read a consent statement (checked by speech recognition and voice match) or the operator has recorded who authorised it.
- **Marked and labelled.** Every file carries an AudioSeal watermark and AI-disclosure tags, and is decoded and re-checked after encoding. If the mark or tags did not survive, the file is not delivered.
- **Agent-ready.** An MCP server and an agent skill, so Claude Code, Codex, Cursor and others can speak in a consented voice.

## Quick start

```bash
# 1. Install (needs Python 3.10 to 3.13; uv fetches one if you lack it. No uv? https://docs.astral.sh/uv/)
uv tool install git+https://github.com/fergo5002/voicesmith
voicesmith doctor                       # checks ffmpeg, models, hardware, and tells you how to fix anything
voicesmith engines install recommended  # Chatterbox; adds Qwen3-TTS on NVIDIA machines

# 2. Create a voice and get consent
voicesmith voice create sam --speaker "Sam Example"
voicesmith consent request sam          # prints a statement with a one-off code
#    Sam reads it aloud. A voice note recorded on a phone is fine.
voicesmith consent verify sam sam-consent.m4a

# 3. Give it audio of Sam speaking (5 to 30 minutes of solo speech is plenty)
voicesmith ingest sam interview.mp3 ./more-recordings https://www.youtube.com/watch?v=...

# 4. Let it find the best engine and reference for Sam (run once)
voicesmith tune sam

# 5. Speak
voicesmith say sam "Thanks for coming in yesterday. Here's the plan." -o note.m4a
```

Every render writes a JSON manifest next to the audio with the scores, engine, reference, seed and watermark check.

## Use it from an agent

**Claude Code**

```bash
claude mcp add voicesmith -- voicesmith mcp
```

**Codex** (`~/.codex/config.toml`)

```toml
[mcp_servers.voicesmith]
command = "voicesmith"
args = ["mcp"]
```

**Cursor, Windsurf and others:** add a stdio server whose command is `voicesmith mcp`.

Tools: `list_voices`, `voice_info`, `speak`, `job_status`, `verify_audio`, `consent_steps`, `doctor`. `speak` returns file paths and quality scores. Long renders come back as a job id to poll, so agents with short tool timeouts still work. Agents cannot create consent; `consent_steps` tells the human what to do.

There is also a skill at [`skills/voicesmith/SKILL.md`](skills/voicesmith/SKILL.md) for agents that use skills. Copy it into your agent's skills folder (for Claude Code, `~/.claude/skills/voicesmith/`).

## How it works

```
sources ──► decode (ffmpeg) ──► speech detection (Silero) ──► transcript + word timings (Parakeet)
        ──► clips cut at pauses, 4 to 14 s ──► speaker embeddings (TitaNet)
        ──► keep the consenting / dominant speaker ──► gates: clipping, bandwidth, noise, clarity
        ──► score + diversity ──► sound-event screen (music, laughter...) ──► reference bank

tune:   every installed engine × top references × probe sentences ──► best (engine, reference) per voice

say:    one continuous take where the engine allows it
        ──► best-of-N with early stop: length, transcript, ending, loops, dead air, voice match, prosody
        ──► join (if split) with punctuation-sized pauses and room tone
        ──► trim, linear loudness to -16 LUFS / -1.5 dBTP ──► AudioSeal watermark
        ──► encode with disclosure tags ──► decode and re-verify every file ──► deliver
```

The core never imports PyTorch: all analysis runs on ONNX Runtime. Each synthesis engine lives in its own uv-managed environment, because engines pin conflicting versions of torch and transformers, and talks to the core over a small JSON protocol. Workers stay warm between renders inside the MCP server.

Design notes and sources: [`docs/research/`](docs/research/). Measured numbers: [`docs/benchmarks.md`](docs/benchmarks.md).

## Engines

| Engine | Weights licence | Languages | Why it is here |
|---|---|---|---|
| `chatterbox-turbo` | MIT | English | Best commercially usable cloner on the 2026 controlled-voice arena; built-in PerTh watermark |
| `chatterbox-nano` | MIT | English | 110M-parameter Turbo for slow machines |
| `chatterbox-multilingual` | MIT | 23 languages | Multilingual V3 |
| `qwen3-tts-0.6b` / `qwen3-tts-1.7b` | Apache-2.0 | 10 languages | Transcript-conditioned cloning, strong identity, long text in one pass |
| `sopro` | Apache-2.0 | EN, PT, FR, DE | 120M, int8 on CPU, the quick option |

Only engines whose weights allow commercial use ship by default. `voicesmith engines list` shows what is installed. The weights belong to their makers and carry their own licences.

## Hardware

voicesmith detects what it is running on and installs the right PyTorch build for each engine: CUDA wheels when there is an NVIDIA GPU, the default wheels on macOS (MPS), and CPU wheels everywhere else. It deliberately avoids uv's automatic backend on Intel-graphics laptops, because that picks Intel XPU wheels for GPUs PyTorch XPU does not support.

Be realistic about CPU-only machines: cloning models run several times slower than real time on a laptop CPU (see the benchmarks). Use `--quality fast` for drafts there, and expect a GPU to be an order of magnitude quicker.

## Consent and safety

Read [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md). In short: clone only people who agreed, never pass output off as a real recording, and know that this is MIT code, so the safeguards make the honest path easy rather than making abuse impossible.

## Commands

| Command | What it does |
|---|---|
| `voicesmith doctor` | Check the setup and print fixes |
| `voicesmith engines list / install / remove` | Manage engine environments |
| `voicesmith voice create / show / delete` | Manage voices |
| `voicesmith voices` | List voices |
| `voicesmith consent request / verify / attest / show / revoke` | Consent records |
| `voicesmith ingest <voice> <files, folders, URLs> [--record SECONDS]` | Add audio and rebuild references |
| `voicesmith tune <voice>` | Pick the best engine and reference for a voice |
| `voicesmith say <voice> "text" -o out.m4a [--quality fast/balanced/best]` | Render speech |
| `voicesmith verify <file>` | Check a file for the watermark and disclosure tags |
| `voicesmith mcp` | Run the MCP server on stdio |

Everything lives in `~/.voicesmith` (override with `VOICESMITH_HOME`).

## Troubleshooting

- **First run is slow.** Engines download several GB of weights on first use. On Windows, Defender scans each new model file; adding `~/.voicesmith` and `~/.cache/huggingface` to its exclusions speeds this up.
- **YouTube downloads fail.** yt-dlp now needs a JavaScript runtime for YouTube. Install [deno](https://deno.com).
- **"no usable consent".** Do the consent step. Agents cannot do it for you.
- **A render keeps failing its checks.** The manifest lists every take and why it was rejected. Try `voicesmith tune` again with more references, or ingest cleaner audio.
- **Logs** are in `~/.voicesmith/logs/`.

## Development

```bash
git clone https://github.com/fergo5002/voicesmith && cd voicesmith
uv venv && uv pip install -e ".[dev]"
uv run pytest -q
```

See [`AGENTS.md`](AGENTS.md) for the layout and the rules for new engines.

## Licence

MIT for this code. Models are downloaded from their makers and keep their own licences: Chatterbox (MIT), Qwen3-TTS (Apache-2.0), Sopro (Apache-2.0), Parakeet TDT and TitaNet (CC-BY-4.0, NVIDIA), Silero VAD (MIT), CED (Apache-2.0), AudioSeal (MIT).
