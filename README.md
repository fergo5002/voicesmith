# voicesmith

**Clone a voice on your own laptop, with the owner's permission, and let your AI talk in it.**

voicesmith turns recordings you already have (podcasts, videos, voice notes, a YouTube link) into a voice that Claude Code, Codex, Cursor or your own scripts can speak in. Nothing leaves your machine. It is fussy on purpose: every take gets transcribed and voice-matched before you hear it, and nothing renders at all until the person whose voice it is has said yes.

A real run, from a copy installed straight from this repo, on a CPU-only laptop (only the file paths are shortened):

```console
$ voicesmith say reader1089 "This was rendered by the copy installed straight from GitHub." --engine sopro --quality fast -o live.ogg -o live.m4a
14:48:22 loading Sopro V2 Turbo (Halo Research)
14:48:39 rendering 1 part(s) with reference ref02
14:48:55 part 1/1 take 1: q=0.93
14:48:55 mastering, watermarking and verifying
wrote live.ogg
wrote live.m4a
3.6s of audio in 43s with sopro (1 take(s)); similarity 0.874, word error 0.0%. Manifest: live.voicesmith.json
```

## Why it's different

- **It does the tedious bit for you.** Give it a podcast. It finds the speech, works out who is talking, sets aside the co-host, drops clips with music, laughter, applause or background chatter, and keeps a varied handful of the cleanest clips of the right person. If two people talk about equally and you haven't said which one you mean, it stops and asks instead of guessing.
- **It auditions itself.** `voicesmith tune` tries each installed engine that suits your machine and language against the best clips, and keeps the best-scoring engine and clip (mostly likeness, plus getting the words right). You do that once per voice; later renders use the winner unless you ask for speed with `--quality fast`.
- **It is its own harshest critic.** Each take is transcribed and voice-matched. Dropped words, loops, trailing off, long dead air, or a voice drifting towards someone else's: binned and re-rendered before you hear anything.
- **Consent or nothing.** A voice will not render until its owner has read a consent statement with a one-off code (checked by speech recognition and voice match), or you have recorded who authorised it and how. Through the MCP server, agents can use voices but cannot grant consent.
- **It signs its work.** Every file carries an AudioSeal watermark and "AI-generated" tags, and is decoded and re-checked after encoding. If either did not survive, the file is not delivered. `voicesmith verify` checks any file.
- **Your agent can use it.** One command adds it to Claude Code or Codex as an MCP server (Cursor takes a short config entry), so your assistant can make you voice notes in a voice you chose.
- **Built for ordinary laptops.** The core needs no GPU or PyTorch. Each engine gets the right PyTorch build for your machine (CUDA, Apple Silicon or CPU). So far it has been measured on a CPU-only Windows laptop; see [the benchmarks](docs/benchmarks.md) for honest numbers.

## Give your AI a voice

```bash
uv tool install git+https://github.com/fergo5002/voicesmith    # no uv? https://docs.astral.sh/uv/
voicesmith doctor                                               # checks everything, tells you how to fix it
voicesmith engines install recommended                          # engine weights (several GB) download on first use

claude mcp add --scope user voicesmith -- voicesmith mcp        # Claude Code
codex mcp add voicesmith -- voicesmith mcp                      # Codex (Cursor: see below)
```

Then set up a voice (yours is the obvious first one) with the five steps below, and ask your agent for "a voice note saying the tests passed".

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

Every render writes `<name>.voicesmith.json` next to the audio with the scores, the engine, reference and seed, every take that was tried and why any were rejected, and the post-encode watermark check.

**Several people talking?** If no one clearly dominates the recordings and there is no spoken consent to anchor on, ingest stops and lists the voices it found with example timestamps, rather than guessing. Re-run with `--pick 2`, or pass `--target clip.wav` with a few seconds of only the right person.

**Want the closest match you can get?** `--quality best` tries more takes, and `--polish` also runs Chatterbox's voice conversion over each take, keeping it only when it scores higher. Both cost time; see the benchmarks for what they buy.

## Use it from an agent

**Claude Code** (`--scope user` makes it available in every project, not just the current folder)

```bash
claude mcp add --scope user voicesmith -- voicesmith mcp
```

**Codex**: `codex mcp add voicesmith -- voicesmith mcp`, or add it to `~/.codex/config.toml`:

```toml
[mcp_servers.voicesmith]
command = "voicesmith"
args = ["mcp"]
```

**Cursor, Windsurf and others:** add a stdio server whose command is `voicesmith mcp`.

Tools: `list_voices`, `voice_info`, `speak`, `job_status`, `verify_audio`, `consent_steps`, `doctor`. `speak` returns file paths and quality scores. Long renders come back as a job id to poll, so agents with short tool timeouts still work. Agents write only new files inside `~/.voicesmith/outputs`, never anywhere else, and cannot create or change consent; `consent_steps` tells the human what to do. Engines an agent has not used for 15 minutes are unloaded to free memory.

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
        ──► trim, linear loudness towards -16 LUFS, capped at -1.5 dBTP ──► AudioSeal watermark
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

Be realistic about CPU-only machines: cloning models run between about 1.5x and 10x slower than real time on a laptop CPU (see the benchmarks). `--quality fast` uses the quickest engine whose tuned similarity is within 0.04 of the best. Expect a GPU to be much quicker; GPU speeds have not been measured for this release.

Loudness is set with one linear gain, aiming for -16 LUFS but never pushing true peaks past -1.5 dBTP, so speech with sharp peaks can come out a little quieter than -16. That is deliberate: a limiter would get closer to the number by squashing the voice.

## Consent and safety

Read [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md). In short: clone only people who agreed, never pass output off as a real recording, and know that this is MIT code, so the safeguards make the honest path easy rather than making abuse impossible.

## Commands

| Command | What it does |
|---|---|
| `voicesmith doctor` | Check the setup and print fixes |
| `voicesmith engines list / install / remove` | Manage engine environments |
| `voicesmith voice create / show / delete` | Manage voices |
| `voicesmith voices` | List voices |
| `voicesmith consent request / verify / attest / show / revoke` | Consent records (every change is kept in `consent/history.jsonl`) |
| `voicesmith ingest <voice> <files, folders, URLs> [--record SECONDS] [--pick N] [--target clip]` | Add audio and rebuild references |
| `voicesmith tune <voice>` | Pick the best engine and reference for a voice |
| `voicesmith say <voice> "text" -o out.m4a [--quality fast/balanced/best] [--polish]` | Render speech |
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
