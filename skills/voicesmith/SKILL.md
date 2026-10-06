---
name: voicesmith
description: Render speech in a consented, locally cloned voice with the voicesmith CLI or MCP server, and help a human set up a new voice (create, consent, ingest, tune). Use when the user wants a voice note, narration, or any audio in a specific person's cloned voice, or wants to clone a voice they have permission to use.
---

# voicesmith

voicesmith clones voices locally and only renders voices that have a consent record. Every output is watermarked and tagged as AI-generated.

## Before anything

Run `voicesmith doctor --quick`. If no engine is installed, tell the user to run `voicesmith engines install recommended` (it downloads several GB).

## Rendering speech

```bash
voicesmith voices --json                      # what exists, and which are ready
voicesmith say <voice> "Text to speak" -o out.m4a --json
voicesmith say <voice> -f script.txt -o out.wav -o out.mp3 --quality best --json
```

- `--quality fast|balanced|best` trades time for more candidate takes. `balanced` is the default.
- Each take is transcribed and voice-matched; a take that drops words, loops, or sounds like someone else is rejected and retried.
- The JSON result has `files`, `manifest`, `score.similarity` and `score.wer`. Report those numbers honestly; do not call a take perfect.
- Over MCP, call `speak`. If it returns `"status": "running"`, poll `job_status` with the `job_id`.

## Setting up a new voice (the human must do the consent step)

```bash
voicesmith voice create <name> --speaker "Full Name"
voicesmith consent request <name>      # prints a statement with a one-off code
#   the speaker reads it aloud (a phone voice note is fine), then:
voicesmith consent verify <name> recording.m4a
voicesmith ingest <name> talk1.mp3 ./folder https://youtu.be/...   # 5+ minutes of clean solo speech is plenty
voicesmith tune <name>                 # picks the best engine and reference; run once
```

If the speaker cannot record a statement but has authorised the use another way, the human can run `voicesmith consent attest <name> --by "Their Name" --evidence "how and when"`.

## Rules

- Never create, attest or bypass consent on the user's behalf. If a voice lacks consent, explain the steps and stop.
- Never strip or disguise the watermark or disclosure tags, and never present output as a real recording.
- Do not clone a voice when the user indicates the speaker has not agreed.
- Keep outputs where the user asked; do not upload them anywhere.

## Troubleshooting

- "no usable consent": the consent step above has not been done.
- Slow renders: `voicesmith doctor` shows the device in use; CPU-only machines run at a few times slower than real time. Use `--quality fast` for drafts.
- YouTube downloads failing: yt-dlp needs a JavaScript runtime such as deno.
- Logs: `~/.voicesmith/logs/`.
