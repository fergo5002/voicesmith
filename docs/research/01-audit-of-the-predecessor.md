# Audit of the private predecessor

voicesmith replaces a private, single-speaker voice cloning workflow built in July 2026 for one authorised speaker. That project went through three versions over two days. This is what it did, what worked, and what did not. All figures come from its own run logs.

## What it did

| Version | Engine | Corpus | Result |
|---|---|---|---|
| v1 | Chatterbox Turbo, CPU | A few hand-cut 10 to 13 s clips from YouTube | 1.98% WER, 0.863 speaker similarity (Chatterbox encoder and ECAPA, not comparable with Seed-TTS-eval numbers) |
| v2 | Qwen3-TTS 0.6B Base, beating IndexTTS2, Chatterbox and F5-TTS in a bake-off | 223 videos (170.5 hours) archived; 120 minutes of target-only speech kept from 496 verified clips | 1.75% WER, 0.905 held-out ECAPA identity after mastering |
| v3 | Qwen3-TTS 0.6B and 1.7B vs CosyVoice3 instruct, 21-candidate tournament | Same corpus plus "style banks" of warm, persuasive and playful references | 0.94% WER, 0.905 to 0.943 identity after encoding, 81 tests, one fail-closed command |

## What worked (kept in voicesmith)

- **Filtering the corpus to the target speaker with speaker embeddings.** Going from 170 hours of mixed audio to 2 hours of clean target speech was the biggest single quality lever. voicesmith does this automatically, anchored on the consent recording when there is one.
- **Transcript-conditioned cloning.** Qwen3-TTS in in-context mode, given the exact transcript of the reference, beat the same model in embedding-only mode. voicesmith always has the transcript because it transcribes every clip.
- **One continuous take beats stitched paragraphs.** v2 rendered paragraphs separately and joined them with fixed silence on top of the model's own edge silence. Joins reached 0.69 s, one boundary reset energy by 7.5 dB and pitch by 3.6 semitones. The listener heard "weird pauses". voicesmith renders the whole text in one pass whenever the engine allows it, and when it must split, the pause is the total gap, sized by punctuation and filled with the take's own room tone.
- **Hard gates before ranking.** Wording, identity, truncation and artefact checks rejected bad takes before any ranking happened. voicesmith keeps this order and makes it cheaper: cheap checks run first and a take stops being scored the moment it fails.
- **Verifying the encoded file, not the WAV it came from.** v3 caught real failures this way (codec loaders that could not open M4A, metadata that did not survive). voicesmith decodes every delivered file and re-checks the watermark, the tags and the loudness, and deletes the file if any check fails.
- **A token ceiling.** One v3 take entered a runaway sampling loop and ran for over 15 minutes. voicesmith caps generation length from the text length.
- **Decoding everything through ffmpeg.** torchaudio and libsndfile both failed on some containers on Windows. voicesmith never lets an engine open the user's media directly.

## What did not work (fixed in voicesmith)

- **Hard-wired to one person.** The copy, the required phrases, the thresholds and even one catchphrase repair were specific to one speaker and one message. voicesmith has no speaker-specific code.
- **Six hand-built Python environments, 55 GB.** Each engine pinned different torch and transformers versions, and several needed Windows fixes (`setuptools<81` for `pkg_resources`, a console encoding crash, TorchCodec DLLs). voicesmith creates one uv-managed environment per engine family on demand, with the fixes built in, and keeps the core free of PyTorch.
- **Slow.** On the same Windows laptop, a 29 s take took 2 to 6 minutes with Qwen 0.6B, 3 minutes with 1.7B, and 15 minutes with IndexTTS2. The evaluator alone took over 9 minutes for 16 candidates. The tournament ran for every message. voicesmith tunes each voice once, caches each engine's speaker conditioning, keeps workers warm, and stops a best-of-N search as soon as a take is good enough.
- **Manual reference selection in v1.** The first version picked clips by hand. voicesmith scores and selects references automatically.
- **Heuristics that overfit.** Several v3 repairs (a phonetic stretch of one word, emphasis windows tuned to one ending) solved one message rather than the problem. voicesmith avoids message-specific repairs.
- **No consent mechanism.** Authorisation lived in a chat message. voicesmith will not render without a consent record, and spoken consent is verified against the voice.
- **Not callable from a fresh session.** The workflow lived in one folder with PowerShell scripts and instructions to paste into a new chat. voicesmith is a CLI, an MCP server and an agent skill.
