# From messy recordings to a voice profile

Research behind `voicesmith ingest`. Labels: **[S]** sourced (read 2026-10-06), **[O]** read in an engine's own source, **[G]** a guess to be calibrated.

## The headline findings

1. **Choose good segments rather than repairing bad ones.** Two papers found that running a speech enhancer over a usable prompt lowers the clone's speaker similarity [S14][S15]. Enhancement only helps when the prompt is genuinely noisy [S16]: on noisy prompts, MP-SENet lifted similarity MOS from 2.80 to 3.85 and cut WER from 21.4% to 2.5%. So voicesmith rejects bad segments and does not enhance by default.
2. **Run cheap analysis on everything and expensive work on a shortlist.** Demucs separation is about 1.5x real time on CPU [S9]. Sound-event tagging runs only on the shortlist.
3. **Use the consent recording as the enrolment anchor.** Microsoft and ElevenLabs both check that the consenting voice matches the training audio [S27][S28]. voicesmith uses the same embedding to keep only that speaker's segments from a multi-speaker podcast.
4. **Reference windows are engine-specific and partly hard-coded** [O]: Chatterbox Turbo feeds only 10 s to its decoder; F5 clips at about 12 s; IndexTTS2 truncates at 15 s. Keep clips 4 to 14 s and let each engine use what it uses.

## Pipeline as built

| Stage | Tool | Why |
|---|---|---|
| Acquire | yt-dlp `bestaudio` with no re-encode; local files; microphone via sounddevice | Never transcode lossy to lossy [S1] |
| Decode | ffmpeg to mono float | Independent of libsndfile and torchaudio codec support |
| Voice activity | Silero VAD via sherpa-onnx | MIT, under 1 ms per 30 ms chunk on one thread [S6] |
| Transcript and word timings | Parakeet TDT 0.6B v3 int8 via sherpa-onnx | CC-BY-4.0, 25 languages, native timestamps, about 36x real time on a desktop CPU [S18][S19] |
| Clip cutting | Word gaps of at least 120 ms at both ends, sentence ends preferred, 4 to 14 s | Cuts never land mid-word |
| Speaker filter | TitaNet-small embeddings, anchored on consent or the dominant voice | One mechanism for consent matching and target filtering |
| Gates | clipping, bandwidth (95% roll-off), SNR, ASR confidence, voiced speech | Thresholds in `ingest/pipeline.py` |
| Score | 0.40 centrality + 0.20 clarity + 0.15 SNR + 0.15 prosodic typicality + 0.10 length fit, as within-pool robust z-scores | Absolute values shift with source and encoder; ranks do not |
| Shortlist | Diversity: no two clips within 60 s of each other or above 0.97 cosine | Gives `tune` real alternatives |
| Screen | CED-mini AudioSet tagger for music, laughter, applause, crowd, TV | Clones inherit what is in the prompt |

## Evidence on reference length and variety

- Mega-TTS 2 improved from 3 to 10 s prompts and dropped at 20 s [S33]. A WildSpoof 2026 system found 7.7 s prompts beat 5.5 s [S15].
- Repeating a prompt raises VoiceStar's similarity but sharply worsens F5's WER [S34]; concatenation buys nothing on engines that truncate.
- Expressive Prompting picks prompts in two stages: a static quality and similarity score, then per-text selection [S36]. voicesmith's `tune` is the static stage; per-text selection is future work.
- Default to a typical reference, near the speaker's median pitch and rate [G].

## Fine-tuning versus zero-shot

LoRA on a Qwen-0.5B-based TTS with 3.5 to 18.5 hours per speaker raised similarity from 0.73 to 0.75 up to 0.80 to 0.82 on one 24 GB GPU, and made DNSMOS worse when the data lacked variation [S41]. VoxCPM 2 LoRA needs about 20 GB VRAM [S42]. Zero-shot with a good reference bank is the default at every data size; fine-tuning is future work for CUDA machines only.

## Loudness

References are stored mono 24 kHz at about -23 dB RMS with 15 to 40 ms fades. Outputs are mastered to -16 LUFS integrated with a -1.5 dBTP ceiling (AES TD1008 for spoken word) [S43] by one linear gain.

## Sources

S1 https://github.com/yt-dlp/yt-dlp · S6 https://github.com/snakers4/silero-vad · S9 https://github.com/adefossez/demucs · S14 https://arxiv.org/html/2406.05699 · S15 https://arxiv.org/html/2602.05770 · S16 https://arxiv.org/html/2505.13830 · S18 https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 · S19 https://github.com/istupakov/onnx-asr · S27 https://learn.microsoft.com/en-us/azure/ai-services/speech-service/personal-voice-create-consent · S28 https://elevenlabs.io/docs/eleven-api/guides/how-to/voices/professional-voice-cloning · S33 https://arxiv.org/pdf/2307.07218 · S34 https://arxiv.org/pdf/2505.19462 · S36 https://arxiv.org/abs/2409.18512 · S41 https://arxiv.org/html/2603.10904 · S42 https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html · S43 AES TD1008 via https://thepodcasthost.com/recording-skills/how-loud-should-a-podcast-be/
