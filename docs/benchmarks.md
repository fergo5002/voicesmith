# Benchmarks

Everything here was measured, on the machine and under the conditions stated. Re-run with [`scripts/bench_engines.py`](../scripts/bench_engines.py) on your own hardware; vendor speed claims did not survive contact with ours.

## Machine

Windows 11 laptop, Intel Core i7-13700H (6 performance and 8 efficiency cores), 32 GB RAM, Intel Iris Xe graphics (not used), no NVIDIA GPU. **The machine was shared with other heavy work during these runs** (70 to 80% total CPU busy before our jobs started, and memory often under 4 GB free), so absolute speeds are pessimistic. Relative speeds between engines run back to back are more trustworthy than the absolute numbers.

## Engines on LibriSpeech test-clean (6 October 2026)

Six speakers (three male, three female). For each, one utterance of about 10 s is the reference and two other utterances' text is rendered, so every render has a real recording of the same sentence to compare against. One seed per case, 12 renders per engine, CPU only.

- **WER / CER:** Parakeet TDT 0.6B v3 transcript of the render against the script.
- **SIM:** TitaNet-small cosine between the render and the centroid of up to 20 *other* recordings of the speaker. This is not the Seed-TTS-eval WavLM metric, so these numbers are not comparable with published tables.
- **RTF:** generation seconds per second of audio (lower is faster; 1.0 is real time).

| Engine | WER | CER | SIM (held-out) | worst SIM | RTF |
|---|---|---|---|---|---|
| Real recordings of the same sentences (ceiling) | 2.2% | 0.8% | **0.896** | 0.866 | n/a |
| Sopro V2 Turbo, int8 | 1.4% | 0.1% | 0.759 | 0.602 | **2.2** |
| Sopro V2 Turbo, fp32 | 1.4% | 0.4% | 0.759 | 0.602 | 5.3 |
| Qwen3-TTS 0.6B Base | 1.1% | 0.3% | 0.752 | 0.625 | 10.2 |
| Chatterbox Turbo | **0.5%** | 0.2% | 0.747 | 0.668 | 11.4 |
| Chatterbox Nano | 3.2% | 2.0% | 0.733 | 0.627 | 3.8 |

What this does and does not show:

- **The engines are close on similarity.** With 12 renders each and one seed, differences of 0.01 to 0.03 are within noise. The large and real gap is between every engine and the speaker's own recordings (about 0.75 against 0.90). Closing that gap is what reference selection, per-voice tuning and best-of-N are for.
- **Clones match their reference clip more closely than the speaker's real recordings do** (mean cosine to the reference: 0.82 to 0.84 for clones, 0.74 for real recordings). They copy that one clip's room and delivery. This is why voicesmith scores against held-out audio rather than against the reference: scoring against the reference would reward copying a recording over sounding like the person.
- **Speeds were measured under contention.** The same Chatterbox Turbo took about RTF 4 on this laptop in July 2026 when it was quieter. Treat the RTF column as a ranking, not a promise.
- **Not measured yet:** GPU speeds, Apple Silicon, Chatterbox Multilingual, Qwen3-TTS 1.7B, and more than one seed per case.

## Ingest

A 12.3 minute synthetic two-speaker "podcast" built from LibriSpeech: the target speaker (68% of the speech), a second speaker (32%), a music bed under 0.8 minutes of the target's speech, and noise added to 0.4 minutes.

| Stage | Time |
|---|---|
| Decode, speech detection and speaker fingerprinting of 9.9 min of speech (148 regions) | 10 s |
| Transcription of the best 5.3 minutes | 68 s |
| Clip measurement, scoring, sound-event screen, writing references | 7 s |
| **Total** | **95 s** |

- Target speaker identified at 67% of the speech (truth 68%); 3.4 minutes of the other speaker set aside (truth 3.6).
- 8 of 8 selected references were the target speaker, checked against the ground-truth timeline. None came from the music or noise stretches.
- Peak memory about 3 GB. An early version that batched long regions for transcription peaked at 16 GB; ingest now caps transcription at 10 minutes of the best speech, so its cost no longer grows with input length.

## End-to-end render

`voicesmith say` on the ingested voice, Sopro, `--quality fast`, one 13-word sentence: 4.7 s of audio in 46 s wall time including model load, first take accepted (q 0.95, WER 0%, similarity 0.88 to the voice's held-out clips). Delivered as M4A, WAV and Ogg: AudioSeal watermark detected at probability 1.0 with the correct payload in all three after encoding; a real LibriSpeech recording used as a control read 0.0. Loudness came out at -18.8 LUFS rather than the -16 target, because the gain is linear and stops at the -1.5 dBTP ceiling instead of limiting.
