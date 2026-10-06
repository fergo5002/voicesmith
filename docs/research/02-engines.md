# Which cloning engines, as of 6 October 2026

Research notes behind voicesmith's engine choices. Labels: **[S]** sourced (list at the end, all read 2026-10-06), **[D]** derived from sourced numbers, **[G]** a guess not yet tested. voicesmith's own measurements live in [`../benchmarks.md`](../benchmarks.md) and override anything here.

Two similarity scales appear below and neither should be trusted alone. **Vendor SIM** is each maker's self-reported speaker similarity on Seed-TTS-eval English (x100). **Bench SIM** is tts-bench, a community benchmark that scores every model with one similarity model but only five prompts each [2]. Speeds are tts-bench's "times faster than real time" on an Apple M4 CPU and an RTX 3090 [3].

## Engines with permissive or near-permissive weights

| Engine | Params | Weights | Languages | Reference | Vendor SIM | Bench SIM | M4 CPU / 3090 | Notes |
|---|---|---|---|---|---|---|---|---|
| Chatterbox Turbo (Resemble, Dec 2025) | 350M | MIT | EN | over 5 s, uses the first 10 to 15 s, no transcript | n/a | 0.666 | 1.14x / 4.66x | PerTh watermark built in; pins torch 2.6.0 and transformers 5.2.0 [7][9][10] |
| Chatterbox Nano (Jul 2026) | 110M | MIT | EN | about 10 s | n/a | n/a | "3x real time on 8 CPU threads" (vendor) | same package, git install [11][12] |
| Chatterbox Multilingual V3 (Jun 2026) | 500M | MIT | 23+ | as Turbo | n/a | n/a | n/a | watermark built in [14] |
| LongCat-AudioDiT 1B / 3.5B (Meituan, Mar 2026) | 1B / 3.5B | MIT | EN, ZH | transcript required | 76.2 / 78.6 | **0.870** / 0.834 | n/a / 8.65x | research drop, CUDA documented only [16] |
| AuK / AuK-Flash (Tencent, Sep 2026) | 1.5B | MIT | multilingual | transcript optional | n/a | 0.824 / 0.851 | n/a / 9.49x | 17.5 to 25 GB peak memory [17] |
| dots.tts (RedNote, Jun 2026) | 2B | Apache-2.0 | 24 | about 10 s | 77.1 / 80.0 | 0.718 | n/a / 0.81x | [18] |
| VoxCPM2 (OpenBMB, Apr 2026) | 2B | Apache-2.0 | 30 | transcript optional | 75.3 | 0.533 | 0.17x / 2.10x | official LoRA [19] |
| Qwen3-TTS 1.7B / 0.6B (Alibaba, Jan 2026) | 1.7B / 0.6B | Apache-2.0 | 10 | 3 s+, transcript for best results | 71.7 | 0.604 | 0.24x / 3.04x | GGUF, MLX, OpenVINO [20][21] |
| CosyVoice 3 0.5B | 0.5B | Apache-2.0 | 9 | one clip | 71.8 | 0.723 | n/a / 1.99x | no pip package, Linux docs [25] |
| MOSS-TTS-Nano | 100M | Apache-2.0 | 20 | about 3 s | n/a | 0.642 | 1.93x / 2.08x | built for CPU [26] |
| Sopro V2 Turbo (Aug 2026) | 120M | Apache-2.0 | EN, PT, FR, DE | 5 to 20 s | 65.5 (vendor) | 0.629 | about 4x on M3 CPU (vendor) | int8 on CPU [27] |
| Pocket TTS (Kyutai) | 100M | CC-BY-4.0, gated | 6 to 7 | one clip | n/a | 0.513 | 7.82x (Ryzen) | [28] |
| IndexTTS2 | 1.5B | bilibili licence (restricted) | ZH, EN | clip plus emotion clip | 70.6 | 0.810 | 0.14x / 1.08x | [29] |

## Non-commercial or restricted (opt-in at most)

F5-TTS and E2-TTS (code MIT, **weights CC-BY-NC** because of Emilia training data), OmniVoice (weights CC-BY-NC), XTTS-v2 (CPML), Fish Audio S2 Pro and OpenAudio S1-mini (research and NC-SA licences), Spark-TTS (moved to CC-BY-NC-SA), MaskGCT and Llasa (CC-BY-NC), Echo-TTS (NC-SA terms cover outputs), Higgs Audio v3 (non-commercial), Breeze TTS 2 (non-commercial; top of Artificial Analysis's open-weights board), Voxtral TTS (CC-BY-NC and the cloning encoder is withheld) [31] to [40].

## Ruled out

- **Cannot clone:** Kokoro, KittenTTS, Magpie Multilingual, Orpheus (no official zero-shot), VibeVoice (TTS code pulled), MegaTTS3 (encoder withheld) [41] to [50].
- **Clones poorly:** OpenVoice v2 (bench 0.242), NeuTTS.
- **Too slow or too narrow for a default:** Zonos, Sesame CSM-1B, Dia2, Step-Audio-EditX, DramaBox.

## What voicesmith ships with, and why

- **Chatterbox (Turbo, Nano, Multilingual).** Best-placed commercially usable cloner on the July 2026 controlled-voice arena, MIT weights, watermark built in, runs on CPU, CUDA and MPS.
- **Qwen3-TTS (0.6B, 1.7B).** Apache-2.0, transcript-conditioned cloning, and in the private predecessor it beat Chatterbox Turbo, IndexTTS2 and F5-TTS on held-out identity for the target speaker. Handles longer text in one pass.
- **Sopro V2 Turbo.** Apache-2.0, 120M parameters, int8 on CPU; the fast option for weak machines.

Every engine is scored by voicesmith's own evaluator on the voice being cloned, and `voicesmith tune` picks per voice. Published rankings disagree with each other (dots.tts is 80.0 on its own report and 0.718 on the community bench), so the user's own speaker decides.

Candidates for the next adapters: LongCat-AudioDiT 1B for CUDA machines, VoxCPM2 for LoRA fine-tuning, MOSS-TTS-Nano and Pocket TTS for CPU.

## Multiple references and fine-tuning

- Similarity rises with reference length and flattens around 10 s (VALL-E 2, Voicebox) [51]. Chatterbox throws away anything past its window [10].
- Only XTTS-v2 averages a list of references natively [52], and its weights are non-commercial. Everywhere else, "multiple references" means trying several and keeping the best, which is what `tune` does.
- Fine-tuning with 10+ minutes of audio: VoxCPM2 has official LoRA (about 20 GB VRAM). Baseten found Qwen3-TTS fine-tuning "didn't materially beat zero-shot on similarity", with gains in prosody [24]. Treat it as a CUDA-only prosody upgrade.

## Sources

1. VoxCPM2 tech report: https://arxiv.org/html/2606.06928v1 · 2. tts-bench scores: https://5uck1ess.github.io/tts-bench/scores.html · 3. tts-bench speed: https://5uck1ess.github.io/tts-bench/speed.html · 4. Artificial Analysis: https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice/open-weights · 5. Pinggy roundup: https://pinggy.io/blog/best_open_source_self_hosted_text_to_speech_models/ · 7. https://github.com/resemble-ai/chatterbox · 9. https://raw.githubusercontent.com/resemble-ai/chatterbox/master/pyproject.toml · 10. https://raw.githubusercontent.com/resemble-ai/chatterbox/master/src/chatterbox/tts_turbo.py · 11. https://www.resemble.ai/resources/chatterbox-nano-and-flash-speed-at-the-edge-throughput-at-scale · 12. https://huggingface.co/ResembleAI/chatterbox-nano · 14. https://www.resemble.ai/resources/chatterbox-multilingual-v3-tts-with-embedded-watermarking-for-25-languages · 16. https://github.com/meituan-longcat/LongCat-AudioDiT · 17. https://github.com/Tencent-Hunyuan/AuK · 18. https://arxiv.org/abs/2606.07080 · 19. https://github.com/OpenBMB/VoxCPM · 20. https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base · 21. https://github.com/huggingface/optimum-intel/pull/1765 · 24. https://www.baseten.co/blog/fine-tuning-qwen3-tts-for-high-quality-voice-cloning/ · 25. https://github.com/FunAudioLLM/CosyVoice · 26. https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano · 27. https://huggingface.co/samuel-vitorino/sopro-v2-turbo · 28. https://github.com/kyutai-labs/pocket-tts · 29. https://github.com/index-tts/index-tts · 31. https://github.com/SWivid/F5-TTS · 32. https://huggingface.co/k2-fsa/OmniVoice · 33. https://huggingface.co/fishaudio/s2-pro · 34. https://huggingface.co/mistralai/Voxtral-4B-TTS-2603 · 37. https://huggingface.co/coqui/XTTS-v2 · 38. https://huggingface.co/SparkAudio/Spark-TTS-0.5B · 41. https://github.com/bytedance/MegaTTS3 · 42. https://github.com/myshell-ai/OpenVoice · 50. https://github.com/Blaizzy/mlx-audio · 51. https://arxiv.org/pdf/2406.05370 · 52. Coqui XTTS multi-reference source.
