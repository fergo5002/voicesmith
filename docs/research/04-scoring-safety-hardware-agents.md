# Scoring, safety, hardware and agents

Research behind voicesmith's evaluator, watermarking, hardware detection, packaging and MCP server. Labels: **[S]** sourced (read 2026-10-06), **[O]** checked directly on PyPI, GitHub or in source, **[G]** a guess.

## Scoring takes

- Seed-TTS-eval measures similarity with a fine-tuned WavLM-large speaker verifier (about 316M parameters) [S1]. On VoxSim's human similarity ratings, plain ECAPA-TDNN (LCC 0.768) did as well as WavLM-ECAPA (0.752) without fine-tuning [S2], so a small encoder is not a compromise.
- Speaker embeddings capture timbre and pitch range and miss rhythm [S3], hence voicesmith's separate prosody term.
- Quality-score models (UTMOS, DNSMOS) rank strong systems poorly (system-level SRCC 0.118 and 0.091 in one study) [S6]. They should reject, never pick the winner.
- voicesmith's gates, in cost order: signal (non-finite, clipping, implausible length), ASR (character error, missing last word, repeated phrases, dead air), similarity floor. Survivors are ranked by `Q = 0.50 * similarity (normalised between a stranger and the speaker's own self-similarity) + 0.35 * exp(-CER / 0.03) + 0.15 * prosody match`.
- Best-of-N stops as soon as a take reaches the quality preset's threshold, so the expected number of renders is about one over the pass rate rather than a fixed N.

## Watermarks and provenance

| Option | Licence | State | Notes |
|---|---|---|---|
| AudioSeal | MIT, code and weights [S16] | 0.2.0, Dec 2025 [O] | 16-bit payload, works on any waveform; voicesmith uses it on every output |
| PerTh (resemble-perth) | MIT | 1.0.1, May 2025; its `pkg_resources` fix is merged but unreleased [O] | built into Chatterbox; kept |
| SilentCipher, WavMark | MIT | inactive since 2024 [O] | not used |

Robustness is limited: a 2026 benchmark found detection "approaching chance level" under some common edits for AudioSeal, WavMark, Timbre and PerTh [S21], and neural codecs are the hardest attack [S20]. A watermark is a signal for honest pipelines, not proof against a determined attacker, so voicesmith also writes disclosure tags and a manifest, and the README says plainly that an MIT fork can remove all of it.

C2PA content credentials (`c2pa-python` 0.38, MIT or Apache-2.0, Windows wheels) [S23][O] are a planned addition: the EU Code of Practice asks for at least two machine-readable layers [S30].

## Law worth knowing (not legal advice)

- **EU AI Act Article 50** transparency duties apply from 2 August 2026; the free and open-source exemption in Article 2(12) does not cover Article 50 [S28][S30].
- **US NO FAKES Act** (S.4591 and H.R.8915) cleared the Senate Judiciary Committee on 18 June 2026 and is not law as of the last report [S32].
- **Tennessee ELVIS Act** (in force since 1 July 2024) covers tools whose primary purpose is producing someone's voice without authorisation [S33].
- **California AB 2602 and AB 1836** apply from 1 January 2025 [S37]. **China's labelling measures** from 1 September 2025 [S35]. **South Korea's AI Basic Act** from 22 January 2026 [S36]. **Denmark's** proposed likeness right was not confirmed as passed [S34].

## Hardware

- Dependable on Windows in 2026: CUDA, and plain CPU PyTorch and ONNX Runtime [S39][S40].
- DirectML is in maintenance mode; `torch-directml` last shipped in September 2024 [O]. Not used.
- PyTorch XPU does not list Iris Xe, **and uv's `--torch-backend auto` picks XPU wheels whenever any Intel display adapter exists** [O, uv `accelerator.rs`]. voicesmith only uses `auto` when an NVIDIA GPU is present and forces `cpu` otherwise.
- Apple: MPS through PyTorch; MLX ports exist (mlx-audio 0.5.8) [O].
- Vendor CPU speeds do not transfer between machines (see `../benchmarks.md`), so speed has to be measured where it runs.

## Packaging

- Core installs with `uv tool install voicesmith` and has no PyTorch. Each engine family gets its own uv environment because Chatterbox pins torch 2.6.0 and transformers 5.2.0 while others need newer versions [O]. uv hard-links from its cache on Windows, so shared wheels are not duplicated.
- Hugging Face on Windows: set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` so a broken saved login never blocks public downloads; without Developer Mode, caches fall back to plain copies [S45].

## Agents

- FastMCP 4.0.x on SDK v2 [S49][O], stdio by default. Codex allows 60 s per tool call by default [S51]; Claude Code backgrounds MCP calls over 2 minutes [S50]. voicesmith's `speak` waits up to 50 s, then returns a job id.
- Return file paths, never base64 audio: it eats the agent's output budget and the model cannot hear it [S50].
- An agent must never be able to grant consent.
- Prior art: Voicebox (MIT, 56.5k stars) has seven engines and MCP but no consent or watermark [S53]; Chatterbox-TTS-Extended validates per chunk but exposes a switch to turn the watermark off [S54].

## Sources

S1 https://github.com/BytedanceSpeech/seed-tts-eval · S2 https://arxiv.org/abs/2407.18505 · S3 https://arxiv.org/abs/2507.02176 · S6 https://arxiv.org/abs/2603.24430 · S16 https://github.com/facebookresearch/audioseal · S20 https://arxiv.org/abs/2505.19663 · S21 https://arxiv.org/abs/2606.15187 · S23 https://github.com/contentauth/c2pa-python · S28 https://artificialintelligenceact.eu/article/50/ · S30 https://www.paulweiss.com/insights/client-memos/eu-finalises-transparency-rules-for-ai-generated-content · S32 https://www.congress.gov/bill/119th-congress/senate-bill/4591 · S33 https://www.insideglobaltech.com/2024/04/02/tennessee-enacts-legislation-to-protect-musicians-from-ai-generated-voice-impersonations/ · S34 https://globallawexperts.com/denmark-deepfake-law-2026/ · S35 https://www.loeb.com/en/insights/publications/2025/03/chinas-ai-labeling-measures-and-mandatory-national-standards-take-effect-september-1 · S36 https://www.cooley.com/news/insight/2026/2026-01-27-south-koreas-ai-basic-act-overview-and-key-takeaways · S37 California AB 2602 and AB 1836 summary (Skadden, Sept 2024) · S39 https://github.com/microsoft/DirectML · S40 https://docs.pytorch.org/docs/2.13/notes/get_start_xpu.html · S45 https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables · S49 https://gofastmcp.com/changelog · S50 https://code.claude.com/docs/en/mcp · S51 https://developers.openai.com/codex/mcp · S53 https://github.com/jamiepine/voicebox · S54 https://github.com/petermg/Chatterbox-TTS-Extended
