"""Turn raw recordings into a ranked bank of reference clips.

The approach, in order of cost:

1. Find speech (Silero VAD) and transcribe it with word timings (Parakeet).
2. Cut candidate clips at sentence ends and pauses, 4 to 14 seconds long.
3. Embed every clip, find the target speaker (the consent recording if there
   is one, otherwise the dominant voice), and drop everyone else.
4. Reject clips that clip, are noisy, low-bandwidth, mumbled, or have music,
   laughter or crosstalk in them.
5. Score the survivors on how typical they are of the speaker and how clean,
   then pick a diverse shortlist.

Choosing good clips beats repairing bad ones: enhancement lowers the speaker
similarity of clones made from clean prompts (see docs/research).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from voicesmith import audio, consent, ffmpeg, voices
from voicesmith.analysis import asr, events, prosody, speaker, vad

Log = Callable[[str], None]
SR = 16_000

MIN_CLIP, MAX_CLIP, IDEAL = 4.0, 14.0, (7.0, 11.0)
MIN_GAP = 0.12  # a cut must land in a pause at least this long
ASR_BUDGET_SECONDS = 600.0  # transcribe at most this much of the best speech
IN_MEMORY_HOURS = 2.0  # decoded sources kept in RAM (about 230 MB per hour); beyond this, decode on demand


@dataclass
class Clip:
    source: str
    start: float
    end: float
    text: str
    confidence: float
    emb: np.ndarray | None = None
    metrics: dict = field(default_factory=dict)
    sim: float = 0.0
    score: float = 0.0
    rejected: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def candidate_clips(words: list[asr.Word], source: str) -> list[Clip]:
    """Group words into clips that start and end in pauses, preferring sentence ends."""
    clips: list[Clip] = []
    i = 0
    n = len(words)
    while i < n:
        # A clip must start after a pause too, or it begins mid-phrase.
        if i > 0 and words[i].start - words[i - 1].end < MIN_GAP:
            i += 1
            continue
        best_j = None
        for j in range(i, n):
            dur = words[j].end - words[i].start
            if dur > MAX_CLIP:
                break
            gap_after = (words[j + 1].start - words[j].end) if j + 1 < n else 1.0
            if dur >= MIN_CLIP and gap_after >= MIN_GAP:
                sentence_end = words[j].text[-1:] in ".?!"
                if sentence_end and IDEAL[0] <= dur:
                    best_j = j
                    if dur >= IDEAL[1] - 1.0:
                        break
                elif best_j is None and gap_after >= 0.25:
                    best_j = j
        if best_j is None:
            i += 1
            continue
        seg = words[i : best_j + 1]
        clips.append(
            Clip(
                source=source,
                start=seg[0].start,
                end=seg[-1].end,
                text=" ".join(w.text for w in seg),
                confidence=float(np.exp(np.mean([w.logprob for w in seg]))),
            )
        )
        i = best_j + 1
    return clips


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    if mad < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - med) / mad, -3, 3)


class Sources:
    """Decoded audio by source name.

    16 kHz float audio is about 230 MB per hour, so sources stay in memory only
    up to ``IN_MEMORY_HOURS`` in total; beyond that, segments are decoded from
    disk on demand with a seek, which is slower but keeps memory flat.
    """

    def __init__(self, files: dict[str, Path]):
        self.files = files
        self.cache: dict[str, np.ndarray] = {}
        self.cached_seconds = 0.0

    def keep(self, name: str, wav: np.ndarray) -> None:
        if self.cached_seconds + len(wav) / SR <= IN_MEMORY_HOURS * 3600:
            self.cache[name] = wav
            self.cached_seconds += len(wav) / SR

    def segment(self, name: str, start: float, end: float) -> np.ndarray:
        if name in self.cache:
            return self.cache[name][int(start * SR) : int(end * SR)]
        return ffmpeg.decode(self.files[name], SR, start=start, duration=end - start)


def scan_source(path: Path, log: Log) -> tuple[list[Clip], np.ndarray]:
    """Cheap pass over a whole source: speech regions with a speaker fingerprint and
    quick signal checks. No transcription yet."""
    wav = audio.load(path, SR)
    regions = vad.speech_regions(wav, max_speech=20.0)
    speech_s = sum(r.duration for r in regions)
    log(f"{path.name}: {len(wav) / SR / 60:.1f} min of audio, {speech_s / 60:.1f} min of speech in {len(regions)} regions")
    out: list[Clip] = []
    for r in regions:
        if r.duration < 1.5:
            continue
        seg = wav[int(r.start * SR) : int(r.end * SR)]
        c = Clip(source=path.name, start=r.start, end=r.end, text="", confidence=1.0, emb=speaker.embed(seg))
        c.metrics = {
            "clipping": audio.clipping_ratio(seg, 0.99),
            "snr_db": audio.snr_db(seg, SR),
            "bandwidth_hz": audio.bandwidth_hz(seg, SR),
        }
        out.append(c)
    return out, wav


def transcribe_regions(regions: list[Clip], waves: Sources, budget_s: float, log: Log) -> list[Clip]:
    """Transcribe the best regions first, stopping at ``budget_s`` of audio, and cut clips from them.

    Transcription is the only expensive step in ingest, so it runs on a bounded
    shortlist rather than on every minute of a long podcast.
    """
    clips: list[Clip] = []
    used = 0.0
    for r in regions:
        if used >= budget_s:
            break
        seg = waves.segment(r.source, r.start, r.end)
        tr = asr.transcribe(seg, offset=r.start)
        used += r.duration
        clips += candidate_clips(tr.words, r.source)
    log(f"transcribed {used / 60:.1f} min of the best speech into {len(clips)} candidate clips")
    return clips


def measure(clip: Clip, seg: np.ndarray) -> None:
    """``seg`` is the clip's own audio at 16 kHz."""
    clip.emb = speaker.embed(seg)
    p = prosody.measure(seg, words=len(clip.text.split()), speech_seconds=clip.duration)
    clip.metrics = {
        "clipping": audio.clipping_ratio(seg, 0.99),
        "snr_db": audio.snr_db(seg, SR),
        "bandwidth_hz": audio.bandwidth_hz(seg, SR),
        "f0_median": p.f0_median,
        "f0_range_st": p.f0_range_st,
        "rate_wps": p.rate_wps,
        "rms_db": audio.rms_db(seg),
    }


class AmbiguousSpeaker(RuntimeError):
    """No consent anchor or target clip, and no single voice clearly dominates."""


SAME_SPEAKER = 0.5  # TitaNet cosine
DOMINANT_SHARE = 0.6


def _fmt_time(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


def _clusters(clips: list[Clip]) -> list[tuple[float, list[int]]]:
    """Greedy speaker clusters by duration share: densest medoid first, then the rest."""
    embs = np.stack([c.emb for c in clips])
    durs = np.array([c.duration for c in clips])
    if len(clips) < 3:
        gram = embs @ embs.T
        if np.all(gram >= SAME_SPEAKER):
            return [(1.0, list(range(len(clips))))]
        return [(float(d / durs.sum()), [i]) for i, d in enumerate(durs)]
    left = np.ones(len(clips), dtype=bool)
    out = []
    while left.sum() >= 3 and len(out) < 4:
        idx = np.flatnonzero(left)
        sub = embs[idx]
        density = ((sub @ sub.T) >= SAME_SPEAKER).astype(float) @ durs[idx]
        medoid = sub[int(np.argmax(density))]
        members = idx[(sub @ medoid) >= SAME_SPEAKER]
        out.append((float(durs[members].sum() / durs.sum()), members.tolist()))
        left[members] = False
    return out


def find_target(
    clips: list[Clip], anchor: np.ndarray | None, log: Log, target: np.ndarray | None = None, pick: int | None = None
) -> np.ndarray:
    embs = np.stack([c.emb for c in clips])
    for name, ref in (("the consent recording", anchor), ("the --target clip", target)):
        if ref is not None:
            log(f"matching speakers against {name}")
            sims = embs @ ref
            seed = embs[sims >= 0.45] if (sims >= 0.45).sum() >= 3 else embs[sims >= np.quantile(sims, 0.8)]
            return speaker.centroid(seed)
    clusters = _clusters(clips)
    if pick is not None:
        if not 1 <= pick <= len(clusters):
            raise ValueError(f"--pick {pick} is out of range: found {len(clusters)} voice(s)")
        share, members = clusters[pick - 1]
        log(f"using voice {pick} as chosen ({share:.0%} of the speech)")
        return speaker.centroid(embs[members])
    share, members = clusters[0]
    second = clusters[1][0] if len(clusters) > 1 else 0.0
    if share < DOMINANT_SHARE or (second > 0.2 and share < 2 * second):
        lines = []
        for i, (sh, mem) in enumerate(clusters, 1):
            ex = ", ".join(f"{clips[m].source} at {_fmt_time(clips[m].start)}" for m in mem[:3])
            lines.append(f"  voice {i}: {sh:.0%} of the speech, e.g. {ex}")
        raise AmbiguousSpeaker(
            "more than one person speaks a lot in these recordings and there is no consent recording to say "
            "which one to clone:\n" + "\n".join(lines) + "\nRe-run with --pick N to choose one, verify spoken "
            "consent first (it anchors the speaker), or pass --target with a short clip of only the right person."
        )
    log(f"target speaker covers {share:.0%} of the speech" + (f"; next voice {second:.0%}" if second else ""))
    return speaker.centroid(embs[members])


def _signal_gate(m: dict) -> str:
    if m["clipping"] > 0.001:
        return "clipped"
    if m["bandwidth_hz"] < 6000:  # measured: clean 16 kHz speech 6.5k+, phone 4.3k, 16 kbps MP3 5.5k
        return "low bandwidth (phone or heavy compression)"
    if m["snr_db"] < 18:
        return "noisy"
    return ""


def gate(c: Clip) -> str:
    m = c.metrics
    if why := _signal_gate(m):
        return why
    if c.confidence < 0.55:
        return "unclear speech"
    if m["f0_median"] <= 0:
        return "no voiced speech"
    return ""


def select(clips: list[Clip], k: int, log: Log) -> list[Clip]:
    """Score clips within the pool and take a diverse top ``k``."""
    if not clips:
        return []
    sims = np.array([c.sim for c in clips])
    snr = np.array([c.metrics["snr_db"] for c in clips])
    conf = np.array([c.confidence for c in clips])
    f0 = np.array([c.metrics["f0_median"] for c in clips])
    rate = np.array([c.metrics["rate_wps"] for c in clips])
    dur = np.array([c.duration for c in clips])
    f0_dev = np.abs(12 * np.log2(f0 / np.median(f0)))
    rate_dev = np.abs(np.log(np.maximum(rate, 0.1) / max(0.1, float(np.median(rate)))))
    typical = -(_robust_z(f0_dev) + _robust_z(rate_dev)) / 2
    length_fit = -np.maximum(0, np.maximum(IDEAL[0] - dur, dur - IDEAL[1]))
    score = (
        0.40 * _robust_z(sims) + 0.20 * _robust_z(conf) + 0.15 * _robust_z(snr) + 0.15 * typical + 0.10 * _robust_z(length_fit)
    )
    for c, s in zip(clips, score):
        c.score = float(s)
    chosen: list[Clip] = []
    for c in sorted(clips, key=lambda c: -c.score):
        if len(chosen) >= k:
            break
        too_close = any(
            (o.source == c.source and abs(o.start - c.start) < 60) or float(o.emb @ c.emb) > 0.97 for o in chosen
        )
        if not too_close:
            chosen.append(c)
    return chosen


def run(
    voice: voices.Voice, *, k: int = 8, log: Log = print, target_clip: str | None = None, pick: int | None = None
) -> voices.Voice:
    files = [voice.dir / "sources" / src["path"] for src in voice.sources]
    if not files:
        raise ValueError(f"voice {voice.name!r} has no sources; add some with: voicesmith ingest {voice.name} <file|url>")
    t0 = time.time()
    regions: list[Clip] = []
    waves = Sources({f.name: f for f in files})
    for f in files:
        rs, wav = scan_source(f, log)
        waves.keep(f.name, wav)
        del wav
        regions += rs
    if not regions:
        raise RuntimeError("no speech found in the sources")
    target_emb = speaker.embed(audio.load(target_clip, SR)) if target_clip else None
    target = find_target(regions, consent.anchor(voice), log, target_emb, pick)
    for r in regions:
        r.sim = float(r.emb @ target)
    sims = np.array([r.sim for r in regions])
    keep_sim = max(0.5, float(np.median(sims[sims >= 0.5])) - 0.25) if (sims >= 0.5).any() else 0.5
    target_regions = [r for r in regions if r.sim >= keep_sim]
    if len(target_regions) < len(regions):
        others = sum(r.duration for r in regions if r.sim < keep_sim) / 60
        log(f"set aside {others:.1f} min that sounds like someone else (similarity < {keep_sim:.2f})")
    if not target_regions:
        raise RuntimeError("no speech matched the target speaker")
    centroid = speaker.centroid([r.emb for r in target_regions])
    # Quick signal gates, then the cleanest, most typical regions go to transcription first.
    usable = [r for r in target_regions if not _signal_gate(r.metrics)]
    if not usable:
        raise RuntimeError("all of the target speaker's audio failed the signal checks (clipped, noisy or low bandwidth)")
    for r in usable:
        r.sim = float(r.emb @ centroid)
    order = 0.6 * _robust_z(np.array([r.sim for r in usable])) + 0.4 * _robust_z(
        np.array([r.metrics["snr_db"] for r in usable])
    )
    ranked = [usable[i] for i in np.argsort(-order)]
    candidates = transcribe_regions(ranked, waves, ASR_BUDGET_SECONDS, log)
    if not candidates:
        raise RuntimeError("no clean 4 to 14 second stretches with pauses at both ends were found")
    for c in candidates:
        measure(c, waves.segment(c.source, c.start, c.end))
        c.sim = float(c.emb @ centroid)
        c.rejected = gate(c)
    good = [c for c in candidates if not c.rejected]
    reasons: dict[str, int] = {}
    for c in candidates:
        if c.rejected:
            reasons[c.rejected] = reasons.get(c.rejected, 0) + 1
    for why, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        log(f"rejected {count} clips: {why}")
    if not good:
        raise RuntimeError("every candidate clip failed the quality gates; try cleaner recordings")
    shortlist = select(good, k * 2, log)
    # Sound-event screening is the slowest per-clip check, so it runs on the shortlist only.
    screened = []
    for c in shortlist:
        seg = waves.segment(c.source, c.start, c.end)
        bad = {name: prob for name, prob in events.contaminants(seg).items() if prob >= 0.2}
        if bad:
            c.rejected = "contains " + ", ".join(sorted(bad))
            log(f"rejected a clip at {_fmt_time(c.start)}: {c.rejected}")
        else:
            screened.append(c)
    chosen = select(screened, k, log)
    log(f"selected {len(chosen)} references from {len(good)} clean clips")
    _write(voice, chosen, [c for c in good if c not in chosen], target_regions, log)
    voice.stats.update(
        {
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seconds_to_ingest": round(time.time() - t0, 1),
            "speech_minutes": round(sum(r.duration for r in regions) / 60, 2),
            "target_minutes": round(sum(r.duration for r in target_regions) / 60, 2),
            "candidate_clips": len(candidates),
            "clean_clips": len(good),
            "clean_minutes": round(sum(c.duration for c in good) / 60, 2),
            "self_similarity": round(speaker.self_similarity(np.stack([c.emb for c in good])), 4),
            "f0_median": round(float(np.median([c.metrics["f0_median"] for c in good])), 1),
            "f0_range_st": round(float(np.median([c.metrics["f0_range_st"] for c in good])), 2),
            "rate_wps": round(float(np.median([c.metrics["rate_wps"] for c in good])), 3),
        }
    )
    voice.save()
    match = consent.check_against_references(voice)
    if match is not None:
        log(f"consent recording matches the references at {match:.2f}")
    return voice


def _write(voice: voices.Voice, chosen: list[Clip], rest: list[Clip], target: list[Clip], log: Log) -> None:
    refs_dir = voice.dir / "refs"
    if refs_dir.exists():
        for f in refs_dir.iterdir():
            f.unlink()
    refs_dir.mkdir(parents=True, exist_ok=True)
    refs: list[voices.Reference] = []
    for i, c in enumerate(chosen, 1):
        src = voice.dir / "sources" / c.source
        lead, tail = 0.15, 0.3
        start = max(0.0, c.start - lead)
        wav = ffmpeg.decode(src, audio.REFERENCE_SR, start=start, duration=c.duration + lead + tail)
        wav = audio.fade(wav, audio.REFERENCE_SR, 15, 40)
        gain = 10 ** ((-23.0 - audio.rms_db(wav)) / 20)
        wav = wav * min(gain, 10 ** ((-1.0 - audio.peak_db(wav)) / 20))
        rid = f"ref{i:02d}"
        rel = f"refs/{rid}.wav"
        audio.save_wav(voice.dir / rel, wav, audio.REFERENCE_SR, subtype="PCM_16")
        refs.append(
            voices.Reference(
                id=rid, path=rel, text=c.text, duration=round(len(wav) / audio.REFERENCE_SR, 2), source=c.source,
                start=round(c.start, 2), score=round(c.score, 3), similarity=round(c.sim, 4),
                snr_db=round(c.metrics["snr_db"], 1), f0_median=round(c.metrics["f0_median"], 1),
                rate_wps=round(c.metrics["rate_wps"], 2),
            )
        )
    voice.references = refs
    np.save(voice.dir / "centroid.npy", speaker.centroid([c.emb for c in target]))
    holdout = rest if len(rest) >= 3 else target
    np.save(voice.dir / "holdout.npy", speaker.centroid([c.emb for c in holdout]))
