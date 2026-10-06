"""Benchmark installed engines on LibriSpeech test-clean through voicesmith's own worker layer.

    python scripts/bench_engines.py --librispeech /path/to/LibriSpeech/test-clean \
        --engines chatterbox-turbo,qwen3-tts-0.6b --out bench.json

For each of six speakers it takes one ~10 s utterance as the reference and
renders two other utterances' text, so every render has a real recording of the
same sentence to compare against. Reported per engine:

* WER and CER of the render (Parakeet ASR) against the script,
* SIM: TitaNet cosine to the speaker's held-out centroid (other utterances),
* RTF: generation seconds per second of audio, after a warm-up render,
* the machine's CPU load before and after, because a busy machine makes RTF lie.

LibriSpeech is public-domain LibriVox audio (CC-BY-4.0 corpus), the standard
benchmark for zero-shot cloning.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from voicesmith import audio, ffmpeg, hardware, text
from voicesmith.analysis import asr, speaker
from voicesmith.engines import client, registry

SPEAKERS = ["1089", "2300", "61", "121", "4446", "8463"]  # three male, three female


def cases(root: Path, speakers: list[str]) -> list[dict]:
    out = []
    for spk in speakers:
        utts = []
        for tf in sorted((root / spk).glob("*/*.trans.txt")):
            for line in tf.read_text().splitlines():
                key, words = line.split(" ", 1)
                flac = tf.parent / f"{key}.flac"
                utts.append((key, words.capitalize(), flac, ffmpeg.probe(flac).duration or 0))
        ref = min((u for u in utts if 8 <= u[3] <= 14), key=lambda u: abs(u[3] - 10.5))
        targets = [u for u in utts if u[0] != ref[0] and 12 <= len(u[1].split()) <= 30][:2]
        holdout = [u[2] for u in utts if u[0] != ref[0] and u not in targets][:20]
        for i, t in enumerate(targets):
            out.append({"id": f"{spk}_{i}", "speaker": spk, "ref": ref[2], "ref_text": ref[1], "text": t[1] + ".",
                        "truth": t[2], "holdout": holdout})
    return out


def score(wav16: np.ndarray, case: dict, centroids: dict) -> dict:
    tr = asr.transcribe(wav16)
    e = speaker.embed(wav16)
    return {"wer": text.wer(case["text"], tr.text), "cer": text.cer(case["text"], tr.text),
            "sim": speaker.cosine(e, centroids[case["speaker"]])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", type=Path, required=True)
    ap.add_argument("--engines", default=",".join(registry.ENGINES))
    ap.add_argument("--speakers", default=",".join(SPEAKERS))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("bench.json"))
    a = ap.parse_args()

    machine = hardware.detect()
    cs = cases(a.librispeech, a.speakers.split(","))
    centroids = {}
    for c in cs:
        if c["speaker"] not in centroids:
            centroids[c["speaker"]] = speaker.centroid([speaker.embed(audio.load(p)) for p in c["holdout"]])
    results = {"machine": machine.as_dict(), "engines": {}}
    truth = [score(audio.load(c["truth"]), c, centroids) for c in cs]
    results["engines"]["ground-truth"] = {"rows": truth}

    with tempfile.TemporaryDirectory() as tmp:
        for eng in a.engines.split(","):
            busy_before = hardware.busy_percent(2.0)
            t0 = time.time()
            try:
                w = client.engine_worker(eng, threads=a.threads)
            except Exception as exc:
                results["engines"][eng] = {"error": str(exc)}
                print(f"{eng}: skipped ({exc})")
                continue
            load_s = time.time() - t0
            # Warm-up render so one-off graph and cache costs do not land on the first case.
            ref_wav = audio.save_wav(Path(tmp) / "ref0.wav", audio.load(cs[0]["ref"], 24_000), 24_000)
            w.call("synthesize", {"text": "This is a warm up.", "ref_wav": str(ref_wav), "ref_text": cs[0]["ref_text"],
                                  "seed": 0, "out_wav": str(Path(tmp) / "warm.wav")})
            rows = []
            for c in cs:
                ref = audio.save_wav(Path(tmp) / f"{c['speaker']}.wav", audio.load(c["ref"], 24_000), 24_000)
                out = Path(tmp) / f"{eng}-{c['id']}.wav"
                gen = w.call("synthesize", {"text": c["text"], "ref_wav": str(ref), "ref_text": c["ref_text"],
                                            "seed": 0, "out_wav": str(out)}, timeout=1800)
                wav, sr = audio.read_wav(out)
                row = score(audio.resample(wav, sr, 16_000), c, centroids)
                row.update(id=c["id"], rtf=gen["seconds"] / max(0.1, gen["duration"]), seconds=gen["seconds"])
                rows.append(row)
                print(f"  {eng} {c['id']}: RTF {row['rtf']:.2f}  SIM {row['sim']:.3f}  WER {row['wer']:.1%}", flush=True)
            results["engines"][eng] = {"rows": rows, "load_s": load_s, "busy_before": busy_before,
                                       "busy_after": hardware.busy_percent(2.0), "threads": w.info.get("threads")}
            a.out.write_text(json.dumps(results, indent=1, default=float))

    print("\n| engine | WER | CER | SIM (held-out) | min SIM | RTF (CPU) | CPU busy before |")
    print("|---|---|---|---|---|---|---|")
    for name, r in results["engines"].items():
        if "rows" not in r:
            continue
        rows = r["rows"]
        rtf = f"{np.median([x['rtf'] for x in rows]):.2f}" if "rtf" in rows[0] else "-"
        busy = f"{r['busy_before']:.0f}%" if "busy_before" in r else "-"
        print(f"| {name} | {np.mean([x['wer'] for x in rows]):.1%} | {np.mean([x['cer'] for x in rows]):.1%} | "
              f"{np.mean([x['sim'] for x in rows]):.3f} | {np.min([x['sim'] for x in rows]):.3f} | {rtf} | {busy} |")
    a.out.write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
