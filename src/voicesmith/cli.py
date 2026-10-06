"""Command line interface."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from voicesmith import __version__

app = typer.Typer(
    name="voicesmith",
    help="Local, consent-first voice cloning. Start with: voicesmith doctor",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
engines_app = typer.Typer(help="Install and inspect synthesis engines.", no_args_is_help=True)
voice_app = typer.Typer(help="Create, inspect and delete voices.", no_args_is_help=True)
consent_app = typer.Typer(help="Record, verify, attest or revoke consent for a voice.", no_args_is_help=True)
app.add_typer(engines_app, name="engines")
app.add_typer(voice_app, name="voice")
app.add_typer(consent_app, name="consent")

out = Console()
err = Console(stderr=True)


def _fail(msg: str, code: int = 1) -> None:
    err.print(f"[bold red]error:[/] {msg}")
    raise typer.Exit(code)


def _log(msg: str) -> None:
    err.print(f"[dim]{time.strftime('%H:%M:%S')}[/] {msg}")


@app.callback(invoke_without_command=True)
def _root(version: Annotated[bool, typer.Option("--version", help="Show the version.")] = False) -> None:
    if version:
        out.print(__version__)
        raise typer.Exit()


# ---------------------------------------------------------------- doctor

@app.command()
def doctor(
    json_out: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    quick: Annotated[bool, typer.Option("--quick", help="Do not start engine workers.")] = False,
) -> None:
    """Check this machine and say how to fix anything that is wrong."""
    from voicesmith import doctor as doc

    checks, machine = doc.run(probe_engines=not quick)
    if json_out:
        out.print_json(json.dumps({"machine": machine.as_dict(), "checks": [c.__dict__ for c in checks]}))
        return
    for c in checks:
        mark = "[green]ok[/]" if c.ok else "[red]!![/]" if c.ok is False else "[blue]--[/]"
        out.print(f" {mark} [bold]{c.name}[/]: {c.detail}")
        if c.fix:
            out.print(f"      fix: [yellow]{c.fix}[/]")
    if any(c.ok is False for c in checks):
        raise typer.Exit(1)


# ---------------------------------------------------------------- engines

@engines_app.command("list")
def engines_list() -> None:
    """Show every engine, its licence and whether it is installed."""
    from voicesmith.engines import manager, registry

    t = Table("engine", "family", "installed", "licence", "params", "languages", "notes")
    for e in registry.ENGINES.values():
        t.add_row(
            e.name, e.family, "yes" if manager.installed(e.family) else "no", e.weights_licence, f"{e.params_m}M",
            ",".join(e.languages[:6]) + ("..." if len(e.languages) > 6 else ""), ", ".join(e.tags),
        )
    out.print(t)


@engines_app.command("install")
def engines_install(
    names: Annotated[list[str], typer.Argument(help="Family or engine names, or 'recommended'.")],
    force: Annotated[bool, typer.Option(help="Rebuild even if installed.")] = False,
) -> None:
    """Install engine environments (each is isolated, so versions never clash)."""
    from voicesmith import hardware
    from voicesmith.engines import manager, registry

    machine = hardware.detect()
    families: list[str] = []
    for n in names:
        if n == "recommended":
            families += ["chatterbox", "qwen3"] if machine.nvidia else ["chatterbox"]
        elif n == "all":
            families += list(registry.FAMILIES)
        elif n in registry.FAMILIES:
            families.append(n)
        elif n in registry.ENGINES:
            families.append(registry.ENGINES[n].family)
        else:
            _fail(f"unknown engine or family {n!r}; see: voicesmith engines list")
    for fam in dict.fromkeys(families):
        try:
            manager.install(fam, log=_log, machine=machine, force=force)
        except Exception as exc:
            _fail(str(exc))
    out.print("[green]done.[/] Model weights download on first use.")


@engines_app.command("remove")
def engines_remove(family: str, yes: Annotated[bool, typer.Option("--yes")] = False) -> None:
    """Delete an engine environment (model weights in the Hugging Face cache are kept)."""
    from voicesmith.engines import manager

    if not yes and not typer.confirm(f"Delete the {family} environment?"):
        raise typer.Exit()
    manager.remove(family)
    out.print(f"removed {family}")


# ---------------------------------------------------------------- voices

@app.command("voices")
def voices_list(json_out: Annotated[bool, typer.Option("--json")] = False) -> None:
    """List voices."""
    from voicesmith import consent, voices

    rows = []
    for v in voices.all_voices():
        c = consent.load(v)
        rows.append({
            "name": v.name, "speaker": v.speaker, "language": v.language,
            "consent": c.status if c else "none", "references": len(v.references),
            "clean_minutes": v.stats.get("clean_minutes"), "tuned_engine": v.tuning.engine if v.tuning else None,
        })
    if json_out:
        out.print_json(json.dumps(rows))
        return
    if not rows:
        out.print("No voices yet. Create one: voicesmith voice create <name> --speaker \"Full Name\"")
        return
    t = Table("name", "speaker", "lang", "consent", "refs", "clean min", "tuned engine")
    for r in rows:
        t.add_row(r["name"], r["speaker"], r["language"], r["consent"], str(r["references"]),
                  str(r["clean_minutes"] or "-"), r["tuned_engine"] or "-")
    out.print(t)


@voice_app.command("create")
def voice_create(
    name: str,
    speaker: Annotated[str, typer.Option(help="The real person's full name.")],
    language: Annotated[str, typer.Option(help="ISO 639-1 code.")] = "en",
) -> None:
    """Create an empty voice."""
    from voicesmith import voices

    try:
        v = voices.create(name, speaker, language)
    except (ValueError, FileExistsError) as exc:
        _fail(str(exc))
    out.print(f"created voice [bold]{v.name}[/] for {v.speaker}.")
    out.print(f"next: voicesmith consent request {v.name}   and   voicesmith ingest {v.name} <files or URLs>")


@voice_app.command("show")
def voice_show(name: str) -> None:
    """Everything known about a voice."""
    from dataclasses import asdict

    from voicesmith import consent, voices

    try:
        v = voices.load(name)
    except FileNotFoundError as exc:
        _fail(str(exc))
    c = consent.load(v)
    data = asdict(v)
    data["consent"] = asdict(c) if c else None
    out.print_json(json.dumps(data, default=str))


@voice_app.command("delete")
def voice_delete(name: str, yes: Annotated[bool, typer.Option("--yes")] = False) -> None:
    """Delete a voice, its references, sources and consent record."""
    from voicesmith import voices

    if not yes and not typer.confirm(f"Permanently delete voice {name!r} and everything in it?"):
        raise typer.Exit()
    voices.delete(name)
    out.print(f"deleted {name}")


# ---------------------------------------------------------------- consent

@consent_app.command("request")
def consent_request(name: str, scope: Annotated[str, typer.Option(help="What the voice may be used for.")] = "") -> None:
    """Make a consent statement for the speaker to read aloud."""
    from voicesmith import consent, voices

    v = voices.load(name)
    c = consent.request(v, scope)
    out.print(f"Ask [bold]{v.speaker}[/] to read this aloud, clearly, in one take (a phone voice note is fine):\n")
    out.print(f"  [bold cyan]{c.statement}[/]\n")
    out.print(f"The code expires {c.expires}. Then run:")
    out.print(f"  voicesmith consent verify {v.name} <recording>      or      voicesmith consent verify {v.name} --record 25")


@consent_app.command("verify")
def consent_verify(
    name: str,
    recording: Annotated[Path | None, typer.Argument(help="Audio file of the speaker reading the statement.")] = None,
    record: Annotated[float, typer.Option("--record", help="Record this many seconds from the microphone instead.")] = 0,
) -> None:
    """Check a spoken consent recording and mark the voice as consented."""
    from voicesmith import consent, voices
    from voicesmith.ingest import acquire

    v = voices.load(name)
    if record:
        c0 = consent.load(v)
        if c0:
            out.print(f"Read aloud:\n\n  [bold cyan]{c0.statement}[/]\n")
        input("Press Enter to start recording...")
        rec_dir = v.dir / "consent"
        rec_dir.mkdir(parents=True, exist_ok=True)
        meta = acquire.record(rec_dir, record, _log)
        recording = rec_dir / meta["path"]
    if recording is None:
        _fail("give a recording file or --record SECONDS")
    try:
        c = consent.verify_recording(v, recording)
    except (ValueError, FileNotFoundError) as exc:
        _fail(str(exc))
    out.print(f"[green]consent verified[/] for {v.speaker} (character error {c.asr_cer:.0%}"
              + (f", voice match {c.speaker_match:.2f}" if c.speaker_match is not None else "") + ")")


@consent_app.command("attest")
def consent_attest(
    name: str,
    by: Annotated[str, typer.Option(help="Your name: the person taking responsibility.")],
    evidence: Annotated[str, typer.Option(help="How and when the speaker authorised this.")],
    scope: Annotated[str, typer.Option(help="What the voice may be used for.")] = "",
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Record that the speaker authorised cloning when they cannot record a statement."""
    from voicesmith import consent, voices

    v = voices.load(name)
    msg = (f"I, {by}, confirm that {v.speaker} has authorised a synthetic copy of their voice "
           f"({evidence}). Outputs will be labelled as made under an attestation.")
    if not yes and not typer.confirm(msg + " Continue?"):
        raise typer.Exit()
    consent.attest(v, by=by, evidence=evidence, scope=scope)
    out.print("[green]attestation recorded.[/]")


@consent_app.command("show")
def consent_show(name: str) -> None:
    from dataclasses import asdict

    from voicesmith import consent, voices

    c = consent.load(voices.load(name))
    out.print_json(json.dumps(asdict(c) if c else None))


@consent_app.command("revoke")
def consent_revoke(name: str) -> None:
    """Withdraw consent. The voice can no longer render until consent is given again."""
    from voicesmith import consent, voices

    consent.revoke(voices.load(name))
    out.print("consent revoked.")


# ---------------------------------------------------------------- ingest / tune / say

@app.command()
def ingest(
    name: str,
    sources: Annotated[list[str], typer.Argument(help="Audio or video files, folders, or URLs.")] = None,  # type: ignore[assignment]
    record: Annotated[float, typer.Option("--record", help="Also record this many seconds from the microphone.")] = 0,
    max_items: Annotated[int, typer.Option(help="Most items to fetch from a playlist or channel URL.")] = 25,
    refs: Annotated[int, typer.Option(help="How many reference clips to keep.")] = 8,
    reanalyse: Annotated[bool, typer.Option(help="Re-run analysis over existing sources only.")] = False,
) -> None:
    """Add recordings of the speaker and rebuild the voice's reference clips."""
    from voicesmith import voices
    from voicesmith.ingest import acquire, pipeline

    v = voices.load(name)
    dest = v.dir / "sources"
    dest.mkdir(parents=True, exist_ok=True)
    added: list[dict] = []
    try:
        for s in sources or []:
            added += acquire.from_url(s, dest, _log, max_items) if acquire.is_url(s) else acquire.from_path(s, dest, _log)
        if record:
            input(f"Press Enter and speak naturally for {record:.0f} seconds...")
            added.append(acquire.record(dest, record, _log))
    except Exception as exc:
        _fail(str(exc))
    if not added and not reanalyse and not v.sources:
        _fail("give at least one file, folder or URL, or --record SECONDS")
    v.sources += added
    v.save()
    try:
        v = pipeline.run(v, k=refs, log=_log)
    except Exception as exc:
        _fail(str(exc))
    st = v.stats
    out.print(f"[green]{v.name}[/]: {len(v.references)} references from {st['clean_minutes']} clean minutes "
              f"(self-similarity {st['self_similarity']:.2f}).")
    from voicesmith import consent

    c = consent.load(v)
    if not c or not c.usable:
        out.print(f"next: consent. voicesmith consent request {v.name}")
    else:
        out.print(f"next: voicesmith tune {v.name}")


@app.command()
def tune(
    name: str,
    engines: Annotated[str | None, typer.Option(help="Comma-separated engines to try (default: all installed).")] = None,
    refs: Annotated[int, typer.Option(help="References to try per engine.")] = 3,
    probes: Annotated[int, typer.Option(help="Probe sentences per reference.")] = 2,
    budget: Annotated[float | None, typer.Option(help="Stop after this many minutes.")] = None,
) -> None:
    """Find the best engine and reference for a voice (run once, after ingest)."""
    from voicesmith import tune as tuner

    try:
        t = tuner.tune(name, engines=engines.split(",") if engines else None, n_refs=refs, n_probes=probes,
                       budget_minutes=budget, log=_log)
    except Exception as exc:
        _fail(str(exc))
    out.print(f"[green]tuned[/]: {t.engine} with {t.reference} (similarity {t.similarity:.3f}, CER {t.cer:.1%}, "
              f"{t.rtf:.1f}x real time)")


@app.command()
def say(
    name: str,
    text: Annotated[str | None, typer.Argument(help="What to say. Use --file for long scripts.")] = None,
    file: Annotated[Path | None, typer.Option("--file", "-f", help="Read the script from a text file.")] = None,
    output: Annotated[list[Path] | None, typer.Option("--out", "-o", help="Output file(s): .wav .mp3 .m4a .ogg .flac")] = None,
    quality: Annotated[str, typer.Option(help="fast, balanced or best.")] = "balanced",
    engine: Annotated[str | None, typer.Option(help="Override the tuned engine.")] = None,
    ref: Annotated[str | None, typer.Option(help="Override the tuned reference clip id.")] = None,
    seed: Annotated[int | None, typer.Option(help="Make the takes reproducible.")] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print a JSON result.")] = False,
) -> None:
    """Speak text in a voice. Every take is checked, the best is mastered and watermarked."""
    from voicesmith.synth import render

    script = file.read_text(encoding="utf-8") if file else text
    if not script:
        _fail("give the text to say, or --file")
    if quality not in render.QUALITY:
        _fail(f"quality must be one of {', '.join(render.QUALITY)}")
    outs = output or [Path.cwd() / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.wav"]
    try:
        r = render.render(name, script, outputs=[p.resolve() for p in outs], engine=engine, reference=ref,
                          quality=quality, seed=seed, progress=_log)
    except Exception as exc:
        _fail(str(exc))
    if json_out:
        out.print_json(render.result_json(r))
        return
    s = r.score
    for f in r.files:
        out.print(f"[green]wrote[/] {f}")
    out.print(f"{r.duration:.1f}s of audio in {r.seconds:.0f}s with {r.engine} ({r.takes} take(s)); "
              f"similarity {s['similarity']:.3f}, word error {s['wer']:.1%}. Manifest: {r.manifest}")


@app.command()
def verify(path: Path) -> None:
    """Check a file for voicesmith's watermark and disclosure tags."""
    import tempfile

    from voicesmith import audio, ffmpeg, provenance
    from voicesmith.engines import client

    if not path.exists():
        _fail(f"no such file: {path}")
    meta = ffmpeg.read_metadata(path)
    tags_ok = provenance.check_tags(meta)
    w = client.any_watermark_worker()
    det = None
    if w is not None:
        with tempfile.TemporaryDirectory() as tmp:
            wav = audio.save_wav(Path(tmp) / "x.wav", ffmpeg.decode(path, 24_000), 24_000, subtype="FLOAT")
            det = w.call("detect", {"wav": str(wav)})
    found = bool(det and det.get("audioseal_prob", 0) >= 0.5 and det.get("audioseal_payload_ok"))
    out.print_json(json.dumps({"file": str(path), "disclosure_tags": tags_ok, "watermark": det,
                               "voicesmith_watermark_found": found, "tags": {k: v for k, v in meta.items() if "voicesmith" in k or k in ("comment", "ai_generated")}}))
    if w is None:
        err.print("no engine installed, so the watermark could not be checked")


@app.command()
def mcp() -> None:
    """Run the MCP server on stdio (for Claude Code, Codex, Cursor and others)."""
    from voicesmith import mcp_server

    mcp_server.serve()


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        err.print("interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
