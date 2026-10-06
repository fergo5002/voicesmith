"""Consent records. Nothing renders in a voice without one.

Two routes:

* **Spoken** (preferred). ``consent request`` makes a statement with a one-off
  code. The speaker reads it aloud, live or as a voice note sent from their
  phone, and ``consent verify`` checks the words by ASR and, once references
  exist, that the voice matches. The recording also anchors ingest, so only
  the consenting speaker's audio is kept from a multi-speaker source.
* **Attested**. For archive material where the speaker cannot record, the
  operator records who authorised what, how, and when. It is weaker evidence,
  so every output made under it says so in its metadata.

Agents cannot create consent: the MCP server only reads it.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from voicesmith import audio, text, voices

STATEMENT = (
    "I, {speaker}, give permission for a synthetic copy of my voice to be made with voicesmith. "
    "I understand it can say things I never said, and that I can withdraw this permission. "
    "My consent code is {code}."
)
REQUEST_DAYS = 14
MAX_CER = 0.2  # allows for ASR slips on unusual names
SPEAKER_MATCH = 0.45  # TitaNet cosine; same speaker ~0.8, different ~0.05 on our checks

_WORDS = (
    "amber anchor apple arrow aspen autumn badger bamboo basil beacon birch bison bramble breeze bronze cactus "
    "canyon cedar cherry cinder clover cobalt comet coral cotton cricket crystal daisy delta desert ember falcon "
    "fennel fern fjord forest fossil garnet ginger glacier granite harbour hazel heron honey indigo island ivory "
    "jasper juniper kestrel lagoon lantern lemon lilac linen lotus magnet maple marble meadow meteor mint mosaic "
    "nectar nutmeg oasis olive onyx orbit orchid otter pebble pepper pine planet plum prism quartz quill raven "
    "reef ripple river robin saffron sage salmon sapphire shadow silver sparrow spruce summit swallow thistle "
    "thunder tiger timber topaz tulip velvet violet walnut willow winter zephyr"
).split()


@dataclass
class Consent:
    kind: str  # "spoken" | "attested"
    status: str  # "pending" | "verified" | "attested" | "revoked"
    speaker: str
    statement: str = ""
    code: str = ""
    created: str = ""
    expires: str = ""
    recording: str = ""
    recording_sha256: str = ""
    asr_text: str = ""
    asr_cer: float | None = None
    speaker_match: float | None = None
    attested_by: str = ""
    evidence: str = ""
    scope: str = ""
    revoked_at: str = ""

    @property
    def usable(self) -> bool:
        return self.status in ("verified", "attested")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _path(voice: voices.Voice) -> Path:
    return voice.dir / "consent" / "consent.json"


def load(voice: voices.Voice) -> Consent | None:
    p = _path(voice)
    if not p.exists():
        return None
    return Consent(**json.loads(p.read_text(encoding="utf-8")))


def save(voice: voices.Voice, c: Consent) -> Consent:
    p = _path(voice)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    tmp.write_text(json.dumps(asdict(c), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    # Append-only history: a later record can never erase that consent was once withdrawn.
    with open(p.parent / "history.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": _now(), **asdict(c)}, ensure_ascii=False) + "\n")
    return c


def new_code() -> str:
    return f"{secrets.choice(_WORDS)} {secrets.choice(_WORDS)} {secrets.choice(_WORDS)} {secrets.randbelow(90) + 10}"


def request(voice: voices.Voice, scope: str = "") -> Consent:
    code = new_code()
    expires = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + REQUEST_DAYS * 86400))
    c = Consent(
        kind="spoken",
        status="pending",
        speaker=voice.speaker,
        statement=STATEMENT.format(speaker=voice.speaker, code=code),
        code=code,
        created=_now(),
        expires=expires,
        scope=scope,
    )
    return save(voice, c)


def verify_recording(voice: voices.Voice, recording: str | Path) -> Consent:
    """Check a spoken consent recording against the pending statement."""
    from voicesmith.analysis import asr, speaker

    c = load(voice)
    if c is None or c.kind != "spoken" or not c.statement:
        raise ValueError(f"no pending spoken consent for {voice.name!r}; run: voicesmith consent request {voice.name}")
    if c.status == "revoked":
        raise ValueError("consent was revoked; request a new one")
    if c.expires and time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) > c.expires:
        raise ValueError("the consent code has expired; request a new one")
    wav16 = audio.load(recording, 16_000)
    if len(wav16) < 16_000 * 4:
        raise ValueError("recording is too short to contain the statement")
    tr = asr.transcribe(wav16)
    full_cer = text.cer(c.statement, tr.text)
    code_cer = text.cer(c.code, _best_window(c.code, tr.text))
    if code_cer > 0.25:
        raise ValueError(f"the consent code was not heard clearly (heard: {tr.text!r})")
    if full_cer > MAX_CER:
        raise ValueError(f"the statement did not match closely enough (character error {full_cer:.0%}; heard: {tr.text!r})")
    dest = voice.dir / "consent" / ("consent" + Path(recording).suffix.lower())
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(recording, dest)
    emb = speaker.embed(wav16)
    np.save(voice.dir / "consent" / "anchor.npy", emb)
    c.recording = str(dest.relative_to(voice.dir))
    c.recording_sha256 = hashlib.sha256(dest.read_bytes()).hexdigest()
    c.asr_text = tr.text
    c.asr_cer = round(full_cer, 4)
    centroid = voice.centroid()
    if centroid is not None:
        c.speaker_match = round(speaker.cosine(emb, centroid), 4)
        if c.speaker_match < SPEAKER_MATCH:
            save(voice, c)
            raise ValueError(
                f"the consent recording does not sound like the voice's references (match {c.speaker_match:.2f})"
            )
    c.status = "verified"
    return save(voice, c)


def check_against_references(voice: voices.Voice) -> float | None:
    """After ingest, confirm the consenting speaker is the one in the references."""
    from voicesmith.analysis import speaker

    c = load(voice)
    anchor_p = voice.dir / "consent" / "anchor.npy"
    centroid = voice.centroid()
    if c is None or c.kind != "spoken" or not anchor_p.exists() or centroid is None:
        return None
    c.speaker_match = round(speaker.cosine(np.load(anchor_p), centroid), 4)
    save(voice, c)
    return c.speaker_match


def anchor(voice: voices.Voice) -> np.ndarray | None:
    p = voice.dir / "consent" / "anchor.npy"
    return np.load(p) if p.exists() else None


def attest(voice: voices.Voice, *, by: str, evidence: str, scope: str = "", override_revocation: bool = False) -> Consent:
    if not by.strip() or not evidence.strip():
        raise ValueError("an attestation needs who is attesting (--by) and what the authorisation is (--evidence)")
    current = load(voice)
    if current is not None and current.status == "revoked" and not override_revocation:
        raise ValueError(
            f"{voice.speaker} revoked consent on {current.revoked_at}. Only attest again if they have given new "
            "permission since, and say so with --override-revocation and evidence of the new permission."
        )
    c = Consent(
        kind="attested",
        status="attested",
        speaker=voice.speaker,
        created=_now(),
        attested_by=by.strip(),
        evidence=evidence.strip(),
        scope=scope,
        statement=f"{by.strip()} attests that {voice.speaker} authorised a synthetic copy of their voice: {evidence.strip()}",
    )
    return save(voice, c)


def revoke(voice: voices.Voice) -> Consent:
    c = load(voice)
    if c is None:
        raise ValueError(f"voice {voice.name!r} has no consent record")
    c.status = "revoked"
    c.revoked_at = _now()
    return save(voice, c)


def require(voice: voices.Voice) -> Consent:
    c = load(voice)
    if c is None or not c.usable:
        state = "missing" if c is None else c.status
        raise PermissionError(
            f"voice {voice.name!r} has no usable consent ({state}). "
            f"Run 'voicesmith consent request {voice.name}' and have {voice.speaker} read the statement, "
            f"or 'voicesmith consent attest {voice.name} --by ... --evidence ...'."
        )
    if c.kind == "spoken" and c.speaker_match is not None and c.speaker_match < SPEAKER_MATCH:
        raise PermissionError(
            f"the consent recording for {voice.name!r} does not match the voice's references "
            f"(match {c.speaker_match:.2f}); the references may be someone else"
        )
    return c


def _best_window(needle: str, hay: str) -> str:
    """The stretch of the transcript that best matches the code, word-aligned."""
    n = len(text.normalise(needle).split())
    words = text.normalise(hay).split()
    if len(words) <= n:
        return " ".join(words)
    best, best_score = "", 1e9
    for i in range(len(words) - n + 1):
        cand = " ".join(words[i : i + n])
        s = text.cer(needle, cand)
        if s < best_score:
            best, best_score = cand, s
    return best
