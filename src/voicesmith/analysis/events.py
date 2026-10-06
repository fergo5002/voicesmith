"""Sound-event screening (CED-mini AudioSet tagger via sherpa-onnx).

Used to keep music beds, laughter, applause and crosstalk out of reference clips.
"""

from __future__ import annotations

import numpy as np

from voicesmith.analysis import runtime

SR = 16_000

# AudioSet labels that make a clip a poor cloning reference.
BAD = {
    "Music",
    "Musical instrument",
    "Singing",
    "Laughter",
    "Giggle",
    "Chuckle, chortle",
    "Applause",
    "Cheering",
    "Crowd",
    "Chatter",
    "Hubbub, speech noise, speech babble",
    "Television",
    "Radio",
    "Static",
    "Noise",
    "Wind noise (microphone)",
    "Echo",
    "Reverberation",
}


def contaminants(audio: np.ndarray) -> dict[str, float]:
    """Probabilities for the problem labels that appear in the top predictions."""
    tg = runtime.tagger()
    stream = tg.create_stream()
    stream.accept_waveform(SR, audio.astype(np.float32))
    events = tg.compute(stream)
    return {e.name: float(e.prob) for e in events if e.name in BAD}
