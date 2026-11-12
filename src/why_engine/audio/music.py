"""Procedural music bed.

The spec forbids catalogued music: a Content ID claim is not something this
pipeline can dispute, and a licensed track is a standing liability on every
video ever published. Synthesising the bed removes the surface entirely --
there is nothing to match against, permanently, for zero marginal cost.

Level and coverage are measured from the reference video. Its bed sits about
23 dB under speech (gap RMS 0.0136 against speech RMS 0.1865) and runs
continuously while speech is active only ~65% of the time, so the bed is what
carries the silence between sentences.

Ducking is scheduled, not detected. The narration timing map already says
exactly where every sentence starts and ends, so the bed can be pulled down
before a line begins rather than reacting after it has started -- no pumping,
no detector to tune.
"""
from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050

#: Bed level under speech, measured from the reference.
DUCK_DB = -23.0
#: Bed level in the gaps, where it is the only thing playing.
OPEN_DB = -17.0
#: Duck ramp. Long enough not to click, short enough to be under the first word.
RAMP_SECONDS = 0.35

#: A natural-minor scale in semitones. Minor because the subject matter is
#: usually something going wrong; the reference's bed is likewise not cheerful.
SCALE = (0, 2, 3, 5, 7, 8, 10)
ROOT_HZ = 55.0  # A1
BPM = 68.0


@dataclass(frozen=True)
class Window:
    """A stretch of the timeline where speech is active."""

    start: float
    end: float


def _db(x: float) -> float:
    return 10.0 ** (x / 20.0)


#: Every array in this module is sized to the whole video -- a 21.7-minute bed
#: is 28.7 million samples, and at float64 that is 230MB each. write_bed chained
#: enough of them to peak at 1.18GB and was killed for memory three times. The
#: bed is written as 16-bit PCM, so float32 carries far more precision than
#: survives the write.
DTYPE = np.float32


def _sine(freq: float, n: int, sr: int, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n, dtype=DTYPE) / sr
    return np.sin(2 * np.pi * freq * t + phase)


def _pluck(freq: float, n: int, sr: int, decay: float = 2.6) -> np.ndarray:
    """A decaying tone with a little second harmonic -- enough to read as an
    instrument rather than a test tone, without needing a sampler."""
    t = np.arange(n, dtype=DTYPE) / sr
    env = np.exp(-decay * t)
    return env * (np.sin(2 * np.pi * freq * t) + 0.22 * np.sin(4 * np.pi * freq * t))


def bed(duration: float, seed: int = 0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Render `duration` seconds of bed, deterministic for a given seed."""
    n = int(duration * sr)
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=DTYPE)

    # Drone: root and fifth, slightly detuned so they beat slowly against each
    # other instead of sitting still.
    for freq, gain in ((ROOT_HZ, 0.50), (ROOT_HZ * 1.5, 0.28), (ROOT_HZ * 2, 0.20)):
        out += gain * _sine(freq * rng.uniform(0.998, 1.002), n, sr, rng.uniform(0, 6.28))

    # Slow swell, so the bed breathes rather than sitting at one level.
    t = np.arange(n, dtype=DTYPE) / sr
    out *= 0.75 + 0.25 * np.sin(2 * np.pi * t / 17.0)

    # Sparse plucks on the beat grid, two octaves up, skipping most positions.
    beat = 60.0 / BPM
    note_n = int(beat * 3.2 * sr)
    for i in range(int(duration / beat)):
        if rng.random() > 0.28:
            continue
        start = int(i * beat * sr)
        degree = SCALE[rng.integers(0, len(SCALE))]
        freq = ROOT_HZ * 4 * (2 ** (degree / 12))
        note = _pluck(freq, min(note_n, n - start), sr)
        out[start : start + len(note)] += 0.16 * note

    peak = np.abs(out).max()
    return out / peak if peak > 0 else out


def duck_envelope(duration: float, windows: list[Window], sr: int = SAMPLE_RATE,
                  ramp: float = RAMP_SECONDS) -> np.ndarray:
    """Gain curve: OPEN_DB in the gaps, DUCK_DB under speech, ramped between.

    Built by scheduling from the timing map, so the bed is already down by the
    time the first syllable lands.
    """
    n = int(duration * sr)
    env = np.full(n, _db(OPEN_DB), dtype=DTYPE)
    for w in windows:
        a, b = int(max(w.start, 0) * sr), int(min(w.end, duration) * sr)
        if b <= a:
            continue
        env[a:b] = _db(DUCK_DB)
    # Smoothing in both directions gives a symmetric ramp with no phase lag,
    # so the duck neither lands late nor releases early. Pad with edge values
    # first: numpy's "same" mode pads with ZEROS, which silently fades the bed
    # out at the start and end of every video.
    k = max(int(ramp * sr), 1)
    kernel = np.ones(k, dtype=DTYPE) / k
    for _ in range(2):
        padded = np.pad(env, k, mode="edge")
        env = np.convolve(padded, kernel, mode="same")[k:-k]
    return env


def write_bed(path: Path | str, duration: float, windows: list[Window],
              seed: int = 0, sr: int = SAMPLE_RATE) -> Path:
    """Render the ducked bed to a 16-bit mono WAV."""
    # In place throughout: every intermediate here is another full-length array,
    # and the chain used to hold four of them at once.
    signal = bed(duration, seed, sr)
    signal *= duck_envelope(duration, windows, sr)
    np.clip(signal, -1.0, 1.0, out=signal)
    signal *= 32767
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(signal.astype(np.int16).tobytes())
    return Path(path)


def windows_from_segments(segments) -> list[Window]:
    """Speech windows from a narration timing map."""
    return [Window(s.start, s.end) for s in segments]
