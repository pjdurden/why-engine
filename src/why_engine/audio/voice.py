"""Narration voicing, fitted to the reference video's measured spectrum.

Piper's en_US-ryan-high already sits at the reference's pitch -- median F0
167 Hz against the reference's 171 Hz -- so the thin sound was never pitch.
It was spectral tilt: the reference puts 44% of its speech energy in
160-320 Hz where Piper puts 13%, and Piper's peak sits an octave higher, in
320-640 Hz. That is the difference between a chest voice and a throat one.

BAND_TARGET below is measured from the reference's speech frames only; its
quiet bed is bright (2.5-11 kHz) and does not contaminate the low bands.
CHAIN was fitted against that target to a sum-of-squared-error of 38.5,
down from 792 for a first hand-guessed attempt. tools/voice_eq.py re-runs
the fit.
"""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

BANDS = ((20, 80), (80, 160), (160, 320), (320, 640),
         (640, 1280), (1280, 2560), (2560, 5120), (5120, 11000))

#: Percent of speech-frame energy per band, measured from the reference.
BAND_TARGET = (0.00, 5.61, 44.38, 29.86, 15.04, 4.31, 0.49, 0.30)

#: Fitted chain. asetrate/atempo drop pitch and formants ~4% for chest
#: weight; the peaking filters do the rest of the tilt; the compressor and
#: limiter hold a steady level without the clipping the raw Piper output had.
CHAIN = ",".join((
    "asetrate=22050*0.96",
    "aresample=22050",
    "atempo=1.0417",
    "highpass=f=90",
    "equalizer=f=250:width_type=q:w=1.5:g=10",
    "equalizer=f=480:width_type=q:w=1.3:g=-5",
    "equalizer=f=900:width_type=q:w=1.5:g=-1",
    "highshelf=f=2500:g=-5",
    "acompressor=threshold=-16dB:ratio=2.5:attack=15:release=200:makeup=1",
    "alimiter=limit=0.94",
))


def voice(src: Path | str, dst: Path | str, chain: str = CHAIN) -> Path:
    """Apply the fitted chain to a Piper WAV, writing 22050 Hz mono."""
    dst = Path(dst)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-af", chain,
         "-ar", "22050", "-ac", "1", "-y", str(dst)],
        check=True,
    )
    return dst


def band_profile(path: Path | str) -> list[float]:
    """Percent of energy per band over the loudest 30% of frames."""
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    frame, hop = 2048, 1024
    count = (len(x) - frame) // hop
    if count < 1:
        raise ValueError(f"{path} is too short to profile")
    fr = np.stack([x[i * hop : i * hop + frame] for i in range(count)])
    rms = np.sqrt((fr**2).mean(axis=1))
    loud = fr[rms > np.percentile(rms, 70)]
    spec = np.abs(np.fft.rfft(loud * np.hanning(frame), axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame, 1 / sr)
    total = spec.sum()
    return [100 * spec[:, (freqs >= lo) & (freqs < hi)].sum() / total for lo, hi in BANDS]


def fit_error(path: Path | str) -> float:
    """Sum of squared per-band error against the reference. Lower is closer."""
    return float(sum((t - g) ** 2 for t, g in zip(BAND_TARGET, band_profile(path))))


#: Median F0 of the reference narrator, measured over its speech frames.
REF_F0_HZ = 170.9

#: Narration target. Below the reference's 171 Hz: a deeper read was asked for
#: after hearing a build at reference pitch. Landed by ear over three builds --
#: 154 Hz read as not deep enough, 142 Hz as too deep, 150 Hz as right. Keep
#: clear of ~120 Hz and below, where the resampler starts showing on consonants.
NARRATION_F0_HZ = 150.0


def median_f0(path: Path | str, fmin: float = 60.0, fmax: float = 320.0) -> float:
    """Median fundamental frequency over voiced frames, by autocorrelation.

    Used to tell voices apart before choosing one: pitch is the cheapest
    reliable signal for whether a model is male or female, and an unlabelled
    community model rarely says.
    """
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    frame, hop = 2048, 1024
    count = (len(x) - frame) // hop
    if count < 1:
        raise ValueError(f"{path} is too short to measure")
    fr = np.stack([x[i * hop : i * hop + frame] for i in range(count)])
    rms = np.sqrt((fr**2).mean(axis=1))
    lo, hi = int(sr / fmax), int(sr / fmin)
    out = []
    for f in fr[rms > np.percentile(rms, 45)]:
        f = f - f.mean()
        if not f.any():
            continue
        ac = np.correlate(f, f, mode="full")[len(f) - 1 :]
        if ac[0] <= 0:
            continue
        seg = (ac / ac[0])[lo:hi]
        k = int(np.argmax(seg))
        if seg[k] > 0.35:
            out.append(sr / (lo + k))
    if not out:
        raise ValueError(f"{path} has no voiced frames to measure")
    return float(np.median(out))


def pitch_chain(ratio: float, sr: int = 22050) -> str:
    """Resample-and-restore pair that shifts pitch without changing duration.

    Lowering the sample rate drops pitch AND formants together, which is what
    makes the result read as a different-sized speaker rather than a slowed
    recording. atempo puts the duration back.
    """
    return f"asetrate={sr}*{ratio:.4f},aresample={sr},atempo={1/ratio:.4f}"


def voice_at_reference_pitch(src: Path | str, dst: Path | str,
                             target_hz: float = REF_F0_HZ) -> tuple[Path, float]:
    """Apply the fitted chain, first shifting the source onto the target pitch.

    The EQ in CHAIN was fitted against en_US-ryan-high at 167 Hz. A voice that
    sits somewhere else needs moving onto that pitch first, or the fixed
    filter frequencies land on the wrong part of its spectrum.
    """
    measured = median_f0(src)
    ratio = max(min(target_hz / measured, 1.35), 0.70)
    # CHAIN opens with its own resample pair; replace it rather than stacking a
    # second one, which would compound the tempo correction.
    body = CHAIN.split("atempo=1.0417,", 1)[1]
    chain = f"{pitch_chain(ratio)},{body}"
    voice(src, dst, chain)
    return Path(dst), measured
