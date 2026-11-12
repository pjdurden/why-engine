"""Pitch modulation: how much a voice moves, independent of how deep it is.

A monotone read and an expressive one can share a median F0. What separates
them is spread. Reporting the interquartile range as a FRACTION of the median
keeps the measure scale-invariant, so a deep voice and a high one are compared
on the same footing and lowering the pitch does not flatter the number.
"""
import sys, wave
import numpy as np


def f0_track(path, fmin=60.0, fmax=340.0):
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float) / 32768
    frame, hop = 2048, 512
    fr = np.stack([x[i*hop:i*hop+frame] for i in range((len(x)-frame)//hop)])
    rms = np.sqrt((fr**2).mean(axis=1))
    lo, hi = int(sr/fmax), int(sr/fmin)
    out = []
    for f in fr[rms > np.percentile(rms, 55)]:
        f = f - f.mean()
        if not f.any():
            continue
        ac = np.correlate(f, f, mode="full")[len(f)-1:]
        if ac[0] <= 0:
            continue
        seg = (ac/ac[0])[lo:hi]
        k = int(np.argmax(seg))
        if seg[k] > 0.35:
            out.append(sr/(lo+k))
    return np.array(out)


def modulation(path):
    f0 = f0_track(path)
    med = np.median(f0)
    iqr = np.percentile(f0, 75) - np.percentile(f0, 25)
    return dict(median=med, p25=np.percentile(f0, 25), p75=np.percentile(f0, 75),
                iqr=iqr, relative=iqr/med, n=len(f0))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        m = modulation(p)
        print(f"{p:<46} median {m['median']:5.1f}Hz  p25 {m['p25']:5.1f}  p75 {m['p75']:5.1f}  "
              f"IQR {m['iqr']:5.1f}  relative {m['relative']:.3f}")
