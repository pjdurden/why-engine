"""Measure spectral balance and pitch of a mono WAV, for voice matching."""
import sys, wave, numpy as np

BANDS = [(20,80),(80,160),(160,320),(320,640),(640,1280),(1280,2560),(2560,5120),(5120,11000)]

def load(path):
    with wave.open(path) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    return sr, x

def voiced_frames(x, sr, frame=2048, hop=1024):
    """Frames whose RMS is in the top 55% -- speech, not gaps."""
    n = (len(x) - frame) // hop
    fr = np.stack([x[i*hop:i*hop+frame] for i in range(n)])
    rms = np.sqrt((fr**2).mean(axis=1))
    keep = rms > np.percentile(rms, 45)
    return fr[keep]

def band_energy(fr, sr):
    win = np.hanning(fr.shape[1])
    spec = np.abs(np.fft.rfft(fr * win, axis=1))**2
    freqs = np.fft.rfftfreq(fr.shape[1], 1/sr)
    total = spec.sum()
    return [(lo, hi, 100.0 * spec[:, (freqs>=lo)&(freqs<hi)].sum() / total) for lo, hi in BANDS]

def pitch(fr, sr, fmin=60, fmax=300):
    """Per-frame autocorrelation F0; median over frames with a clear peak."""
    out = []
    lo, hi = int(sr/fmax), int(sr/fmin)
    for f in fr:
        f = f - f.mean()
        if not f.any():
            continue
        ac = np.correlate(f, f, mode='full')[len(f)-1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        seg = ac[lo:hi]
        k = int(np.argmax(seg))
        if seg[k] > 0.35:
            out.append(sr / (lo + k))
    return np.array(out)

for path in sys.argv[1:]:
    sr, x = load(path)
    fr = voiced_frames(x, sr)
    print(f"\n=== {path}  ({len(x)/sr:.1f}s, {sr} Hz, {len(fr)} voiced frames) ===")
    for lo, hi, pct in band_energy(fr, sr):
        print(f"  {lo:>5}-{hi:<5} Hz  {pct:6.2f}%  {'#' * int(pct)}")
    f0 = pitch(fr, sr)
    if len(f0):
        print(f"  F0 median {np.median(f0):.1f} Hz   p25 {np.percentile(f0,25):.1f}   p75 {np.percentile(f0,75):.1f}   n={len(f0)}")
    print(f"  RMS {np.sqrt((x**2).mean()):.4f}  peak {np.abs(x).max():.3f}")
