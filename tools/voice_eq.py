"""Try a voice-processing chain, measure it against the reference band profile."""
import subprocess, sys, wave, numpy as np

BANDS = [(20,80),(80,160),(160,320),(320,640),(640,1280),(1280,2560),(2560,5120),(5120,11000)]
# measured from the reference narration, speech frames only
TARGET = [0.00, 5.61, 44.38, 29.86, 15.04, 4.31, 0.49, 0.30]

def profile(path):
    with wave.open(path) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)/32768.0
    frame, hop = 2048, 1024
    fr = np.stack([x[i*hop:i*hop+frame] for i in range((len(x)-frame)//hop)])
    rms = np.sqrt((fr**2).mean(axis=1))
    f = fr[rms > np.percentile(rms, 70)]
    S = np.abs(np.fft.rfft(f*np.hanning(frame), axis=1))**2
    fq = np.fft.rfftfreq(frame, 1/sr)
    tot = S.sum()
    return [100*S[:, (fq>=lo)&(fq<hi)].sum()/tot for lo, hi in BANDS], np.abs(x).max()

def apply(src, dst, chain):
    subprocess.run(["ffmpeg","-v","error","-i",src,"-af",chain,"-ar","22050","-ac","1","-y",dst], check=True)

if __name__ == "__main__":
    src, dst, chain = sys.argv[1], sys.argv[2], sys.argv[3]
    apply(src, dst, chain)
    got, peak = profile(dst)
    err = 0.0
    print(f"{'band':>14} {'target':>7} {'got':>7}  delta")
    for (lo,hi), t, g in zip(BANDS, TARGET, got):
        err += (t-g)**2
        print(f"{lo:>6}-{hi:<7} {t:6.2f}% {g:6.2f}%  {g-t:+6.2f}")
    print(f"  SSE {err:8.1f}   peak {peak:.3f}")
