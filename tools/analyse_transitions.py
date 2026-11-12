"""Detect blur-based transitions by temporal sharpness collapse at shot boundaries.

A whip/blur transition makes one or two frames dramatically softer than the
frames on either side of it. A hard cut does not.
"""
import io, re, subprocess, numpy as np
from PIL import Image

VIDEO = "build/Why We Don't Eat Hippo meat _.mp4"
OFFSETS = [-0.30, -0.20, -0.14, -0.10, -0.06, -0.03, 0.03, 0.10, 0.20, 0.30]

def sharpness(t):
    p = subprocess.run(["ffmpeg","-v","error","-ss",f"{max(t,0):.3f}","-i",VIDEO,"-frames:v","1",
                        "-vf","scale=320:-2","-f","image2pipe","-vcodec","png","-"],
                       capture_output=True)
    if not p.stdout:
        return None
    g = np.asarray(Image.open(io.BytesIO(p.stdout)).convert("L"), dtype=np.float64)
    return np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean()

times = [float(m) for m in re.findall(r"pts_time:([\d.]+)", open("build/ANALYSIS_scenes.txt").read())]
sample = times[::5][:45]

drops, blurred = [], 0
for t in sample:
    s = {o: sharpness(t + o) for o in OFFSETS}
    s = {o: v for o, v in s.items() if v is not None}
    if len(s) < 6:
        continue
    lo = min(s.values())
    o_lo = min(s, key=s.get)
    baseline = np.median([s[o] for o in s if abs(o - o_lo) > 0.12])
    ratio = lo / max(baseline, 1e-6)
    drops.append(ratio)
    if ratio < 0.55:
        blurred += 1
        print(f"  t={t:7.2f}  blur trough at {o_lo:+.2f}s  sharpness {ratio:.2f}x baseline  BLUR TRANSITION")

print(f"\nsampled {len(drops)} boundaries")
print(f"  blur transitions : {blurred}  ({100*blurred/len(drops):.0f}%)")
print(f"  sharpness-trough ratio: median {np.median(drops):.2f}  p10 {np.percentile(drops,10):.2f}  min {min(drops):.2f}")
