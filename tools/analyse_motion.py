"""Measure the camera-motion and transition grammar of a reference video.

This is the measurement half of the reverse-engineering pass on the reference
video. It answers "what is the camera physically doing, frame by frame" so that
a separate pass can answer "when does the editor choose which device".

Stage 1 (``measure``) walks a pre-extracted greyscale frame stack and, for every
frame pair, estimates a similarity model -- horizontal translation ``tx``,
vertical translation ``ty`` and fractional scale change ``z`` -- by running
phase correlation independently on a 3x3 grid of overlapping blocks and
least-squares fitting the three parameters to the nine measured block shifts.
Blocks are used rather than one whole-frame correlation because a whole-frame
correlation collapses pan and zoom into a single ambiguous peak: a zoom moves
the left blocks left and the right blocks right, which a single peak cannot
represent, whereas nine peaks with known lever arms separate the two cleanly.

Two temporal baselines are measured at once:

  lag 1  (33ms)  -- fast events. Whips, cuts, and the exact frame a move starts.
  lag 10 (333ms) -- slow events. A 2%/s zoom displaces a corner pixel by 0.1px
                    in one frame, which is below the subpixel noise floor, but
                    by ~1px over a third of a second, which is measurable.

Alongside the motion it records per-frame sharpness split by axis (mean absolute
horizontal gradient vs vertical gradient), because a horizontal whip smears
vertical edges and collapses the horizontal gradient specifically, and the ratio
of the two discriminates a directional smear from a globally soft frame.

Stage 2 (``classify``) lives in analyse_motion_report.py and consumes the npz
this writes.

Usage:
    python tools/analyse_motion.py extract     # ffmpeg -> raw greyscale stack
    python tools/analyse_motion.py measure     # raw stack -> build/MOT_motion.npz
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

VIDEO = "build/Why We Don't Eat Hippo meat _.mp4"

# Frame stack geometry. 30fps is enough to put 3-6 samples inside a 0.1-0.2s
# whip; 256px wide keeps a 1% scale change at ~1.3px at the frame edge, which
# phase correlation can resolve, while the whole stack still fits on disk.
FPS = 30
W, H = 256, 142
RAW = f"build/MOT_gray_{FPS}fps_{W}x{H}.raw"
OUT = "build/MOT_motion.npz"

# 3x3 grid of 128x72 blocks. Overlapping, so each block still has enough
# texture to correlate, and the lever arms (+-64px in x, +-35px in y) are long
# enough that a scale change shows up as a differential shift between them.
BLOCK_W, BLOCK_H = 128, 72
BLOCK_X = (0, 64, 128)
BLOCK_Y = (0, 35, 70)
LAGS = (1, 10)


def extract_frames() -> None:
    """Decode the reference to a flat greyscale raw stack (idempotent)."""
    if os.path.exists(RAW):
        print(f"{RAW} already present ({os.path.getsize(RAW)/1e9:.2f} GB)")
        return
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", VIDEO,
         "-vf", f"fps={FPS},scale={W}:{H},format=gray",
         "-f", "rawvideo", RAW, "-y"],
        check=True,
    )
    print(f"wrote {RAW} ({os.path.getsize(RAW)/1e9:.2f} GB)")


# --------------------------------------------------------------------------
# phase correlation
# --------------------------------------------------------------------------

_HANN = np.outer(np.hanning(BLOCK_H), np.hanning(BLOCK_W))
# Block centres relative to the frame centre. These are the lever arms the
# scale term is fitted against.
_CENTRES = np.array(
    [[(bx + BLOCK_W / 2) - W / 2, (by + BLOCK_H / 2) - H / 2]
     for by in BLOCK_Y for bx in BLOCK_X],
    dtype=np.float64,
)  # (9, 2) as (cx, cy)


def block_fft(frame: np.ndarray) -> np.ndarray:
    """Windowed rfft2 of all nine blocks at once -> (9, BLOCK_H, BLOCK_W//2+1)."""
    blocks = np.stack(
        [frame[by:by + BLOCK_H, bx:bx + BLOCK_W] for by in BLOCK_Y for bx in BLOCK_X]
    ).astype(np.float64)
    blocks -= blocks.mean(axis=(1, 2), keepdims=True)
    blocks *= _HANN
    return np.fft.rfft2(blocks, axes=(-2, -1))


def _parabolic(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Subpixel offset of the peak of the parabola through (-1,a) (0,b) (1,c)."""
    denom = a - 2.0 * b + c
    off = np.where(np.abs(denom) < 1e-12, 0.0, 0.5 * (a - c) / np.where(denom == 0, 1.0, denom))
    return np.clip(off, -1.0, 1.0)


def block_shifts(f_prev: np.ndarray, f_curr: np.ndarray):
    """Per-block (dx, dy) and correlation peak height between two block FFTs.

    Sign convention: positive dx means image content moved to the RIGHT going
    from f_prev to f_curr, i.e. the camera panned LEFT.
    """
    cross = f_curr * np.conj(f_prev)
    mag = np.abs(cross)
    cross /= np.where(mag < 1e-9, 1.0, mag)
    corr = np.fft.irfft2(cross, s=(BLOCK_H, BLOCK_W), axes=(-2, -1))

    flat = corr.reshape(corr.shape[0], -1)
    idx = np.argmax(flat, axis=1)
    peak = flat[np.arange(flat.shape[0]), idx]
    py, px = np.divmod(idx, BLOCK_W)

    n = np.arange(corr.shape[0])
    ox = _parabolic(corr[n, py, (px - 1) % BLOCK_W], peak, corr[n, py, (px + 1) % BLOCK_W])
    oy = _parabolic(corr[n, (py - 1) % BLOCK_H, px], peak, corr[n, (py + 1) % BLOCK_H, px])

    dx = np.where(px > BLOCK_W // 2, px - BLOCK_W, px) + ox
    dy = np.where(py > BLOCK_H // 2, py - BLOCK_H, py) + oy
    return dx, dy, peak


def fit_similarity(dx, dy, peak):
    """Least-squares fit of tx, ty, z to nine block shifts, with outlier rejection.

    Model, for a block whose centre sits at (cx, cy) from the frame centre:
        dx = tx + z * cx
        dy = ty + z * cy
    so z > 0 is a push-in (content expands away from the centre).

    Returns (tx, ty, z, inlier_count, median_peak, residual_rms).
    """
    cx, cy = _CENTRES[:, 0], _CENTRES[:, 1]
    good = peak > 0.0
    for _ in range(2):
        if good.sum() < 4:
            break
        A = np.concatenate([
            np.stack([np.ones(good.sum()), np.zeros(good.sum()), cx[good]], axis=1),
            np.stack([np.zeros(good.sum()), np.ones(good.sum()), cy[good]], axis=1),
        ])
        b = np.concatenate([dx[good], dy[good]])
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        tx, ty, z = sol
        rx = dx - (tx + z * cx)
        ry = dy - (ty + z * cy)
        res = np.hypot(rx, ry)
        mad = np.median(np.abs(res - np.median(res))) + 1e-6
        new_good = good & (np.abs(res - np.median(res)) < 3.0 * mad + 0.5)
        if new_good.sum() < 4 or (new_good == good).all():
            good = new_good if new_good.sum() >= 4 else good
            break
        good = new_good
    if good.sum() < 4:
        return 0.0, 0.0, 0.0, int(good.sum()), float(np.median(peak)), 99.0
    rx = dx[good] - (tx + z * cx[good])
    ry = dy[good] - (ty + z * cy[good])
    rms = float(np.sqrt(np.mean(rx ** 2 + ry ** 2)))
    return float(tx), float(ty), float(z), int(good.sum()), float(np.median(peak)), rms


def measure() -> None:
    n_frames = os.path.getsize(RAW) // (W * H)
    stack = np.memmap(RAW, dtype=np.uint8, mode="r", shape=(n_frames, H, W))
    print(f"{n_frames} frames at {FPS}fps ({n_frames/FPS:.1f}s)")

    max_lag = max(LAGS)
    ring: list[np.ndarray | None] = [None] * (max_lag + 1)
    ring_gray: list[np.ndarray | None] = [None] * (max_lag + 1)

    res = {
        lag: {k: np.full(n_frames, np.nan) for k in
              ("tx", "ty", "z", "peak", "rms", "inliers")}
        for lag in LAGS
    }
    sharp_h = np.zeros(n_frames)
    sharp_v = np.zeros(n_frames)
    absdiff = np.zeros(n_frames)
    brightness = np.zeros(n_frames)

    t0 = time.time()
    for i in range(n_frames):
        g = stack[i].astype(np.float64)
        sharp_h[i] = np.abs(np.diff(g, axis=1)).mean()
        sharp_v[i] = np.abs(np.diff(g, axis=0)).mean()
        brightness[i] = g.mean()
        f = block_fft(g)
        ring[i % (max_lag + 1)] = f
        prev_gray = ring_gray[(i - 1) % (max_lag + 1)]
        if prev_gray is not None:
            absdiff[i] = np.abs(g - prev_gray).mean()
        ring_gray[i % (max_lag + 1)] = g

        for lag in LAGS:
            j = i - lag
            if j < 0:
                continue
            f_prev = ring[j % (max_lag + 1)]
            dx, dy, peak = block_shifts(f_prev, f)
            tx, ty, z, inl, mp, rms = fit_similarity(dx, dy, peak)
            r = res[lag]
            r["tx"][i], r["ty"][i], r["z"][i] = tx, ty, z
            r["peak"][i], r["rms"][i], r["inliers"][i] = mp, rms, inl

        if i % 3000 == 0 and i:
            el = time.time() - t0
            print(f"  {i}/{n_frames}  {el:.0f}s  eta {el/i*(n_frames-i):.0f}s", flush=True)

    payload = {"fps": np.array(FPS), "n_frames": np.array(n_frames),
               "width": np.array(W), "height": np.array(H),
               "sharp_h": sharp_h, "sharp_v": sharp_v,
               "absdiff": absdiff, "brightness": brightness}
    for lag in LAGS:
        for k, v in res[lag].items():
            payload[f"lag{lag}_{k}"] = v
    np.savez_compressed(OUT, **payload)
    print(f"wrote {OUT} in {time.time()-t0:.0f}s")


def selftest() -> None:
    """Synthetic check that the sign conventions and the scale fit are right."""
    rng = np.random.default_rng(0)
    base = rng.random((H * 2, W * 2))
    # smooth it so it has structure at the scale phase correlation likes
    for _ in range(3):
        base = (base + np.roll(base, 1, 0) + np.roll(base, 1, 1)) / 3.0
    base = (base - base.min()) / (base.max() - base.min()) * 255.0

    def sample(shift_x, shift_y, scale):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        # inverse map: where in the source does this destination pixel come from
        sx = (xx - W / 2 - shift_x) / scale + W
        sy = (yy - H / 2 - shift_y) / scale + H
        x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
        fx, fy = sx - x0, sy - y0
        x0 = np.clip(x0, 0, base.shape[1] - 2)
        y0 = np.clip(y0, 0, base.shape[0] - 2)
        return (base[y0, x0] * (1 - fx) * (1 - fy) + base[y0, x0 + 1] * fx * (1 - fy)
                + base[y0 + 1, x0] * (1 - fx) * fy + base[y0 + 1, x0 + 1] * fx * fy)

    for want_tx, want_ty, want_s in [(0, 0, 1.0), (3.0, 0, 1.0), (-2.0, 1.5, 1.0),
                                     (0, 0, 1.02), (0, 0, 0.98), (1.0, 0, 1.01)]:
        a = sample(0, 0, 1.0)
        b = sample(want_tx, want_ty, want_s)
        fa, fb = block_fft(a), block_fft(b)
        dx, dy, peak = block_shifts(fa, fb)
        tx, ty, z, inl, mp, rms = fit_similarity(dx, dy, peak)
        ok = (abs(tx - want_tx) < 0.35 and abs(ty - want_ty) < 0.35
              and abs(z - (want_s - 1)) < 0.004)
        print(f"  want tx={want_tx:+.2f} ty={want_ty:+.2f} z={want_s-1:+.4f} | "
              f"got tx={tx:+.2f} ty={ty:+.2f} z={z:+.4f} peak={mp:.3f} rms={rms:.2f} "
              f"{'OK' if ok else 'FAIL'}")

    # a hard cut: two unrelated frames should collapse the peak height
    other = rng.random((H, W)) * 255.0
    fa, fb = block_fft(sample(0, 0, 1.0)), block_fft(other)
    _, _, peak = block_shifts(fa, fb)
    print(f"  uncorrelated frames -> median peak {np.median(peak):.3f} "
          f"(matched frames are ~0.1-0.9)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "measure"
    {"extract": extract_frames, "measure": measure, "selftest": selftest}[cmd]()
