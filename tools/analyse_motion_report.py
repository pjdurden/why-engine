"""Turn the raw per-frame motion measurements into a shot/transition grammar.

Consumes build/MOT_motion.npz (written by tools/analyse_motion.py) plus a
low-res RGB stack for content classification, and produces:

  * a shot list derived from the pixels rather than from ffmpeg's scene
    detector, which at threshold 0.20 misses 35 cuts between similar white
    diagram cards
  * a motion class per shot (static hold / push-in / pull-out / pan / local
    element animation) with the rate, the total travel, and where in the shot
    the move starts and stops -- the last of these is what separates a
    shot-long Ken Burns push from a hold that only starts moving partway in
  * a transition class per boundary (hard cut / whip / punch) with duration,
    blur depth and the zoom ramps on either side
  * the device-to-device transition matrix, the P(device | shot length) and
    P(device | content) tables, and a runs test -- i.e. the evidence for and
    against a pattern, not just the summary

Findings are written up in
docs/superpowers/specs/2026-09-16-reference-motion-analysis.md.
Everything is printed; nothing is written back into the repo.

Usage:
    python tools/analyse_motion_report.py            # full report
    python tools/analyse_motion_report.py dump FILE  # also dump per-shot TSV
"""
from __future__ import annotations

import collections
import os
import re
import sys

import numpy as np

NPZ = "build/MOT_motion.npz"
SCENES = "build/ANALYSIS_scenes.txt"
SCENES_LOW = "build/MOT_scenes_006.txt"
RGB_RAW = "build/MOT_rgb_4fps_160x89.raw"
RGB_FPS, RGB_W, RGB_H = 4, 160, 89


def load():
    d = dict(np.load(NPZ))
    d["fps"] = int(d["fps"])
    return d


def scene_times(path: str) -> list[float]:
    if not os.path.exists(path):
        return []
    txt = open(path).read()
    return [float(m) for m in re.findall(r"pts_time:([\d.]+)", txt)]


def paper_fraction(t: float) -> float:
    """Fraction of the frame at time t that is near-white unsaturated paper.

    Cheap proxy for content type: a line-art diagram on a white card scores
    above ~0.72, a full-bleed painted scene below ~0.40.
    """
    n = os.path.getsize(RGB_RAW) // (RGB_W * RGB_H * 3)
    i = int(np.clip(round(t * RGB_FPS), 0, n - 1))
    stack = np.memmap(RGB_RAW, dtype=np.uint8, mode="r", shape=(n, RGB_H, RGB_W, 3))
    f = np.asarray(stack[i], dtype=np.float64)
    lum = f.mean(axis=2)
    sat = f.max(axis=2) - f.min(axis=2)
    return float(((lum > 185) & (sat < 45)).mean())


def content_type(paper: float) -> str:
    return "paper/diagram" if paper > 0.72 else ("mixed" if paper > 0.40 else "full-bleed")


# --------------------------------------------------------------------------
# transition events and shots
# --------------------------------------------------------------------------

CORR_COLLAPSE = 0.5     # phase-correlation peak below this == content discontinuity
EVENT_MERGE = 6         # frames; joins the blurred frames of one whip into one event
FROZEN = 0.02           # mean abs frame difference below this == bit-identical frame
RAMP_PCT_S = 3.0        # %/s; above this a zoom is a transition ramp, not a Ken Burns move
ZOOM_MIN_PCT_S = 0.03   # %/s; below this a zoom is measurement noise


def find_events(d):
    """Transition events = runs of phase-correlation collapse.

    One hard cut gives exactly one frame of collapse. A whip gives a run,
    because every blurred frame decorrelates from its neighbour.

    The absdiff guard matters: on a near-blank card (a white title slate) the
    phase-correlation surface is flat and the peak height collapses even though
    the frame is bit-identical to the one before it. Requiring real pixel change
    as well removes that whole false-positive family.
    """
    p1, ad = d["lag1_peak"], d["absdiff"]
    idx = np.where((p1 < CORR_COLLAPSE) & (ad > 0.5))[0]
    events, start, prev = [], idx[0], idx[0]
    for k in idx[1:]:
        if k - prev > EVENT_MERGE:
            events.append((start, prev + 1))
            start = k
        prev = k
    events.append((start, prev + 1))
    return events


def classify_event(d, a, b, fps=30):
    """Measure one transition. Returns a feature dict including its class.

    Motion estimates *inside* the collapse window are meaningless for a hard cut
    (two unrelated images produce a random correlation peak), so the class is
    decided from evidence that survives: the number of collapsed frames, whether
    the sharpness dips below BOTH neighbouring shots, and the zoom ramp measured
    in the frames on either side where correlation is still valid.
    """
    sh, sv, z1, tx1, ty1 = d["sharp_h"], d["sharp_v"], d["lag1_z"], d["lag1_tx"], d["lag1_ty"]
    n = len(sh)
    w = int(0.4 * fps)
    pre = slice(max(a - w, 0), max(a - 1, 1))
    post = slice(min(b + 1, n - 1), min(b + w, n))
    base_h = min(np.median(sh[pre]), np.median(sh[post]))
    aniso_base = np.median([np.median(sh[pre]) / max(np.median(sv[pre]), 1e-6),
                            np.median(sh[post]) / max(np.median(sv[post]), 1e-6)])
    core = slice(max(a - 1, 0), min(b + 1, n))
    blur = float(sh[core].min() / max(base_h, 1e-6))
    j = int(np.argmin(sh[core])) + max(a - 1, 0)
    aniso = float((sh[j] / max(sv[j], 1e-6)) / max(aniso_base, 1e-6))

    rate = z1 * fps * 100.0
    pre_ramp = float(np.abs(rate[pre]).max()) if pre.stop > pre.start else 0.0
    post_ramp = float(np.abs(rate[post]).max()) if post.stop > post.start else 0.0
    pre_sign = float(np.sign(rate[pre][np.argmax(np.abs(rate[pre]))])) if pre.stop > pre.start else 0.0
    post_sign = float(np.sign(rate[post][np.argmax(np.abs(rate[post]))])) if post.stop > post.start else 0.0
    slide_x = float(np.sum(tx1[a:b]))
    slide_y = float(np.sum(ty1[a:b]))

    n_collapsed = b - a
    # The sign of the ramp on the far side of the cut separates the two dynamic
    # devices cleanly, and agrees with eyeball inspection on all 7 instances:
    # a whip decelerates out of the cut (push-in, smear, then pull-out to rest),
    # whereas a punch keeps accelerating through it (push-in on both sides).
    if n_collapsed >= 2 and blur < 0.6:
        kind = "whip" if post_sign < 0 else "punch_dissolve"
    elif max(pre_ramp, post_ramp) >= RAMP_PCT_S:
        kind = "punch_cut"
    else:
        kind = "hard_cut"
    return dict(a=a, b=b, t=a / fps, frames=n_collapsed, dur=n_collapsed / fps,
                blur=blur, aniso=aniso, pre_ramp=pre_ramp, post_ramp=post_ramp,
                pre_sign=pre_sign, post_sign=post_sign,
                slide_x=slide_x, slide_y=slide_y, kind=kind)


def shot_motion(d, s, e, fps=30):
    """Classify the camera move inside one shot, ignoring transition ramps.

    Returns the dominant device plus, for a zoom, its rate, its total travel and
    where in the shot the move starts and stops -- which is what distinguishes a
    shot-long Ken Burns push from a hold that only starts moving partway in.
    """
    ad, z1, tx1, ty1, p1 = d["absdiff"], d["lag1_z"], d["lag1_tx"], d["lag1_ty"], d["lag1_peak"]
    pad = int(0.2 * fps)          # skip the ease-in/ease-out belonging to the transitions
    a, b = s + pad, e - pad
    if b - a < int(0.3 * fps):
        return dict(kind="too_short", dur=(e - s) / fps)

    rate = z1[a:b] * fps * 100.0
    moving = (ad[a:b] > FROZEN) & (p1[a:b] > 0.6)
    zooming = moving & (np.abs(rate) > ZOOM_MIN_PCT_S) & (np.abs(tx1[a:b]) < 0.3) & (np.abs(ty1[a:b]) < 0.3)
    panning = moving & ((np.abs(tx1[a:b]) > 0.3) | (np.abs(ty1[a:b]) > 0.3))

    out = dict(dur=(e - s) / fps, frac_frozen=float((ad[a:b] <= FROZEN).mean()),
               frac_zoom=float(zooming.mean()), frac_pan=float(panning.mean()),
               frac_local=float((moving & ~zooming & ~panning).mean()))
    if zooming.sum() >= int(0.25 * fps):
        zr = float(np.median(rate[zooming]))
        first, last = np.where(zooming)[0][[0, -1]]
        out.update(kind="zoom_in" if zr > 0 else "zoom_out", rate=zr,
                   total=float(np.sum(rate[zooming]) / fps),
                   onset=float(first / fps), offset=float((b - a - 1 - last) / fps),
                   move_dur=float((last - first + 1) / fps))
    elif panning.sum() >= int(0.25 * fps):
        out.update(kind="pan", rate=float(np.median(tx1[a:b][panning]) * fps))
    elif out["frac_frozen"] > 0.9:
        out.update(kind="hold")
    else:
        out.update(kind="local_anim")
    return out


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def build(d):
    """Return (events, shots) with every shot carrying its in/out transition."""
    fps = d["fps"]
    n = int(d["n_frames"])
    events = [classify_event(d, a, b, fps) for a, b in find_events(d)]
    bounds = [(0, events[0]["a"])]
    for i in range(len(events) - 1):
        bounds.append((events[i]["b"], events[i + 1]["a"]))
    bounds.append((events[-1]["b"], n))
    shots = []
    for i, (s, e) in enumerate(bounds):
        m = shot_motion(d, s, e, fps)
        m.update(idx=i, s=s, e=e, t=s / fps,
                 in_tr=events[i - 1]["kind"] if i > 0 else None,
                 out_tr=events[i]["kind"] if i < len(events) else None)
        shots.append(m)
    return events, shots


def pct(x, n):
    return f"{x:4d} ({100*x/max(n,1):5.1f}%)"


def main():
    d = load()
    fps = d["fps"]
    events, shots = build(d)
    n = len(shots)
    print(f"{len(events)} transitions, {n} shots, {int(d['n_frames'])/fps:.1f}s\n")

    print("== shot devices ==")
    c = collections.Counter(s["kind"] for s in shots)
    for k, v in c.most_common():
        print(f"  {k:12s} {pct(v, n)}")

    print("\n== transition devices ==")
    ce = collections.Counter(e["kind"] for e in events)
    for k, v in ce.most_common():
        print(f"  {k:12s} {pct(v, len(events))}")

    zs = [s for s in shots if s["kind"].startswith("zoom")]
    print(f"\n== zoom rates (%/s), {len(zs)} zoom shots ==")
    r = np.array([s["rate"] for s in zs])
    tot = np.array([s["total"] for s in zs])
    for lo, hi in [(-99, -1.25), (-1.25, -0.9), (-0.9, -0.55), (-0.55, -0.3),
                   (-0.3, 0), (0, 0.13), (0.13, 0.23), (0.23, 0.35),
                   (0.35, 0.55), (0.55, 0.9), (0.9, 99)]:
        m = (r >= lo) & (r < hi)
        if m.sum():
            print(f"  {lo:+6.2f}..{hi:+6.2f}: {m.sum():4d}  median total travel {np.median(tot[m]):+6.2f}%")
    print(f"  overall: push-in {sum(1 for s in zs if s['rate']>0)}, pull-out {sum(1 for s in zs if s['rate']<0)}")
    print(f"  |total travel| median {np.median(np.abs(tot)):.2f}%  p10 {np.percentile(np.abs(tot),10):.2f}  p90 {np.percentile(np.abs(tot),90):.2f}")

    print("\n== where the move sits inside the shot ==")
    on = np.array([s["onset"] for s in zs])
    off = np.array([s["offset"] for s in zs])
    dur = np.array([s["dur"] for s in zs])
    print(f"  onset after shot start: median {np.median(on):.2f}s  p75 {np.percentile(on,75):.2f}s  p90 {np.percentile(on,90):.2f}s")
    print(f"  settle before shot end : median {np.median(off):.2f}s  p75 {np.percentile(off,75):.2f}s  p90 {np.percentile(off,90):.2f}s")
    print(f"  onset > 0.5s (static-then-move): {sum(on > 0.5)}/{len(zs)} ({100*np.mean(on>0.5):.0f}%)")
    print(f"  move covers >85% of shot       : {sum((on<0.3)&(off<0.3))}/{len(zs)} ({100*np.mean((on<0.3)&(off<0.3)):.0f}%)")

    print("\n== device frequency per 2-minute bucket ==")
    print(f"  {'bucket':>12s} {'shots':>6s} {'hold':>6s} {'zoom_in':>8s} {'zoom_out':>9s} {'local':>7s} {'whip':>5s} {'punch':>6s} {'cuts/min':>9s} {'medlen':>7s}")
    for b0 in range(0, 1001, 120):
        sub = [s for s in shots if b0 <= s["t"] < b0 + 120]
        bev = [e for e in events if b0 <= e["t"] < b0 + 120]
        if not sub:
            continue
        cc = collections.Counter(s["kind"] for s in sub)
        cv = collections.Counter(e["kind"] for e in bev)
        print(f"  {b0:5d}-{b0+120:5d} {len(sub):6d} {cc['hold']:6d} {cc['zoom_in']:8d} {cc['zoom_out']:9d} "
              f"{cc['local_anim']:7d} {cv['whip']:5d} "
              f"{cv['punch_cut']+cv['punch_dissolve']:6d} {len(bev)/2.0:9.1f} "
              f"{np.median([s['dur'] for s in sub]):7.2f}")

    print("\n== transition matrix: shot device N -> device N+1 ==")
    keys = ["hold", "zoom_in", "zoom_out", "local_anim"]
    mat = collections.Counter((a["kind"], b["kind"]) for a, b in zip(shots, shots[1:]))
    header = "from/to"
    print(f"  {header:>12s}" + "".join(f"{k:>12s}" for k in keys) + f"{'n':>6s}")
    for k in keys:
        row = [mat[(k, j)] for j in keys]
        tot_r = sum(mat[(k, j)] for j in keys)
        print(f"  {k:>12s}" + "".join(f"{v:>6d}{'':>1s}{100*v/max(tot_r,1):4.0f}%" for v in row) + f"{tot_r:>6d}")
    print("  (compare against the marginal, i.e. an independence null)")
    marg = collections.Counter(s["kind"] for s in shots)
    tt = sum(marg[k] for k in keys)
    print("   marginal  " + "".join(f"{100*marg[k]/tt:>11.0f}%" for k in keys))

    print("\n== zoom presets: total travel is fixed, the rate is derived ==")
    real = [s for s in zs if abs(s["total"]) > 0.3]
    fams = {"gentle push": [s for s in real if 0.5 < s["total"] < 2.5],
            "strong push": [s for s in real if s["total"] >= 2.5],
            "pull-out": [s for s in real if s["total"] < -0.5]}
    print(f"  {'family':12s}{'n':>4s}{'total%':>9s}{'sd':>6s}{'rate%/s':>9s}"
          f"{'head/dur':>10s}{'move/dur':>10s}{'tail/dur':>10s}{'shotlen':>9s}")
    for name, g in fams.items():
        if not g:
            continue
        print(f"  {name:12s}{len(g):4d}{np.mean([x['total'] for x in g]):+9.2f}"
              f"{np.std([x['total'] for x in g]):6.2f}{np.median([x['rate'] for x in g]):+9.2f}"
              f"{np.median([x['onset']/x['dur'] for x in g]):10.2f}"
              f"{np.median([x['move_dur']/x['dur'] for x in g]):10.2f}"
              f"{np.median([x['offset']/x['dur'] for x in g]):10.2f}"
              f"{np.median([x['dur'] for x in g]):9.2f}")

    print("\n== P(move | shot length), body only (t > 164s, after the last whip) ==")
    body = [s for s in shots if s["t"] > 164]
    for lo, hi in [(0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5), (5, 7), (7, 10), (10, 99)]:
        sub = [s for s in body if lo <= s["dur"] < hi]
        if not sub:
            continue
        mv = sum(1 for s in sub if s["kind"].startswith("zoom"))
        print(f"  {lo:4.1f}-{hi:4.1f}s  n={len(sub):4d}  moves {mv:3d}  ({mv/len(sub):5.0%})")

    print("\n== P(device | content type) ==")
    for s in shots:
        s["paper"] = paper_fraction(s["t"] + s["dur"] * 0.5)
    tab = collections.Counter((content_type(s["paper"]), s["kind"]) for s in shots)
    print(f"  {'content':>14s}" + "".join(f"{k:>12s}" for k in keys) + f"{'n':>6s}")
    for ct in ("paper/diagram", "mixed", "full-bleed"):
        tot_c = sum(tab[(ct, k)] for k in keys)
        print(f"  {ct:>14s}" + "".join(f"{tab[(ct,k)]:>6d}{100*tab[(ct,k)]/max(tot_c,1):5.0f}%"
                                      for k in keys) + f"{tot_c:>6d}")
    print(f"  {'marginal':>14s}" + "".join(f"{'':>6s}{100*marg[k]/tt:5.0f}%" for k in keys))

    print("\n== runs test: is the hold/move sequence anything but independent? ==")
    b = np.array([0 if s["kind"] == "hold" else 1 for s in shots])
    runs = 1 + int((np.diff(b) != 0).sum())
    n1, n0 = int(b.sum()), int(len(b) - b.sum())
    exp = 1 + 2 * n0 * n1 / len(b)
    sd = np.sqrt(2 * n0 * n1 * (2 * n0 * n1 - len(b)) / (len(b) ** 2 * (len(b) - 1)))
    print(f"  runs={runs} expected={exp:.1f} sd={sd:.1f} z={(runs-exp)/sd:+.2f} "
          f"({'no structure' if abs((runs-exp)/sd) < 2 else 'STRUCTURE'})")

    print("\n== zoom-direction alternation across a cut ==")
    pairs = [(a["kind"], b["kind"]) for a, b in zip(shots, shots[1:])
             if a["kind"].startswith("zoom") and b["kind"].startswith("zoom")]
    same = sum(1 for a, b in pairs if a == b)
    print(f"  consecutive zoom shots: {len(pairs)}; same direction {same} ({100*same/max(len(pairs),1):.0f}%), reversed {len(pairs)-same}")

    print("\n== shot length by device ==")
    for k in keys:
        v = [s["dur"] for s in shots if s["kind"] == k]
        if v:
            print(f"  {k:12s} n={len(v):4d} median {np.median(v):5.2f}s  p10 {np.percentile(v,10):5.2f}  p90 {np.percentile(v,90):5.2f}")

    print("\n== outgoing shot length before each transition type ==")
    for k in ce:
        v = [shots[i]["dur"] for i, e in enumerate(events) if e["kind"] == k]
        if v:
            print(f"  {k:12s} n={len(v):4d} median {np.median(v):5.2f}s  range {min(v):.2f}-{max(v):.2f}")

    if len(sys.argv) > 2 and sys.argv[1] == "dump":
        with open(sys.argv[2], "w") as fh:
            fh.write("idx\tt\tdur\tkind\trate\ttotal\tonset\toffset\tin_tr\tout_tr\n")
            for s in shots:
                fh.write(f"{s['idx']}\t{s['t']:.2f}\t{s['dur']:.2f}\t{s['kind']}\t"
                         f"{s.get('rate',0):.3f}\t{s.get('total',0):.3f}\t"
                         f"{s.get('onset',-1):.2f}\t{s.get('offset',-1):.2f}\t{s['in_tr']}\t{s['out_tr']}\n")
        print(f"\nwrote {sys.argv[2]}")


if __name__ == "__main__":
    main()
