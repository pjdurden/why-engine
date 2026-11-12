# Reference video: camera motion and transition grammar

Measured 2026-09-16 against `build/Why We Don't Eat Hippo meat _.mp4`
(1000.96s, 1280x714, 60fps). Video only. Every number below is derived from
pixels, not from the scene detector.

Code: `tools/analyse_motion.py` (stage 1, per-frame motion) and
`tools/analyse_motion_report.py` (stage 2, shots, transitions, pattern tests).
All intermediates live under `build/` and are gitignored.

---

## 0. Headline

The reference is **far more static than the previous note claimed, and far less
varied**. It has exactly three camera devices and effectively one transition.

| | share |
|---|---|
| Frames bit-identical to the previous frame | **76.0%** (760.7s) |
| Frames in a linear zoom | 20.8% (208.2s) |
| Frames with local element animation, camera locked | 1.8% (17.6s) |
| Frames inside a transition | 1.25% (12.5s) |
| Frames panning or sliding | **0.07%** (0.7s — all of it noise) |

- There is **no pan, no drift, no slide and no swipe anywhere in the video.**
  Not slow, not fast. The camera only ever holds or scales about the frame centre.
- **97.6% of transitions are single-frame hard cuts.** There are 4 whips and
  3 punches in the whole video, and **all 7 occur before t=164s** (p = 5e-6).
- The "slow zoom on every shot, alternating push and pull" model is wrong.
  **81% of shots never move at all.** A move happens only on long shots.
- The previously reported "31% of boundaries show a blur trough" is a
  measurement artifact. Reproduced exactly (31.2%), and 86 of the 92 flagged
  boundaries are plain hard cuts. See §7.

---

## 1. Methodology

### 1.1 Frame stack

`ffmpeg -vf "fps=30,scale=256:142,format=gray" -f rawvideo` → 30027 frames,
1.09 GB, at `build/MOT_gray_30fps_256x142.raw`. 30fps puts 3-6 samples inside a
0.1-0.2s whip; 256px wide keeps a 1% scale change at ~1.3px at the frame edge,
which phase correlation resolves. A parallel 4fps RGB stack at 160x89
(`build/MOT_rgb_4fps_160x89.raw`) is used only for content classification.

### 1.2 Global motion: block phase correlation + similarity fit

A whole-frame phase correlation cannot represent a zoom — a zoom moves the left
of the frame left and the right of the frame right, which a single correlation
peak cannot encode. So each frame is split into a **3x3 grid of overlapping
128x72 blocks** (x offsets 0/64/128, y offsets 0/35/70), each Hann-windowed and
mean-subtracted, and each correlated independently against the same block in an
earlier frame. The nine (dx, dy) peaks, with subpixel parabolic refinement, are
least-squares fitted to

```
dx_i = tx + z * cx_i
dy_i = ty + z * cy_i
```

where (cx_i, cy_i) is the block centre relative to the frame centre. `z` is the
fractional scale change; **z > 0 is a push-in**. Two rounds of MAD-based outlier
rejection. Sign convention: positive `tx` means image content moved right, i.e.
the camera panned left.

Two temporal baselines are computed in one pass, using a ring buffer of block
FFTs: **lag 1** (33ms, for cuts/whips/motion-onset timing) and **lag 10**
(333ms, kept for reference). All the results below use lag 1, integrated.

Validated on synthetic data (`python tools/analyse_motion.py selftest`): pure
translations of 2-3px recovered to within 0.02px, scale changes of ±1% and ±2%
recovered to within 0.001, and two uncorrelated frames drop the median peak
height to 0.055 versus ~0.6-1.0 for matched frames.

### 1.3 Sharpness, split by axis

Mean absolute gradient, computed separately along x (`sharp_h`) and y
(`sharp_v`) — the same statistic as `why_engine.art.transitions.sharpness`, just
not summed. A horizontal smear destroys the horizontal gradient and leaves the
vertical one, so the ratio discriminates a directional smear from a globally
soft frame — *in principle*. In practice it fails on near-white cards; see §8.

### 1.4 Boundary detection (not the scene detector)

A transition is a **run of frames where the lag-1 phase-correlation peak
collapses below 0.5 AND the mean absolute frame difference exceeds 0.5**, with
runs merged across gaps of up to 6 frames. Both conditions are needed: on a
near-blank white slate the correlation surface is flat and the peak collapses
even though the frame is bit-identical to its predecessor (this produced a
spurious 0.87s "event" at t=799.9 before the guard was added).

**This found 295 boundaries against the cached detector's 260.** Every one of
the 260 scene-detector boundaries has a correlation collapse — the detector
produced no false positives. The 35 extra are cuts the detector at threshold
0.20 missed (e.g. 98.25, 100.15, 101.75, 106.58, 109.15, 414.98, 415.78,
416.72, 472.58, 473.58, 474.32, 475.28, 904.32, 905.42, 906.82, 908.58),
almost all of them cuts between two similar white diagram cards.

Resulting shot statistics (295 transitions, 296 shots): median 2.77s,
mean 3.34s, p10 1.37s, p90 6.30s, max 14.97s.

### 1.5 Shot motion classification

For each shot, 0.2s is trimmed from each end to exclude any transition ramp.
Interior frames are then labelled frozen (absdiff ≤ 0.02), zooming
(|rate| > 0.03 %/s, |tx| and |ty| < 0.3px, correlation peak > 0.6), panning
(|tx| or |ty| > 0.3px), or local animation (pixels change but the global model
is identity). A shot is a zoom if ≥ 0.25s of it is zooming; the reported rate is
the median over the zooming frames, and onset/settle are the gaps between the
shot ends and the first/last zooming frame.

### 1.6 Limitations

- **Motion estimates inside a hard cut are meaningless.** Two unrelated images
  give a random correlation peak. `tx = -23px` at t=490.47 is noise, not a
  swipe; the frames on either side are bit-identical to their neighbours. Any
  direction or magnitude quoted for a *hard cut* would be fiction, and none is.
- Per-frame zoom sensitivity floor is roughly 0.03 %/s. A push slower than that
  would read as a hold. The slowest real move found is 0.11 %/s, comfortably above.
- 256px wide means sub-0.05px translations are invisible; a pan slower than
  ~2 px/s in the 1280px original would be missed. Given that the measured pan
  share is 0.07% and the *fastest* candidate pans are only a few px/s over
  several seconds, this is not hiding a swipe device.
- Content classification (paper vs scene) uses a single mid-shot frame at
  160x89, so it is coarse; it distinguishes "white background with line art"
  from "full-bleed painted scene" reliably and nothing finer.
- The axis-split sharpness ratio is unreliable when both shots are near-white
  cards with little vertical structure (§8, t=61.40).

---

## 2. Motion taxonomy — what the shots actually do

### 2.1 Shot device counts (n = 296)

| device | n | share | median shot length |
|---|---|---|---|
| **static hold** (camera and content both frozen) | 240 | 81.1% | 2.53s |
| **push-in** (linear scale up, centred) | 29 | 9.8% | 6.27s |
| **pull-out** (linear scale down, centred) | 12 | 4.1% | 3.75s |
| **local element animation** (camera locked, one element moves/appears) | 15 | 5.1% | 1.43s |
| pan / slide / drift | **0** | **0%** | — |

The 15 local-animation shots are short beats where the camera never moves but an
element pops in or animates (a check mark, a label, a stamp). Median 1.43s, with
only 23% of the shot's frames changing. They are almost all on white diagram
cards (mean paper fraction 0.72, the highest of any device).

### 2.2 The moves are linear, not eased

The per-frame rate inside a move is a **flat plateau with an abrupt start and an
abrupt stop**, not an ease curve. Example, the 14.5s shot at t=371.5:

```
  0- 20% of shot   rate 0.000 %/s   100% of frames bit-identical
 20- 25%           rate +0.037      switching on
 25- 95%           rate +0.109 .. +0.115 %/s, dead flat
 95-100%           0, frozen again
```

Drift across a move is under 15% of the rate (pull-outs accelerate slightly,
+0.44 → +0.39 %/s for strong pushes). For implementation, treat it as linear.

### 2.3 Three zoom presets, each with a fixed total travel

Total travel quantises hard; **the rate is whatever makes the fixed total fit
the shot**, which is why a naive per-frame rate histogram looks like a
continuum.

| family | n | total travel | rate | head hold | move | tail hold | shot length |
|---|---|---|---|---|---|---|---|
| **gentle push** | 21 | **+1.19% ± 0.30** | +0.23 %/s (median) | **19% of shot** | 73% | 1% | 5.93s |
| **strong push** | 7 | **+5.26% ± 2.02** | +0.59 %/s | 5% | 90% | 2% | 7.57s |
| **pull-out** | 11 | **−3.66% ± 0.20** | −1.16 %/s | 2% | 78% | **10%** | 3.73s |

Two more zoom shots are degenerate (|total| < 0.3%) and one is the t=20.9 punch;
they are excluded from the family table but counted in §2.1.

The rate is not a free parameter. For the gentle push, `r(onset, shot length) =
+0.93` and `r(move duration, shot length) = +0.99` while the total stays pinned
at +1.19%. The editor is placing a fixed scale delta across a proportional
window, not picking a speed.

### 2.4 Static-then-move is real, and it is specific to the push-in

- **54% of all zoom shots (22/41) start moving more than 0.5s after the cut.**
- The head hold is **proportional, not fixed**: median 19% of the shot for a
  gentle push, absolute onsets ranging 0.00s to 3.17s.
- **Pull-outs do not do this.** They start at the cut (head 2% of shot) and
  instead hold at the *end*, settling and then sitting still for the last 10%.

So the shape is, per family:

```
gentle push:  [hold 19%][push +1.2% over 73%][hold 8%]
strong push:  [hold  5%][push +5.3% over 90%][hold 5%]
pull-out:     [pull -3.7% over 78%][hold 22%]
```

Only 12% of zoom shots (5/41) have the move covering more than 85% of the shot.

---

## 3. Transition taxonomy

### 3.1 Counts (n = 295)

| transition | n | share | duration | where |
|---|---|---|---|---|
| **hard cut** | 288 | 97.6% | 1 frame (0.033s) | everywhere |
| **whip** | 4 | 1.4% | 0.60–0.90s total | t = 9.3, 61.4, 70.5, 163.4 |
| **punch dissolve** | 2 | 0.7% | 0.35–0.50s | t = 48.0, 93.7 |
| **punch cut** | 1 | 0.3% | 0.30s ramp + cut | t = 15.1 |
| dissolve / crossfade (standalone) | **0** | 0% | — | — |
| slow swipe / slide | **0** | 0% | — | — |

### 3.2 The whip, anatomically

Measured at t=61.65 and confirmed on all four instances:

| phase | duration | measurement |
|---|---|---|
| outgoing shot accelerates into a push-in | 0.13–0.20s | rate ramps 0.2 → +28 to +49 %/s |
| horizontal motion smear + the cut, hidden inside it | 0.13–0.23s | sharp_h falls to **0.07–0.16x** the softer of the two shots |
| incoming shot decelerates out of a pull-out | 0.20–0.50s | rate −28 to −114 %/s decaying to 0 |
| **total** | **0.60–0.90s** | |

The incoming shot always **enters already zoomed in and pulls out to rest**;
this is the same ease shape on both sides of the cut. Blur direction is
**horizontal in all four cases** on eyeball inspection. Three of the four show
large negative translation through the smear (−32.2, −86.6, −30.0 px of
accumulated content motion at 256px width, i.e. content moves left, camera whips
right); the fourth (t=163.4) is too blurred for phase correlation to give a sign.
**No vertical whip, no up/down, no rightward-content whip anywhere.**

### 3.3 The punch, and the discriminator that works

The two punch dissolves (t=48.0, 93.7) and the punch cut (t=15.1) also start
with a fast push-in ramp on the outgoing shot, but then **keep pushing in on the
incoming shot** instead of pulling out, and the softness at the cut is isotropic
(sharp_h and sharp_v drop by the same factor, 0.39/0.43 and 0.40/0.40) rather
than directional. Eyeballed at 48.0 and 93.7, both show a genuine ~0.15-0.20s
cross-fade where the outgoing and incoming cards are both faintly visible. The
punch cut at 15.1 has the ramp but a clean 1-frame swap.

**The sign of the zoom ramp after the cut separates the two devices perfectly
(7/7, agreeing with eyeball inspection):** whip → ramp reverses (push then
pull); punch → ramp continues (push then push). This is a far more reliable
discriminator than blur anisotropy.

---

## 4. Transition matrix: shot device N → device N+1

Row-normalised, n = 295 adjacent pairs.

| from \ to | hold | push-in | pull-out | local anim | n |
|---|---|---|---|---|---|
| **hold** | 81% | 10% | 3% | 6% | 240 |
| **push-in** | 79% | 7% | 11% | 4% | 28 |
| **pull-out** | 92% | 8% | 0% | 0% | 12 |
| **local anim** | 80% | 13% | 7% | 0% | 15 |
| *marginal (independence null)* | *81%* | *10%* | *4%* | *5%* | 295 |

**Every row matches the marginal.** There is no cycle, no alternation, no
memory. A Wald-Wolfowitz runs test on the binary moving-vs-holding sequence
gives runs = 92 against an expected 91.8 (sd 5.3), **z = +0.04** — the sequence
is statistically indistinguishable from independent coin flips at the base rate.

The only non-null cell is `pull-out → pull-out = 0/12`, which is suggestive of
a no-repeat rule on pull-outs but is not significant at n = 12 (expected 0.5).

---

## 5. The pattern, stated as implementable rules

Each rule carries its evidence and sample size. "Body" means t > 164s, the
region after the last dynamic transition.

### R1 — Shot length selects the device. This is the main rule.

Nothing else predicts a move nearly as well.

| body shot length | n | carries a move |
|---|---|---|
| < 3.5s | **134** | **0 (0%)** |
| 3.5 – 5.0s | 57 | 4 (7%) |
| 5.0 – 7.0s | 21 | 9 (43%) |
| 7.0 – 10.0s | 16 | 8 (50%) |
| > 10.0s | 5 | **5 (100%)** |

> **A shot shorter than ~3.5s is always a static hold. 0 of 134.**
> Above 3.5s the probability of a move climbs monotonically to 1.0 by 10s.

Median shot length: 5.70s for moving shots versus 2.50s for held shots,
difference +3.20s, permutation p < 1e-4 (n = 296).

### R2 — Dynamic transitions exist only in the opening 164 seconds.

All 7 whips/punches fall in the first 62 of 295 transitions. Permutation
p = **5e-6**. After t = 163.4s there are 233 consecutive transitions and every
single one is a plain hard cut. The opening also cuts faster: 24.0 cuts/min in
the first 2 minutes versus 15.5–18.5/min for the rest of the video.

> **Whips and punches are an opening-hook device. Budget ~7 of them, all inside
> the first 3 minutes, and none after.**

### R3 — Full-bleed painted scenes move; white diagram cards hold.

| content (mid-shot frame) | n | carries a move |
|---|---|---|
| full-bleed painted scene (paper fraction < 0.40) | 84 | 21 (**25%**) |
| mixed (0.40–0.72) | 24 | 3 (13%) |
| white paper / diagram (> 0.72) | 188 | 17 (**9%**) |

Permutation p = 0.0010 for the scene enrichment and p = 0.0018 for the paper
depletion (20k resamples each). The local-animation device is the mirror image:
it lands almost exclusively on white diagram cards.

> **A painted scene is ~2.8x more likely to carry a zoom than a diagram card.**

### R4 — Three zoom presets with fixed total travel, chosen by family.

Push-in outnumbers pull-out **29 : 12** overall and **20 : 6** in the body.
Implement as: pick a family, pick the shot length, and derive the rate.

| family | total travel | head hold | move span | tail hold |
|---|---|---|---|---|
| gentle push (n=21, the default) | +1.2% | 19% of shot | 73% | 8% |
| strong push (n=7, long shots ≥ 7s) | +5.3% | 5% | 90% | 5% |
| pull-out (n=11) | −3.7% | 2% | 78% | 20% |

Rate = total ÷ move span. Linear, no easing.

### R5 — Push-ins hold first; pull-outs hold last.

A push-in sits still for the first ~19% of the shot then starts. A pull-out
starts at the cut and sits still for the last ~10-20%. 22 of 41 zoom shots have
a head hold over 0.5s.

### R6 — The whip is a compound, not a blur.

Push-in ramp (0.15s, to ~+40 %/s) → 0.15-0.2s horizontal smear containing the
cut → pull-out ramp (0.3s, from ~−50 %/s). Total 0.6-0.9s. Both sides are static
holds before and after. The incoming shot always arrives zoomed in and settles
outward.

### R7 — Everything else is a 1-frame hard cut.

288 of 295. No handles, no dissolve, no blur, no zoom into or out of it. The
default transition is nothing at all.

---

## 6. Hypotheses tested and NOT confirmed

These are negative results, listed because they are load-bearing.

1. **"Slow swipes and fast swipes are distinct devices."**
   **FAILED — there are no swipes at all.** Panning/sliding accounts for 22 of
   30027 frames (0.07%), and inspecting all 22 shows they are noise on
   near-blank frames, not translation. The largest sustained non-transition
   translation anywhere in the video is ~6px at 256px width across several
   seconds. Directional translation exists *only* inside the 4 whips, where it
   lasts 0.13–0.23s. There is no slow directional slide, no push-off, no
   card-slide, no wipe.

2. **"Shots carry a slow zoom, alternating push-in and pull-out."**
   **FAILED on both halves.** 81% of shots have zero camera motion. And there
   is no alternation: consecutive zoom shots are same-direction 2/6 and
   reversed 4/6, which at n = 6 is uninformative. The transition matrix (§4) is
   flat against the marginal and the runs test gives z = +0.04.

3. **"A fast whip is used when the outgoing shot ran long."**
   **FAILED.** Outgoing-shot length before a whip: median 2.08s, range
   1.43–3.23s (n = 4); across all 7 dynamic transitions, range 0.87–3.77s.
   Compare 2.77s median before a hard cut (n = 288). Whips follow *shorter*
   than average shots if anything, and never one longer than 3.8s.

4. **"The device correlates with whether the incoming shot is a text card."**
   **FAILED for transitions.** 5 of the 7 whip/punch incoming shots are
   white/diagram, against a 64% base rate — nothing. Outgoing shots: 4 of 7.
   The content effect is real for *shot* devices (R3) but not for transition
   choice. n = 7 is small, but the effect is not even directionally present.

5. **"There is a repeating cycle, e.g. static → zoom → whip → static."**
   **FAILED.** See §4. Device N carries no information about device N+1.

6. **"Motion direction reverses consistently across a transition."**
   **UNTESTABLE / no evidence.** There is no pan direction to reverse. For zoom
   direction there are only 6 adjacent zoom-zoom pairs in the entire video, so
   the question cannot be answered; do not build on it.

7. **"Shot length trends over the video (opening fast, body slow)."**
   **PARTIAL.** The *cut rate* does drop (24.0/min in minutes 0-2 versus
   15.5-18.5/min afterwards), but median shot length per 2-minute bucket
   afterwards is flat and non-monotone: 3.08, 3.43, 2.07, 2.73, 3.27, 3.07,
   2.70, 3.10. There is no gradual arc — just a faster opening.

8. **"31% of boundaries are blur transitions."**
   **FAILED — artifact.** See §7.

---

## 7. Why the earlier 31% blur figure was wrong

`tools/analyse_transitions.py` takes the softest frame in a window around a
boundary and divides it by the **median sharpness pooled across both sides**.
When a detailed shot cuts to a sparse one, that pooled median sits *between* the
two sharpness levels, so the new shot's own, perfectly sharp, lower-detail
frames read as a trough.

Reproducing that exact metric over all 295 boundaries gives **31.2%** below
0.55 — matching the figure in §9b of the design spec. Of those 92 boundaries,
**86 are plain hard cuts** and only 6 are real whips.

The confound is measurable: across hard cuts the median sharpness *step* between
the two shots is 43%, and the correlation between that step and the "trough"
ratio is **r = −0.58**. The metric was measuring scene detail, not blur.

Dividing instead by the **minimum** of the two one-sided baselines (so a frame
only counts as blurred if it is softer than *both* neighbouring shots) gives
**2.0% (6 of 295), and all 6 are the genuine whips.** That is the number to use.

---

## 8. Validation

Frames were extracted at full resolution, tiled, and inspected by eye. Sheets
are under `build/MOT_val/`.

| boundary / shot | classifier said | eye said | agree |
|---|---|---|---|
| t=61.65 | whip: push-in ramp, 0.2s horizontal smear, pull-out ramp | exactly that — "?" card pushes in, smears into horizontal streaks, savannah enters large and settles | ✅ |
| t=490.47 | hard cut, both sides frozen, tx = −23px is noise | instantaneous swap between two frames, no intermediate frame, no blur, no slide | ✅ |
| t=38.9–44.0 | push-in, +2.35% over 5.0s | hippo diagram grows slightly, arrows spread outward | ✅ |
| t=176.2–187.6 | push-in, onset 2.03s, +1.22% | 176.3/177.0/177.9 pixel-identical, growth visible from 178.4 | ✅ |
| t=9.27 | whip, horizontal | horizontal streaks, supermarket → "?" card | ✅ |
| t=47.97 | punch dissolve, isotropic | outgoing pushes in, then a genuine ~0.2s cross-fade with both cards visible at 48.17 | ✅ |
| t=70.50 | whip, horizontal | horizontal streaks | ✅ |
| t=93.70 | punch dissolve, isotropic | outgoing hippo punches in fast, cross-fade to "NEVER FARMED", which fades up over ~0.2s | ✅ |
| t=163.40 | whip, horizontal | horizontal streaks | ✅ |
| t=15.10 | punch cut | outgoing card accelerates into the cut, clean swap, incoming static | ✅ |
| t=812.6–816.2 | pull-out, −3.78%, starts at the cut | map shrinks steadily from the first frame, white margin grows | ✅ |
| t=371.5–386.0 | push-in, onset 3.17s, +1.19% | 371.6 → 374.4 identical; growth from 375.0 on | ✅ |

**One disagreement.** At t=61.40 the axis-split blur-direction metric reports
*vertical* (sharp_v drops to 0.01x baseline versus 0.13x for sharp_h), but the
frames are unambiguously horizontally smeared. Cause: both the outgoing and the
incoming frame are near-white cards with almost no vertical structure, so
`sharp_v` has essentially no baseline to fall from and the ratio is unstable.
The accumulated translation through the smear (−86.6px) gives the right answer
for that case. **Do not trust the anisotropy ratio on white-card boundaries**;
prefer accumulated translation, or the post-cut ramp sign for whip-vs-punch.

12 boundaries/shots inspected, 12 classifier calls correct, 1 sub-attribute
(blur axis) wrong.

---

## 9. What this means for why-engine

Concretely, to reproduce this grammar:

1. Default every shot to a **frozen frame**. 81% of the reference is that.
2. Cut with **no transition at all** (1 frame, no handles). 97.6% of boundaries.
3. If a shot is **under 3.5s, never move the camera.** No exceptions in 134 body shots.
4. If a shot runs **over ~5s**, roll for a move at about 50%, rising to certain
   by 10s. Prefer painted scenes over diagram cards by ~2.8x.
5. A move is a **centred linear scale with a fixed total**: +1.2% (default),
   +5.3% (long shots), or −3.7% (pull-out, ~1 in 3 of all moves). No easing.
   Rate = total ÷ span.
6. A push-in **holds for the first ~19% of the shot** before starting;
   a pull-out starts immediately and **holds for the last ~10-20%**.
7. Spend the whole whip/punch budget — about **7 instances — inside the first
   3 minutes**, and cut faster there (~24/min versus ~17/min). After that,
   hard cuts only.
8. A whip is: 0.15s push-in ramp to ~+40 %/s → 0.15-0.2s horizontal smear with
   the cut inside it → 0.3s pull-out ramp from ~−50 %/s, total 0.6-0.9s, and the
   incoming shot enters zoomed and settles out.
9. **Do not implement pans, slides or swipes.** They are not in the reference.
