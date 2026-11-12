"""Camera motion, measured from the reference video.

See docs/superpowers/specs/2026-09-16-reference-motion-analysis.md. Three
findings drive this module, and all three contradict what the composer was
doing before it existed:

1. 81% of shots never move at all, and a shot shorter than 3.5s never moves --
   0 of 134 measured. Motion is earned by duration, not applied by default.
2. There are no pans, slides or drifts anywhere. The camera only holds or
   scales about the frame centre.
3. A move is a fixed *total travel* placed across a proportional window, not a
   fixed rate. The editor picks a preset and a shot length; the rate is
   whatever makes the total fit. Motion is dead linear -- abrupt start, abrupt
   stop, no easing.

The head and tail holds are the "static-then-move" shape: a push-in sits still
for the first fifth of the shot before starting, a pull-out starts at the cut
and settles early, then sits still.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

#: A body shot below this length is always a static hold (0 of 134 measured).
STATIC_BELOW_SECONDS = 3.5


@dataclass(frozen=True)
class ZoomPreset:
    """A measured zoom family.

    `total` is the fractional scale change across the whole move, signed:
    positive pushes in, negative pulls out. `head` and `tail` are the frozen
    fractions of the shot before and after the move. They sum to 1.0 with the
    move span, which is derived rather than stored so the three cannot drift
    apart.
    """

    total: float
    head: float
    tail: float

    @property
    def span(self) -> float:
        return 1.0 - self.head - self.tail


#: Measured families. Counts and standard deviations are in the analysis, S2.3.
PRESETS = {
    # n=21, +1.19% +/- 0.30, median shot 5.93s. The default.
    "gentle_push": ZoomPreset(total=+0.0119, head=0.19, tail=0.08),
    # n=7, +5.26% +/- 2.02, median shot 7.57s. Long shots only.
    "strong_push": ZoomPreset(total=+0.0526, head=0.05, tail=0.05),
    # n=11, -3.66% +/- 0.20, median shot 3.73s. Holds at the end, not the start.
    "pull_out": ZoomPreset(total=-0.0366, head=0.02, tail=0.20),
}

#: Push-in outnumbers pull-out 20:6 in the body.
PUSH_TO_PULL_RATIO = 20 / 6


def shot_moves(duration: float) -> bool:
    """Whether a shot of this length is allowed to carry a move at all."""
    return duration >= STATIC_BELOW_SECONDS


def zoom_curve(duration: float, fps: int, preset: ZoomPreset | str | None) -> list[float]:
    """Absolute scale factor per frame, always >= 1.0.

    Rendering only ever crops inward, so a pull-out is expressed as starting
    zoomed in and ending at 1.0 rather than as a scale below unity. Passing
    None -- or a duration too short to earn a move -- returns a flat 1.0
    curve, which is the common case.
    """
    n = max(int(round(duration * fps)), 1)
    if preset is None or not shot_moves(duration):
        return [1.0] * n

    if isinstance(preset, str):
        preset = PRESETS[preset]

    magnitude = abs(preset.total)
    if preset.total >= 0:
        start, end = 1.0, 1.0 + magnitude
    else:
        start, end = 1.0 + magnitude, 1.0

    head = int(round(preset.head * n))
    tail = int(round(preset.tail * n))
    move = max(n - head - tail, 1)
    # Reclaim from the tail first: losing the settle reads better than losing
    # the move, and the head hold is the device's signature.
    if head + tail + move > n:
        tail = max(n - head - move, 0)

    curve = [start] * head
    for i in range(move):
        curve.append(start + (end - start) * (i / max(move - 1, 1)))
    curve += [end] * tail
    return (curve + [end] * n)[:n]


class Zoomer:
    """Renders zoom frames from one still.

    Holds a single upscaled copy of the source so a shot's frames do not each
    pay to resample the original, and so the crop has pixels to spend at the
    tightest zoom in the curve.
    """

    def __init__(self, img: Image.Image, size: tuple[int, int], headroom: float = 1.25):
        self.size = size
        self._base = img.resize(
            (int(size[0] * headroom), int(size[1] * headroom)), Image.LANCZOS
        )

    def frame(self, zoom: float) -> Image.Image:
        w, h = self._base.size
        cw, ch = int(w / zoom), int(h / zoom)
        x, y = (w - cw) // 2, (h - ch) // 2
        return self._base.crop((x, y, x + cw, y + ch)).resize(self.size, Image.LANCZOS)

    def shot(self, curve: list[float]):
        """Yield a shot's frames, reusing the frame for runs of equal zoom.

        The reference is 76-82% bit-identical frames, and the holds are why.
        Reusing the object keeps that property exactly rather than
        approximately -- resampling the same crop twice can differ in the last
        bit, and the encoder notices.

        A generator, not a list. A static shot costs one image however long it
        runs, but a shot carrying a move is distinct every frame: ten seconds of
        push at 1920x1080 is 300 frames and 1.9GB held at once.
        """
        last_zoom, last_frame = None, None
        for z in curve:
            if last_frame is None or z != last_zoom:
                last_frame, last_zoom = self.frame(z), z
            yield last_frame


# --- local element animation -------------------------------------------------
#
# The reference's fifth device, and the only one that is animation rather than
# camera work: 15 shots (5.1%), camera locked, one element popping in while
# everything else holds. Median 1.43s with only 23% of the shot's frames
# changing, almost always on a white diagram card.
#
# The overshoot is what makes it read as drawn rather than faded: the element
# arrives slightly too large and settles back, the way a stamp bounces.

POP_SECONDS = 0.20
POP_OVERSHOOT = 1.10


def pop_scale(progress: float, overshoot: float = POP_OVERSHOOT) -> float:
    """Scale factor for a pop-in at `progress` through its animation.

    Rises past 1.0 to `overshoot` at three quarters, then settles to exactly
    1.0. Returns 1.0 for any progress at or past the end so callers can hold.
    """
    if progress >= 1.0:
        return 1.0
    if progress <= 0.0:
        return 0.0
    if progress < 0.75:
        return overshoot * (progress / 0.75)
    return overshoot + (1.0 - overshoot) * ((progress - 0.75) / 0.25)


def pop_in(base, element, center, frames: int, delay: int = 0,
           pop_frames: int | None = None, fps: int = 30):
    """Composite `element` onto `base` with a pop-in, returning `frames` frames.

    `base` is the shot without the element. Frames before `delay` and after the
    pop are the same objects repeated, so the shot stays bit-identical wherever
    nothing is moving -- the property the reference has for 76-82% of its
    runtime.
    """
    from PIL import Image

    pop_frames = pop_frames or max(int(round(POP_SECONDS * fps)), 2)
    cx, cy = center
    out, cache = [], {}
    for i in range(frames):
        p = 0.0 if i < delay else min((i - delay) / pop_frames, 1.0)
        s = round(pop_scale(p), 3)
        if s not in cache:
            if s <= 0.0:
                cache[s] = base
            else:
                frame = base.copy()
                w, h = max(int(element.width * s), 1), max(int(element.height * s), 1)
                scaled = element.resize((w, h), Image.LANCZOS)
                frame.paste(scaled, (int(cx - w / 2), int(cy - h / 2)), scaled)
                cache[s] = frame
        out.append(cache[s])
    return out
