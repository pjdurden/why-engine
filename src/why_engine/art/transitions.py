"""Whip transitions, measured from the reference video.

At roughly a third of its shot boundaries the reference slams one shot into
the next with a fast directional slide under heavy motion blur: sharpness
collapses to 0.15-0.5x its normal level for about 0.15s, then snaps back.
The remaining boundaries are hard cuts. Measurement lives in
tools/analyse_transitions.py.

The blur is not decoration. It is proportional to slide velocity, which is
why it peaks mid-transition and vanishes at both ends -- the same reason a
real whip pan smears. Deriving it from velocity rather than tuning it by
hand keeps the two in step when the duration changes.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# Peak blur as a fraction of the frame's long edge. Tuned so the mid-whip
# frame lands inside the reference's measured 0.15-0.5x sharpness trough.
MAX_BLUR_FRAC = 0.045
DEFAULT_FRAMES = 5  # ~0.17s at 30fps, matching the reference's 0.10-0.20s


def _smoothstep(p: float) -> float:
    return p * p * (3.0 - 2.0 * p)


def _velocity(p: float) -> float:
    """Derivative of smoothstep, normalised to 1.0 at its midpoint peak."""
    return (6.0 * p * (1.0 - p)) / 1.5


def directional_blur(img: Image.Image, radius: int, axis: str = "x") -> Image.Image:
    """Box-blur along one axis only, leaving the other axis sharp."""
    if radius < 1:
        return img
    a = np.asarray(img.convert("RGB"), dtype=np.float64)
    if axis == "y":
        a = a.transpose(1, 0, 2)
    n = 2 * radius + 1
    pad = np.pad(a, ((0, 0), (radius + 1, radius), (0, 0)), mode="edge")
    c = np.cumsum(pad, axis=1)
    out = (c[:, n:, :] - c[:, :-n, :]) / n
    if axis == "y":
        out = out.transpose(1, 0, 2)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def whip_frames(
    a: Image.Image,
    b: Image.Image,
    frames: int = DEFAULT_FRAMES,
    axis: str = "x",
    reverse: bool = False,
) -> list[Image.Image]:
    """Render the whip from shot `a` into shot `b`.

    `a` slides out as `b` slides in, both under velocity-proportional blur.
    The endpoints are never emitted: frame 0 would be plain `a` and frame
    `frames` would be plain `b`, and the caller already has both.
    """
    if a.size != b.size:
        raise ValueError(f"whip needs matching sizes, got {a.size} and {b.size}")
    w, h = a.size
    span = w if axis == "x" else h
    max_blur = int(MAX_BLUR_FRAC * max(w, h))
    sign = -1 if reverse else 1

    out = []
    for i in range(1, frames + 1):
        p = i / (frames + 1)
        shift = int(_smoothstep(p) * span) * sign
        canvas = Image.new("RGB", (w, h), (253, 253, 253))
        if axis == "x":
            canvas.paste(a, (-shift, 0))
            canvas.paste(b, (span * sign - shift, 0))
        else:
            canvas.paste(a, (0, -shift))
            canvas.paste(b, (0, span * sign - shift))
        out.append(directional_blur(canvas, int(_velocity(p) * max_blur), axis))
    return out


def sharpness(img: Image.Image) -> float:
    """Mean absolute gradient -- the statistic the reference was measured with."""
    g = np.asarray(img.convert("L"), dtype=np.float64)
    return float(np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean())


# --- the compound whip -------------------------------------------------------
#
# Measured in the analysis, S3 and R6. The bare smear above is only the middle
# third of the device. The reference wraps it in two zoom ramps:
#
#   [push-in ramp 0.15s, accelerating to ~+40%/s]
#   [horizontal smear 0.15-0.20s, containing the cut]
#   [pull-out ramp 0.30s, decelerating from ~-50%/s]
#
# so the outgoing shot lunges at the camera, the cut hides inside the smear,
# and the incoming shot arrives already zoomed in and settles outward. Both
# sides are static holds before and after. Total 0.6-0.9s.
#
# Two budget rules travel with it, because getting them wrong is what made the
# previous build look nothing like the reference:
#   - Whips are an OPENING device. All 7 in the reference fall before t=164s;
#     the following 233 transitions are plain hard cuts. See WHIP_BUDGET.
#   - All 4 measured whips are horizontal. There are no vertical whips.

RAMP_IN_SECONDS = 0.15
RAMP_OUT_SECONDS = 0.30
SMEAR_SECONDS = 0.17
RAMP_IN_PEAK_RATE = 0.40   # scale fraction per second, reached at the smear
RAMP_OUT_PEAK_RATE = 0.50  # scale fraction per second, at the smear

#: Whips permitted per video, and the timestamp after which none may appear.
WHIP_BUDGET = 7
WHIP_WINDOW_SECONDS = 164.0


def whip_travel(peak_rate: float, seconds: float) -> float:
    """Total scale change of a linear velocity ramp -- the area under it."""
    return 0.5 * peak_rate * seconds


def whip_sequence(a, b, fps: int = 30, size=None) -> list:
    """The full compound whip from shot `a` into shot `b`.

    `a` and `b` are stills. Returns every frame of the device, ready to splice
    between the two shots' own frames.
    """
    from why_engine.art.motion import Zoomer

    size = size or a.size
    za, zb = Zoomer(a, size), Zoomer(b, size)

    n_in = max(int(round(RAMP_IN_SECONDS * fps)), 1)
    n_out = max(int(round(RAMP_OUT_SECONDS * fps)), 1)
    n_smear = max(int(round(SMEAR_SECONDS * fps)), 1)

    peak_in = 1.0 + whip_travel(RAMP_IN_PEAK_RATE, RAMP_IN_SECONDS)
    peak_out = 1.0 + whip_travel(RAMP_OUT_PEAK_RATE, RAMP_OUT_SECONDS)

    frames = []
    # Velocity ramps linearly, so displacement is quadratic in time.
    for i in range(1, n_in + 1):
        p = i / n_in
        frames.append(za.frame(1.0 + (peak_in - 1.0) * p * p))
    frames += whip_frames(za.frame(peak_in), zb.frame(peak_out), frames=n_smear, axis="x")
    for i in range(n_out):
        p = i / n_out
        frames.append(zb.frame(1.0 + (peak_out - 1.0) * (1.0 - p) ** 2))
    return frames
