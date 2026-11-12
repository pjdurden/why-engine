"""Colour matching against the reference video.

The reference is not the white-paper world the composer was building. Sampling
100 frames across it: 62% are paper-dominant diagram cards, but **38% are
full-bleed colour scenes** -- flat fills with a heavy black outline in a muted
earthy range. A build made entirely of white cards is missing over a third of
the video's visual vocabulary, which is why it reads as clinical.

The paper itself is genuinely near-white, RGB (250.8, 250.7, 250.4) measured
over the light pixels of card frames, so PAPER below is not a warm tint -- the
cards were never the problem.

Generated scenes come out flatter than the reference. Measured over non-paper
pixels, the reference scenes average 0.348 mean saturation and 0.583 at the
90th percentile; fal's output averages 0.251 and 0.451. SATURATION_GAIN closes
that gap.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

#: Measured mean of the light pixels on the reference's diagram cards.
PAPER = (251, 251, 250)

#: Reference scene saturation, over non-paper pixels.
REF_SAT_MEAN = 0.348
REF_SAT_P90 = 0.583

#: Ratio needed to bring fal's flat output up to the reference.
SATURATION_GAIN = 1.35

#: Non-paper cutoff. Below this the pixel is background, not artwork.
BODY_VALUE = 40


def saturation(img: Image.Image) -> tuple[float, float]:
    """Mean and 90th-percentile saturation over the image's non-paper pixels."""
    a = np.asarray(img.convert("RGB"), dtype=np.float64)
    mx, mn = a.max(axis=2), a.min(axis=2)
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)
    body = s[mx > BODY_VALUE]
    if body.size == 0:
        return 0.0, 0.0
    return float(body.mean()), float(np.percentile(body, 90))


def saturate(img: Image.Image, gain: float = SATURATION_GAIN) -> Image.Image:
    """Scale each pixel's distance from its own grey, preserving luminance.

    Pulling colours away from per-pixel grey rather than from a global mean
    keeps the flat fills flat: a fill that was one colour stays one colour, and
    the black outline -- which has no saturation to scale -- stays black.
    """
    a = np.asarray(img.convert("RGB"), dtype=np.float64)
    grey = a.mean(axis=2, keepdims=True)
    return Image.fromarray(np.clip(grey + (a - grey) * gain, 0, 255).astype(np.uint8))


def match_reference(img: Image.Image, tolerance: float = 0.02, max_gain: float = 2.5) -> Image.Image:
    """Saturate until the image's mean saturation reaches the reference's.

    Solved by bisection rather than by a fixed gain because clipping makes the
    relationship between gain and resulting saturation non-linear, and a source
    that is already vivid should be left alone.
    """
    current, _ = saturation(img)
    if current <= 0 or current >= REF_SAT_MEAN - tolerance:
        return img
    lo, hi = 1.0, max_gain
    best = img
    for _ in range(12):
        mid = (lo + hi) / 2
        candidate = saturate(img, mid)
        got, _ = saturation(candidate)
        best = candidate
        if abs(got - REF_SAT_MEAN) <= tolerance:
            return candidate
        if got < REF_SAT_MEAN:
            lo = mid
        else:
            hi = mid
    return best
