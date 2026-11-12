"""Caption placement into a frame's empty space.

Every frame in the reference carries three to six words of hand-lettered
text sitting in whatever space the artwork left over -- a statement in ink
and, where there is a verdict, one line in red. Composed frames do not know
in advance where that space is, so this module finds it: build an ink mask,
then pick the lowest-coverage position a text block fits in.

An integral image makes each candidate position O(1), so an exhaustive
stride search over a 1920x1080 frame stays cheap.
"""
from __future__ import annotations

import math

from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PAPER_THRESHOLD = 236  # matches the transparency cutoff used when placing assets
Box = tuple[int, int, int, int]


def zoom_safe_margin(width: int, height: int, max_zoom: float, base: int = 40) -> int:
    """Inset that survives the shot's zoom.

    Shots are rendered with a slow push or pull, so a caption sitting in the
    outer band of the frame is cropped away at peak zoom -- silently, since
    nothing errors. This returns the inset that keeps text on screen for the
    whole shot.
    """
    if max_zoom <= 1.0:
        return base
    cropped = (1.0 - 1.0 / max_zoom) / 2.0
    # ceil, not int: truncating leaves the caption a fraction of a pixel
    # outside the crop, which shows up as a clipped glyph edge.
    return base + math.ceil(cropped * max(width, height))


def ink_mask(img: Image.Image, threshold: int = PAPER_THRESHOLD, dilate: int = 0) -> np.ndarray:
    """True where the frame carries marks rather than paper.

    `dilate` grows the mask by that many pixels in every direction. Without
    it a caption can score zero coverage while sitting flush against a
    headline -- technically empty, visually a collision. Dilation turns the
    breathing room a reader needs into space the placer can see.
    """
    mask = np.asarray(img.convert("L"), dtype=np.uint8) < threshold
    return _dilate(mask, dilate) if dilate > 0 else mask


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Box dilation via an integral image: any window holding ink becomes ink."""
    h, w = mask.shape
    integral = np.zeros((h + 1, w + 1), dtype=np.int64)
    integral[1:, 1:] = mask.astype(np.int64).cumsum(axis=0).cumsum(axis=1)
    y0 = np.clip(np.arange(h) - radius, 0, h)[:, None]
    y1 = np.clip(np.arange(h) + radius + 1, 0, h)[:, None]
    x0 = np.clip(np.arange(w) - radius, 0, w)[None, :]
    x1 = np.clip(np.arange(w) + radius + 1, 0, w)[None, :]
    total = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    return total > 0


def find_empty_box(
    mask: np.ndarray,
    width: int,
    height: int,
    margin: int = 48,
    occupied: Iterable[Box] = (),
    stride: int = 12,
    max_coverage: float = 0.004,
    anchor: tuple[float, float] | None = None,
    pull: float = 0.02,
) -> Box | None:
    """Lowest-ink position for a `width` x `height` block, or None if none is clear.

    `occupied` boxes are treated as fully inked so successive calls do not
    stack captions on top of each other.

    A frame usually has far more clear space than any one caption needs, so
    coverage alone leaves the choice arbitrary -- in practice every caption
    piles into the first empty band scanned. `anchor` is where the caller
    wants the text, in relative (x, y) coordinates; `pull` weights distance
    from it against ink coverage. Keep `pull` small: it should break ties
    between clear positions, never justify drawing over artwork.
    """
    h, w = mask.shape
    if width + 2 * margin > w or height + 2 * margin > h:
        return None

    blocked = mask.astype(np.float64)
    for ox, oy, ow, oh in occupied:
        x0, y0 = max(0, ox), max(0, oy)
        blocked[y0 : oy + oh, x0 : ox + ow] = 1.0

    integral = np.zeros((h + 1, w + 1), dtype=np.float64)
    integral[1:, 1:] = blocked.cumsum(axis=0).cumsum(axis=1)

    def coverage(x: int, y: int) -> float:
        total = (
            integral[y + height, x + width]
            - integral[y, x + width]
            - integral[y + height, x]
            + integral[y, x]
        )
        return total / (width * height)

    ax = ay = None
    if anchor is not None:
        ax, ay = anchor[0] * w, anchor[1] * h
        diagonal = float(np.hypot(w, h))

    best, best_score, best_cov = None, float("inf"), 1.0
    for y in range(margin, h - height - margin + 1, stride):
        for x in range(margin, w - width - margin + 1, stride):
            cov = coverage(x, y)
            score = cov
            if ax is not None:
                cx, cy = x + width / 2, y + height / 2
                score += pull * float(np.hypot(cx - ax, cy - ay)) / diagonal
            if score < best_score:
                best, best_score, best_cov = (x, y, width, height), score, cov
    return best if best_cov <= max_coverage else None


def _measure(draw: ImageDraw.ImageDraw, lines: Sequence[str], font: ImageFont.FreeTypeFont,
             leading: int) -> tuple[int, int, list[int]]:
    widths, heights = [], []
    for line in lines:
        x0, y0, x1, y1 = draw.textbbox((0, 0), line, font=font)
        widths.append(x1 - x0)
        heights.append(y1 - y0)
    block_h = sum(heights) + leading * (len(lines) - 1)
    return max(widths), block_h, heights


def place_caption(
    img: Image.Image,
    lines: Sequence[str],
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    occupied: list[Box] | None = None,
    margin: int = 48,
    leading: int = 14,
    pad: int = 24,
    anchor: tuple[float, float] | None = None,
    gap: int = 26,
) -> Box | None:
    """Draw `lines` into the emptiest region that fits them.

    `gap` is a preference, not a requirement: a busy frame is retried at half
    the gap and then at none, because a caption placed tight beats a caption
    silently dropped. Returns the box consumed, appending it to `occupied` so
    the next caption avoids it, or None when even the tight pass finds no room.
    """
    draw = ImageDraw.Draw(img)
    text_w, text_h, heights = _measure(draw, lines, font, leading)
    box = None
    for attempt in (gap, gap // 2, 0):
        box = find_empty_box(
            ink_mask(img, dilate=attempt),
            text_w + 2 * pad,
            text_h + 2 * pad,
            margin=margin,
            occupied=occupied or (),
            anchor=anchor,
        )
        if box is not None:
            break
    if box is None:
        return None

    x, y, bw, _ = box
    cx = x + bw // 2
    cursor = y + pad
    for line, lh in zip(lines, heights):
        draw.text((cx, cursor), line, font=font, fill=fill, anchor="ma")
        cursor += lh + leading
    if occupied is not None:
        occupied.append(box)
    return box
