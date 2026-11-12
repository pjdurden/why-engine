"""The deterministic style pass every asset flows through.

FLUX schnell has a strong prior toward clean polished illustration and leaks
small amounts of colour. Prompting against both was measured and failed, so the
hand-drawn character is applied here instead, identically to every asset.
"""
import numpy as np
from PIL import Image, ImageFilter

# Style contract. Measured from a contact sheet; not per-call knobs.
WOBBLE = 5.0        # displacement amplitude in output pixels
PRESSURE = 0.35     # stroke thickening; higher blobs the line
BLUR_FRAC = 0.045   # displacement smoothness as a fraction of WORK
WORK = 1536         # warp here, then downscale: what makes tremor, not noise
ROTATE_DEG = 1.2
INK_THRESHOLD = 200


def is_monochrome(img: Image.Image) -> bool:
    a = np.array(img.convert("RGB"), dtype=np.int16)
    return bool((a.max(axis=2) - a.min(axis=2)).max() == 0)


def _field(rng, shape, amp: float, blur: int) -> np.ndarray:
    n = rng.normal(0, 1, shape).astype(np.float32)
    n = (n - n.min()) / (n.max() - n.min() + 1e-9) * 255.0
    n = np.array(
        Image.fromarray(n.astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(blur)),
        dtype=np.float32,
    )
    n -= n.mean()
    return n / (np.abs(n).max() + 1e-9) * amp


def humanise(img: Image.Image, seed: int = 0) -> Image.Image:
    out_w, out_h = img.size
    side = max(out_w, out_h)
    off_x, off_y = (side - out_w) // 2, (side - out_h) // 2

    src_img = img
    if src_img.mode in ("RGBA", "LA", "P"):
        # convert("L") discards alpha outright, so a fully-transparent pixel
        # becomes an arbitrary grey/black value instead of background. Composite
        # onto white first so transparency reads as blank paper, not ink.
        src_img = src_img.convert("RGBA")
        flat = Image.new("RGBA", src_img.size, (255, 255, 255, 255))
        src_img = Image.alpha_composite(flat, src_img)

    # Pad to square rather than stretch: warping a stretched portrait gives the
    # tremor a different amplitude AND wavelength on each axis.
    square = Image.new("L", (side, side), 255)
    square.paste(src_img.convert("L"), (off_x, off_y))
    src = square.resize((WORK, WORK), Image.LANCZOS)
    ink = (np.array(src, dtype=np.float32) < INK_THRESHOLD).astype(np.float32)

    rng = np.random.default_rng(seed)
    blur = max(3, int(WORK * BLUR_FRAC))
    scale = WORK / side

    yy, xx = np.meshgrid(np.arange(WORK), np.arange(WORK), indexing="ij")
    xs = np.clip(xx + _field(rng, (WORK, WORK), WOBBLE * scale, blur), 0, WORK - 1).astype(np.int32)
    ys = np.clip(yy + _field(rng, (WORK, WORK), WOBBLE * scale, blur), 0, WORK - 1).astype(np.int32)
    k = ink[ys, xs]

    radius = int(3 * scale) | 1
    thick = (
        np.array(
            Image.fromarray((k * 255).astype(np.uint8), "L").filter(ImageFilter.MaxFilter(radius)),
            dtype=np.float32,
        )
        / 255.0
    )
    merged = np.clip(k + thick * PRESSURE, 0, 1)

    grey = Image.fromarray(((1 - merged) * 255).astype(np.uint8), "L").resize(
        (side, side), Image.LANCZOS
    )
    rotated = grey.rotate(
        rng.uniform(-ROTATE_DEG, ROTATE_DEG), fillcolor=255, resample=Image.BICUBIC
    )
    cropped = rotated.crop((off_x, off_y, off_x + out_w, off_y + out_h))
    # Threshold last so the resample and rotation cannot reintroduce grey.
    return Image.fromarray(
        np.where(np.array(cropped) < INK_THRESHOLD, 0, 255).astype(np.uint8), "L"
    )
