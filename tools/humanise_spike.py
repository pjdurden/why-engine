"""SPIKE, not production code. Finds humanise() parameter values by rendering a
contact sheet. The chosen values become the fixed style contract in Plan 2."""
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

WORK = 1536  # warp at high res, then downscale: that is what turns a smooth
             # displacement into a convincing wobble instead of noise

def humanise(path, wobble=6.0, blur_frac=0.045, skip=0.0, pressure=0.9, seed=3, out_size=512):
    src = Image.open(path).convert("L").resize((WORK, WORK), Image.LANCZOS)
    ink = (np.array(src, dtype=np.float32) < 200).astype(np.float32)
    h, w = ink.shape
    rng = np.random.default_rng(seed)
    blur = max(3, int(WORK * blur_frac))

    def field(amp):
        n = rng.normal(0, 1, (h, w)).astype(np.float32)
        n = (n - n.min()) / (n.max() - n.min()) * 255.0
        n = np.array(Image.fromarray(n.astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(blur)), dtype=np.float32)
        n -= n.mean()
        return n / (np.abs(n).max() + 1e-6) * amp

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    px = WORK / out_size
    xs = np.clip(xx + field(wobble * px), 0, w - 1).astype(np.int32)
    ys = np.clip(yy + field(wobble * px), 0, h - 1).astype(np.int32)
    k = ink[ys, xs]
    if skip:
        k = k * (rng.random((h, w)) > skip)
    t = np.array(Image.fromarray((k * 255).astype(np.uint8), "L").filter(ImageFilter.MaxFilter(int(3 * px) | 1)), dtype=np.float32) / 255.0
    o = np.clip(k + t * pressure, 0, 1)
    img = Image.fromarray(((1 - o) * 255).astype(np.uint8)).resize((out_size, out_size), Image.LANCZOS)
    return img.rotate(rng.uniform(-1.2, 1.2), fillcolor=255, resample=Image.BICUBIC)

SRC = "assets/candidates/messy_m4.jpg"
CELL, COLS = 300, 3
settings = [
    ("A wobble 2",  dict(wobble=2.0)),
    ("B wobble 5",  dict(wobble=5.0)),
    ("C wobble 9",  dict(wobble=9.0)),
    ("D w5 loose",  dict(wobble=5.0, blur_frac=0.09)),
    ("E w5 tight",  dict(wobble=5.0, blur_frac=0.02)),
    ("F w9 loose",  dict(wobble=9.0, blur_frac=0.09)),
    ("G w5 +skip",  dict(wobble=5.0, skip=0.03)),
    ("H w5 light",  dict(wobble=5.0, pressure=0.35)),
    ("I w5 heavy",  dict(wobble=5.0, pressure=1.4)),
]
sheet = Image.new("L", (CELL * COLS, CELL * 3 + 3 * 22), 255)
d = ImageDraw.Draw(sheet)
for i, (label, kw) in enumerate(settings):
    r, c = divmod(i, COLS)
    y = r * (CELL + 22)
    sheet.paste(humanise(SRC, out_size=CELL, **kw), (c * CELL, y + 22))
    d.text((c * CELL + 6, y + 6), label, fill=0)
sheet.save("assets/candidates/CONTACT_SHEET.png")
print("wrote assets/candidates/CONTACT_SHEET.png")
