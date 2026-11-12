"""SPIKE, not production. Audit build for the four corrections measured off the
reference video: bass-weighted voice, whip transitions, continuous zoom carried
through the cut, and a short caption in every frame's empty space.

The facts on these cards are placeholder copy for a look-and-feel audit. They
have NOT been through the grounding gate.

Plan 2B replaces this with archetypes.py + compose.py + text.py.
"""
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from why_engine.art.annotate import place_caption, zoom_safe_margin
from why_engine.art.transitions import whip_frames
from why_engine.audio.voice import voice

W, H = 1920, 1080
FPS = 30
PAPER, INK, RED, GREY = (253, 253, 253), (26, 26, 26), (200, 38, 42), (150, 150, 150)
IC, LIB, OUT = Path("build/icons"), Path("assets/library"), Path("build/v4")
FRAMES = OUT / "frames"
FRAMES.mkdir(parents=True, exist_ok=True)
HEAD = "assets/fonts/PermanentMarker-Regular.ttf"
CAP = "assets/fonts/ArchitectsDaughter-Regular.ttf"
F = lambda p, s: ImageFont.truetype(p, s)
canvas = lambda: Image.new("RGB", (W, H), PAPER)


def place(c, path, cx, cy, h):
    im = Image.open(path).convert("RGBA")
    w = int(im.width * h / im.height)
    im = im.resize((w, h), Image.LANCZOS)
    px = im.load()
    for y in range(im.height):
        for x in range(im.width):
            r, g, b, a = px[x, y]
            if r > 236 and g > 236 and b > 236:
                px[x, y] = (r, g, b, 0)
    c.paste(im, (int(cx - w / 2), int(cy - h / 2)), im)


def caps(c, xy, t, f, fill, anchor="mm"):
    ImageDraw.Draw(c).text(xy, t.upper(), font=f, fill=fill, anchor=anchor)


shots = []
Z_LO, Z_HI = 1.00, 1.10
SAFE = zoom_safe_margin(W, H, Z_HI)


def shot(img, dur, notes):
    """Drop each caption into the clear space nearest its anchor.

    The anchors come from the composition: the caller knows where the
    artwork is, so it says roughly where the text belongs and the placer
    finds the actual gap. SAFE keeps every caption inside the zoom crop.
    """
    occupied = []
    for lines, colour, size, anchor in notes:
        place_caption(img, [l.upper() for l in lines], F(CAP, size), colour,
                      occupied=occupied, margin=SAFE, pad=20, anchor=anchor, gap=30)
    shots.append((img, dur))


# 1 text_card
s = canvas()
caps(s, (960, 380), "WHY NOT CARP?", F(HEAD, 155), RED)
ImageDraw.Draw(s).line([(470, 485), (1450, 485)], fill=RED, width=11)
caps(s, (960, 575), "HALF THE WORLD EATS IT", F(CAP, 72), INK)
place(s, IC / "carp.png", 960, 800, 230)
shot(s, 2.2, [(["INVASIVE IN", "45 US STATES"], INK, 56, (0.15, 0.80)),
              (["TWO BILLION", "PEOPLE EAT IT"], RED, 56, (0.85, 0.80))])

# 2 icon_multiply
s = canvas()
place(s, IC / "carp.png", 300, 400, 290)
caps(s, (520, 400), "=", F(CAP, 150), INK)
for i in range(12):
    r, col = divmod(i, 6)
    place(s, IC / "pig.png", 740 + col * 190, 320 + r * 185, 155)
caps(s, (960, 720), "MORE PROTEIN PER ACRE THAN TWELVE PIGS", F(CAP, 62), INK)
shot(s, 2.4, [(["ONE ACRE.", "4,000 POUNDS."], RED, 54, (0.16, 0.72)),
              (["PIGS NEED GRAIN.", "CARP DO NOT."], INK, 52, (0.82, 0.86))])

# 3 measure_compare
s = canvas()
d = ImageDraw.Draw(s)
place(s, IC / "carp.png", 520, 380, 290)
place(s, IC / "cow.png", 1400, 380, 290)
d.line([(150, 580), (1770, 580)], fill=INK, width=9)
d.rectangle([340, 620, 700, 706], fill=RED)
caps(s, (520, 663), "1.2 KG FEED", F(CAP, 52), PAPER)
caps(s, (1400, 663), "8.0 KG FEED", F(CAP, 52), INK)
caps(s, (960, 830), "CARP WINS", F(HEAD, 98), RED)
shot(s, 2.2, [(["FEED PER POUND", "OF MEAT"], INK, 52, (0.13, 0.20)),
              (["SIX TIMES", "LESS INPUT"], RED, 54, (0.87, 0.80))])

# 4 versus_split
s = canvas()
d = ImageDraw.Draw(s)
d.line([(960, 170), (960, 860)], fill=INK, width=9)
place(s, LIB / "everyman_disgusted.png", 470, 490, 540)
place(s, LIB / "bureaucrat_stamping.png", 1420, 490, 580)
caps(s, (960, 470), "VS", F(HEAD, 118), RED)
caps(s, (470, 900), "THE EATER", F(CAP, 54), INK)
caps(s, (1420, 900), "THE COMMISSION", F(CAP, 54), INK)
shot(s, 2.2, [(["HE WILL", "NOT TRY IT"], INK, 52, (0.14, 0.17)),
              (["THEY STOCKED", "THE RIVERS"], RED, 52, (0.86, 0.14))])

# 5 archival_plate
s = canvas()
p = Image.open("assets/archival/archival.png").convert("RGB")
h = 500
w = int(p.width * h / p.height)
p = p.resize((w, h), Image.LANCZOS)
fr = Image.new("RGB", (w + 34, h + 34), (252, 252, 250))
fr.paste(p, (17, 17))
rot = fr.rotate(-2.2, expand=True, fillcolor=PAPER, resample=Image.BICUBIC)
s.paste(rot, (int(960 - rot.width / 2), int(400 - rot.height / 2)))
caps(s, (960, 760), "US FISH COMMISSION PLATE, 1880", F(CAP, 54), INK)
caps(s, (960, 830), "PUBLIC DOMAIN", F(CAP, 36), GREY)
shot(s, 2.6, [(["SHIPPED FREE", "TO 25 STATES"], RED, 54, (0.15, 0.88)),
              (["NOBODY ASKED", "FOR THEM"], INK, 50, (0.85, 0.88))])

# 6 text_card close
s = canvas()
caps(s, (960, 400), "A BONE.", F(HEAD, 165), INK)
caps(s, (960, 580), "AND A MOLECULE THAT SMELLS LIKE RAIN", F(CAP, 62), RED)
ImageDraw.Draw(s).line([(400, 640), (1520, 640)], fill=RED, width=8)
shot(s, 2.8, [(["TWO REASONS.", "BOTH FIXABLE."], INK, 56, (0.15, 0.82))])


def zoom_frames(img, n, z0, z1):
    """Crop-and-scale zoom. Rendering frames ourselves is what lets the whip
    interpolate already-zoomed content, so motion carries through the cut."""
    big = img.resize((int(W * 1.25), int(H * 1.25)), Image.LANCZOS)
    out = []
    for i in range(n):
        z = z0 + (z1 - z0) * (i / max(n - 1, 1))
        cw = int(big.width / z)
        ch = int(big.height / z)
        x, y = (big.width - cw) // 2, (big.height - ch) // 2
        out.append(big.crop((x, y, x + cw, y + ch)).resize((W, H), Image.LANCZOS))
    return out


# 31% of the reference's boundaries are whips; the rest are hard cuts.
WHIP_AT = {1: "x", 3: "y"}

timeline, prev_last = [], None
for i, (img, dur) in enumerate(shots):
    n = int(dur * FPS)
    z0, z1 = (Z_LO, Z_HI) if i % 2 == 0 else (Z_HI, Z_LO)
    frames = zoom_frames(img, n, z0, z1)
    if i in WHIP_AT and prev_last is not None:
        timeline += whip_frames(prev_last, frames[0], axis=WHIP_AT[i], reverse=(i % 4 == 3))
    timeline += frames
    prev_last = frames[-1]

for i, f in enumerate(timeline):
    f.save(FRAMES / f"{i:05d}.png")
print(f"{len(timeline)} frames ({len(timeline)/FPS:.2f}s), {len(WHIP_AT)} whips")

voice("build/narration_v3.wav", OUT / "narration_bass.wav")
subprocess.run(
    ["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(FRAMES / "%05d.png"),
     "-i", str(OUT / "narration_bass.wav"),
     "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
     "-c:a", "aac", "-b:a", "128k", "-shortest", "build/audit_v4.mp4"],
    check=True, capture_output=True)
print("wrote build/audit_v4.mp4")

sheet = Image.new("RGB", (W // 2 * 3, H // 2 * 2), PAPER)
for i, (img, _) in enumerate(shots):
    sheet.paste(img.resize((W // 2, H // 2), Image.LANCZOS), (i % 3 * (W // 2), i // 3 * (H // 2)))
sheet.save(OUT / "AUDIT_SHEET.png")
print("wrote", OUT / "AUDIT_SHEET.png")
