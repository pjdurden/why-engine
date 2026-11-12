"""SPIKE, not production. Audit build adding colour and animation.

Two changes from v5, one measured and one a deliberate departure:

MEASURED -- full-bleed colour scenes. 38% of sampled reference frames are
colour scenes rather than white diagram cards, and v5 had none, which is why an
all-white build reads as clinical. Here 4 of 10 shots are scenes, saturation
matched to the reference by palette.match_reference.

DEPARTURE -- more motion than the reference's 19%. The measured grammar holds
81% of shots still; this build moves or animates 6 of 10. That is the user's
call, made after seeing the faithful version. It is recorded here rather than
in the spec, because the spec records what the reference does.

Cast are line-art on white, so they are composited only onto cards, never onto
a colour scene, until a colour-preserving humanise exists (Plan 2B task 1).

The facts on these cards are placeholder copy. They have NOT been through the
grounding gate.
"""
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from why_engine.art.annotate import place_caption, zoom_safe_margin
from why_engine.art.motion import PRESETS, Zoomer, pop_in, shot_moves, zoom_curve
from why_engine.art.palette import PAPER, match_reference
from why_engine.art.transitions import whip_sequence

W, H, FPS = 1920, 1080, 30
INK, RED, GREY, CREAM = (26, 26, 26), (200, 38, 42), (150, 150, 150), (247, 241, 226)
IC, LIB, SC, OUT = Path("build/icons"), Path("assets/library"), Path("assets/scenes"), Path("build/v6")
FRAMES = OUT / "frames"
FRAMES.mkdir(parents=True, exist_ok=True)
HEAD = "assets/fonts/PermanentMarker-Regular.ttf"
CAP = "assets/fonts/ArchitectsDaughter-Regular.ttf"
F = lambda p, s: ImageFont.truetype(p, s)
card = lambda: Image.new("RGB", (W, H), PAPER)
SAFE = zoom_safe_margin(W, H, 1.0 + max(abs(p.total) for p in PRESETS.values()))


def scene(name):
    """A full-bleed colour scene, saturation-matched to the reference."""
    im = match_reference(Image.open(SC / f"{name}.png").convert("RGB"))
    return im.resize((W, H), Image.LANCZOS)


def band(img, lines, colour=None, at=0.80, size=76):
    """A caption band across a colour scene -- text needs a field to sit on."""
    d = ImageDraw.Draw(img, "RGBA")
    h = int(size * 1.5 * len(lines))
    y = int(H * at) - h // 2
    d.rectangle([0, y, W, y + h], fill=(255, 255, 255, 228))
    for i, line in enumerate(lines):
        d.text((W // 2, y + int(h * (i + 0.5) / len(lines))), line.upper(),
               font=F(CAP, size), fill=colour or INK, anchor="mm")
    return img


def place(c, path, cx, cy, h, drop_white=True):
    im = Image.open(path).convert("RGBA")
    w = int(im.width * h / im.height)
    im = im.resize((w, h), Image.LANCZOS)
    if drop_white:
        px = im.load()
        for y in range(im.height):
            for x in range(im.width):
                r, g, b, a = px[x, y]
                if r > 236 and g > 236 and b > 236:
                    px[x, y] = (r, g, b, 0)
    c.paste(im, (int(cx - w / 2), int(cy - h / 2)), im)
    return im


def caps(c, xy, t, f, fill, anchor="mm"):
    ImageDraw.Draw(c).text(xy, t.upper(), font=f, fill=fill, anchor=anchor)


shots = []


def shot(img, dur, notes=(), zoom=None, pop=None):
    occupied = []
    for lines, colour, size, anchor in notes:
        place_caption(img, [l.upper() for l in lines], F(CAP, size), colour,
                      occupied=occupied, margin=SAFE, pad=20, anchor=anchor, gap=30)
    shots.append({"img": img, "dur": dur, "zoom": zoom, "pop": pop})


# 1  SCENE  river of carp                                      3.6s  push
s = band(scene("scene_river_carp"), ["HALF THE WORLD EATS THIS FISH"], INK, 0.84, 70)
caps(s, (960, 150), "WHY NOT CARP?", F(HEAD, 140), RED)
shot(s, 3.6, zoom="gentle_push")

# 2  CARD   icon_multiply, with the caption popping in          3.0s  pop
s = card()
place(s, IC / "carp.png", 270, 380, 360)
caps(s, (520, 380), "=", F(CAP, 170), INK)
for i in range(12):
    r, col = divmod(i, 6)
    place(s, IC / "pig.png", 720 + col * 225, 280 + r * 215, 195)
verdict = Image.new("RGBA", (1100, 110), (0, 0, 0, 0))
ImageDraw.Draw(verdict).text((550, 55), "MORE PROTEIN PER ACRE THAN TWELVE PIGS",
                             font=F(CAP, 62), fill=INK, anchor="mm")
shot(s, 3.0, [(["ONE ACRE.", "4,000 POUNDS."], RED, 54, (0.16, 0.88))],
     pop=(verdict, (960, 730), 14))

# 3  CARD   measure_compare, red banner pops                    2.8s  pop
s = card()
d = ImageDraw.Draw(s)
place(s, IC / "carp.png", 520, 360, 380)
place(s, IC / "cow.png", 1400, 360, 380)
d.line([(150, 580), (1770, 580)], fill=INK, width=9)
caps(s, (1400, 663), "8.0 KG FEED", F(CAP, 52), INK)
caps(s, (960, 850), "CARP WINS", F(HEAD, 98), RED)
tag = Image.new("RGBA", (360, 86), RED + (255,))
ImageDraw.Draw(tag).text((180, 43), "1.2 KG FEED", font=F(CAP, 52), fill=PAPER, anchor="mm")
shot(s, 2.8, [(["FEED PER POUND", "OF MEAT"], INK, 52, (0.13, 0.17)),
              (["SIX TIMES", "LESS INPUT"], RED, 54, (0.87, 0.17))],
     pop=(tag, (520, 663), 18))

# 4  CARD   the date                                            2.2s  static  -> WHIP
s = card()
caps(s, (960, 430), "1877", F(HEAD, 240), INK)
caps(s, (960, 640), "THE GOVERNMENT SHIPPED THEM FREE", F(CAP, 70), RED)
ImageDraw.Draw(s).line([(430, 705), (1490, 705)], fill=RED, width=9)
for i in range(5):
    place(s, IC / "carp.png", 460 + i * 250, 880, 150)
shot(s, 2.2, [(["TO 25 STATES"], INK, 56, (0.5, 0.30))])

# 5  SCENE  the depot                                           4.2s  push
s = band(scene("scene_depot_1877"), ["SHIPPED BY RAIL, IN MILK CANS"], INK, 0.85, 68)
shot(s, 4.2, zoom="gentle_push")

# 6  CARD   archival_plate                                      4.0s  pull
s = card()
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
shot(s, 4.0, [(["NOBODY ASKED", "FOR THEM"], RED, 54, (0.15, 0.90))], zoom="pull_out")

# 7  SCENE  the commission office                               3.8s  push
s = band(scene("scene_commission"), ["NOBODY ASKED FOR THEM"], RED, 0.85, 70)
shot(s, 3.8, zoom="gentle_push")

# 8  CARD   versus_split, the VS pops                           2.6s  pop
s = card()
d = ImageDraw.Draw(s)
d.line([(960, 170), (960, 860)], fill=INK, width=9)
place(s, LIB / "everyman_disgusted.png", 450, 480, 640)
place(s, LIB / "bureaucrat_stamping.png", 1440, 480, 690)
caps(s, (470, 900), "THE EATER", F(CAP, 54), INK)
caps(s, (1420, 900), "THE COMMISSION", F(CAP, 54), INK)
vs = Image.new("RGBA", (260, 180), (0, 0, 0, 0))
ImageDraw.Draw(vs).text((130, 90), "VS", font=F(HEAD, 118), fill=RED, anchor="mm")
shot(s, 2.6, [(["HE WILL", "NOT TRY IT"], INK, 52, (0.14, 0.14)),
              (["THEY STOCKED", "THE RIVERS"], RED, 52, (0.86, 0.14))],
     pop=(vs, (960, 470), 20))

# 9  SCENE  the choked river                                    4.0s  push
s = band(scene("scene_choked_river"), ["THIRTY YEARS LATER"], RED, 0.18, 84)
band(s, ["THE SAME GOVERNMENT PAID TO REMOVE THEM"], INK, 0.86, 62)
shot(s, 4.0, zoom="gentle_push")

# 10 CARD   close                                               4.2s  pull
s = card()
caps(s, (960, 400), "A BONE.", F(HEAD, 165), INK)
caps(s, (960, 580), "AND A MOLECULE THAT SMELLS LIKE RAIN", F(CAP, 62), RED)
ImageDraw.Draw(s).line([(400, 640), (1520, 640)], fill=RED, width=8)
place(s, IC / "carp.png", 960, 850, 260)
shot(s, 4.2, [(["TWO REASONS.", "BOTH FIXABLE."], INK, 56, (0.18, 0.80))], zoom="pull_out")

WHIP_BEFORE = 4

timeline, log = [], []
for i, sh in enumerate(shots):
    if i == WHIP_BEFORE:
        timeline += whip_sequence(shots[i - 1]["img"], sh["img"], fps=FPS, size=(W, H))
        log.append("whip (horizontal, compound)")
    n = int(round(sh["dur"] * FPS))
    if sh["pop"]:
        element, centre, delay = sh["pop"]
        timeline += pop_in(sh["img"], element, centre, frames=n, delay=delay, fps=FPS)
        kind = "POP-IN"
    else:
        curve = zoom_curve(sh["dur"], FPS, sh["zoom"])
        timeline += list(Zoomer(sh["img"], (W, H)).shot(curve))
        kind = ("MOVE " + sh["zoom"]) if max(curve) - min(curve) > 1e-9 else "static"
    log.append(f"shot {i+1:2d}: {sh['dur']:.1f}s  {kind}")

for line in log:
    print("  " + line)

scenes_n = sum(1 for s in shots if s["zoom"] and s["pop"] is None and s["dur"] >= 3.5)
animated = sum(1 for s in shots if s["pop"] or (s["zoom"] and shot_moves(s["dur"])))
print(f"\n{len(timeline)} frames, {len(timeline)/FPS:.2f}s, {len(shots)} shots, "
      f"{animated}/{len(shots)} carry motion ({100*animated/len(shots):.0f}%; reference 19%), "
      f"4/{len(shots)} colour scenes (40%; reference 38%)")

# Raw frames straight down a pipe. Writing ~1000 PNGs at 1080p dominated the
# build; this keeps the exact same bytes and skips the disk round trip.
X264 = "psy-rd=0.0:0.0,aq-mode=0,no-mbtree=1,bframes=0,ip-ratio=1.0"
proc = subprocess.Popen(
    ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
     "-framerate", str(FPS), "-i", "-", "-i", "build/v5_narration.wav",
     "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-x264-params", X264,
     "-c:a", "aac", "-b:a", "128k", "-shortest", "build/audit_v6.mp4"],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
last = None
for f in timeline:
    if f is not last:          # holds repeat the same object; encode its bytes once
        payload = f.convert("RGB").tobytes()
        last = f
    proc.stdin.write(payload)
proc.stdin.close()
if proc.wait() != 0:
    raise SystemExit("ffmpeg failed")
print("wrote build/audit_v6.mp4")

# Render each shot's FINAL state: a pop-in shot's base frame is missing the
# element that pops, so pasting sh["img"] showed a card with a hole in it.
cols, rows = 3, (len(shots) + 2) // 3
sheet = Image.new("RGB", (W // 3 * cols, H // 3 * rows), PAPER)
for i, sh in enumerate(shots):
    final = sh["img"]
    if sh["pop"]:
        final = pop_in(sh["img"], sh["pop"][0], sh["pop"][1], frames=1, delay=0, pop_frames=1)[0]
    sheet.paste(final.resize((W // 3, H // 3), Image.LANCZOS), (i % cols * (W // 3), i // cols * (H // 3)))
sheet.save(OUT / "AUDIT_SHEET.png")
print("wrote", OUT / "AUDIT_SHEET.png")
