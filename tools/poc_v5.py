"""SPIKE, not production. Audit build under the MEASURED motion grammar from
docs/superpowers/specs/2026-09-16-reference-motion-analysis.md.

What changed from v4, and why each thing was wrong:
  - v4 zoomed every shot. The reference holds 81% of them, and a shot under
    3.5s never moves. Here 7 of 9 shots are frozen.
  - v4 alternated push and pull every shot. That pattern is not in the data;
    there were only 6 adjacent zoom-zoom pairs in the whole reference, too few
    to support any rule. Dropped.
  - v4 used 2 whips in 14s, one of them vertical. The reference spends 7 in
    16.7 minutes, all before t=164s, all horizontal. Here: 1, horizontal.
  - v4's whip was a bare 0.17s smear. The real device wraps the smear in a
    push-in ramp and a pull-out ramp, 0.6-0.9s total.

The facts on these cards are placeholder copy for a look-and-feel audit. They
have NOT been through the grounding gate.
"""
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from why_engine.art.annotate import place_caption, zoom_safe_margin
from why_engine.art.motion import PRESETS, Zoomer, shot_moves, zoom_curve
from why_engine.art.transitions import whip_sequence

W, H, FPS = 1920, 1080, 30
PAPER, INK, RED, GREY = (253, 253, 253), (26, 26, 26), (200, 38, 42), (150, 150, 150)
IC, LIB, OUT = Path("build/icons"), Path("assets/library"), Path("build/v5")
FRAMES = OUT / "frames"
FRAMES.mkdir(parents=True, exist_ok=True)
HEAD = "assets/fonts/PermanentMarker-Regular.ttf"
CAP = "assets/fonts/ArchitectsDaughter-Regular.ttf"
F = lambda p, s: ImageFont.truetype(p, s)
canvas = lambda: Image.new("RGB", (W, H), PAPER)
# Peak zoom across every preset, so no caption is ever cropped by a move.
SAFE = zoom_safe_margin(W, H, 1.0 + max(abs(p.total) for p in PRESETS.values()))


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


def shot(img, dur, notes, zoom=None):
    occupied = []
    for lines, colour, size, anchor in notes:
        place_caption(img, [l.upper() for l in lines], F(CAP, size), colour,
                      occupied=occupied, margin=SAFE, pad=20, anchor=anchor, gap=30)
    shots.append((img, dur, zoom))


# 1  text_card                                    3.0s  static
s = canvas()
caps(s, (960, 380), "WHY NOT CARP?", F(HEAD, 155), RED)
ImageDraw.Draw(s).line([(470, 485), (1450, 485)], fill=RED, width=11)
caps(s, (960, 575), "HALF THE WORLD EATS IT", F(CAP, 72), INK)
place(s, IC / "carp.png", 960, 800, 230)
shot(s, 3.0, [(["INVASIVE IN", "45 US STATES"], INK, 56, (0.15, 0.80)),
              (["AMERICA", "WILL NOT"], RED, 56, (0.85, 0.80))])

# 2  icon_multiply                                2.6s  static
s = canvas()
place(s, IC / "carp.png", 300, 400, 290)
caps(s, (520, 400), "=", F(CAP, 150), INK)
for i in range(12):
    r, col = divmod(i, 6)
    place(s, IC / "pig.png", 740 + col * 190, 320 + r * 185, 155)
caps(s, (960, 720), "MORE PROTEIN PER ACRE THAN TWELVE PIGS", F(CAP, 62), INK)
shot(s, 2.6, [(["ONE ACRE.", "4,000 POUNDS."], RED, 54, (0.16, 0.88)),
              (["PIGS NEED GRAIN.", "CARP DO NOT."], INK, 52, (0.84, 0.88))])

# 3  measure_compare                              2.8s  static
s = canvas()
d = ImageDraw.Draw(s)
place(s, IC / "carp.png", 520, 380, 290)
place(s, IC / "cow.png", 1400, 380, 290)
d.line([(150, 580), (1770, 580)], fill=INK, width=9)
d.rectangle([340, 620, 700, 706], fill=RED)
caps(s, (520, 663), "1.2 KG FEED", F(CAP, 52), PAPER)
caps(s, (1400, 663), "8.0 KG FEED", F(CAP, 52), INK)
caps(s, (960, 830), "CARP WINS", F(HEAD, 98), RED)
shot(s, 2.8, [(["FEED PER POUND", "OF MEAT"], INK, 52, (0.13, 0.17)),
              (["SIX TIMES", "LESS INPUT"], RED, 54, (0.87, 0.17))])

# 4  text_card, the date                          2.2s  static   -> WHIP into 5
s = canvas()
caps(s, (960, 430), "1877", F(HEAD, 240), INK)
caps(s, (960, 640), "THE GOVERNMENT SHIPPED THEM FREE", F(CAP, 70), RED)
ImageDraw.Draw(s).line([(430, 705), (1490, 705)], fill=RED, width=9)
shot(s, 2.2, [(["TO 25 STATES"], INK, 56, (0.5, 0.88))])

# 5  archival_plate                               4.9s  gentle push
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
shot(s, 4.9, [(["NOBODY ASKED", "FOR THEM"], RED, 54, (0.15, 0.90)),
              (["SHIPPED BY RAIL", "IN MILK CANS"], INK, 50, (0.85, 0.90))], zoom="gentle_push")

# 6  versus_split                                 2.4s  static
s = canvas()
d = ImageDraw.Draw(s)
d.line([(960, 170), (960, 860)], fill=INK, width=9)
place(s, LIB / "everyman_disgusted.png", 470, 490, 540)
place(s, LIB / "bureaucrat_stamping.png", 1420, 490, 580)
caps(s, (960, 470), "VS", F(HEAD, 118), RED)
caps(s, (470, 900), "THE EATER", F(CAP, 54), INK)
caps(s, (1420, 900), "THE COMMISSION", F(CAP, 54), INK)
shot(s, 2.4, [(["HE WILL", "NOT TRY IT"], INK, 52, (0.14, 0.14)),
              (["THEY STOCKED", "THE RIVERS"], RED, 52, (0.86, 0.14))])

# 7  text_card, the turn                          2.0s  static
s = canvas()
caps(s, (960, 460), "THIRTY YEARS LATER", F(HEAD, 130), INK)
caps(s, (960, 630), "THE SAME GOVERNMENT PAID TO REMOVE THEM", F(CAP, 64), RED)
ImageDraw.Draw(s).line([(340, 692), (1580, 692)], fill=RED, width=9)
shot(s, 2.0, [])

# 8  icon shot, the cost                          3.8s  static (under the bar)
s = canvas()
place(s, IC / "coins.png", 640, 430, 300)
caps(s, (960, 430), "->", F(HEAD, 110), RED)
place(s, IC / "carp.png", 1310, 430, 300)
caps(s, (960, 720), "PUBLIC MONEY, SPENT BACKWARDS", F(CAP, 66), INK)
shot(s, 3.8, [(["REMOVAL COSTS", "STILL RUN TODAY"], RED, 52, (0.5, 0.90))])

# 9  text_card close                              4.1s  pull-out
s = canvas()
caps(s, (960, 400), "A BONE.", F(HEAD, 165), INK)
caps(s, (960, 580), "AND A MOLECULE THAT SMELLS LIKE RAIN", F(CAP, 62), RED)
ImageDraw.Draw(s).line([(400, 640), (1520, 640)], fill=RED, width=8)
shot(s, 4.1, [(["TWO REASONS.", "BOTH FIXABLE."], INK, 56, (0.5, 0.88))], zoom="pull_out")

# The reference spends its 7 whips inside the opening 164s, ~11% of that
# window's boundaries. Nine shots give eight boundaries, so exactly one.
WHIP_BEFORE = 4  # index of the shot the whip lands ON

timeline = []
log = []
for i, (img, dur, zoom) in enumerate(shots):
    if i == WHIP_BEFORE:
        timeline += whip_sequence(shots[i - 1][0], img, fps=FPS, size=(W, H))
        log.append(("whip", len(timeline)))
    curve = zoom_curve(dur, FPS, zoom)
    timeline += list(Zoomer(img, (W, H)).shot(curve))
    moved = max(curve) - min(curve) > 1e-9
    log.append((f"shot {i+1}: {dur:.1f}s  {'MOVE ' + zoom if moved else 'static'}"
                f"{'' if shot_moves(dur) else '  (under 3.5s, gate forces static)'}", len(timeline)))

for i, f in enumerate(timeline):
    f.save(FRAMES / f"{i:05d}.png")
for name, end in log:
    print(f"  {name:<58} -> t={end/FPS:5.2f}s")
moving = sum(1 for _, d, z in shots if z and shot_moves(d))
print(f"\n{len(timeline)} frames, {len(timeline)/FPS:.2f}s, {len(shots)} shots, "
      f"{moving}/{len(shots)} move ({100*moving/len(shots):.0f}%, reference 19%), "
      f"1 whip, {len(shots)-1} boundaries = {60*(len(shots)-1)/(len(timeline)/FPS):.1f} cuts/min")

# x264's defaults re-dither held frames: 81.8% of the composed frames are
# bit-identical to their predecessor but only 21.6% survive a default encode,
# which shows up as each shot visibly settling for a few frames after its cut.
# Disabling the psycho-visual passes takes that to 55.9% and shrinks the file.
# The rest is x264 converging on each new shot and is still open -- eight
# configurations were measured, none beat this one.
X264 = "psy-rd=0.0:0.0,aq-mode=0,no-mbtree=1,bframes=0,ip-ratio=1.0"
subprocess.run(
    ["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(FRAMES / "%05d.png"),
     "-i", "build/v5_narration.wav",
     "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-x264-params", X264,
     "-c:a", "aac", "-b:a", "128k", "-shortest", "build/audit_v5.mp4"],
    check=True, capture_output=True)
print("wrote build/audit_v5.mp4")

cols = 3
rows = (len(shots) + cols - 1) // cols
sheet = Image.new("RGB", (W // 3 * cols, H // 3 * rows), PAPER)
for i, (img, _, _) in enumerate(shots):
    sheet.paste(img.resize((W // 3, H // 3), Image.LANCZOS), (i % cols * (W // 3), i // cols * (H // 3)))
sheet.save(OUT / "AUDIT_SHEET.png")
print("wrote", OUT / "AUDIT_SHEET.png")
