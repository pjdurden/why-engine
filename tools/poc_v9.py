"""SPIKE, not production. Audit build with narration-aligned cuts.

The drift the previous builds had -- picture running ahead of the voice, worse
the further in you got -- came from picking shot durations by hand and laying a
single narration track over them. Nothing tied the two together.

Here every shot is cut to its own sentence: narrate() renders each sentence
separately, measures what it actually came out at, and that measurement is the
shot's length. Alignment holds by construction and cannot accumulate error.

Colour: 5 of 14 shots are full-bleed scenes (36%; the reference is 38%).
Motion: a deliberate departure from the measured 19%, at the user's request
after seeing the faithful version.

The facts on these cards are placeholder copy. They have NOT been through the
grounding gate.
"""
import os
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from why_engine.art.annotate import place_caption, zoom_safe_margin
from why_engine.art.motion import PRESETS, Zoomer, pop_in, shot_moves, zoom_curve
from why_engine.art.palette import PAPER, match_reference
from why_engine.art.transitions import whip_sequence
from why_engine.audio.narrate import narrate, shot_durations
from why_engine.audio.voice import median_f0, pitch_chain, CHAIN, NARRATION_F0_HZ

W, H, FPS = 1920, 1080, 30
INK, RED, GREY = (26, 26, 26), (200, 38, 42), (150, 150, 150)
IC, LIB, SC, OUT = Path("build/icons"), Path("assets/library"), Path("assets/scenes"), Path("build/v9")
OUT.mkdir(parents=True, exist_ok=True)
HEAD, CAP = "assets/fonts/PermanentMarker-Regular.ttf", "assets/fonts/ArchitectsDaughter-Regular.ttf"
F = lambda p, s: ImageFont.truetype(p, s)
cardimg = lambda: Image.new("RGB", (W, H), PAPER)
SAFE = zoom_safe_margin(W, H, 1.0 + max(abs(p.total) for p in PRESETS.values()))

SCRIPT = Path("build/v6_script.txt").read_text()

# Voice is swappable: pass WHY_VOICE to A/B a model without editing this file.
# US English is the choice for a channel about American systems: a native voice
# carries authority on the subject and avoids any comprehension friction in the
# first thirty seconds, which is where retention is decided. The Indian English
# model stays fetched and one env var away.
VOICE = os.environ.get("WHY_VOICE", "assets/voices/en_US-ryan-high.onnx")

# CHAIN's filter frequencies were fitted against a 167 Hz source. A different
# model sits somewhere else, so measure it once on a probe and shift it onto
# the reference pitch before the fixed EQ runs -- otherwise the peaking filters
# land on the wrong part of its spectrum.
probe = OUT / "probe.wav"
probe.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(["piper", "-m", VOICE, "-f", str(probe)],
               input="The United States Fish Commission shipped carp to twenty five states.",
               text=True, check=True, capture_output=True)
measured = median_f0(probe)
ratio = max(min(NARRATION_F0_HZ / measured, 1.35), 0.70)
chain = f"{pitch_chain(ratio)},{CHAIN.split('atempo=1.0417,', 1)[1]}"
print(f"voice {Path(VOICE).stem}: F0 {measured:.0f} Hz -> {NARRATION_F0_HZ:.0f} Hz (ratio {ratio:.3f})")

narration, segs = narrate(SCRIPT, VOICE, OUT / "audio", chain=chain)
DUR = shot_durations(segs)
print(f"{len(segs)} sentences -> {len(DUR)} shots, {sum(DUR):.2f}s total")


def scene(name):
    return match_reference(Image.open(SC / f"{name}.png").convert("RGB")).resize((W, H), Image.LANCZOS)


def band(img, lines, colour=None, at=0.84, size=68):
    d = ImageDraw.Draw(img, "RGBA")
    h = int(size * 1.5 * len(lines))
    y = int(H * at) - h // 2
    d.rectangle([0, y, W, y + h], fill=(255, 255, 255, 230))
    for i, line in enumerate(lines):
        d.text((W // 2, y + int(h * (i + 0.5) / len(lines))), line.upper(),
               font=F(CAP, size), fill=colour or INK, anchor="mm")
    return img


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


def text_el(text, font, fill, size=(1200, 130)):
    el = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(el).text((size[0] // 2, size[1] // 2), text.upper(), font=font, fill=fill, anchor="mm")
    return el


shots = []
def shot(img, notes=(), zoom=None, pop=None):
    i = len(shots)
    occupied = []
    for lines, colour, size, anchor in notes:
        place_caption(img, [l.upper() for l in lines], F(CAP, size), colour,
                      occupied=occupied, margin=SAFE, pad=20, anchor=anchor, gap=30)
    shots.append({"img": img, "dur": DUR[i], "zoom": zoom, "pop": pop})


# 0  "Half the world eats carp."
s = band(scene("scene_river_carp"), ["HALF THE WORLD EATS THIS FISH"])
caps(s, (960, 140), "WHY NOT CARP?", F(HEAD, 140), RED)
shot(s, zoom="gentle_push")

# 1  "America doesn't."
s = cardimg()
caps(s, (960, 430), "AMERICA DOESN'T.", F(HEAD, 175), RED)
ImageDraw.Draw(s).line([(340, 545), (1580, 545)], fill=RED, width=12)
for i in range(5):
    place(s, IC / "carp.png", 460 + i * 250, 780, 170)
shot(s, [(["45 STATES CALL IT", "AN INVASIVE SPECIES"], INK, 52, (0.5, 0.92))])

# 2  "And that is not an accident."
s = band(scene("scene_fish_counter"), ["IT IS NOT ON THE COUNTER"], RED)
shot(s, zoom="gentle_push")

# 3  "One acre of carp produces more protein than twelve pigs."
s = cardimg()
place(s, IC / "carp.png", 270, 380, 360)
caps(s, (520, 380), "=", F(CAP, 170), INK)
for i in range(12):
    r, col = divmod(i, 6)
    place(s, IC / "pig.png", 700 + col * 215, 300 + r * 215, 185)
shot(s, [(["ONE ACRE.", "4,000 POUNDS."], RED, 54, (0.14, 0.86))],
     pop=(text_el("MORE PROTEIN PER ACRE", F(CAP, 66), INK), (960, 800), 12))

# 4  "A cow needs eight kilograms of feed..."
s = cardimg()
d = ImageDraw.Draw(s)
place(s, IC / "cow.png", 960, 400, 430)
d.line([(260, 660), (1660, 660)], fill=INK, width=9)
caps(s, (960, 760), "8.0 KG FEED", F(HEAD, 110), INK)
shot(s, [(["PER KILOGRAM", "OF MEAT"], INK, 54, (0.14, 0.24))])

# 5  "A carp needs one point two."
s = cardimg()
place(s, IC / "carp.png", 960, 400, 430)
ImageDraw.Draw(s).line([(260, 660), (1660, 660)], fill=INK, width=9)
tag = Image.new("RGBA", (620, 140), RED + (255,))
ImageDraw.Draw(tag).text((310, 70), "1.2 KG FEED", font=F(HEAD, 100), fill=PAPER, anchor="mm")
shot(s, [(["SIX TIMES", "LESS INPUT"], RED, 56, (0.85, 0.24))], pop=(tag, (960, 770), 10))

# 6  "In 1877 the Fish Commission shipped carp free to 25 states."   -> WHIP in
s = cardimg()
caps(s, (960, 400), "1877", F(HEAD, 250), INK)
caps(s, (960, 620), "SHIPPED FREE TO 25 STATES", F(CAP, 76), RED)
ImageDraw.Draw(s).line([(410, 690), (1510, 690)], fill=RED, width=10)
shot(s, [(["US FISH COMMISSION"], INK, 54, (0.5, 0.88))])

# 7  "They went out by rail, packed in milk cans."
s = band(scene("scene_depot_1877"), ["BY RAIL, IN MILK CANS"])
shot(s, zoom="gentle_push")

# 8  "Nobody outside that office had asked for them at all."
s = band(scene("scene_commission"), ["NOBODY ASKED FOR THEM"], RED)
shot(s, zoom="gentle_push")

# 9  "Within thirty years the same government was paying to pull them out."
s = band(scene("scene_choked_river"), ["THIRTY YEARS LATER"], RED, 0.16, 80)
band(s, ["PAYING CREWS TO PULL THEM BACK OUT"], INK, 0.86, 62)
shot(s, zoom="gentle_push")

# 10 "The fish had not changed."
s = cardimg()
p = Image.open("assets/archival/archival.png").convert("RGB")
h = 540
w = int(p.width * h / p.height)
p = p.resize((w, h), Image.LANCZOS)
fr = Image.new("RGB", (w + 34, h + 34), (252, 252, 250))
fr.paste(p, (17, 17))
rot = fr.rotate(-2.2, expand=True, fillcolor=PAPER, resample=Image.BICUBIC)
s.paste(rot, (int(960 - rot.width / 2), int(420 - rot.height / 2)))
caps(s, (960, 800), "US FISH COMMISSION PLATE, 1880", F(CAP, 54), INK)
caps(s, (960, 868), "PUBLIC DOMAIN", F(CAP, 36), GREY)
shot(s, [(["THE FISH", "DID NOT CHANGE"], RED, 54, (0.14, 0.18))], zoom="pull_out")

# 11 "What changed was what Americans were willing to put on a plate."
s = cardimg()
d = ImageDraw.Draw(s)
d.line([(960, 150), (960, 880)], fill=INK, width=9)
place(s, LIB / "everyman_disgusted.png", 450, 470, 660)
place(s, LIB / "bureaucrat_stamping.png", 1440, 470, 710)
caps(s, (450, 930), "THE EATER", F(CAP, 56), INK)
caps(s, (1440, 930), "THE COMMISSION", F(CAP, 56), INK)
shot(s, [(["HE WILL", "NOT TRY IT"], INK, 52, (0.14, 0.13)),
         (["THEY STOCKED", "THE RIVERS"], RED, 52, (0.86, 0.13))],
     pop=(text_el("VS", F(HEAD, 130), RED, (300, 200)), (960, 460), 14))

# 12 "And the reason comes down to two things."
s = cardimg()
caps(s, (960, 440), "TWO REASONS.", F(HEAD, 175), INK)
ImageDraw.Draw(s).line([(430, 560), (1490, 560)], fill=RED, width=12)
shot(s, [(["BOTH OF THEM FIXABLE"], RED, 60, (0.5, 0.80))])

# 13 "A bone, and a molecule that smells like rain."
s = cardimg()
caps(s, (960, 380), "A BONE.", F(HEAD, 170), INK)
caps(s, (960, 560), "AND A MOLECULE THAT SMELLS LIKE RAIN", F(CAP, 64), RED)
ImageDraw.Draw(s).line([(390, 625), (1530, 625)], fill=RED, width=9)
place(s, IC / "carp.png", 960, 830, 280)
shot(s, zoom="pull_out")

WHIP_BEFORE = 6

# Frame boundaries from CUMULATIVE time, not per-shot rounding. Rounding each
# shot independently lets up to half a frame of error per shot accumulate; over
# fourteen shots that is a quarter second of drift by the end, in the build
# whose entire purpose is that the picture and the voice stay together.
edges, t = [0], 0.0
for d in DUR:
    t += d
    edges.append(int(round(t * FPS)))

timeline, log = [], []
for i, sh in enumerate(shots):
    n = edges[i + 1] - edges[i]
    if i == WHIP_BEFORE:
        # Spent from the incoming shot's budget, never added on top: appending
        # it pushes every later shot out of sync by the whip's own length.
        whip = whip_sequence(shots[i - 1]["img"], sh["img"], fps=FPS, size=(W, H))
        timeline += whip
        n -= len(whip)
        log.append(f"      whip ({len(whip)} frames, taken from shot {i}'s budget)")
    if sh["pop"]:
        el, centre, delay = sh["pop"]
        timeline += pop_in(sh["img"], el, centre, frames=n, delay=delay, fps=FPS)
        kind = "POP-IN"
    else:
        # Curve from the ADJUSTED frame count, not the nominal duration -- the
        # zoom branch used to recompute n from sh["dur"] and silently discard
        # the whip subtraction.
        curve = zoom_curve(n / FPS, FPS, sh["zoom"])[:n]
        curve += [curve[-1]] * (n - len(curve))
        timeline += list(Zoomer(sh["img"], (W, H)).shot(curve))
        kind = ("MOVE " + sh["zoom"]) if max(curve) - min(curve) > 1e-9 else "static"
    log.append(f"  {i:2d} {sh['dur']:5.2f}s  {kind:18s} \"{segs[i].text[:44]}\"")

assert len(timeline) == edges[-1], f"timeline {len(timeline)} != {edges[-1]} frames"

for line in log:
    print(line)

X264 = "psy-rd=0.0:0.0,aq-mode=0,no-mbtree=1,bframes=0,ip-ratio=1.0"
proc = subprocess.Popen(
    ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
     "-framerate", str(FPS), "-i", "-", "-i", str(narration),
     "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-x264-params", X264,
     "-c:a", "aac", "-b:a", "128k", "build/audit_v9.mp4"],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
last, payload = None, None
try:
    for f in timeline:
        if f is not last:
            payload, last = f.convert("RGB").tobytes(), f
        proc.stdin.write(payload)
    proc.stdin.close()
except BrokenPipeError:
    pass
if proc.wait() != 0:
    raise SystemExit("ffmpeg failed")

moving = sum(1 for s in shots if s["pop"] or (s["zoom"] and shot_moves(s["dur"])))
scenes = 5
print(f"\n{len(timeline)} frames, {len(timeline)/FPS:.2f}s video, {sum(DUR):.2f}s audio, "
      f"{len(shots)} shots = {60*(len(shots)-1)/(sum(DUR)):.1f} cuts/min\n"
      f"{moving}/{len(shots)} carry motion, {scenes}/{len(shots)} colour scenes "
      f"({100*scenes/len(shots):.0f}%; reference 38%)")
print("wrote build/audit_v9.mp4")

cols, rows = 4, (len(shots) + 3) // 4
sheet = Image.new("RGB", (W // 4 * cols, H // 4 * rows), PAPER)
for i, sh in enumerate(shots):
    # frames=2 with pop_frames=1 so index 1 is progress 1.0 -- the settled
    # state. Asking for frame 0 gives scale 0 and drops the element entirely.
    final = sh["img"] if not sh["pop"] else \
        pop_in(sh["img"], sh["pop"][0], sh["pop"][1], frames=2, delay=0, pop_frames=1)[1]
    sheet.paste(final.resize((W // 4, H // 4), Image.LANCZOS), (i % cols * (W // 4), i // cols * (H // 4)))
sheet.save(OUT / "AUDIT_SHEET.png")
print("wrote", OUT / "AUDIT_SHEET.png")
