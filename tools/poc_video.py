"""SPIKE, not production. Hand-assembled proof of concept for the look and pace.
Plan 2B replaces this with art_director + composition + assembler."""
import subprocess, math
from pathlib import Path
from PIL import Image, ImageDraw

W, H = 1920, 1080
LIB, OUT = Path("assets/library"), Path("build/shots")
OUT.mkdir(parents=True, exist_ok=True)

def paper():
    """Off-white notebook page with faint ruled lines and a margin."""
    img = Image.new("RGB", (W, H), (250, 249, 244))
    d = ImageDraw.Draw(img)
    for y in range(120, H, 58):
        d.line([(0, y), (W, y)], fill=(225, 232, 238), width=2)
    d.line([(210, 0), (210, H)], fill=(238, 214, 214), width=3)
    return img

def place(canvas, path, cx, cy, target_h, mirror=False):
    """Composite a black-on-white drawing, dropping its white to transparent."""
    a = Image.open(path).convert("L")
    if mirror: a = a.transpose(Image.FLIP_LEFT_RIGHT)
    w = int(a.width * target_h / a.height)
    a = a.resize((w, target_h), Image.LANCZOS)
    rgba = Image.new("RGBA", a.size, (20, 20, 20, 255))
    rgba.putalpha(a.point(lambda p: 255 - p))          # ink opaque, paper clear
    canvas.paste(rgba, (int(cx - w / 2), int(cy - target_h / 2)), rgba)

def tape_in(canvas, path, cx, cy, target_h, angle=-3.5):
    """Archival photo presented as if taped into the notebook: rotated, shadowed,
    with tape strips. The contrast is the aesthetic, so it must look authored."""
    p = Image.open(path).convert("L").convert("RGB")
    w = int(p.width * target_h / p.height)
    p = p.resize((w, target_h), Image.LANCZOS)
    framed = Image.new("RGB", (w + 28, target_h + 28), (252, 252, 250))
    framed.paste(p, (14, 14))
    rot = framed.rotate(angle, expand=True, fillcolor=(250, 249, 244), resample=Image.BICUBIC)
    shadow = Image.new("RGBA", rot.size, (0, 0, 0, 55))
    canvas.paste(shadow, (int(cx - rot.width/2) + 8, int(cy - rot.height/2) + 8), shadow)
    canvas.paste(rot, (int(cx - rot.width / 2), int(cy - rot.height / 2)))
    d = ImageDraw.Draw(canvas, "RGBA")
    for dx in (-rot.width//2 + 30, rot.width//2 - 30):
        d.rectangle([cx+dx-48, cy-rot.height//2-12, cx+dx+48, cy-rot.height//2+26],
                    fill=(236, 226, 190, 190))

# --- shots: 4 cuts over ~9s of narration, average ~2.3s ---
shots = []

s = paper(); place(s, LIB/"everyman_neutral.png", 780, 620, 620)
place(s, LIB/"prop_document.png", 1330, 700, 300); s.save(OUT/"01.png"); shots.append(("01.png", 2.4, "in"))

s = paper(); place(s, LIB/"everyman_confused.png", 960, 600, 720); s.save(OUT/"02.png"); shots.append(("02.png", 2.2, "in"))

s = paper(); tape_in(s, Path("assets/archival/archival.png"), 960, 540, 620)
shots.append(("03.png", 2.6, "in")); s.save(OUT/"03.png")

s = paper(); place(s, LIB/"bureaucrat_stamping.png", 700, 600, 760)
place(s, LIB/"prop_chart.png", 1350, 640, 340); s.save(OUT/"04.png"); shots.append(("04.png", 2.2, "out"))

# --- ffmpeg: each still becomes a clip with a slow push, then concat + mux ---
clips = []
for name, dur, direction in shots:
    out = OUT / name.replace(".png", ".mp4")
    frames = int(dur * 30)
    z = ("min(1.0001+0.0009*on,1.12)" if direction == "in" else "max(1.12-0.0009*on,1.0001)")
    subprocess.run(["ffmpeg","-y","-loop","1","-i",str(OUT/name),"-t",str(dur),
        "-vf",f"scale=2400:-1,zoompan=z='{z}':d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps=30,format=yuv420p",
        "-r","30",str(out)], check=True, capture_output=True)
    clips.append(out)

with open("build/concat.txt","w") as f:
    for c in clips: f.write(f"file '{c.resolve()}'\n")
subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i","build/concat.txt",
    "-i","build/narration.wav","-c:v","libx264","-crf","20","-c:a","aac","-shortest",
    "build/proof_of_concept.mp4"], check=True, capture_output=True)
print("wrote build/proof_of_concept.mp4")
