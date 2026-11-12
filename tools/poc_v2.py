"""SPIKE, not production. Hand-composed demo of the shot archetypes from spec S9a.
Plan 2B replaces this with archetypes.py + compose.py + text.py."""
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
PAPER, INK, RED = (253,253,253), (26,26,26), (200,38,42)
IC, OUT = Path("build/icons"), Path("build/v2"); OUT.mkdir(parents=True, exist_ok=True)
HEAD = "assets/fonts/PermanentMarker-Regular.ttf"
CAP  = "assets/fonts/ArchitectsDaughter-Regular.ttf"
F = lambda p,s: ImageFont.truetype(p, s)

def canvas(): return Image.new("RGB",(W,H),PAPER)

def icon(c, name, cx, cy, h):
    im = Image.open(IC/f"{name}.png").convert("RGBA")
    w = int(im.width*h/im.height); im = im.resize((w,h), Image.LANCZOS)
    px = im.load()
    for y in range(im.height):
        for x in range(im.width):
            r,g,b,a = px[x,y]
            if r>238 and g>238 and b>238: px[x,y]=(r,g,b,0)
    c.paste(im,(int(cx-w/2),int(cy-h/2)),im)

def caps(c, xy, t, font, fill, anchor="mm"):
    ImageDraw.Draw(c).text(xy, t.upper(), font=font, fill=fill, anchor=anchor)

shots=[]

# 1. text_card
s=canvas(); caps(s,(960,430),"WHY NOT CARP?",F(HEAD,140),RED)
ImageDraw.Draw(s).line([(520,530),(1400,530)],fill=RED,width=10)
caps(s,(960,630),"HALF THE WORLD EATS IT",F(CAP,64),INK); s.save(OUT/"01.png"); shots.append(("01.png",2.0))

# 2. icon_multiply
s=canvas(); icon(s,"carp",400,430,300); caps(s,(680,430),"=",F(CAP,140),INK)
for i in range(12):
    r,col=divmod(i,6); icon(s,"pig",900+col*170,350+r*180,140)
caps(s,(960,800),"MORE PROTEIN THAN 12 PIGS",F(CAP,70),INK); s.save(OUT/"02.png"); shots.append(("02.png",2.2))

# 3. versus_split
s=canvas(); d=ImageDraw.Draw(s); d.line([(960,170),(960,860)],fill=INK,width=8)
icon(s,"carp",520,470,340); icon(s,"cow",1400,470,340)
caps(s,(960,470),"VS",F(HEAD,110),RED)
icon(s,"cross",520,830,120); icon(s,"check",1400,830,120); s.save(OUT/"03.png"); shots.append(("03.png",2.0))

# 4. archival_plate
s=canvas()
p=Image.open("assets/archival/archival.png").convert("RGB")
h=600; w=int(p.width*h/p.height); p=p.resize((w,h),Image.LANCZOS)
fr=Image.new("RGB",(w+34,h+34),(252,252,250)); fr.paste(p,(17,17))
rot=fr.rotate(-2.2,expand=True,fillcolor=PAPER,resample=Image.BICUBIC)
s.paste(rot,(int(960-rot.width/2),int(450-rot.height/2)))
caps(s,(960,880),"US FISH COMMISSION, 1880",F(CAP,52),INK); s.save(OUT/"04.png"); shots.append(("04.png",2.4))

# 5. text_card close
s=canvas(); caps(s,(960,460),"IT WAS A BONE.",F(HEAD,150),INK)
caps(s,(960,620),"AND A MOLECULE THAT SMELLS LIKE RAIN",F(CAP,58),RED); s.save(OUT/"05.png"); shots.append(("05.png",2.0))

clips=[]
for i,(n,dur) in enumerate(shots):
    o=OUT/n.replace(".png",".mp4"); fr=int(dur*30)
    z = f"min(1.0001+0.0011*on,1.10)" if i%2==0 else f"max(1.10-0.0011*on,1.0001)"
    subprocess.run(["ffmpeg","-y","-loop","1","-i",str(OUT/n),"-t",str(dur),
      "-vf",f"scale=2400:-1,zoompan=z='{z}':d={fr}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps=30,format=yuv420p",
      "-r","30",str(o)],check=True,capture_output=True)
    clips.append(o)
open("build/v2/concat.txt","w").write("".join(f"file '{c.resolve()}'\n" for c in clips))
subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i","build/v2/concat.txt",
  "-i","build/narration_v2.wav","-c:v","libx264","-crf","20","-c:a","aac","-shortest",
  "build/reference_v2.mp4"],check=True,capture_output=True)
print("wrote build/reference_v2.mp4")
