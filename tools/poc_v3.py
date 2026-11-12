"""SPIKE, not production. Audit build demonstrating the S9a shot archetypes.
Plan 2B replaces this with archetypes.py + compose.py + text.py."""
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
PAPER, INK, RED, GREEN = (253,253,253), (26,26,26), (200,38,42), (60,140,70)
IC, LIB, OUT = Path("build/icons"), Path("assets/library"), Path("build/v3")
OUT.mkdir(parents=True, exist_ok=True)
HEAD, CAP = "assets/fonts/PermanentMarker-Regular.ttf", "assets/fonts/ArchitectsDaughter-Regular.ttf"
F = lambda p,s: ImageFont.truetype(p,s)
canvas = lambda: Image.new("RGB",(W,H),PAPER)

def place(c, path, cx, cy, h):
    im = Image.open(path).convert("RGBA")
    w = int(im.width*h/im.height); im = im.resize((w,h), Image.LANCZOS)
    px = im.load()
    for y in range(im.height):
        for x in range(im.width):
            r,g,b,a = px[x,y]
            if r>236 and g>236 and b>236: px[x,y]=(r,g,b,0)
    c.paste(im,(int(cx-w/2),int(cy-h/2)),im)

def caps(c, xy, t, f, fill, anchor="mm"):
    ImageDraw.Draw(c).text(xy, t.upper(), font=f, fill=fill, anchor=anchor)

shots=[]
def shot(img, name, dur): img.save(OUT/name); shots.append((name,dur))

# 1 text_card — tighter: headline higher, rule, subhead close under
s=canvas(); caps(s,(960,400),"WHY NOT CARP?",F(HEAD,155),RED)
ImageDraw.Draw(s).line([(470,505),(1450,505)],fill=RED,width=11)
caps(s,(960,600),"HALF THE WORLD EATS IT",F(CAP,72),INK)
place(s,IC/"carp.png",960,830,230); shot(s,"01.png",2.0)

# 2 icon_multiply — fills the frame, caption tight under
s=canvas(); place(s,IC/"carp.png",330,430,300); caps(s,(560,430),"=",F(CAP,150),INK)
for i in range(12):
    r,col=divmod(i,6); place(s,IC/"pig.png",760+col*195,340+r*195,160)
caps(s,(960,760),"MORE PROTEIN PER ACRE THAN TWELVE PIGS",F(CAP,66),INK); shot(s,"02.png",2.4)

# 3 measure_compare — values banded, verdict under
s=canvas(); d=ImageDraw.Draw(s)
place(s,IC/"carp.png",520,400,300); place(s,IC/"cow.png",1400,400,300)
d.line([(150,610),(1770,610)],fill=INK,width=9)
d.rectangle([340,650,700,740],fill=RED); caps(s,(520,695),"1.2 KG FEED",F(CAP,54),PAPER)
caps(s,(1400,695),"8.0 KG FEED",F(CAP,54),INK)
caps(s,(960,880),"CARP WINS",F(HEAD,100),RED); shot(s,"03.png",2.2)

# 4 versus_split with the cast
s=canvas(); d=ImageDraw.Draw(s); d.line([(960,150),(960,900)],fill=INK,width=9)
place(s,LIB/"everyman_disgusted.png",480,500,560)
place(s,LIB/"bureaucrat_stamping.png",1420,500,600)
caps(s,(960,480),"VS",F(HEAD,120),RED)
caps(s,(480,930),"THE EATER",F(CAP,56),INK); caps(s,(1420,930),"THE COMMISSION",F(CAP,56),INK)
shot(s,"04.png",2.0)

# 5 archival_plate
s=canvas()
p=Image.open("assets/archival/archival.png").convert("RGB")
h=620; w=int(p.width*h/p.height); p=p.resize((w,h),Image.LANCZOS)
fr=Image.new("RGB",(w+34,h+34),(252,252,250)); fr.paste(p,(17,17))
rot=fr.rotate(-2.2,expand=True,fillcolor=PAPER,resample=Image.BICUBIC)
s.paste(rot,(int(960-rot.width/2),int(430-rot.height/2)))
caps(s,(960,860),"US FISH COMMISSION PLATE, 1880",F(CAP,56),INK)
caps(s,(960,940),"PUBLIC DOMAIN",F(CAP,38),(150,150,150)); shot(s,"05.png",2.6)

# 6 text_card close
s=canvas(); caps(s,(960,420),"A BONE.",F(HEAD,165),INK)
caps(s,(960,600),"AND A MOLECULE THAT SMELLS LIKE RAIN",F(CAP,64),RED)
ImageDraw.Draw(s).line([(400,660),(1520,660)],fill=RED,width=8); shot(s,"06.png",2.8)

clips=[]
for i,(n,dur) in enumerate(shots):
    o=OUT/n.replace(".png",".mp4"); fr=int(dur*30)
    z = "min(1.0001+0.0010*on,1.09)" if i%2==0 else "max(1.09-0.0010*on,1.0001)"
    subprocess.run(["ffmpeg","-y","-loop","1","-i",str(OUT/n),"-t",str(dur),
      "-vf",f"scale=2400:-1,zoompan=z='{z}':d={fr}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps=30,format=yuv420p",
      "-r","30",str(o)],check=True,capture_output=True)
    clips.append(o)
open(OUT/"concat.txt","w").write("".join(f"file '{c.resolve()}'\n" for c in clips))
subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(OUT/"concat.txt"),
  "-i","build/narration_v3.wav","-c:v","libx264","-crf","20","-c:a","aac","-shortest",
  "build/audit.mp4"],check=True,capture_output=True)
print("wrote build/audit.mp4")
