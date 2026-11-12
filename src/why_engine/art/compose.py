"""Composer: a Shot becomes a frame.

Spec S7.10. Everything the spikes in tools/ placed by hand is here as a
function of the shot, so a script the pipeline has never seen composes without
anyone laying out a card.

Layout constants are measured, not chosen. Card backgrounds are the
reference's own paper colour; captions go wherever the frame is actually empty
(art/annotate.py) rather than at fixed coordinates, because a generated shot
does not know in advance where its own artwork will land.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from why_engine.art.annotate import place_caption, zoom_safe_margin
from why_engine.art.direct import Shot
from why_engine.art.motion import PRESETS
from why_engine.art.palette import PAPER, match_reference

W, H = 1920, 1080
INK, RED, GREY = (26, 26, 26), (200, 38, 42), (150, 150, 150)
HEAD_FONT = "assets/fonts/PermanentMarker-Regular.ttf"
CAP_FONT = "assets/fonts/ArchitectsDaughter-Regular.ttf"

#: Inset that survives the largest zoom preset, so no caption is ever cropped.
SAFE = zoom_safe_margin(W, H, 1.0 + max(abs(p.total) for p in PRESETS.values()))
#: Icon grid geometry. Six per row keeps them large enough to count at a glance.
GRID_COLS = 6


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _fit(draw, text, path, start, max_width):
    """Largest size at which `text` fits `max_width`.

    Hand-lettered headlines vary hugely in length between scripts. Picking one
    size and hoping is how a headline ends up running off the frame.
    """
    size = start
    while size > 24:
        f = _font(path, size)
        x0, _, x1, _ = draw.textbbox((0, 0), text, font=f)
        if x1 - x0 <= max_width:
            return f
        size -= 4
    return _font(path, 24)


def card() -> Image.Image:
    return Image.new("RGB", (W, H), PAPER)


def _caps(img, xy, text, font, fill, anchor="mm"):
    ImageDraw.Draw(img).text(xy, text.upper(), font=font, fill=fill, anchor=anchor)


def _rule(img, y, half_width, colour=RED, width=10):
    ImageDraw.Draw(img).line([(W // 2 - half_width, y), (W // 2 + half_width, y)],
                             fill=colour, width=width)


def paste_asset(img, path, cx, cy, height, drop_white=True):
    """Place an asset, knocking out its white field so it sits on the paper."""
    asset = Image.open(path).convert("RGBA")
    w = int(asset.width * height / asset.height)
    asset = asset.resize((w, height), Image.LANCZOS)
    if drop_white:
        px = asset.load()
        for y in range(asset.height):
            for x in range(asset.width):
                r, g, b, a = px[x, y]
                if r > 236 and g > 236 and b > 236:
                    px[x, y] = (r, g, b, 0)
    img.paste(asset, (int(cx - w / 2), int(cy - height / 2)), asset)
    return asset


def _annotate(img, shot: Shot, anchors):
    """Drop the shot's captions into whatever space the artwork left."""
    occupied: list = []
    for text, anchor in zip(shot.captions, anchors):
        place_caption(img, [text.upper()], _font(CAP_FONT, 54), RED if anchor[0] > 0.5 else INK,
                      occupied=occupied, margin=SAFE, pad=20, anchor=anchor, gap=30)


def compose_text_card(shot: Shot, assets: dict) -> Image.Image:
    img = card()
    d = ImageDraw.Draw(img)
    head = shot.headline.upper() or shot.sentence.upper()[:24]
    font = _fit(d, head, HEAD_FONT, 175, W - 2 * SAFE - 120)
    _caps(img, (W // 2, 400), head, font, INK if shot.numbers else RED)
    x0, _, x1, _ = d.textbbox((0, 0), head, font=font)
    _rule(img, 500, min((x1 - x0) // 2 + 30, W // 2 - SAFE))
    if shot.subhead:
        sub = _fit(d, shot.subhead.upper(), CAP_FONT, 70, W - 2 * SAFE - 200)
        _caps(img, (W // 2, 600), shot.subhead, sub, INK)
    icon = assets.get("default_icon")
    if icon and Path(icon).exists():
        paste_asset(img, icon, W // 2, 840, 240)
    _annotate(img, shot, [(0.5, 0.90)])
    return img


def compose_icon_multiply(shot: Shot, assets: dict) -> Image.Image:
    img = card()
    subject = assets.get("subject_icon")
    unit = assets.get("count_icon") or subject
    if subject and Path(subject).exists():
        paste_asset(img, subject, 280, 380, 340)
        _caps(img, (500, 380), "=", _font(CAP_FONT, 160), INK)
    rows = (shot.count + GRID_COLS - 1) // GRID_COLS
    cols = min(shot.count, GRID_COLS)
    # Fit the grid to the space actually left, so a count of 30 does not run
    # off the right edge the way a fixed pitch does.
    left, right = 700, W - SAFE
    pitch = min((right - left) / max(cols, 1), 215)
    size = int(pitch * 0.88)
    for i in range(shot.count):
        r, c = divmod(i, GRID_COLS)
        if unit and Path(unit).exists():
            paste_asset(img, unit, left + pitch * (c + 0.5), 300 + r * (size + 24), size)
    d = ImageDraw.Draw(img)
    cap = _fit(d, shot.headline.upper(), CAP_FONT, 64, W - 2 * SAFE - 120)
    _caps(img, (W // 2, 300 + rows * (size + 24) + 90), shot.headline, cap, INK)
    _annotate(img, shot, [(0.16, 0.88), (0.84, 0.88)])
    return img


def compose_measure_compare(shot: Shot, assets: dict) -> Image.Image:
    img = card()
    d = ImageDraw.Draw(img)
    left_icon, right_icon = assets.get("left_icon"), assets.get("right_icon")
    for icon, cx in ((left_icon, 520), (right_icon, 1400)):
        if icon and Path(icon).exists():
            paste_asset(img, icon, cx, 370, 350)
    d.line([(SAFE, 590), (W - SAFE, 590)], fill=INK, width=9)
    values = (shot.captions + ["", ""])[:2]
    # The smaller value is the point being made, so it takes the red field.
    lo = 0 if _numeric(values[0]) <= _numeric(values[1]) else 1
    for i, (text, cx) in enumerate(zip(values, (520, 1400))):
        f = _font(CAP_FONT, 54)
        if i == lo:
            x0, y0, x1, y1 = d.textbbox((cx, 665), text.upper(), font=f, anchor="mm")
            d.rectangle([x0 - 26, y0 - 18, x1 + 26, y1 + 18], fill=RED)
            _caps(img, (cx, 665), text, f, PAPER)
        else:
            _caps(img, (cx, 665), text, f, INK)
    if shot.headline:
        hf = _fit(d, shot.headline.upper(), HEAD_FONT, 96, W - 2 * SAFE - 200)
        _caps(img, (W // 2, 840), shot.headline, hf, RED)
    return img


def _numeric(text: str) -> float:
    for part in text.replace(",", "").split():
        try:
            return float(part)
        except ValueError:
            continue
    return float("inf")


def compose_versus_split(shot: Shot, assets: dict) -> Image.Image:
    img = card()
    ImageDraw.Draw(img).line([(W // 2, 170), (W // 2, 870)], fill=INK, width=9)
    for key, cx in (("left_cast", 460), ("right_cast", 1440)):
        path = assets.get(key)
        if path and Path(path).exists():
            paste_asset(img, path, cx, 480, 620)
    for key, cx in (("left_label", 460), ("right_label", 1440)):
        if assets.get(key):
            _caps(img, (cx, 930), assets[key], _font(CAP_FONT, 54), INK)
    _annotate(img, shot, [(0.14, 0.14), (0.86, 0.14)])
    return img


def compose_painted_scene(shot: Shot, assets: dict) -> Image.Image:
    path = assets.get("scene")
    if not path or not Path(path).exists():
        return compose_text_card(shot, assets)
    img = match_reference(Image.open(path).convert("RGB")).resize((W, H), Image.LANCZOS)
    lines = shot.captions or [shot.headline]
    if lines and lines[0]:
        band(img, lines[0], at=0.85)
    return img


def band(img, text, at=0.85, size=68, colour=INK):
    """A caption band across a scene. Hand-lettering needs a field to sit on;
    over a painted background it is unreadable without one."""
    d = ImageDraw.Draw(img, "RGBA")
    f = _fit(d, text.upper(), CAP_FONT, size, W - 160)
    h = int(size * 1.5)
    y = int(H * at) - h // 2
    d.rectangle([0, y, W, y + h], fill=(255, 255, 255, 230))
    d.text((W // 2, y + h // 2), text.upper(), font=f, fill=colour, anchor="mm")
    return img


def compose_archival_plate(shot: Shot, assets: dict) -> Image.Image:
    path = assets.get("archival")
    if not path or not Path(path).exists():
        return compose_text_card(shot, assets)
    img = card()
    plate = Image.open(path).convert("RGB")
    h = 540
    w = int(plate.width * h / plate.height)
    plate = plate.resize((w, h), Image.LANCZOS)
    framed = Image.new("RGB", (w + 34, h + 34), (252, 252, 250))
    framed.paste(plate, (17, 17))
    rot = framed.rotate(-2.2, expand=True, fillcolor=PAPER, resample=Image.BICUBIC)
    img.paste(rot, (int(W / 2 - rot.width / 2), int(420 - rot.height / 2)))
    _caps(img, (W // 2, 800), assets.get("archival_caption", "ARCHIVAL PLATE"),
          _font(CAP_FONT, 54), INK)
    # The licence line is not decoration: it is the visible half of the gate
    # that keeps a copyrighted image out of the video.
    _caps(img, (W // 2, 868), assets.get("archival_licence", "PUBLIC DOMAIN"),
          _font(CAP_FONT, 36), GREY)
    _annotate(img, shot, [(0.15, 0.18)])
    return img


COMPOSERS = {
    "text_card": compose_text_card,
    "icon_multiply": compose_icon_multiply,
    "measure_compare": compose_measure_compare,
    "versus_split": compose_versus_split,
    "painted_scene": compose_painted_scene,
    "archival_plate": compose_archival_plate,
    "then_now": compose_text_card,
    "quantity_viz": compose_icon_multiply,
}


def compose(shot: Shot, assets: dict | None = None) -> Image.Image:
    """Render `shot`. Unknown archetypes fall back to a card rather than raising:
    a missing layout should cost one plain shot, not the whole video."""
    return COMPOSERS.get(shot.archetype, compose_text_card)(shot, assets or {})
