"""The live art direction.

`library.py` still carries the superseded one -- thin black ink on white paper,
no colour -- and every asset in assets/library was built from it. The direction
now is the reference video's: flat colour, heavy uniform black outline, muted
earthy palette, red reserved as the only accent. It lives here rather than
inside a tool script so the next topic does not have to rediscover it, and so a
test can assert the two are actually different.

humanise() is not applied to anything generated from this direction: that pass
converts to greyscale and its tremor is tuned for thin ink line art, not flat
colour fills.
"""
from __future__ import annotations

from PIL import Image

ICON_PX = 460

#: Full-bleed painted scenes. Measured from the reference: flat fills, heavy
#: uniform outline, no gradients, sage/ochre/tan/warm brown/pale sky blue.
COLOUR_STYLE = (
    ", flat colour cartoon illustration, heavy uniform black outline, "
    "simple flat colour fills with no gradients and no shading, "
    "muted earthy palette of sage green, ochre, tan, warm brown and pale sky blue, "
    "full bleed edge to edge composition, hand drawn storybook look, no text, no words"
)

#: Diagram icons. Same drawing, but isolated: an icon carries its own
#: background onto the white card it is pasted on, so the background has to be
#: white in the generated frame rather than cropped away afterwards.
ICON_STYLE = (
    ", isolated on a pure white background, cut out, nothing else in the frame, "
    "no room, no interior, no wall, no floor, no ground, no horizon, no shadow, "
    "flat colour cartoon illustration, heavy uniform black outline, simple flat "
    "colour fills with no gradients and no shading, muted earthy palette, "
    "no text, no words"
)

#: How much of an icon's border has to be white before it will paste cleanly.
ICON_BORDER_WHITE = 0.90
_BORDER_PX = 12
_WHITE = 235


def to_square_icon(img: Image.Image, size: int = ICON_PX) -> Image.Image:
    """Centre-crop a 16:9 frame to a square icon at the library's size.

    fal bills per megapixel rounded up, so only the two 16:9 presets are ever
    generated; square_hd would bill double for less usable area. The subject is
    prompted into the middle of the frame, so the centre is what survives.
    """
    side = min(img.size)
    left = (img.width - side) // 2
    top = (img.height - side) // 2
    return img.convert("RGB").crop((left, top, left + side, top + side)).resize(
        (size, size), Image.LANCZOS)


def icon_is_isolated(img: Image.Image) -> bool:
    """Will this icon paste onto a white card without showing its own edges?

    Asking the model for a white background is not enough -- eight of the first
    twelve school-bus icons came back with a room behind the subject. The border
    is what gives that away: a cut-out subject leaves white there, a scene does
    not. The middle is deliberately ignored, because a dark subject filling the
    centre is the normal case.
    """
    import numpy as np

    a = np.asarray(img.convert("L"), dtype=float)
    b = _BORDER_PX
    border = np.concatenate([a[:b].ravel(), a[-b:].ravel(),
                             a[:, :b].ravel(), a[:, -b:].ravel()])
    return float((border > _WHITE).mean()) >= ICON_BORDER_WHITE
