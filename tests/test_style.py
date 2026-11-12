"""The live art direction, and the crop that makes an icon square.

`library.py` still carries the superseded direction -- thin black ink, no
colour -- and everything in assets/library was built from it. The colour
direction lives here so a second topic does not have to rediscover it inside a
tool script.
"""
from PIL import Image

from why_engine.art.style import COLOUR_STYLE, ICON_STYLE, to_square_icon


def test_the_colour_direction_is_not_the_dead_monochrome_one():
    from why_engine.art.library import STYLE_SUFFIX
    assert "no colour" in STYLE_SUFFIX          # the superseded direction
    assert "no colour" not in COLOUR_STYLE
    assert "flat colour" in COLOUR_STYLE


def test_an_icon_prompt_asks_for_an_isolated_object():
    """Icons are pasted onto white diagram cards, so a scene behind one shows up
    as a rectangle of background sitting on the card."""
    assert "white background" in ICON_STYLE


def test_a_landscape_frame_becomes_a_square_icon():
    assert to_square_icon(Image.new("RGB", (1024, 576), "white")).size == (460, 460)


def test_a_portrait_frame_becomes_a_square_icon():
    assert to_square_icon(Image.new("RGB", (576, 1024), "white")).size == (460, 460)


def test_the_crop_is_centred():
    """A subject generated in the middle of the frame must survive the crop."""
    img = Image.new("RGB", (1024, 576), "white")
    # A block, not a pixel: LANCZOS blends a single pixel with its white
    # neighbours, which would make this test about the resampler.
    img.paste((255, 0, 0), (512 - 40, 288 - 40, 512 + 40, 288 + 40))
    out = to_square_icon(img)
    assert out.getpixel((230, 230)) == (255, 0, 0)
    assert out.getpixel((5, 5)) == (255, 255, 255)


def test_the_icon_has_no_alpha_like_the_existing_ones():
    """The composer pastes icons onto white cards; the ones already in the
    library are RGB, and a mode change would be a silent difference."""
    assert to_square_icon(Image.new("RGB", (1024, 576), "white")).mode == "RGB"


def test_an_icon_on_white_is_isolated():
    from why_engine.art.style import icon_is_isolated
    assert icon_is_isolated(Image.new("RGB", (460, 460), "white"))


def test_an_icon_with_a_scene_behind_it_is_not_isolated():
    """Eight of the twelve school-bus icons came back with a room behind them
    despite the prompt asking for white. Pasted onto a white diagram card, that
    reads as a rectangle of someone else's picture sitting on the card, so the
    prompt cannot be the only thing enforcing this."""
    from why_engine.art.style import icon_is_isolated
    assert not icon_is_isolated(Image.new("RGB", (460, 460), (120, 140, 110)))


def test_a_drawn_subject_does_not_make_an_icon_look_unisolated():
    """The check looks at the border, not the middle: a dark subject filling the
    centre of the frame is the normal case, not a failure."""
    from why_engine.art.style import icon_is_isolated
    img = Image.new("RGB", (460, 460), "white")
    img.paste((30, 30, 30), (120, 120, 340, 340))
    assert icon_is_isolated(img)
