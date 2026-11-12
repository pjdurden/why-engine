import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from why_engine.art.annotate import find_empty_box, ink_mask, place_caption

W, H = 640, 360
PAPER = (253, 253, 253)
FONT = ImageFont.load_default(size=16)


def blank() -> Image.Image:
    return Image.new("RGB", (W, H), PAPER)


def with_blob(box) -> Image.Image:
    img = blank()
    ImageDraw.Draw(img).rectangle(box, fill=(20, 20, 20))
    return img


def test_ink_mask_sees_marks_not_paper():
    mask = ink_mask(with_blob([100, 100, 199, 199]))
    assert mask[150, 150]
    assert not mask[10, 10]
    assert mask.sum() == pytest.approx(100 * 100, rel=0.02)


def test_find_empty_box_avoids_ink():
    """Artwork fills the left half; the box must land on the right."""
    mask = ink_mask(with_blob([0, 0, W // 2, H]))
    box = find_empty_box(mask, 120, 40)
    assert box is not None
    assert box[0] >= W // 2


def test_find_empty_box_returns_none_when_the_frame_is_full():
    mask = ink_mask(with_blob([0, 0, W, H]))
    assert find_empty_box(mask, 120, 40) is None


def test_find_empty_box_respects_the_margin():
    box = find_empty_box(ink_mask(blank()), 100, 30, margin=60)
    x, y, bw, bh = box
    assert x >= 60 and y >= 60
    assert x + bw <= W - 60 and y + bh <= H - 60


def test_find_empty_box_returns_none_when_the_block_cannot_fit():
    assert find_empty_box(ink_mask(blank()), W, H) is None


def test_find_empty_box_treats_occupied_boxes_as_full():
    """Two captions must not stack on the same clear patch."""
    mask = ink_mask(with_blob([0, 0, W // 2, H]))
    first = find_empty_box(mask, 120, 40)
    second = find_empty_box(mask, 120, 40, occupied=[first])
    assert second is not None
    fx, fy, fw, fh = first
    sx, sy, sw, sh = second
    assert not (fx < sx + sw and sx < fx + fw and fy < sy + sh and sy < fy + fh)


def test_place_caption_draws_into_the_clear_half():
    img = with_blob([0, 0, W // 2, H])
    before = ink_mask(img).sum()
    box = place_caption(img, ["CARP WINS"], FONT, (200, 38, 42))
    assert box is not None
    after = ink_mask(img)
    assert after.sum() > before
    assert after[:, W // 2 :].sum() > 0


def test_place_caption_appends_to_occupied():
    img = blank()
    occupied = []
    place_caption(img, ["ONE"], FONT, (0, 0, 0), occupied=occupied)
    place_caption(img, ["TWO"], FONT, (0, 0, 0), occupied=occupied)
    assert len(occupied) == 2


def test_place_caption_draws_nothing_when_there_is_no_room():
    img = with_blob([0, 0, W, H])
    before = np.asarray(img).copy()
    assert place_caption(img, ["NO ROOM"], FONT, (200, 38, 42)) is None
    assert np.array_equal(np.asarray(img), before)


def test_place_caption_stacks_multiple_lines_downward():
    img = blank()
    box = place_caption(img, ["FIRST LINE", "SECOND LINE"], FONT, (0, 0, 0))
    one = place_caption(blank(), ["FIRST LINE"], FONT, (0, 0, 0))
    assert box[3] > one[3]


def test_anchor_pulls_the_box_toward_the_requested_corner():
    """On a blank frame every position is equally clear, so the anchor decides."""
    mask = ink_mask(blank())
    left = find_empty_box(mask, 100, 30, anchor=(0.1, 0.5))
    right = find_empty_box(mask, 100, 30, anchor=(0.9, 0.5))
    assert left[0] < right[0]


def test_anchor_does_not_override_ink():
    """The anchor breaks ties between clear spots; it never wins against artwork."""
    mask = ink_mask(with_blob([0, 0, W // 2, H]))
    box = find_empty_box(mask, 120, 40, anchor=(0.05, 0.5))
    assert box[0] >= W // 2


def test_anchored_captions_do_not_all_pile_into_one_band():
    img = blank()
    occupied = []
    place_caption(img, ["TOP"], FONT, (0, 0, 0), occupied=occupied, anchor=(0.5, 0.15))
    place_caption(img, ["BOTTOM"], FONT, (0, 0, 0), occupied=occupied, anchor=(0.5, 0.85))
    assert occupied[1][1] - occupied[0][1] > H // 3


def test_zoom_safe_margin_grows_with_zoom():
    from why_engine.art.annotate import zoom_safe_margin
    assert zoom_safe_margin(1920, 1080, 1.0) == 40
    assert zoom_safe_margin(1920, 1080, 1.10) > zoom_safe_margin(1920, 1080, 1.05) > 40


def test_zoom_safe_margin_keeps_text_inside_the_crop():
    """A caption at the returned inset must still be on screen at peak zoom."""
    from why_engine.art.annotate import zoom_safe_margin
    zoom = 1.10
    m = zoom_safe_margin(W, H, zoom, base=0)
    visible_left = (W - W / zoom) / 2
    assert m >= visible_left


def test_dilate_grows_the_mask_outward():
    plain = ink_mask(with_blob([100, 100, 199, 199]))
    grown = ink_mask(with_blob([100, 100, 199, 199]), dilate=10)
    assert grown.sum() > plain.sum()
    assert grown[95, 150] and not plain[95, 150]


def test_dilate_leaves_a_blank_frame_blank():
    assert not ink_mask(blank(), dilate=20).any()


def test_dilate_does_not_run_off_the_edge():
    img = with_blob([0, 0, 20, 20])
    grown = ink_mask(img, dilate=30)
    assert grown.shape == (H, W)
    assert grown[0, 0]


def test_caption_keeps_a_gap_from_existing_marks():
    """Zero ink coverage is not enough -- text flush against artwork reads as a collision."""
    img = with_blob([0, 0, W // 2, H])
    box = place_caption(img, ["CLEAR OF IT"], FONT, (0, 0, 0), gap=30)
    assert box is not None
    assert box[0] >= W // 2 + 30


def test_caption_falls_back_to_a_tighter_gap_rather_than_dropping():
    """A busy frame should still get its line, just with less breathing room."""
    img = blank()
    ImageDraw.Draw(img).rectangle([0, 0, W, H // 2 - 30], fill=(20, 20, 20))
    ImageDraw.Draw(img).rectangle([0, H // 2 + 30, W, H], fill=(20, 20, 20))
    # a 60px slot: too tight for the requested gap, wide enough at gap=0
    assert place_caption(img, ["X"], FONT, (200, 38, 42), gap=200, margin=0, pad=0) is not None


def test_caption_still_returns_none_when_the_tight_pass_also_fails():
    img = with_blob([0, 0, W, H])
    assert place_caption(img, ["NO ROOM"], FONT, (200, 38, 42), gap=60) is None
