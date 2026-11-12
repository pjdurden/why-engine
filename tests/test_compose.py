"""Composer. Every archetype must render something usable from a bare shot."""
import numpy as np
import pytest
from PIL import Image

from why_engine.art.compose import H, SAFE, W, band, compose, _numeric
from why_engine.art.direct import Shot

ink = lambda img: (np.asarray(img.convert("L")) < 200).mean()


def shot(archetype, **kw):
    kw.setdefault("sentence", "The fish had not changed.")
    kw.setdefault("headline", "THE FISH")
    return Shot(index=0, segment_id=1, archetype=archetype, **kw)


@pytest.mark.parametrize("archetype", [
    "text_card", "icon_multiply", "measure_compare", "versus_split",
    "painted_scene", "archival_plate", "then_now", "quantity_viz",
])
def test_every_archetype_renders_a_full_frame(archetype):
    img = compose(shot(archetype, count=6, captions=["8 KG", "1.2 KG"]))
    assert img.size == (W, H)
    assert img.mode == "RGB"


@pytest.mark.parametrize("archetype", ["text_card", "measure_compare", "icon_multiply"])
def test_frames_are_not_blank(archetype):
    assert ink(compose(shot(archetype, count=4, captions=["8 KG", "1.2 KG"]))) > 0.001


def test_unknown_archetype_falls_back_rather_than_raising():
    """A missing layout should cost one plain shot, not the whole video."""
    img = compose(shot("no_such_archetype"))
    assert img.size == (W, H) and ink(img) > 0.001


def test_missing_assets_do_not_raise():
    """Assets can be absent mid-run; the composer must still produce a frame."""
    img = compose(shot("icon_multiply", count=8), {"subject_icon": "/nope/missing.png"})
    assert img.size == (W, H)


def test_painted_scene_without_a_scene_falls_back_to_a_card():
    img = compose(shot("painted_scene", captions=["BY RAIL"]), {"scene": "/nope.png"})
    assert ink(img) > 0.001


def test_archival_plate_without_an_image_falls_back_to_a_card():
    img = compose(shot("archival_plate"), {"archival": "/nope.png"})
    assert ink(img) > 0.001


def test_a_long_headline_is_shrunk_to_fit():
    long = "THE SAME GOVERNMENT THAT GAVE THEM AWAY WAS NOW PAYING TO TAKE THEM BACK AGAIN"
    img = compose(shot("text_card", headline=long))
    col = np.asarray(img.convert("L")) < 200
    xs = np.where(col.any(axis=0))[0]
    assert xs.min() >= SAFE - 40 and xs.max() <= W - SAFE + 40


def test_a_large_icon_count_stays_inside_the_frame():
    """A fixed pitch runs a count of thirty off the right edge."""
    img = compose(shot("icon_multiply", count=30),
                  {"count_icon": "build/icons/pig.png", "subject_icon": "build/icons/carp.png"})
    col = np.asarray(img.convert("L")) < 200
    xs = np.where(col.any(axis=0))[0]
    assert xs.max() < W


def test_measure_compare_marks_the_smaller_value_in_red():
    img = compose(shot("measure_compare", captions=["8 KG", "1.2 KG"]))
    a = np.asarray(img)
    reddish = (a[:, :, 0] > 150) & (a[:, :, 1] < 90) & (a[:, :, 2] < 90)
    right = reddish[600:730, W // 2:].sum()
    left = reddish[600:730, : W // 2].sum()
    assert right > left  # 1.2 is on the right and is the smaller value


def test_numeric_reads_the_number_out_of_a_label():
    assert _numeric("1.2 KG") == pytest.approx(1.2)
    assert _numeric("4,000 LB") == pytest.approx(4000)
    assert _numeric("NO NUMBER") == float("inf")


def test_band_writes_over_a_scene():
    img = Image.new("RGB", (W, H), (60, 90, 70))
    before = ink(img)
    band(img, "BY RAIL, IN MILK CANS")
    assert ink(img) != before


def test_band_text_fits_the_frame():
    """Measured inside the band only: the scene behind it is dark everywhere,
    so a whole-frame threshold just re-detects the background."""
    img = Image.new("RGB", (W, H), (60, 90, 70))
    band(img, "A VERY LONG CAPTION THAT WOULD OTHERWISE RUN STRAIGHT OFF THE EDGE OF THE FRAME")
    a = np.asarray(img.convert("L"))
    rows = np.where((a > 200).any(axis=1))[0]        # the white band
    strip = a[rows.min():rows.max() + 1]
    xs = np.where((strip < 120).any(axis=0))[0]      # glyphs inside it
    assert xs.min() > 0 and xs.max() < W - 1


def test_compose_is_deterministic():
    a = np.asarray(compose(shot("text_card", headline="ONE EIGHT SEVEN SEVEN")))
    b = np.asarray(compose(shot("text_card", headline="ONE EIGHT SEVEN SEVEN")))
    assert np.array_equal(a, b)
