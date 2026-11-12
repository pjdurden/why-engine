"""Colour matching. Targets are measured from the reference; see palette.py."""
import numpy as np
import pytest
from PIL import Image

from why_engine.art.palette import (
    PAPER, REF_SAT_MEAN, SATURATION_GAIN, match_reference, saturate, saturation,
)


def solid(rgb, size=(64, 64)):
    return Image.new("RGB", size, rgb)


def test_grey_has_no_saturation():
    assert saturation(solid((128, 128, 128)))[0] == pytest.approx(0.0)


def test_pure_hue_is_fully_saturated():
    assert saturation(solid((200, 0, 0)))[0] == pytest.approx(1.0, abs=0.01)


def test_saturation_ignores_near_black_pixels():
    """Outlines carry no colour; including them would drag every mean down."""
    img = Image.new("RGB", (64, 64), (5, 5, 5))
    img.paste(solid((200, 40, 40), (32, 64)), (0, 0))
    assert saturation(img)[0] == pytest.approx(0.8, abs=0.02)


def test_saturate_increases_saturation():
    before = saturation(solid((150, 120, 100)))[0]
    after = saturation(saturate(solid((150, 120, 100)), 1.5))[0]
    assert after > before


def test_saturate_leaves_grey_grey():
    """A fill with no colour must not acquire one."""
    out = np.asarray(saturate(solid((128, 128, 128)), 2.0))
    assert out.min() == out.max() == 128


def test_saturate_leaves_black_outlines_black():
    out = np.asarray(saturate(solid((0, 0, 0)), 2.0))
    assert out.max() == 0


def test_saturate_preserves_flat_fills_as_flat():
    """One colour in, one colour out -- the flat-fill look depends on this."""
    out = np.asarray(saturate(solid((160, 120, 90)), 1.6)).reshape(-1, 3)
    assert len(np.unique(out, axis=0)) == 1


def test_saturate_gain_of_one_is_identity():
    src = solid((160, 120, 90))
    assert np.array_equal(np.asarray(saturate(src, 1.0)), np.asarray(src))


def test_match_reference_hits_the_measured_target():
    img = solid((150, 138, 120))
    assert saturation(match_reference(img))[0] == pytest.approx(REF_SAT_MEAN, abs=0.03)


def test_match_reference_leaves_an_already_vivid_image_alone():
    vivid = solid((220, 20, 20))
    assert np.array_equal(np.asarray(match_reference(vivid)), np.asarray(vivid))


def test_match_reference_leaves_a_greyscale_image_alone():
    """There is no colour to amplify; bisection must not spin on it."""
    grey = solid((140, 140, 140))
    assert np.array_equal(np.asarray(match_reference(grey)), np.asarray(grey))


def test_paper_is_near_white_not_a_warm_tint():
    """Measured RGB (250.8, 250.7, 250.4) -- the cards were never the problem."""
    assert min(PAPER) >= 248
    assert max(PAPER) - min(PAPER) <= 2


def test_default_gain_matches_the_measured_shortfall():
    assert SATURATION_GAIN == pytest.approx(0.348 / 0.251, abs=0.05)
