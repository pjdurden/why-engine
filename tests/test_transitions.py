import numpy as np
import pytest
from PIL import Image, ImageDraw

from why_engine.art.transitions import (
    DEFAULT_FRAMES, directional_blur, sharpness, whip_frames,
)

SIZE = (320, 180)


def textured(seed: int) -> Image.Image:
    """A frame with detail on both axes, so blur on either is measurable."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (SIZE[1], SIZE[0], 3), dtype=np.uint8)
    return Image.fromarray(a)


def striped(vertical: bool) -> Image.Image:
    img = Image.new("RGB", SIZE, (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i in range(0, 320 if vertical else 180, 8):
        if vertical:
            d.line([(i, 0), (i, 180)], fill=(0, 0, 0), width=3)
        else:
            d.line([(0, i), (320, i)], fill=(0, 0, 0), width=3)
    return img


def test_directional_blur_on_x_destroys_vertical_stripes():
    before = striped(vertical=True)
    after = directional_blur(before, 12, axis="x")
    assert sharpness(after) < 0.25 * sharpness(before)


def test_directional_blur_on_x_spares_horizontal_stripes():
    """The whole point of directional blur: the other axis survives."""
    before = striped(vertical=False)
    after = directional_blur(before, 12, axis="x")
    assert sharpness(after) > 0.9 * sharpness(before)


def test_directional_blur_radius_zero_is_identity():
    img = textured(1)
    assert np.array_equal(np.asarray(directional_blur(img, 0)), np.asarray(img))


def test_whip_emits_the_requested_frame_count():
    frames = whip_frames(textured(1), textured(2), frames=7)
    assert len(frames) == 7
    assert all(f.size == SIZE for f in frames)


def test_whip_sharpness_trough_matches_the_reference():
    """The reference's blur transitions measured 0.15-0.5x baseline sharpness."""
    a, b = textured(1), textured(2)
    baseline = min(sharpness(a), sharpness(b))
    trough = min(sharpness(f) for f in whip_frames(a, b))
    assert 0.10 <= trough / baseline <= 0.55


def test_whip_ends_are_less_blurred_than_its_middle():
    """Blur tracks slide velocity, so it peaks mid-transition."""
    frames = whip_frames(textured(1), textured(2), frames=5)
    mid = sharpness(frames[2])
    assert mid < sharpness(frames[0])
    assert mid < sharpness(frames[-1])


def test_whip_carries_content_from_a_toward_b():
    """Early frames hold mostly A; late frames mostly B."""
    a = Image.new("RGB", SIZE, (255, 0, 0))
    b = Image.new("RGB", SIZE, (0, 0, 255))
    frames = whip_frames(a, b, frames=5)
    redness = [np.asarray(f, dtype=float)[..., 0].mean() for f in frames]
    assert redness[0] > redness[-1]
    assert redness == sorted(redness, reverse=True)


def test_whip_on_y_axis_slides_vertically():
    a = Image.new("RGB", SIZE, (255, 0, 0))
    b = Image.new("RGB", SIZE, (0, 0, 255))
    mid = np.asarray(whip_frames(a, b, frames=3, axis="y")[1], dtype=float)
    top_red = mid[: SIZE[1] // 3, :, 0].mean()
    bottom_red = mid[-SIZE[1] // 3 :, :, 0].mean()
    assert top_red > bottom_red


def test_whip_reverse_slides_the_other_way():
    a = Image.new("RGB", SIZE, (255, 0, 0))
    b = Image.new("RGB", SIZE, (0, 0, 255))
    fwd = np.asarray(whip_frames(a, b, frames=3)[1], dtype=float)[..., 0]
    rev = np.asarray(whip_frames(a, b, frames=3, reverse=True)[1], dtype=float)[..., 0]
    assert fwd[:, : SIZE[0] // 3].mean() != pytest.approx(rev[:, : SIZE[0] // 3].mean(), abs=5)


def test_whip_is_deterministic():
    args = (textured(1), textured(2))
    first = [np.asarray(f) for f in whip_frames(*args)]
    second = [np.asarray(f) for f in whip_frames(*args)]
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_whip_rejects_mismatched_sizes():
    with pytest.raises(ValueError, match="matching sizes"):
        whip_frames(textured(1), Image.new("RGB", (100, 100)))


def test_default_frame_count_lands_in_the_measured_window():
    """0.10-0.20s at 30fps is 3-6 frames."""
    assert 3 <= DEFAULT_FRAMES <= 6


# --- the compound whip -------------------------------------------------------

from why_engine.art.transitions import (  # noqa: E402
    RAMP_IN_SECONDS, RAMP_OUT_SECONDS, SMEAR_SECONDS, WHIP_BUDGET,
    WHIP_WINDOW_SECONDS, whip_sequence, whip_travel,
)


def test_whip_sequence_lasts_the_measured_duration():
    """The reference's whips run 0.6-0.9s end to end."""
    n = len(whip_sequence(textured(1), textured(2), fps=30))
    assert 0.6 <= n / 30 <= 0.9


def test_whip_sequence_is_longer_than_its_smear():
    """The bare smear is only the middle third -- that was the earlier mistake."""
    seq = len(whip_sequence(textured(1), textured(2), fps=30))
    assert seq > 3 * int(round(SMEAR_SECONDS * 30))


def test_whip_sequence_trough_lands_in_the_measured_band():
    """0.15-0.50x baseline, measured on the reference's 6 genuine whips.

    Structured input, not noise: the reference is line art on paper, and pure
    noise is an unrepresentative worst case for a blur -- being entirely high
    frequency, it loses proportionally more sharpness than any real frame.
    """
    a, b = striped(vertical=True), striped(vertical=False)
    base = min(sharpness(a), sharpness(b))
    trough = min(sharpness(f) for f in whip_sequence(a, b, fps=30))
    assert 0.10 <= trough / base <= 0.55


def test_whip_sequence_ends_settled_at_unity():
    """The incoming shot arrives zoomed in and settles outward."""
    b = striped(vertical=True)
    last = whip_sequence(textured(1), b, fps=30)[-1]
    assert sharpness(last) == pytest.approx(sharpness(b), rel=0.12)


def test_whip_sequence_incoming_arrives_zoomed_in():
    """Right after the smear the new shot is larger than it ends up."""
    b = striped(vertical=True)
    seq = whip_sequence(textured(1), b, fps=30)
    after_smear = int(round((RAMP_IN_SECONDS + SMEAR_SECONDS) * 30))
    arriving = np.asarray(seq[after_smear].convert("L"), dtype=float)
    settled = np.asarray(seq[-1].convert("L"), dtype=float)
    # A zoomed-in crop of the same stripes has fewer stripe edges per row.
    edges = lambda a: (np.abs(np.diff(a, axis=1)) > 40).sum()
    assert edges(arriving) < edges(settled)


def test_whip_travel_is_the_area_under_the_velocity_ramp():
    assert whip_travel(0.40, 0.15) == pytest.approx(0.030)
    assert whip_travel(0.50, 0.30) == pytest.approx(0.075)


def test_ramp_out_is_longer_than_ramp_in():
    """The device lunges fast and settles slow."""
    assert RAMP_OUT_SECONDS > RAMP_IN_SECONDS


def test_whip_budget_matches_the_measurement():
    """All 7 whips fall before t=164s; the following 233 cuts are plain."""
    assert WHIP_BUDGET == 7
    assert WHIP_WINDOW_SECONDS == pytest.approx(164.0)
