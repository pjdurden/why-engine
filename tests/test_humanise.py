import numpy as np
from PIL import Image

from why_engine.art.humanise import BLUR_FRAC, PRESSURE, WOBBLE, WORK, humanise, is_monochrome


def _drawing(size=256):
    """A white canvas with a black cross: survives thresholding, shows warping."""
    a = np.full((size, size), 255, dtype=np.uint8)
    a[size // 2 - 1 : size // 2 + 1, 20 : size - 20] = 0
    a[20 : size - 20, size // 2 - 1 : size // 2 + 1] = 0
    return Image.fromarray(a, "L")


def _coloured(size=256):
    img = _drawing(size).convert("RGB")
    px = img.load()
    for x in range(40, 60):
        for y in range(40, 60):
            px[x, y] = (255, 190, 200)  # the blush FLUX leaks
    return img


def test_output_is_strictly_black_and_white():
    out = humanise(_drawing())
    assert set(np.unique(np.array(out))) <= {0, 255}


def test_colour_is_removed_not_merely_reduced():
    """Prompting FLUX not to add colour was measured and failed; this is the guarantee."""
    assert is_monochrome(humanise(_coloured())) is True


def test_is_monochrome_rejects_a_colour_image():
    assert is_monochrome(_coloured()) is False


def test_output_keeps_the_input_size():
    assert humanise(_drawing(256)).size == (256, 256)


def test_ink_survives_the_pass():
    """A warp that erases the drawing is worse than no warp."""
    before = (np.array(_drawing()) < 128).sum()
    after = (np.array(humanise(_drawing())) < 128).sum()
    assert after > before * 0.5


def test_the_same_seed_gives_a_byte_identical_result():
    """Consistent imperfection reads as one hand; varying imperfection reads worse
    than clean, so the pass must be reproducible."""
    a = np.array(humanise(_drawing(), seed=4))
    b = np.array(humanise(_drawing(), seed=4))
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    a = np.array(humanise(_drawing(), seed=1))
    b = np.array(humanise(_drawing(), seed=2))
    assert not np.array_equal(a, b)


def test_the_style_contract_matches_the_measured_setting():
    """These came off a contact sheet. Changing them changes every asset."""
    assert (WOBBLE, PRESSURE, BLUR_FRAC, WORK) == (5.0, 0.35, 0.045, 1536)


def test_an_rgb_input_is_accepted():
    assert humanise(_coloured()).mode == "L"


def _cross(w, h):
    """A horizontal and a vertical line. Each line's deviation from straight
    measures the wobble on one axis."""
    a = np.full((h, w), 255, dtype=np.uint8)
    a[h // 2 - 1 : h // 2 + 1, :] = 0
    a[:, w // 2 - 1 : w // 2 + 1] = 0
    return Image.fromarray(a, "L")


def _line_deviation(ink, along_axis, near, exclude, window):
    """Std-dev of a line's centre position, sampled over a fixed-length window
    so both axes get the same number of samples."""
    positions = []
    span = ink.shape[1] if along_axis == 0 else ink.shape[0]
    lo = span // 2 - window // 2
    for i in range(lo, lo + window, 7):
        if abs(i - exclude) < 40:
            continue
        strip = ink[:, i] if along_axis == 0 else ink[i, :]
        hits = np.where(strip)[0]
        hits = hits[np.abs(hits - near) < 30]
        if len(hits):
            positions.append(hits.mean())
    return float(np.std(positions))


def test_wobble_is_isotropic_on_a_non_square_input():
    """Warping through a stretched square gives a portrait ~44% less tremor on
    its short axis (measured: x=2.81px vs y=5.00px, ratio 1.78). Square fixtures
    cannot catch it because the axes coincide.

    Averaged over seeds: the blur correlation length is ~46 output px, so a
    576px axis carries only ~12 independent wiggles and a single-seed ratio
    swings between 1.0 and 2.6 even when the wobble is perfectly isotropic.
    """
    w, h = 576, 1024
    xs, ys = [], []
    for seed in range(10):
        ink = np.array(humanise(_cross(w, h), seed=seed)) < 128
        ys.append(_line_deviation(ink, 0, near=h // 2, exclude=w // 2, window=w))
        xs.append(_line_deviation(ink, 1, near=w // 2, exclude=h // 2, window=w))
    x_dev, y_dev = float(np.mean(xs)), float(np.mean(ys))
    assert x_dev > 0 and y_dev > 0
    ratio = max(x_dev, y_dev) / min(x_dev, y_dev)
    assert ratio < 1.25, f"anisotropic wobble: x={x_dev:.2f} y={y_dev:.2f} ratio={ratio:.3f}"


def test_a_non_square_input_keeps_its_size():
    assert humanise(_cross(576, 1024)).size == (576, 1024)


def test_a_transparent_rgba_input_does_not_become_solid_black():
    """convert('L') discards alpha, turning transparency into black."""
    # Black-transparent, not white-transparent: PIL's RGBA->L ignores alpha and
    # reads the RGB channels, so a white-transparent fixture yields ink 0.00 on
    # the UNFIXED code too and guards nothing. Only dark RGB reproduces the bug.
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    out = humanise(img, seed=2)
    ink = (np.array(out.convert("L")) < 200).mean()
    assert ink < 0.5, f"transparent input rendered {ink:.0%} ink"
