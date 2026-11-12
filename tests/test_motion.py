"""Motion rules measured from the reference. Numbers cite the analysis doc."""
import numpy as np
import pytest
from PIL import Image

from why_engine.art.motion import (
    PRESETS, STATIC_BELOW_SECONDS, Zoomer, ZoomPreset, shot_moves, zoom_curve,
)

FPS = 30


def flat(curve):
    return max(curve) - min(curve) < 1e-9


def test_short_shots_never_move():
    """0 of 134 body shots under 3.5s carried a move."""
    assert not shot_moves(2.4)
    assert not shot_moves(STATIC_BELOW_SECONDS - 0.01)
    assert shot_moves(STATIC_BELOW_SECONDS)
    assert shot_moves(6.0)


def test_a_short_shot_gets_a_flat_curve_even_when_a_preset_is_asked_for():
    """The duration gate outranks the caller -- this is the rule that was violated."""
    assert flat(zoom_curve(2.4, FPS, "gentle_push"))


def test_no_preset_gives_a_flat_curve():
    assert flat(zoom_curve(8.0, FPS, None))


def test_curve_has_one_entry_per_frame():
    assert len(zoom_curve(6.0, FPS, "gentle_push")) == 180
    assert len(zoom_curve(2.0, FPS, None)) == 60


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_presets_partition_the_shot(name):
    p = PRESETS[name]
    assert p.span > 0
    assert p.head + p.span + p.tail == pytest.approx(1.0)


def test_gentle_push_travels_the_measured_total():
    """+1.19% +/- 0.30, the most common family."""
    c = zoom_curve(6.0, FPS, "gentle_push")
    assert max(c) - min(c) == pytest.approx(0.0119, abs=1e-4)


def test_strong_push_travels_further_than_gentle():
    strong = zoom_curve(8.0, FPS, "strong_push")
    gentle = zoom_curve(8.0, FPS, "gentle_push")
    assert (max(strong) - min(strong)) > 4 * (max(gentle) - min(gentle))


def test_push_starts_wide_and_ends_tight():
    c = zoom_curve(6.0, FPS, "gentle_push")
    assert c[0] == pytest.approx(1.0)
    assert c[-1] > c[0]


def test_pull_out_starts_tight_and_ends_at_unity():
    """Rendering only crops inward, so a pull-out must start zoomed in."""
    c = zoom_curve(6.0, FPS, "pull_out")
    assert c[0] == pytest.approx(1.0366, abs=1e-4)
    assert c[-1] == pytest.approx(1.0)


def test_push_holds_at_the_head():
    """Static-then-move: a gentle push sits still for the first ~19%."""
    c = zoom_curve(6.0, FPS, "gentle_push")
    head = int(0.19 * len(c))
    assert flat(c[: head - 1])
    assert c[head + 2] > c[0]


def test_pull_out_holds_at_the_tail_not_the_head():
    """Pull-outs start ~at the cut and settle early -- the mirror of a push.

    Measured head hold is 2% against a 10-20% tail, so the asymmetry is the
    finding, not a zero head.
    """
    p = PRESETS["pull_out"]
    assert p.tail > 5 * p.head
    c = zoom_curve(6.0, FPS, "pull_out")
    assert c[int(0.05 * len(c))] < c[0]
    assert flat(c[-int(0.18 * len(c)):])


def test_the_move_is_linear_not_eased():
    """Measured as a flat rate plateau with an abrupt start and stop."""
    c = zoom_curve(6.0, FPS, "gentle_push")
    p = PRESETS["gentle_push"]
    lo = int(p.head * len(c)) + 2
    hi = len(c) - int(p.tail * len(c)) - 2
    steps = [c[i + 1] - c[i] for i in range(lo, hi - 1)]
    assert max(steps) - min(steps) < 0.02 * max(steps)


def test_rate_falls_as_the_shot_lengthens():
    """Total travel is fixed; the rate is whatever makes it fit."""
    short = zoom_curve(4.0, FPS, "gentle_push")
    long = zoom_curve(12.0, FPS, "gentle_push")
    assert max(short) - min(short) == pytest.approx(max(long) - min(long), abs=1e-4)
    rate = lambda c, d: (max(c) - min(c)) / d
    assert rate(short, 4.0) > 2 * rate(long, 12.0)


def test_curve_never_scales_below_unity():
    for name in PRESETS:
        assert min(zoom_curve(6.0, FPS, name)) >= 1.0


def test_a_very_short_moving_shot_still_produces_a_usable_curve():
    c = zoom_curve(STATIC_BELOW_SECONDS, FPS, "strong_push")
    assert len(c) == int(round(STATIC_BELOW_SECONDS * FPS))
    assert max(c) - min(c) == pytest.approx(0.0526, abs=1e-3)


def test_custom_preset_is_accepted():
    c = zoom_curve(6.0, FPS, ZoomPreset(total=+0.02, head=0.1, tail=0.1))
    assert max(c) - min(c) == pytest.approx(0.02, abs=1e-4)


def test_zoomer_at_unity_fills_the_requested_size():
    src = Image.new("RGB", (400, 300), (200, 30, 30))
    assert Zoomer(src, (320, 180)).frame(1.0).size == (320, 180)


def test_zoomer_reuses_the_frame_object_across_a_hold():
    """The reference is 76-82% bit-identical frames; holds must be exact."""
    src = Image.new("RGB", (400, 300), (10, 10, 10))
    frames = list(Zoomer(src, (320, 180)).shot(zoom_curve(6.0, FPS, "gentle_push")))
    head = int(0.19 * len(frames)) - 2
    assert all(f is frames[0] for f in frames[:head])


def test_zoomer_static_shot_is_one_object_repeated():
    src = Image.new("RGB", (400, 300), (10, 10, 10))
    frames = list(Zoomer(src, (320, 180)).shot(zoom_curve(2.0, FPS, None)))
    assert len(frames) == 60
    assert all(f is frames[0] for f in frames)


# --- local element animation -------------------------------------------------

from why_engine.art.motion import POP_OVERSHOOT, pop_in, pop_scale  # noqa: E402


def test_pop_scale_starts_at_nothing_and_ends_at_exactly_one():
    assert pop_scale(0.0) == 0.0
    assert pop_scale(1.0) == 1.0
    assert pop_scale(1.5) == 1.0


def test_pop_scale_overshoots_before_settling():
    """The bounce is what makes it read as drawn rather than faded in."""
    assert pop_scale(0.75) == pytest.approx(POP_OVERSHOOT)
    assert pop_scale(0.75) > 1.0
    assert pop_scale(0.95) < pop_scale(0.75)


def test_pop_scale_rises_monotonically_to_the_overshoot():
    xs = [pop_scale(i / 30) for i in range(23)]
    assert xs == sorted(xs)


def test_pop_in_returns_the_requested_frame_count():
    base = Image.new("RGB", (64, 48), (250, 250, 250))
    el = Image.new("RGBA", (16, 16), (200, 30, 30, 255))
    assert len(pop_in(base, el, (32, 24), frames=30)) == 30


def test_pop_in_adds_ink_once_the_element_lands():
    base = Image.new("RGB", (64, 48), (250, 250, 250))
    el = Image.new("RGBA", (16, 16), (200, 30, 30, 255))
    frames = pop_in(base, el, (32, 24), frames=30, delay=5)
    dark = lambda f: (np.asarray(f.convert("L")) < 200).sum()
    assert dark(frames[0]) == 0
    assert dark(frames[-1]) > 0


def test_pop_in_holds_before_the_delay_and_after_the_pop():
    """The still stretches must be the same object, not equal copies."""
    base = Image.new("RGB", (64, 48), (250, 250, 250))
    el = Image.new("RGBA", (16, 16), (200, 30, 30, 255))
    frames = pop_in(base, el, (32, 24), frames=40, delay=8, pop_frames=6)
    assert all(f is frames[0] for f in frames[:8])
    assert all(f is frames[-1] for f in frames[-20:])


def test_pop_in_with_zero_delay_starts_animating_immediately():
    base = Image.new("RGB", (64, 48), (250, 250, 250))
    el = Image.new("RGBA", (16, 16), (200, 30, 30, 255))
    frames = pop_in(base, el, (32, 24), frames=20, delay=0, pop_frames=6)
    assert frames[3] is not frames[0]


def test_pop_in_element_is_largest_at_the_overshoot_not_at_the_end():
    base = Image.new("RGB", (80, 80), (250, 250, 250))
    el = Image.new("RGBA", (20, 20), (0, 0, 0, 255))
    frames = pop_in(base, el, (40, 40), frames=20, delay=0, pop_frames=8)
    dark = lambda f: (np.asarray(f.convert("L")) < 100).sum()
    assert dark(frames[6]) > dark(frames[-1])


def test_a_shot_yields_frames_one_at_a_time():
    """shot() built the whole shot as a list. Most shots are static and cost one
    image, but a shot carrying a move is distinct every frame: ten seconds of
    push at 1920x1080 is 300 frames and 1.9GB in one list, which is what killed
    the school bus render twice."""
    import inspect
    src = Image.new("RGB", (400, 225), "white")
    assert inspect.isgenerator(Zoomer(src, (320, 180)).shot(zoom_curve(6.0, FPS, "gentle_push")))
