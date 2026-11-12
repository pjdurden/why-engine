"""Assembler. The invariant under test is that picture and voice stay together."""
import pytest

from why_engine.art.direct import Shot
from why_engine.art.motion import STATIC_BELOW_SECONDS
from why_engine.art.transitions import WHIP_WINDOW_SECONDS
from why_engine.assemble import FPS, MOVE_EVERY, plan_motion, plan_whips

mk = lambda a="text_card": Shot(index=0, segment_id=1, sentence="s", archetype=a)


def test_short_shots_are_never_given_a_move():
    """The render would drop it anyway; a plan that disagrees is a lie."""
    durations = [1.0, 2.0, 3.0]
    assert plan_motion([mk() for _ in durations], durations) == [None, None, None]


def test_long_shots_can_carry_a_move():
    durations = [6.0] * 6
    assert any(plan_motion([mk() for _ in durations], durations))


def test_moves_are_rationed_not_applied_to_every_eligible_shot():
    durations = [6.0] * 8
    plan = plan_motion([mk() for _ in durations], durations)
    assert 0 < sum(p is not None for p in plan) < len(plan)


def test_scenes_push_in_and_cards_pull_out():
    durations = [6.0] * 4
    shots = [mk("painted_scene"), mk("painted_scene"), mk("text_card"), mk("text_card")]
    chosen = [p for p in plan_motion(shots, durations) if p]
    assert chosen and all(c in {"gentle_push", "pull_out"} for c in chosen)


def test_move_rationing_counts_only_eligible_shots():
    """Short shots must not consume a slot in the rotation."""
    durations = [1.0, 1.0, 6.0, 6.0]
    plan = plan_motion([mk() for _ in durations], durations)
    assert plan[0] is None and plan[1] is None
    assert sum(p is not None for p in plan[2:]) == 2 // MOVE_EVERY


def test_a_shot_exactly_at_the_threshold_is_eligible():
    durations = [STATIC_BELOW_SECONDS] * (MOVE_EVERY * 2)
    assert any(plan_motion([mk() for _ in durations], durations))


def test_whips_stay_inside_the_opening_window():
    """All of the reference's whips fall before t=164s."""
    durations = [30.0] * 12
    starts, t = [], 0.0
    for d in durations:
        starts.append(t)
        t += d
    for i in plan_whips(durations, budget=3):
        assert starts[i] < WHIP_WINDOW_SECONDS


def test_no_whip_lands_on_the_first_shot():
    """There is nothing to whip from."""
    assert 0 not in plan_whips([3.0] * 8, budget=4)


def test_whip_budget_is_respected():
    assert len(plan_whips([2.0] * 40, budget=3)) <= 3


def test_whip_budget_of_zero_produces_none():
    assert plan_whips([2.0] * 10, budget=0) == set()


def test_a_script_shorter_than_one_shot_gets_no_whips():
    assert plan_whips([4.0]) == set()


def test_whips_do_not_all_cluster_on_consecutive_shots():
    picked = sorted(plan_whips([2.0] * 30, budget=3))
    assert len(picked) < 2 or max(picked) - min(picked) > 1


def test_cumulative_edges_do_not_accumulate_rounding_error():
    """Per-shot rounding drifts half a frame at a time; cumulative does not."""
    durations = [1.017] * 60
    edges, t = [0], 0.0
    for d in durations:
        t += d
        edges.append(int(round(t * FPS)))
    assert edges[-1] == int(round(sum(durations) * FPS))
    naive = sum(int(round(d * FPS)) for d in durations)
    assert abs(naive - edges[-1]) > 0  # the bug this guards against is real


def test_frames_are_produced_lazily_not_accumulated():
    """The assembler built a list of every frame in the video before encoding.
    At 1920x1080 and 30fps a twenty-minute script is 36,540 frames, 6.2MB each
    raw; holds share an object but any shot with motion does not. The render of
    the school bus video was killed for memory at exactly this point."""
    import inspect
    from why_engine.assemble import _frames

    def _boom(shot):
        raise AssertionError("composed a shot before anyone asked for a frame")

    gen = _frames([mk()], _boom, edges=[0, 30], motion=[None], whips={})
    assert inspect.isgenerator(gen)          # nothing composed yet


def test_the_frame_stream_matches_the_timing_map():
    """Picture and voice stay together only if every shot yields exactly the
    frames its measured duration bought."""
    from why_engine.assemble import _frames

    shots = [mk(), mk("painted_scene")]
    edges = [0, 45, 120]
    gen = _frames(shots, lambda s: {}, edges=edges, motion=[None, None], whips={})
    assert sum(1 for _ in gen) == edges[-1]


def test_a_stream_that_undershoots_the_timing_map_is_caught():
    """Silently encoding fewer frames than the audio needs is the drift the
    whole timing map exists to prevent, so it has to raise."""
    from why_engine.assemble import count_frames

    with pytest.raises(ValueError):
        count_frames(iter([object()] * 3), expected=10, sink=lambda f: None)
