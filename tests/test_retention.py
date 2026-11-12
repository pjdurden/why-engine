from why_engine.config import Settings
from why_engine.models import Claim, Script, Segment
from why_engine.script.retention import check_retention, estimate_runtime_s


def _seg(segment_id, act, narration):
    return Segment(segment_id=segment_id, act=act, narration=narration, claims=[], image_brief="b")


def _good_script(body_words=2400):
    """Body is split across five acts so it satisfies the per-act ceiling.
    A single 2400-word act would be ~923s, far over the 300s limit."""
    per_act = max(1, body_words // 5)
    body = [_seg(i, i, " ".join(["word"] * per_act)) for i in range(1, 6)]
    return Script(topic="Why is American bread sweet?", segments=[
        _seg(0, 0, "American bread is legally close to cake."),
        *body,
        _seg(6, 6, "And that is why American bread is legally close to cake."),
    ])


def test_estimate_runtime_uses_the_configured_speaking_rate():
    script = Script(topic="t", segments=[_seg(0, 0, " ".join(["w"] * 260))])
    assert estimate_runtime_s(script, Settings(words_per_second=2.6)) == 100.0


def test_a_well_formed_script_has_no_violations():
    assert check_retention(_good_script(), Settings()) == []


def test_missing_cold_open_is_flagged():
    script = Script(topic="t", segments=[_seg(i, i, " ".join(["w"] * 480)) for i in range(1, 6)])
    assert any("cold open" in e for e in check_retention(script, Settings()))


def test_cold_open_longer_than_fifteen_seconds_is_flagged():
    long_open = " ".join(["word"] * 200)  # ~77s at 2.6 wps
    script = Script(topic="t", segments=[
        _seg(0, 0, long_open),
        *[_seg(i, i, " ".join(["w"] * 480)) for i in range(1, 6)],
        _seg(6, 6, "And that is why."),
    ])
    assert any("cold open" in e and "too long" in e for e in check_retention(script, Settings()))


def test_runtime_below_the_floor_is_flagged():
    assert any("too short" in e for e in check_retention(_good_script(body_words=200), Settings()))


def test_runtime_above_the_ceiling_is_flagged():
    """Asserts on the runtime message specifically: a 9000-word body also trips
    the per-act ceiling, and a bare "too long" match would pass for the wrong reason."""
    violations = check_retention(_good_script(body_words=9000), Settings())
    assert any("runtime too long" in e for e in violations)


def test_an_act_longer_than_the_ceiling_is_flagged():
    """Spec S10 budgets each act at 2-4 minutes. A 12-minute act is a lecture,
    not an act, and retention dies inside it."""
    huge_act = " ".join(["word"] * 2000)  # ~769s at 2.6 wps, over the 300s ceiling
    script = Script(topic="t", segments=[
        _seg(0, 0, "American bread is legally close to cake."),
        _seg(1, 1, huge_act),
        _seg(2, 2, "And that is why American bread is legally close to cake."),
    ])
    assert any("act 1" in e and "too long" in e for e in check_retention(script, Settings()))


def test_act_length_is_summed_across_segments_sharing_an_act():
    half = " ".join(["word"] * 500)  # ~192s each, 384s combined, over the ceiling
    script = Script(topic="t", segments=[
        _seg(0, 0, "American bread is legally close to cake."),
        _seg(1, 3, half),
        _seg(2, 3, half),
        _seg(3, 4, " ".join(["word"] * 1400)),
        _seg(4, 5, "And that is why American bread is legally close to cake."),
    ])
    assert any("act 3" in e and "too long" in e for e in check_retention(script, Settings()))


def test_banned_youtuber_filler_is_flagged():
    script = _good_script()
    script.segments[0].narration = "Hey guys, before we begin, make sure to subscribe."
    violations = check_retention(script, Settings())
    assert any("filler" in e for e in violations)


def test_final_segment_must_echo_the_cold_open():
    script = _good_script()
    script.segments[-1].narration = "Thanks for watching everyone."
    assert any("resolve" in e for e in check_retention(script, Settings()))


def test_cold_open_duration_sums_across_every_act_zero_segment():
    """A second act-0 segment is checked by neither the cold-open rule (which saw
    only segments[0]) nor the per-act ceiling (which skips act 0). Six minutes of
    narration used to pass here with zero violations."""
    script = Script(topic="t", segments=[
        _seg(0, 0, "American bread is legally close to cake."),
        _seg(1, 0, " ".join(["word"] * 900)),
        *[_seg(i, i, " ".join(["word"] * 400)) for i in range(2, 6)],
        _seg(6, 6, "And that is why American bread is legally close to cake."),
    ])
    violations = check_retention(script, Settings())
    assert any("cold open too long" in v for v in violations)


def test_a_cold_open_split_across_two_short_segments_is_still_accepted():
    """Splitting the cold open across two brief beats is legitimate; only the
    combined duration matters."""
    script = Script(topic="t", segments=[
        _seg(0, 0, "American bread is legally close to cake."),
        _seg(1, 0, "Nobody notices."),
        *[_seg(i, i, " ".join(["word"] * 480)) for i in range(2, 7)],
        _seg(7, 7, "And that is why American bread is legally close to cake."),
    ])
    assert check_retention(script, Settings()) == []


def test_narration_that_recites_its_quote_is_rejected():
    """The first real script read the CFR aloud: whole paragraphs of regulation
    text arrived as narration with a citation attached. It passed every check
    there was, because nothing measured the difference between saying a thing
    and reading the document that says it."""
    quote = ("Apply a force of 3,114W newtons horizontally in the forward direction "
             "through the loading bar at the pivot attachment point.")
    script = Script(topic="t", segments=[
        Segment(segment_id=0, act=0, image_brief="b", narration="A belief, and a crack."),
        Segment(segment_id=1, act=1, image_brief="b",
                narration="Apply a force of 3,114W newtons horizontally in the forward "
                          "direction through the loading bar at the pivot attachment point.",
                claims=[Claim(text="t", source_id="s", locator="c0", quote=quote)]),
    ])
    assert any("recites" in v for v in check_retention(script, Settings()))


def test_a_short_deliberate_quote_is_allowed():
    """Reading one striking line out loud is a filmmaking move, not recitation.
    The rule is against paragraphs, not against quoting."""
    quote = ("The purpose of this standard is to reduce the number of deaths and the "
             "severity of injuries that result from the impact of school bus occupants "
             "against structures within the vehicle.")
    script = Script(topic="t", segments=[
        Segment(segment_id=0, act=0, image_brief="b", narration="A belief, and a crack."),
        Segment(segment_id=1, act=1, image_brief="b",
                narration="Read that line again. Against structures within the vehicle. "
                          "The rule assumes the child is already moving.",
                claims=[Claim(text="t", source_id="s", locator="c0", quote=quote)]),
    ])
    assert not any("recites" in v for v in check_retention(script, Settings()))
