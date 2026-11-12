"""Art director. Rules are ordered most-specific first; these pin that order."""
import pytest

from why_engine.art.direct import MAX_ICON_COUNT, Shot, direct, direct_sentence
from why_engine.models import Claim, Script, Segment

SEG = Segment(segment_id=3, act=1, narration="x", claims=[], image_brief="a river bank")


def arch(sentence, segment=SEG, **kw):
    return direct_sentence(sentence, segment, 0, **kw).archetype


def test_two_numbers_sharing_a_unit_become_a_comparison():
    s = direct_sentence("A cow needs eight kilograms and a carp needs one point two kilograms.", SEG, 0)
    assert s.archetype == "measure_compare"
    assert s.captions == ["8 KG", "1.2 KG"]


def test_numbers_with_different_units_are_not_compared():
    """Eight kilograms against twenty five states is not a comparison."""
    assert arch("It took eight kilograms and twenty five states.") != "measure_compare"


def test_equal_values_are_not_compared():
    assert arch("Eight kilograms here and eight kilograms there.") != "measure_compare"


def test_a_small_count_becomes_an_icon_grid():
    s = direct_sentence("More protein than twelve pigs.", SEG, 0)
    assert s.archetype == "icon_multiply" and s.count == 12


def test_a_count_too_large_to_read_does_not_become_an_icon_grid():
    assert arch(f"There were {MAX_ICON_COUNT + 40} of them in the river.") != "icon_multiply"


def test_a_year_is_not_mistaken_for_a_count():
    s = direct_sentence("In eighteen seventy seven they shipped them.", SEG, 0)
    assert s.archetype == "text_card" and s.headline == "1877"


def test_comparison_wins_over_a_year_in_the_same_sentence():
    """Most-specific first: a sentence can carry both, and the numbers matter more."""
    s = direct_sentence(
        "By eighteen eighty it was eight kilograms against one point two kilograms.", SEG, 0)
    assert s.archetype == "measure_compare"


def test_contrast_words_produce_a_versus_split():
    assert arch("They chose pork instead of carp.") == "versus_split"


def test_a_place_word_in_a_long_sentence_produces_a_scene():
    assert arch("They went out by rail to every river and town that asked for them.") \
        == "painted_scene"


def test_a_place_word_in_a_short_sentence_stays_a_card():
    """A three-word beat is a card even if it names a place."""
    assert arch("Into the river.") == "text_card"


def test_a_plain_sentence_falls_through_to_a_card():
    assert arch("The fish had not changed.") == "text_card"


def test_a_short_sentence_gets_no_subhead():
    assert direct_sentence("America does not.", SEG, 0).subhead == ""


def test_a_long_sentence_gets_a_subhead():
    s = direct_sentence(
        "The same government that gave them away was now paying to take them back.", SEG, 0)
    assert s.subhead


def test_headline_is_trimmed_to_fit_a_card():
    s = direct_sentence(
        "The same government that gave them away was now paying to take them back.", SEG, 0)
    assert len(s.headline.split()) <= 6
    assert not s.headline.endswith(".")


def test_a_segment_claim_is_carried_onto_the_shot():
    seg = Segment(segment_id=1, act=1, narration="x", image_brief="b",
                  claims=[Claim(text="t", source_id="usfc", quote="q", locator="p1")])
    s = direct_sentence("The fish had not changed.", seg, 0)
    assert s.claim_source == "usfc" and s.claim_quote == "q"


def test_scene_pool_is_used_when_supplied():
    s = direct_sentence("They went out by rail to every river and town that asked.",
                        SEG, 1, scene_pool=["a", "b"])
    assert s.scene in {"a", "b"}


def test_direct_uses_the_injected_splitter():
    """Director and narrator must split identically or picture and voice diverge."""
    script = Script(topic="t", segments=[
        Segment(segment_id=1, act=1, narration="One. Two.", image_brief="b"),
        Segment(segment_id=2, act=2, narration="Three.", image_brief="b"),
    ])
    shots = direct(script, lambda t: [s.strip() for s in t.split(".") if s.strip()])
    assert [s.sentence for s in shots] == ["One", "Two", "Three"]


def test_direct_numbers_shots_consecutively_across_segments():
    script = Script(topic="t", segments=[
        Segment(segment_id=1, act=1, narration="One. Two.", image_brief="b"),
        Segment(segment_id=2, act=2, narration="Three.", image_brief="b"),
    ])
    shots = direct(script, lambda t: [s.strip() for s in t.split(".") if s.strip()])
    assert [s.index for s in shots] == [0, 1, 2]
    assert [s.segment_id for s in shots] == [1, 1, 2]


def test_direct_is_deterministic():
    script = Script(topic="t", segments=[
        Segment(segment_id=1, act=1, narration="Twelve pigs. In eighteen eighty.", image_brief="b")])
    split = lambda t: [s.strip() + "." for s in t.split(".") if s.strip()]
    a = [(s.archetype, s.headline) for s in direct(script, split)]
    b = [(s.archetype, s.headline) for s in direct(script, split)]
    assert a == b


@pytest.mark.parametrize("sentence,ending", [
    ("The commission shipped them to the", "them"),
    ("It was paid for by the", "paid"),
    ("They were sent out to", "out"),
])
def test_phrase_does_not_end_on_a_dangling_word(sentence, ending):
    """'BY RAIL TO' leaves the reader waiting for a word that never comes."""
    assert direct_sentence(sentence, SEG, 0).headline.split()[-1].lower() == ending


def test_scene_caption_does_not_end_on_a_dangling_word():
    s = direct_sentence("They went out by rail to every river and town.", SEG, 0)
    assert s.archetype == "painted_scene"
    assert s.captions[0].split()[-1].lower() not in {"to", "by", "the", "and"}


def test_phrase_survives_a_sentence_that_is_all_function_words():
    assert direct_sentence("Of the.", SEG, 0).headline
