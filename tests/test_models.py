import pytest
from pydantic import ValidationError

from why_engine.models import Claim, Passage, Script, Segment


def _claim(text="Carp were stocked by the US Fish Commission in 1877."):
    return Claim(text=text, source_id="usfc-1877", quote="In 1877 the Commission distributed carp.", locator="p.14")


def test_script_all_claims_flattens_with_segment_ids():
    script = Script(
        topic="Why don't Americans eat carp?",
        segments=[
            Segment(segment_id=0, act=0, narration="Cold open.", claims=[], image_brief="a river"),
            Segment(segment_id=1, act=1, narration="History.", claims=[_claim()], image_brief="a fish"),
        ],
    )
    flat = script.all_claims()
    assert len(flat) == 1
    assert flat[0][0] == 1
    assert flat[0][1].source_id == "usfc-1877"


def test_segment_allows_empty_claims_for_non_factual_narration():
    seg = Segment(segment_id=2, act=1, narration="But here is the strange part.", claims=[], image_brief="a pause")
    assert seg.claims == []


def test_claim_rejects_empty_quote():
    with pytest.raises(ValidationError):
        Claim(text="Something true.", source_id="x", quote="", locator="p.1")


def test_passage_rejects_empty_text():
    with pytest.raises(ValidationError):
        Passage(source_id="x", locator="p.1", text="", url="https://example.gov/a", title="A")
