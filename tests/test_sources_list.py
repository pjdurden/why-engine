"""The source list published under the video.

Spec section 13 rests the compliance argument on three things, and a published
source list is one of them. Nothing built it.
"""
from why_engine.models import Claim, Passage, Script, Segment
from why_engine.sources.corpus import Corpus
from why_engine.publish import source_list


def _script(*claims):
    return Script(topic="t", segments=[
        Segment(segment_id=0, act=0, narration="n", claims=list(claims), image_brief="b")])


def _corpus():
    return Corpus(passages=[
        Passage(source_id="cfr-571.222", locator="c0", text="x", title="FMVSS 222",
                url="https://www.ecfr.gov/current/title-49/section-571.222"),
        Passage(source_id="cfr-571.220", locator="c3", text="y", title="FMVSS 220",
                url="https://www.ecfr.gov/current/title-49/section-571.220"),
    ])


def _claim(source_id, locator="c0"):
    return Claim(text="t", source_id=source_id, locator=locator, quote="q")


def test_each_cited_source_appears_once_with_its_url():
    out = source_list(_script(_claim("cfr-571.222")), _corpus())
    assert out.count("https://www.ecfr.gov/current/title-49/section-571.222") == 1
    assert "FMVSS 222" in out


def test_a_source_cited_many_times_is_listed_once():
    s = _script(_claim("cfr-571.222"), _claim("cfr-571.222"), _claim("cfr-571.222"))
    assert source_list(s, _corpus()).count("section-571.222") == 1


def test_a_source_never_cited_is_not_listed():
    """The list has to describe this video, not the corpus it was offered."""
    assert "571.220" not in source_list(_script(_claim("cfr-571.222")), _corpus())


def test_sources_are_listed_in_a_stable_order():
    """Two runs of the same script must produce the same description."""
    s = _script(_claim("cfr-571.220", "c3"), _claim("cfr-571.222"))
    assert source_list(s, _corpus()) == source_list(s, _corpus())
    assert source_list(s, _corpus()).index("571.220") < source_list(s, _corpus()).index("571.222")
