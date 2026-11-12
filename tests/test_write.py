from why_engine.models import Claim, Passage, Script, Segment
from why_engine.llm import StubLLM
from why_engine.script.write import WRITE_SYSTEM, validate_provenance, write_script
from why_engine.sources.corpus import Corpus


def _corpus():
    return Corpus(passages=[
        Passage(source_id="gao-24", locator="c0", text="In 1877 the Commission distributed carp.",
                url="https://gao.gov/x", title="Carp Report"),
    ])


def _script_with(claim):
    return Script(topic="t", segments=[
        Segment(segment_id=0, act=1, narration="n", claims=[claim], image_brief="b")])


def test_write_script_returns_the_parsed_script():
    expected = _script_with(Claim(text="Carp were distributed in 1877.", source_id="gao-24",
                                  locator="c0", quote="In 1877 the Commission distributed carp."))
    stub = StubLLM([expected])
    assert write_script("t", _corpus(), stub).topic == "t"


def test_corpus_is_sent_as_the_cached_system_prefix():
    stub = StubLLM([_script_with(Claim(text="a", source_id="gao-24", locator="c0", quote="q"))])
    write_script("t", _corpus(), stub)
    assert stub.calls[0]["cache_system"] is True
    assert "In 1877 the Commission distributed carp." in stub.calls[0]["system"]


def test_topic_goes_in_the_user_turn_not_the_cached_prefix():
    """Volatile content after the cache breakpoint, or the cache never hits."""
    stub = StubLLM([_script_with(Claim(text="a", source_id="gao-24", locator="c0", quote="q"))])
    write_script("Why do we eat tilapia?", _corpus(), stub)
    assert "Why do we eat tilapia?" in stub.calls[0]["user"]
    assert "Why do we eat tilapia?" not in stub.calls[0]["system"]


def test_write_script_with_no_feedback_adds_nothing_extra_to_the_user_turn():
    stub = StubLLM([_script_with(Claim(text="a", source_id="gao-24", locator="c0", quote="q"))])
    write_script("t", _corpus(), stub)
    assert stub.calls[0]["user"] == "Write the script for this question:\n\nt"


def test_write_script_with_feedback_puts_reasons_in_the_user_turn_only():
    """Feedback must never touch the system prefix, or every retry invalidates
    the cached corpus prefix."""
    stub = StubLLM([_script_with(Claim(text="a", source_id="gao-24", locator="c0", quote="q"))])
    write_script("t", _corpus(), stub, feedback=["runtime too short: 100s below 600s"])
    assert "runtime too short: 100s below 600s" in stub.calls[0]["user"]
    assert "runtime too short: 100s below 600s" not in stub.calls[0]["system"]


def test_write_system_forbids_outside_knowledge():
    assert "only" in WRITE_SYSTEM.lower()
    assert "passage" in WRITE_SYSTEM.lower()


def test_validate_provenance_passes_for_a_real_passage_and_verbatim_quote():
    claim = Claim(text="Carp were distributed in 1877.", source_id="gao-24", locator="c0",
                  quote="In 1877 the Commission distributed carp.")
    assert validate_provenance(_script_with(claim), _corpus()) == []


def test_validate_provenance_flags_an_invented_source_id():
    claim = Claim(text="x", source_id="does-not-exist", locator="c0", quote="In 1877 the Commission distributed carp.")
    errors = validate_provenance(_script_with(claim), _corpus())
    assert len(errors) == 1
    assert "does-not-exist" in errors[0]


def test_validate_provenance_flags_a_quote_not_present_in_the_passage():
    """The model paraphrased instead of copying. The quote must be verbatim."""
    claim = Claim(text="x", source_id="gao-24", locator="c0",
                  quote="The Commission handed out carp in 1877.")
    errors = validate_provenance(_script_with(claim), _corpus())
    assert len(errors) == 1
    assert "not verbatim" in errors[0]


def test_validate_provenance_tolerates_whitespace_differences_in_the_quote():
    claim = Claim(text="x", source_id="gao-24", locator="c0",
                  quote="In 1877   the Commission\ndistributed carp.")
    assert validate_provenance(_script_with(claim), _corpus()) == []


def test_validate_provenance_rejects_a_quote_too_short_to_establish_anything():
    """A one-word quote is a substring of almost any passage and proves nothing."""
    claim = Claim(text="x", source_id="gao-24", locator="c0", quote="carp")
    errors = validate_provenance(_script_with(claim), _corpus())
    assert len(errors) == 1
    assert "too short" in errors[0]


def test_validate_provenance_accepts_a_quote_at_the_minimum_length():
    """Exactly MIN_QUOTE_WORDS must pass: the floor excludes degenerates, not
    legitimate short citations."""
    claim = Claim(text="x", source_id="gao-24", locator="c0",
                  quote="the Commission distributed carp.")
    assert validate_provenance(_script_with(claim), _corpus()) == []


def test_validate_provenance_reports_every_bad_claim_in_a_multi_claim_script():
    """Task 11 consumes this with real multi-segment scripts; errors must
    accumulate across claims rather than stopping at the first."""
    good = Claim(text="a", source_id="gao-24", locator="c0",
                 quote="In 1877 the Commission distributed carp.")
    bad_source = Claim(text="b", source_id="nope", locator="c0",
                       quote="In 1877 the Commission distributed carp.")
    bad_quote = Claim(text="c", source_id="gao-24", locator="c0",
                      quote="The Commission handed out carp in 1877.")
    script = Script(topic="t", segments=[
        Segment(segment_id=0, act=1, narration="n", claims=[good, bad_source], image_brief="b"),
        Segment(segment_id=1, act=2, narration="n", claims=[bad_quote], image_brief="b"),
    ])
    errors = validate_provenance(script, _corpus())
    assert len(errors) == 2
    assert any("nope" in e for e in errors)
    assert any("not verbatim" in e for e in errors)
