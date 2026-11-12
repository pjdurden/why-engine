"""The gate, after the adversarial pass was cut.

What survives is what can be checked locally and for free: the citation points
at a real passage (`validate_provenance`, run before the gate), the quote is
verbatim in that passage, and every hard fact in the claim -- number, date,
proper noun -- appears verbatim in the quote. What is gone is the entailment
judgement, which cost a max-effort model call per claim to answer a question
about paraphrase. An invented figure still cannot ship; a quote that is real
but only loosely relevant now can.
"""
from why_engine.config import Settings
from why_engine.grounding.gate import run_gate
from why_engine.models import Claim, Script, Segment


def _claim(text="Baird led it in 1877.", quote="In 1877 Baird led it."):
    return Claim(text=text, source_id="s", quote=quote, locator="p.1")


def _script(*claims):
    return Script(
        topic="t",
        segments=[Segment(segment_id=1, act=1, narration="n", claims=list(claims), image_brief="b")],
    )


def _multi_segment_script(*claims_per_segment):
    """One segment per argument, holding that many claims, so flattened claim
    indices can be distinguished from per-segment ones."""
    return Script(topic="t", segments=[
        Segment(segment_id=10 + offset, act=1 + offset, narration="n",
                claims=[_claim() for _ in range(count)], image_brief="b")
        for offset, count in enumerate(claims_per_segment)
    ])


def test_the_gate_needs_no_model_client():
    """The whole point of the cut: accepting a script costs nothing."""
    assert run_gate(_script(_claim()), Settings()).passed is True


def test_a_claim_whose_facts_are_all_in_its_quote_is_accepted():
    result = run_gate(_script(_claim()), Settings())
    assert result.passed is True
    assert result.failed_indices == []


def test_an_invented_figure_is_rejected():
    """The quote has no 298 in it. This is the check that earns its keep."""
    bad = _claim(text="It reached 298 districts.", quote="It reached many districts.")
    result = run_gate(_script(bad), Settings())
    assert result.passed is False
    assert result.failed_indices == [0]
    assert result.claim_verdicts[0].missing_tokens == ["298"]


def test_an_invented_proper_noun_is_rejected():
    bad = _claim(text="The agency sent Baird west.", quote="The agency sent him west.")
    result = run_gate(_script(bad), Settings())
    assert result.passed is False
    assert "Baird" in result.claim_verdicts[0].missing_tokens


def test_a_proper_noun_opening_a_sentence_is_not_checked():
    """A known hole, and now an unbacked one. `hard_fact_tokens` skips the first
    word of every sentence because a sentence-initial capital is usually grammar,
    not a name -- checking it would reject "Wooden houses dominate" for inventing
    "Wooden". The entailment pass used to cover this case. Nothing does now: a
    fabricated name in the opening position of a sentence ships.
    """
    bad = _claim(text="Baird led it.", quote="He led it.")
    assert run_gate(_script(bad), Settings()).passed is True


def test_one_bad_claim_rejects_the_whole_script():
    """Rejection is not per-claim. Dropping a claim would leave the narration
    sentence asserting it in place, with its evidence quietly stripped."""
    bad = _claim(text="It reached 298 districts.", quote="It reached many districts.")
    result = run_gate(_script(_claim(), bad), Settings())
    assert result.passed is False
    assert result.failed_indices == [1]


def test_a_script_with_no_claims_at_all_is_rejected():
    """A script that asserts nothing is not a documentary, it is filler."""
    assert run_gate(_script(), Settings()).passed is False


def test_every_claim_is_recorded_even_when_the_script_passes():
    """verdicts.json is the audit trail. It has to hold the claims that passed,
    not only the ones that failed."""
    result = run_gate(_multi_segment_script(2, 1), Settings())
    assert [v.claim_index for v in result.claim_verdicts] == [0, 1, 2]
    assert [v.segment_id for v in result.claim_verdicts] == [10, 10, 11]
    assert all(v.strict_ok for v in result.claim_verdicts)


def test_failed_indices_are_flattened_across_segments():
    script = _multi_segment_script(2, 1)
    script.segments[1].claims[0] = _claim(text="It reached 298 districts.",
                                          quote="It reached many districts.")
    result = run_gate(script, Settings())
    assert result.failed_indices == [2]


def test_every_failing_claim_is_reported_not_just_the_first():
    """One regeneration attempt should be told about all of its mistakes."""
    bad = _claim(text="It reached 298 districts.", quote="It reached many districts.")
    other = _claim(text="It was led by Baird.", quote="It was led by him.")
    result = run_gate(_script(bad, other), Settings())
    assert result.failed_indices == [0, 1]
