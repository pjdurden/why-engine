from why_engine.models import TopicCandidate
from why_engine.topics.mine import score_topic, select_next_topic


def _cand(question="Why are American houses made of wood?", demand=0.8,
          incumbent_weakness=0.7, hook_strength=0.9, sources=("https://gao.gov/a",)):
    return TopicCandidate(question=question, demand=demand,
                          incumbent_weakness=incumbent_weakness,
                          hook_strength=hook_strength, source_urls=list(sources))


def test_a_candidate_with_no_sources_scores_zero():
    assert score_topic(_cand(sources=()), published=[]) == 0.0


def test_a_strong_candidate_scores_above_zero():
    assert score_topic(_cand(), published=[]) > 0.0


def test_an_exact_duplicate_of_a_published_question_scores_zero():
    q = "Why are American houses made of wood?"
    assert score_topic(_cand(question=q), published=[q]) == 0.0


def test_a_near_duplicate_is_penalised_below_a_fresh_topic():
    fresh = score_topic(_cand(question="Why is American bread so sweet?"),
                        published=["Why are American houses made of wood?"])
    near = score_topic(_cand(question="Why are American homes made of wood?"),
                       published=["Why are American houses made of wood?"])
    assert near < fresh


def test_higher_demand_scores_higher_all_else_equal():
    assert score_topic(_cand(demand=0.9), []) > score_topic(_cand(demand=0.2), [])


def test_weaker_incumbents_score_higher_all_else_equal():
    assert score_topic(_cand(incumbent_weakness=0.9), []) > score_topic(_cand(incumbent_weakness=0.1), [])


def test_select_next_topic_returns_the_highest_scoring_candidate():
    weak = _cand(question="Why is bread sweet?", demand=0.1, incumbent_weakness=0.1, hook_strength=0.1)
    strong = _cand(question="Why are roads so wide?", demand=0.9, incumbent_weakness=0.9, hook_strength=0.9)
    assert select_next_topic([weak, strong], published=[]).question == "Why are roads so wide?"


def test_select_next_topic_returns_none_when_every_candidate_is_gated_out():
    """No source, no topic. Returning None is correct: the day publishes nothing."""
    assert select_next_topic([_cand(sources=()), _cand(sources=())], published=[]) is None


def test_novelty_uses_the_worst_overlap_not_the_average():
    """One near-duplicate disqualifies, even against an archive of dissimilar
    topics. Under averaging this candidate would score positively."""
    published = [
        "Why are American houses made of wood?",
        "Why is bread sweet?",
        "Why are roads so wide?",
    ]
    candidate = _cand(question="Why are American homes made of wood?")
    assert score_topic(candidate, published) == 0.0


def test_a_mildly_similar_topic_is_penalised_proportionally_not_zeroed():
    """Below the floor, novelty degrades smoothly rather than gating: the score
    must sit strictly between a fully-gated 0.0 and an unpublished-topic score."""
    candidate = _cand(question="Why are American homes made of wood?")
    mild = score_topic(candidate, ["Why are American roads made of concrete?"])
    fresh = score_topic(candidate, [])
    assert 0.0 < mild < fresh


def test_two_different_expensive_american_things_are_not_duplicates():
    """Every topic on this channel says "American", and a lot of them end in
    "so expensive". Counting the scaffolding as evidence of overlap makes any
    two questions in the niche look alike: healthcare and college are not the
    same video."""
    published = ["Why is American healthcare so expensive?"]
    assert score_topic(_cand(question="Why is American college so expensive?"),
                       published) > 0.0


def test_a_plural_and_its_singular_are_the_same_topic():
    """'school buses' and 'the school bus' must not be two videos."""
    published = ["Why are American school buses yellow?"]
    assert score_topic(_cand(question="Why is the school bus yellow?"),
                       published) == 0.0
