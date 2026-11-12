from typing import List, Optional, Set

from why_engine.models import TopicCandidate

DEMAND_WEIGHT = 0.4
INCUMBENT_WEIGHT = 0.35
HOOK_WEIGHT = 0.25

NOVELTY_FLOOR = 0.35  # Jaccard overlap at or above this is treated as a duplicate

#: Question scaffolding, plus the niche itself. Every topic on this channel is
#: about America, so "American" is not evidence that two questions are alike --
#: counting it made "why is American healthcare so expensive" and "why is
#: American college so expensive" look like the same video.
_STOPWORDS = {
    "why", "the", "a", "an", "are", "is", "do", "does", "did", "we", "so", "of",
    "in", "to", "on", "at", "it", "its", "this", "that", "they", "their", "there",
    "and", "but", "not", "was", "were", "be", "been", "has", "have", "had",
    "america", "american", "americans", "us",
}


def _stem(word: str) -> str:
    """Crude and symmetric, which is all a set intersection needs. Strips a
    plural 's' and then a trailing 'e', so bus/buses and house/houses land on
    the same token without dragging in a real stemmer."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def _tokens(question: str) -> Set[str]:
    words = {w.strip("?.,!\"'").lower() for w in question.split()}
    return {_stem(w) for w in words - _STOPWORDS if w}


def is_duplicate(question: str, published: List[str]) -> bool:
    """True when `question` is too close to something already covered.

    Shares its tokeniser with scoring on purpose: a topic the ledger waves
    through must not then be zeroed by the miner for the same reason.
    """
    return _novelty(question, published) == 0.0


def _novelty(question: str, published: List[str]) -> float:
    """1.0 means wholly novel, 0.0 means a duplicate of something already published."""
    candidate = _tokens(question)
    if not candidate:
        return 0.0
    worst = 0.0
    for prior in published:
        prior_tokens = _tokens(prior)
        union = candidate | prior_tokens
        if not union:
            continue
        worst = max(worst, len(candidate & prior_tokens) / len(union))
    if worst >= NOVELTY_FLOOR:
        return 0.0
    return 1.0 - worst


def score_topic(candidate: TopicCandidate, published: List[str]) -> float:
    if not candidate.source_urls:
        return 0.0  # hard gate: no source, no topic

    novelty = _novelty(candidate.question, published)
    if novelty == 0.0:
        return 0.0

    base = (
        DEMAND_WEIGHT * candidate.demand
        + INCUMBENT_WEIGHT * candidate.incumbent_weakness
        + HOOK_WEIGHT * candidate.hook_strength
    )
    return base * novelty


def select_next_topic(candidates: List[TopicCandidate], published: List[str]) -> Optional[TopicCandidate]:
    scored = [(score_topic(c, published), c) for c in candidates]
    viable = [(s, c) for s, c in scored if s > 0.0]
    if not viable:
        return None
    return max(viable, key=lambda pair: pair[0])[1]
