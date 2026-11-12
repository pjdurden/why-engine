import re
from typing import List

from why_engine.models import Claim

#: No trailing \b: the CFR writes "3,114W newtons", and demanding a non-word
#: character after the digits made this backtrack to the first boundary it could
#: find -- yielding the token "3" for a figure the claim had copied correctly.
_NUMBER = re.compile(r"\b\d[\d,.]*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\b[A-Za-z][A-Za-z'\-]*\b")

_STOPWORDS = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "They", "He", "She",
    "But", "And", "Or", "So", "Then", "When", "While", "If", "In", "On", "At",
    "By", "For", "From", "To", "Of", "With", "As", "Its", "Their", "His", "Her",
}

_ABBREVIATIONS = {
    "Dr", "Mr", "Mrs", "Ms", "Prof", "St", "Jr", "Sr", "Gen", "Col", "Lt",
    "Sen", "Rep", "Gov", "Capt", "Sgt", "Rev", "Hon", "Inc", "Ltd", "Co",
    "No", "Vol",
}


def _ends_with_abbreviation(fragment: str) -> bool:
    """True when a fragment ends in an abbreviation rather than a full stop:
    a single initial ("F."), a dotted acronym ("U.S."), or a known title ("Dr.")."""
    parts = fragment.split()
    if not parts:
        return False
    core = parts[-1].rstrip(".")
    if not core:
        return False
    if len(core) == 1 and core.isupper():
        return True
    if "." in core and all(len(p) <= 1 for p in core.split(".")):
        return True
    return core in _ABBREVIATIONS


def _split_sentences(text: str) -> List[str]:
    """Split on terminators, then re-join fragments where the split was caused by
    an abbreviation. Without this, a name following an initial lands at index 0 of
    a false sentence and is discarded as sentence-initial grammar."""
    sentences: List[str] = []
    for part in _SENTENCE_SPLIT.split(text.strip()):
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] = sentences[-1] + " " + part
        else:
            sentences.append(part)
    return sentences


def hard_fact_tokens(text: str) -> List[str]:
    tokens: List[str] = []

    for match in _NUMBER.finditer(text):
        tokens.append(match.group().rstrip(".").replace(",", ""))

    for sentence in _split_sentences(text):
        words = _WORD.findall(sentence)
        for index, word in enumerate(words):
            if index == 0:
                continue  # sentence-initial capital is grammar, not a proper noun
            if len(word) < 2:
                continue  # a bare initial (e.g. the "U" or "S" in "U.S.") is not a token
            if word in _ABBREVIATIONS:
                continue  # "Standard No. 217": No is a title, 217 is the fact
            if word[0].isupper() and word not in _STOPWORDS:
                tokens.append(word)

    seen = set()
    unique = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    return unique


def _present(token: str, haystack: str) -> bool:
    """Is this hard fact actually in the evidence?

    Numbers get digit boundaries rather than word boundaries. The CFR writes
    forces as "3,114W newtons", where W is the seat's occupancy; \b demands a
    non-word character after the digits, so the checker called a figure invented
    that was sitting in the quote. Guarding against adjacent digits instead
    still refuses to find 452 inside 14520, which is the failure that matters.
    """
    if token[0].isdigit():
        pattern = rf"(?<!\d){re.escape(token)}(?!\d)"
    else:
        pattern = rf"\b{re.escape(token)}\b"
    return re.search(pattern, haystack, flags=re.IGNORECASE) is not None


def check_verbatim(claim: Claim) -> List[str]:
    """Which hard facts in the claim are not in the evidence behind it.

    The evidence is the quote plus the citation itself. A claim about "Standard
    222" cites a passage of 571.222, and a regulation does not repeat its own
    section number inside every paragraph it contains; without this, naming the
    standard a video is about was an automatic rejection.
    """
    evidence = f"{claim.quote} {claim.source_id} {claim.locator}".replace(",", "")
    return [t for t in hard_fact_tokens(claim.text) if not _present(t, evidence)]
