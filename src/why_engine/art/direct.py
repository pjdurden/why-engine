"""Art director: a Script becomes a shot list.

Spec S7.6. One shot per SENTENCE, not per segment, because the narrator
renders sentence by sentence and the composer cuts to those measurements --
that is what keeps picture and voice together (see audio/narrate.py).

Archetype choice is rule-based and deterministic rather than a model call.
Three reasons. The rules key off things already extracted and verified
elsewhere -- numbers, claims, the segment's image brief -- so a model would be
re-deriving what the pipeline already knows. A model call here would be
unverifiable: the grounding gate checks claims, not layout, so a hallucinated
shot choice has nothing to catch it. And it is one more paid call per sentence
on a pipeline meant to run daily and unattended.

Ordering matters. The rules are tried most-specific first, so a sentence
carrying two comparable quantities becomes a comparison rather than falling
through to a generic card just because it also contains a year.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from why_engine.art.numbers import Number, parse_numbers
from why_engine.models import Script, Segment

Archetype = Literal[
    "text_card", "icon_multiply", "measure_compare", "then_now",
    "quantity_viz", "versus_split", "painted_scene", "archival_plate",
]

#: Above this an icon grid stops reading as a count and becomes texture.
MAX_ICON_COUNT = 30
#: A sentence this short is a beat, and a beat wants a card, not a diagram.
PUNCH_WORDS = 6
#: Words that mark a sentence as describing a place rather than a fact.
SCENE_WORDS = frozenset({
    "river", "rivers", "town", "street", "field", "fields", "office", "counter",
    "station", "depot", "factory", "farm", "farms", "coast", "lake", "road",
    "kitchen", "store", "warehouse", "harbour", "harbor", "bank", "valley",
})
#: Words that mark a sentence as setting two parties against each other.
VERSUS_WORDS = frozenset({"against", "versus", "instead", "rather", "unlike", "whereas"})


@dataclass
class Shot:
    """One sentence's worth of picture."""

    index: int
    segment_id: int
    sentence: str
    archetype: Archetype
    headline: str = ""
    subhead: str = ""
    captions: list[str] = field(default_factory=list)
    numbers: list[Number] = field(default_factory=list)
    icon: str | None = None
    count: int = 0
    scene: str | None = None
    image_brief: str = ""
    claim_quote: str = ""
    claim_source: str = ""


#: Words a phrase must not end on. Truncating mid-clause leaves the reader
#: waiting for a word that never comes -- "BY RAIL TO" reads as a bug.
_DANGLING = frozenset({
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "and", "or", "but", "that", "than", "as", "into", "over", "under", "its",
    "their", "his", "her", "was", "were", "is", "are", "had", "has", "been",
})


def _key_phrase(sentence: str, limit: int = 6) -> str:
    """The first `limit` words, trimmed so the phrase ends somewhere sensible.

    Headlines are hand-lettered at size; a full sentence will not fit, and
    shrinking it to fit is what makes a card look like a slide.
    """
    words = sentence.replace(",", "").split()[:limit]
    while len(words) > 1 and words[-1].lower().strip(".!?;:") in _DANGLING:
        words.pop()
    return " ".join(words).rstrip(".!?;:")


def _comparable(numbers: list[Number]) -> tuple[Number, Number] | None:
    """Two numbers sharing a unit, which is what makes them worth comparing."""
    for i, a in enumerate(numbers):
        for b in numbers[i + 1:]:
            if a.unit and a.unit == b.unit and a.value != b.value:
                return a, b
    return None


def _countable(numbers: list[Number]) -> Number | None:
    for n in numbers:
        if not n.is_year and n.value.is_integer() and 2 <= n.value <= MAX_ICON_COUNT:
            return n
    return None


def direct_sentence(sentence: str, segment: Segment, index: int,
                    scene_pool: list[str] | None = None) -> Shot:
    """Choose the archetype for one sentence and fill its fields."""
    numbers = parse_numbers(sentence)
    lowered = set(sentence.lower().replace(",", "").replace(".", "").split())
    shot = Shot(index=index, segment_id=segment.segment_id, sentence=sentence,
                archetype="text_card", numbers=numbers,
                image_brief=segment.image_brief)

    pair = _comparable(numbers)
    if pair is not None:
        a, b = pair
        shot.archetype = "measure_compare"
        shot.headline = _key_phrase(sentence, 5)
        shot.captions = [a.display, b.display]
        return shot

    count = _countable(numbers)
    if count is not None:
        shot.archetype = "icon_multiply"
        shot.count = int(count.value)
        shot.headline = _key_phrase(sentence, 7)
        return shot

    if lowered & VERSUS_WORDS:
        shot.archetype = "versus_split"
        shot.headline = _key_phrase(sentence, 5)
        return shot

    year = next((n for n in numbers if n.is_year), None)
    if year is not None:
        shot.archetype = "text_card"
        shot.headline = year.display
        shot.subhead = _key_phrase(sentence, 7)
        return shot

    scene_hit = lowered & SCENE_WORDS
    if scene_hit and len(sentence.split()) > PUNCH_WORDS:
        shot.archetype = "painted_scene"
        shot.scene = (scene_pool or [None])[index % len(scene_pool)] if scene_pool else None
        shot.captions = [_key_phrase(sentence, 6)]
        return shot

    if segment.claims:
        claim = segment.claims[0]
        shot.claim_quote = claim.quote
        shot.claim_source = claim.source_id

    shot.headline = _key_phrase(sentence, PUNCH_WORDS)
    shot.subhead = "" if len(sentence.split()) <= PUNCH_WORDS else _key_phrase(sentence, 12)
    return shot


def direct(script: Script, sentences_for: callable,
           scene_pool: list[str] | None = None) -> list[Shot]:
    """Turn a whole Script into a shot list.

    `sentences_for` splits a segment's narration; it is injected rather than
    imported so the director and the narrator provably agree on the split.
    Two different splitters would silently desynchronise picture from voice,
    and that failure is invisible until someone watches the video.
    """
    shots: list[Shot] = []
    for segment in script.segments:
        for sentence in sentences_for(segment.narration):
            shots.append(direct_sentence(sentence, segment, len(shots), scene_pool))
    return shots
