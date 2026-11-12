from typing import List, Set

from why_engine.config import Settings
from why_engine.models import Script

COLD_OPEN_MAX_S = 15.0

#: Longest run of words the narration may share with a quote it cites.
#: Reading one striking line out loud is a filmmaking move; reading the
#: paragraph is what made the first real script sound like a regulation being
#: recited rather than a story being told.
MAX_RECITED_WORDS = 12

BANNED_PHRASES = (
    "before we begin",
    "make sure to subscribe",
    "hey guys",
    "in this video",
    "don't forget to like",
    "welcome back to the channel",
)

_ECHO_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "of",
    "in", "on", "at", "it", "that", "this", "why", "how", "so", "for", "with", "as",
}


def estimate_runtime_s(script: Script, settings: Settings) -> float:
    return script.word_count() / settings.words_per_second


def _content_words(text: str) -> Set[str]:
    return {w.strip(".,!?;:\"'").lower() for w in text.split()} - _ECHO_STOPWORDS


def _words(text: str) -> List[str]:
    return [w.strip(".,!?;:\"'()").lower() for w in text.split() if w.strip(".,!?;:\"'()")]


def longest_shared_run(a: str, b: str) -> int:
    """Longest run of consecutive words appearing in both texts.

    Word-level rather than character-level so that punctuation and casing do not
    hide a recitation, and so the threshold can be stated in the unit a writer
    actually thinks in.
    """
    x, y = _words(a), _words(b)
    if not x or not y:
        return 0
    best = 0
    prev = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        cur = [0] * (len(y) + 1)
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def check_retention(script: Script, settings: Settings) -> List[str]:
    violations: List[str] = []

    if not script.segments:
        return ["script has no segments"]

    first = script.segments[0]
    if first.act != 0:
        violations.append("missing cold open: first segment must be act 0")
    else:
        cold_open_words = sum(
            len(s.narration.split()) for s in script.segments if s.act == 0
        )
        open_s = cold_open_words / settings.words_per_second
        if open_s > COLD_OPEN_MAX_S:
            violations.append(f"cold open too long: {open_s:.0f}s exceeds {COLD_OPEN_MAX_S:.0f}s")

    runtime = estimate_runtime_s(script, settings)
    if runtime < settings.min_runtime_s:
        violations.append(f"runtime too short: {runtime:.0f}s below {settings.min_runtime_s}s")
    if runtime > settings.max_runtime_s:
        violations.append(f"runtime too long: {runtime:.0f}s above {settings.max_runtime_s}s")

    act_words: dict = {}
    for segment in script.segments:
        if segment.act == 0:
            continue  # the cold open has its own, tighter ceiling
        act_words[segment.act] = act_words.get(segment.act, 0) + len(segment.narration.split())
    for act in sorted(act_words):
        act_s = act_words[act] / settings.words_per_second
        if act_s > settings.max_act_s:
            violations.append(
                f"act {act} too long: {act_s:.0f}s exceeds {settings.max_act_s}s"
            )

    for segment in script.segments:
        for claim in segment.claims:
            run = longest_shared_run(segment.narration, claim.quote)
            if run > MAX_RECITED_WORDS:
                violations.append(
                    f"segment {segment.segment_id}: narration recites its source -- "
                    f"{run} consecutive words shared with the quote, limit is "
                    f"{MAX_RECITED_WORDS}. Say it in your own words and let the "
                    f"claim carry the quote."
                )
                break

    for segment in script.segments:
        lowered = segment.narration.lower()
        for phrase in BANNED_PHRASES:
            if phrase in lowered:
                violations.append(f"segment {segment.segment_id}: banned filler phrase '{phrase}'")

    if len(script.segments) >= 2 and first.act == 0:
        shared = _content_words(first.narration) & _content_words(script.segments[-1].narration)
        if len(shared) < 2:
            violations.append("final segment does not resolve the cold open in its own words")

    return violations
