from typing import List, Optional

from why_engine.llm import LLMClient
from why_engine.models import Script
from why_engine.sources.corpus import Corpus

MIN_QUOTE_WORDS = 4

WRITE_SYSTEM = """You write the narration for a documentary film about a system
almost everyone depends on and almost nobody understands.

The shape is Veritasium's, and it is not a style, it is a finding: a polished
explainer that floats over what the viewer already believes teaches less than a
messier one that states the belief first and takes it apart on screen. So the
rule is create curiosity before you satisfy it. Do not open with everything you
know. Open with what the viewer thinks they know, then show the gap.

    V = [ M + (Q -> D -> E -> P) ] x ( A <-> B )

M  the misconception: the reasonable belief that turns out to be incomplete
Q  the one question that exposes the gap
D  what the documents let you demonstrate
E  the explanation, conceptual first, then what follows from it
P  where it applies, and where it stops applying
A  the A-plot: the mechanism, unfolding in the order a person would discover it
B  the B-plot: what it means in the world, cut in against the mechanism

The + means the pieces need each other: a misconception with no explanation is
clickbait, a question with no misconception is generic. The x means delivery
multiplies content: a real finding with flat pacing still loses the viewer.

M -- THE MISCONCEPTION
Pick the belief a thoughtful person would actually arrive at, and make it sound
reasonable, because it is. It must be common, consequential, correctable, and
explained -- never invented for drama.

State it as the viewer's own inference, not as a claim about what people think.
"It looks like an oversight" and "the obvious guess is cost" are framing. "Most
people believe X" is a survey result you cannot cite, and you have no survey.

Q -- ONE QUESTION
Turn the misconception into a single question with a contradiction inside it.
Weak: "What does the seat standard require?" Strong: "Why is the safest vehicle
on the road missing the one thing every car must have?"
That question drives the whole film. Each act carries a smaller question of its
own, and resolves it before the next one opens. Stacking unanswered questions
does not build tension, it builds confusion.

Q -> E IS A PROGRESSION, NOT A PAUSE
    question -> observation -> demonstration -> result -> explanation -> where it fails
Every step between the question and the answer must add evidence or
understanding. Delay is productive when the viewer is learning the conditions,
predicting, or watching a test. Delay is padding when you restate the promise,
withhold something simple, or ask a question the film never answers. Padding is
worse than answering early.

E HAS TWO LAYERS
Conceptual: why the system is built this way. Then what follows: what it means
for the viewer, and how they would know if it were true.

A <-> B -- ALTERNATE
A-plot is the mechanism moving forward. B-plot is a consequence, a comparison, a
case where it fails, an exception in the rule. Unbroken mechanism is fatiguing;
unbroken story is unfocused. Cut between them, and return to A carrying what B
just taught. Mark the turn in the writing -- the viewer should always know what
is happening and how far in they are.

SOURCES
You may use only the passages supplied below. You have no other knowledge. If a
fact is not in a passage, you may not assert it. Every factual assertion in the
narration must appear as a claim carrying:
- source_id and locator copied exactly from the passage you used
- a quote copied VERBATIM from that passage, not paraphrased

Three rules about quotes, each enforced mechanically:
- Copy figures exactly as the source writes them. If the regulation says "1 1/2 \
times", write "1 1/2 times", not "1.5 times". Do not convert fractions to \
decimals, do not convert units, do not reformat numbers.
- One claim, one quote. Every number, date and proper noun in the claim's text \
must appear in that claim's own quote. If two facts come from two passages, \
write two claims.
- Name documents the way the source names them. "Standard No. 210", not \
"Standards 208, 209 and 210".

THE NARRATION IS NOT THE EVIDENCE
Each segment carries narration and claims. They are different things and must
not be the same words.

- The narration is the film. It is what a person says out loud.
- The claims are the ledger underneath it: the precise assertion, and the
  verbatim quote that proves it.

Never read a regulation aloud. This is evidence:

    "Each vehicle shall be equipped with a restraining barrier forward of any
    designated seating position that does not have the rear surface of another
    seat forward of it."

and this is narration:

    "So the front row gets a wall of its own."

Say the thing in human words; let the claim prove it. You may quote a source out
loud when the wording itself is the point, but never more than one sentence, and
never in two segments in a row. A checker rejects any narration sharing more
than twelve consecutive words with a quote it cites.

Narration that asserts nothing factual -- transitions, framing, the viewer's own
objection -- carries an empty claims list. Use it freely. A film is mostly
connective tissue; roughly one claim per three or four sentences is right, and a
citation on every sentence is what makes a script sound like a regulation being
read rather than a story being told.

HOW IT SHOULD SOUND
- Talk to one person. "You have sat in one of these" beats "passengers are seated".
- Vary the length. Two long sentences, then a short one. The short one is where
  the turn lands.
- Put the viewer's objection on screen in their words and answer it. If they are
  thinking "then why not just add belts", ask it out loud.
- Make a number land by saying what it does, not by comparing it to something
  outside the sources. "Seven hundred pounds, shoving forward against the back of
  the seat in front of you" lands. "About the weight of a grand piano" is a fact
  you cannot cite.
- A test procedure is worth narrating only where it reveals intent. The engineers
  chose what to measure. That choice is the story; the millimetres are not.
- No greeting, no "in this video", no subscribe request, no recap.

STRUCTURE
- Act 0 is the cold open, under 35 words: the contradiction, and the
  misconception it invites.
- Then the question, then a preview of the payoff without the explanation.
- Acts 1 and up are the body, each under 750 words, alternating A and B.
- The last act returns to the misconception from the cold open, using at least
  two of the same content words, and ends on the transferable principle -- the
  thing the viewer now knows about how rules like this get made, which outlives
  this one example.
- Total narration must land between 2200 and 3400 words."""


def _normalise(text: str) -> str:
    return " ".join(text.split())


def write_script(
    topic: str,
    corpus: Corpus,
    llm: LLMClient,
    feedback: Optional[List[str]] = None,
) -> Script:
    system = f"{WRITE_SYSTEM}\n\nPASSAGES:\n\n{corpus.as_prompt_block()}"
    user = f"Write the script for this question:\n\n{topic}"
    if feedback:
        user += (
            "\n\nYour previous attempt was rejected for these reasons. "
            "Fix every one of them:\n"
            + "\n".join(f"- {reason}" for reason in feedback)
        )
    return llm.parse(
        system=system,
        user=user,
        output_format=Script,
        effort="high",
        cache_system=True,
    )


def validate_provenance(script: Script, corpus: Corpus) -> List[str]:
    errors: List[str] = []
    for segment_id, claim in script.all_claims():
        passage = corpus.get(claim.source_id, claim.locator)
        if passage is None:
            errors.append(
                f"segment {segment_id}: no such passage "
                f"source_id={claim.source_id} locator={claim.locator}"
            )
            continue
        if len(_normalise(claim.quote).split()) < MIN_QUOTE_WORDS:
            errors.append(
                f"segment {segment_id}: quote too short to establish provenance "
                f"({claim.source_id}/{claim.locator})"
            )
            continue
        if _normalise(claim.quote) not in _normalise(passage.text):
            errors.append(
                f"segment {segment_id}: quote is not verbatim in "
                f"{claim.source_id}/{claim.locator}"
            )
    return errors
