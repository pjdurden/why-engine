# Up for grabs: why-engine

I built most of a first version of this, then stopped. I'm not going to work
on it any more, so if you have the time, take it. It's MIT-licensed: fork it,
rename it, ship it, no need to ask.

## The idea in one line

An autonomous pipeline that researches, scripts, illustrates, narrates and
assembles one 15 to 20 minute "why is America built this way" explainer video
per day, with every claim grounded in primary sources (US federal regulations
and records) and no human in the loop for each video.

## Why it might matter

Research for the spec found the same topic ("why are American houses wood
not brick") getting 2.3M views on one channel and under 30 on another. The
format and the grounding matter more than the topic. Most automated channels
are templated slop that YouTube's 2026 "inauthentic content" policy targets.
This design asserts real authorship through per-video primary-source research,
verified claim grounding and a published source list.

## What exists (Python)

- Topic mining with a novelty score, plus a ledger of topics already covered.
- Source acquisition (eCFR and other public-domain US federal sources) and
  claim grounding.
- Script generation directed like film narration, with a mechanical check that
  refuses long verbatim recitation of source text.
- Hand-drawn-style art direction with colour and topic-scoped scenes, an icon
  library, and licensed fonts (OFL and Apache, licenses included).
- TTS narration (Piper), a generated music bed, and a streaming frame
  assembler that doesn't hold the whole video in memory.
- A measured analysis of a reference explainer video's camera motion and cut
  grammar (`docs/superpowers/specs/2026-09-16-reference-motion-analysis.md`).

Spec: [`docs/superpowers/specs/2026-09-16-why-engine-design.md`](docs/superpowers/specs/2026-09-16-why-engine-design.md),
including the cost model and kill criteria.

## What's left

- Visual polish (`docs/video-polish-plan.md` has the work list).
- Publishing automation and the first 30 videos. The spec's kill criteria
  apply from there.
- Never tested against real audience numbers.

## Where to start

Read spec sections 2, 3 and 13 (decisions, the core finding, compliance), then
`src/why_engine/run.py`. The `tools/poc_v*.py` scripts show how the look
evolved.
