# why-engine: autonomous grounded explainer-video pipeline

Date: 2026-09-16
Status: approved design, not yet implemented

## 1. Goal

Build an autonomous pipeline that produces and publishes one 15-20 minute
illustrated explainer video per day to a single YouTube channel, with zero
per-video human involvement, targeting US viewers in the Education category.

Revenue target: $2,000-5,000/month net by month 6.
Honest probability estimate: 15-25%. See section 14 for kill criteria.

## 2. Decisions locked

| Decision | Choice |
|---|---|
| Human in loop | None per video. System-level configuration only. |
| Niche | "Why is America built this way": US systems, infrastructure, housing, food, transport, standards, regulation |
| Category | Education |
| Subjects | Historical and settled. Systems and objects, not living people. |
| Format | Static hand-drawn-style illustrations plus TTS narration |
| Footage | None ever. Original illustrations only. |
| Cadence | 1/day |
| Channels | One. Config-driven so a second is a config file. |

## 3. The core finding this design is built around

Research on 2026-09-16 compared channels covering the identical topic
("why are American houses wood not brick"):

| Channel | Views |
|---|---|
| Cheddar | 2,332,899 |
| US Observer (4 months ago) | 900,283 |
| LRN2DIY | 605,176 |
| USA Revealed | 3,917 |
| American Princess | 1,490 |
| This is Wild | 346 |
| Simply Understood | 239 |
| Story Behind The Things | 26 |

A channel-name search returned nine distinct channels named "Everyday Things
Explained", plus four near-variants, all at effectively zero subscribers.

**Topic selection is not the bottleneck.** Every dead channel picked a correct,
high-demand topic. The 900,283-to-26 gap on one topic is execution: script
quality, thumbnail CTR, retention across a 15-minute runtime, and art that does
not read as slop.

Therefore this pipeline is designed as a set of quality gates, not a set of
generation steps. Any component that only generates, without a gate that can
reject its output, is incomplete.

The reference channel (@Bright-Psycho, 14.8K subs, 122 videos, ~1/day, median
17K views, top video 866K) confirms the thesis from the other direction: its
video descriptions cite primary archival research (US Fish Commission reports
1877-1897 via archive.org, USGS NAS database, FAO, named researchers and
dates). Twelve copycat channels replicated its art and title formula without
the research. All twelve are under 100 subscribers.

## 4. Niche definition

Question space: why American everyday systems and objects are shaped the way
they are. Housing and building codes, road and rail design, food supply and
regulation, retail and zoning, utilities, measurement standards, agricultural
policy, shipping and logistics.

Selection rule: a topic qualifies only if a primary, public-domain,
text-retrievable authoritative source exists. Candidate source corpora:

- GAO reports
- DOT, FHWA, FRA, FAA rulemaking and studies
- USDA and FDA regulatory history
- Code of Federal Regulations and Federal Register
- Congressional records and committee hearings
- US Census and BLS series
- NIST publications
- archive.org public-domain government documents

All are US government works, so the source material itself carries no copyright
encumbrance.

Rejected alternatives and why:
- Corporate-collapse history: higher RPM ($8-16) but incumbents are WSJ,
  Bloomberg and Patrick Boyle, who have budget and on-camera talent.
- Engineering disasters: best archive in existence (NTSB, CSB, NIST) but the
  most AI-saturated lane found in research, with 30+ farm channels already
  pointed at it, precisely because that archive is the most scrapeable.
- Media and celebrity: ~$2 RPM versus ~$10, plus AI-persona monetization
  restrictions and defamation exposure on living subjects.

## 5. Constraints

**Tax.** Ad revenue from US viewers is subject to US withholding that depends on the creator's tax residence and treaty status. File the right form (W-8BEN for non-US creators) or backup withholding applies.

**Policy.** YouTube's 15 July 2026 update renamed "repetitious content" to
"inauthentic content" and demonetized thousands of faceless AI channels. Faceless
is permitted. TTS narration is permitted. What is targeted is templated content
reproduced at scale with no real author input and little variation between
videos. The grounding gate and per-video source list are the compliance posture,
not an afterthought.

**Monetization timeline.** YPP requires 1,000 subscribers and 4,000 watch hours
in 12 months, then a review period. At 15-20 minute videos with realistic
retention this is roughly 50,000 views. Expect monetization live around month
3-4, leaving 2-3 earning months before the month-6 checkpoint.

**Content ID.** Zero footage, zero music beds from catalogued libraries. An
autonomous pipeline cannot dispute claims, so it must never generate any.

**Compute.** Local machine is a Ryzen 5 5500U laptop: AVX2-only CPU, 14 GB
shared RAM, and an integrated Vega (Lucienne) GPU that `rocminfo` reports no
compute agent for. **Local image generation is therefore permanently out of
scope**, not deferred. TTS runs local on CPU (Piper, male voice
`en_US-ryan-high`). Image generation is API-only. Composition, text rendering
and assembly are all local and free.

## 6. Architecture

Nine stages. Each stage is independently runnable and testable, reads and writes
a typed artifact, and records its verdict to the run state.

```
  topic_miner  ->  source_acquirer  ->  script_writer
                                            |
                                            v
                                     grounding_gate  --reject--> back to script_writer
                                            |
                                            v
                                    retention_checker --reject--> back to script_writer
                                            |
                                            v
         art_director -> image_gen -> style_gate --reject--> back to image_gen
                                            |
                                            v
                                       narrator (TTS)
                                            |
                                            v
                                        assembler
                                            |
                                            v
                                        publisher
```

Regeneration loops are bounded. Exceeding the bound fails the run and the day
produces no video. **Publishing nothing is always an acceptable outcome.** A
missed upload costs one lottery ticket. A bad upload costs channel trust,
and in the worst case, the channel.

## 7. Components

### 7.1 topic_miner
Maintains a scored backlog of candidate questions.

Score inputs:
- **demand**: YouTube search result volume and view distribution for the
  question phrasing and near variants
- **incumbent weakness**: subscriber counts and upload ages of the top ten
  results. A lane where the top results are old or from small channels scores
  higher.
- **source availability**: does a primary document exist in the corpora in
  section 4. This is a hard gate, not a weight. No source, no topic.
- **hook strength**: does the question read as counterintuitive to a US viewer
  with no domain knowledge
- **novelty**: cosine distance from every topic already published on this
  channel, to enforce the variation the inauthentic-content policy requires

Output: `topic.json` with question, scores, and candidate source URLs.

### 7.2 source_acquirer
Fetches and normalizes the primary documents to text, with per-passage
provenance (document id, page or section, and a verbatim quote span). Caches
locally. A topic whose sources fail to fetch or parse is returned to the
backlog, not worked around.

Output: `corpus.jsonl`, one record per retrievable passage.

### 7.3 script_writer
Retrieval-first. Writes only from retrieved passages, never from parametric
memory. Emits structured segments rather than prose:

```json
{
  "segment_id": 12,
  "act": 3,
  "narration": "...",
  "claims": [
    {"text": "...", "source_id": "gao-24-106221", "quote": "...", "locator": "p.14"}
  ],
  "image_brief": "..."
}
```

Every factual assertion must be carried in a `claims` entry with a verbatim
supporting quote. Narration that asserts nothing factual (transitions, framing)
carries an empty claims array and is allowed.

### 7.4 grounding_gate
The moat. See section 8.

### 7.5 retention_checker
Validates the script against the structural template in section 10 and rejects
on: missing cold open, act longer than the ceiling, an act ending without a
hook, unresolved callback, or estimated runtime outside 14-22 minutes.

### 7.6 art_director
Chooses a **shot archetype** for each beat of the script and fills its slots.
Archetypes are the core abstraction of the visual layer: most shots are composed
from reusable parts, not generated.

### 7.7 icon_library
A reusable set of flat colour icons with bold black outlines: animals, objects,
checkmarks, crosses, arrows, maps, people-generic. Distinct from the cast
library. Generated once, recombined into hundreds of layouts, amortised across
every video. This is what makes a 238-shot video affordable.

### 7.8 cast_library
The two recurring characters, in colour, in a set of poses and expressions.

### 7.9 scene_gen
Full painted establishing scenes, generated per video. Roughly a quarter of
shots. The rest are composed.

### 7.10 composer
Builds a shot from an archetype plus slots, in Pillow: places icons, draws
arrows and rules, renders hand-lettered caption text, applies the colour
contract. Free and local.

### 7.11 archival_fetcher
Fetches real photographs from public-domain archives behind a hard licence gate:
public domain or CC0 only, verified before use, no exceptions. Wikimedia's File
namespace also holds PDFs and DjVu scans, so results are filtered to raster
images. The User-Agent must carry a contact URL or the API returns 403.

### 7.12 narrator
Local CPU TTS, male voice, fixed pacing. Emits audio plus a word-level timing
map, which the assembler cuts on.

### 7.13 assembler
ffmpeg. Cuts on narration beats, applies per-archetype motion, lays a music bed
under the whole runtime, and renders the thumbnail from a dedicated template.

### 7.12 publisher
YouTube Data API upload. Scheduled for US morning, matching the reference
channel's ~06:30 PT slot. Generates title, tags, and a description carrying the
full source list with named figures, dates and URLs, plus the standing
disclaimer that all visuals are original illustrations and no copyrighted
footage is used.

## 8. The grounding gate

**Revised 2026-09-17: the adversarial entailment pass is cut.** What follows is
what the gate does now; the original design is kept at the end of this section
because the reasoning for it still stands and it may come back.

This is the component that separates this pipeline from every failed channel in
section 3, and it is the compliance argument for an autonomous system under the
inauthentic-content policy.

**Mechanism, in two local checks, at zero marginal cost.**

1. **Provenance** (`script/write.py:validate_provenance`, run before the gate).
   The cited passage must exist in the corpus, the quote must be at least
   `MIN_QUOTE_WORDS` long, and the quote must appear verbatim inside that
   passage. A fabricated citation cannot survive this.
2. **Strict mode** (`grounding/gate.py` via `grounding/strict.py`). Every hard
   fact in the claim -- number, date, proper noun, statute reference,
   superlative -- must appear verbatim in the cited quote. Paraphrase is
   acceptable for reasoning; it is not acceptable for a figure.

**Rejection policy.** Whole-script, and there is only one verdict. Any claim
failing strict mode rejects the script and regenerates, with every missing
token fed back so the rewrite is told all of its mistakes at once. Three failed
regenerations abandon the topic. Cutting the offending claim was considered and
rejected: a claim is metadata, so removing it deletes the citation while leaving
the narration sentence that asserted it, and the video would still say the
unverified thing with its evidence quietly stripped.

**Known hole.** `hard_fact_tokens` skips the first word of every sentence,
because a sentence-initial capital is usually grammar rather than a name --
checking it would reject "Wooden houses dominate" for inventing "Wooden". A
fabricated proper noun in the opening position of a sentence therefore is not
caught. The entailment pass used to cover this; nothing does now. Test:
`test_a_proper_noun_opening_a_sentence_is_not_checked`.

**What was cut, and why.** For each claim, an independent verification call in a
fresh context received only the claim and its quote and judged entailment
(`SUPPORTED` / `PARTIAL` / `UNSUPPORTED` / `CONTRADICTED`), because a model
asked to check its own work in the same context agrees with itself. It was
removed on 2026-09-17 for three reasons:

- **Cost.** `AnthropicLLM` never set `prefers_batch`, so the batched path was
  dead code on the real client. Every claim was a separate `claude-opus-5` call
  at `effort="high"` with adaptive thinking, multiplied by up to three
  regeneration attempts. That was essentially the whole $2-6 research-and-script
  line in section 14.
- **A second failure mode.** On an unattended daily run the judge is one more
  thing that can be wrong, time out, or return an unparseable verdict, and its
  mistakes are silent.
- **What it actually caught.** Fabricated figures are the failure that would
  sink the channel, and strict mode catches those mechanically. Entailment
  covered paraphrase drift, which is a quality problem, not a credibility one.

**Testability.** Unchanged and now cheaper: the gate is pure local computation,
so its fixture set of claim/quote pairs runs in milliseconds with no API key and
no `live` marker.

## 9. Style lock

Measured from the reference: **flat colour fills inside bold black outlines**,
with soft shading. Backgrounds vary from full painted scenes to plain white for
icon layouts. Hand-lettered caps for on-screen text, black for body and red for
emphasis.

The earlier monochrome threshold is **removed**. It was the right guarantee for
a hand-drawn ink world and is incompatible with this direction. What survives
from that pass is the displacement warp and stroke variation, which keep the
line hand-made rather than vector-clean.

Consistency now comes from three places rather than a threshold:

- **A fixed palette.** A small named set of fills and the one emphasis red, applied
  by the composer rather than requested in prompts. Generators drift on colour;
  composition does not.
- **A persistent cast and icon set.** Recurring elements are the same PNG every
  time, so they cannot drift at all.
- **Composition over generation.** Roughly three quarters of shots are assembled
  from those fixed parts, so only the painted scenes carry any generation
  variance.

## 9a. Shot taxonomy

Derived by measuring the reference video (16.7 min, 238 hard cuts, one every
4.2s, continuous music bed, speech active 65% of the time). Sampling twenty
frames across it, roughly a quarter are painted scenes and the rest are
illustrated argument. Text appears on nearly every frame and is the primary
information channel, not decoration.

| Archetype | Slots | Example from the reference |
|---|---|---|
| `text_card` | headline, subhead, optional vignette | "SOMEONE TRIED" stamped in red |
| `icon_multiply` | subject, count, icon, caption | one hippo = thirty pigs |
| `ranked_row` | subject, rank, comparison icons | hippo marked 1 beside lion, leopard, crocodile |
| `measure_compare` | two subjects, two values, verdict | 30 MPH vs 28 MPH, "HIPPO WINS" |
| `then_now` | two panels, two labels | THEN. / NOW. |
| `quantity_viz` | two clusters, two labels | dense dots DOMESTICATED vs scattered FAILED |
| `numbered_grid` | six cells, one highlighted | stress-response diagram, one circled red |
| `versus_split` | two subjects, verdict marks | angry hippo VS cow with a green check |
| `map_annotate` | map, arrow, label | migration route with a red arrow |
| `painted_scene` | generated background, optional cast | the zoo hippo pool |
| `archival_plate` | real photograph, caption | a public-domain engraving |

Only `painted_scene` and `archival_plate` require fetching or generating per
video. Every other archetype is composed from the icon and cast libraries plus
text, at zero marginal cost.

## 9b. Motion and transitions

**Superseded 2026-09-16.** The figures previously in this section were derived
from a sharpness-trough detector that compared each boundary against a pooled
median. That detector measured the sharpness *step* between a busy shot and a
sparse one, not motion blur: 86 of the 92 boundaries it flagged are plain hard
cuts. The claims "31% of boundaries are blur transitions" and "shots carry a
slow zoom, alternating push-in and pull-out" were both wrong.

The authority is now
`docs/superpowers/specs/2026-09-16-reference-motion-analysis.md`, measured by
phase correlation over every frame. Binding summary:

| Property | Measured | Binding requirement |
|---|---|---|
| Static shots | 81% of 296; 76-82% of frames bit-identical | hold by default |
| Shot under 3.5s carrying a move | **0 of 134** | never move a short shot |
| Pans, slides, drifts | **0** | the camera holds or scales, nothing else |
| Hard cuts | 288 of 295 (97.6%), 1 frame | the default transition is nothing |
| Whips / punches | 7, **all before t=164s** | opening-hook device, budget 7 |
| Whip axis | all horizontal | no vertical whips |
| Whip shape | ramp in 0.15s, smear 0.15-0.20s, ramp out 0.30s | compound, not a bare blur |
| Zoom presets | +1.19%, +5.26%, -3.66% total travel | fixed total, derived rate |
| Zoom curve | flat rate plateau, abrupt start and stop | linear, no easing |
| Push-in | holds the first ~19% of the shot | static-then-move |
| Pull-out | starts at the cut, holds the last ~10-20% | the mirror of a push |

Zoom direction across adjacent cuts is **undetermined** -- only 6 adjacent
zoom-zoom pairs exist in the reference, too few to support a rule. Do not
alternate push and pull; nothing in the data licenses it.

A fifth device exists and is not yet implemented: **local element animation**,
15 shots (5.1%), camera locked while one element pops in. Median 1.43s, almost
always on white diagram cards.

Implementation: `src/why_engine/art/motion.py`, `src/why_engine/art/transitions.py`.

**Delivery note.** x264's psycho-visual defaults re-dither held frames: a
composition measured at 81.8% bit-identical frames survives a default encode at
only 21.6%, which reads as each shot settling for a few frames after its cut.
`psy-rd=0.0:0.0,aq-mode=0,no-mbtree=1,bframes=0,ip-ratio=1.0` recovers it to
55.9% and shrinks the file. The remaining gap is x264 converging on each new
shot; eight configurations were measured and none beat this one. Open.

## 9c. Frame annotation

Every frame in the reference carries three to six words of hand-lettered text
sitting in whatever space the artwork left over -- black for the statement, red
for the verdict or the emphasised clause, occasionally underlined. Boxed
micro-labels and red arrows attach to diagram elements. Frames without a caption
are rare and always painted narrative beats.

Composed frames do not know in advance where their empty space is, so the
composer finds it: build an ink mask, dilate it so captions keep a reader's gap
from existing marks, and pick the lowest-coverage position that fits. The caller
supplies an anchor hint -- roughly where the text belongs given the composition
-- which breaks ties between equally clear positions. Without it every caption
piles into the first empty band scanned.

Two constraints that are silent when violated:

- Captions must sit inside the **zoom-safe inset**. A shot rendered at peak zoom
  1.10 crops 4.5% from each edge; a caption in the outer band is cropped away
  with no error raised.
- A caption that cannot be placed at the requested gap is retried tighter, then
  at no gap. A line placed tight beats a line silently dropped.

Implementation: `src/why_engine/art/annotate.py`.

## 10. Narration voice

Measured from the reference's speech frames (`tools/analyse_voice.py`). Its
music bed is bright -- 2.5-11 kHz -- so it does not contaminate the low bands
the fit depends on.

Pitch is not the variable. Piper `en_US-ryan-high` has a median F0 of 167 Hz
against the reference's 171 Hz. The difference is spectral tilt: the reference
puts 44% of speech energy in 160-320 Hz where raw Piper puts 13%, and Piper's
peak sits an octave higher, in 320-640 Hz. That is the difference between a
chest voice and a throat one.

| Band (Hz) | Reference | Raw Piper | Voiced |
|---|---|---|---|
| 80-160 | 5.6% | 5.4% | 10.2% |
| 160-320 | 44.4% | 13.5% | 42.8% |
| 320-640 | 29.9% | 43.9% | 30.9% |
| 640-1280 | 15.0% | 27.6% | 11.4% |
| 1280-2560 | 4.3% | 6.6% | 3.7% |
| 2560+ | 0.8% | 2.9% | 1.0% |

Sum of squared per-band error: 1695 raw, 38.5 voiced. The narrator MUST apply
the fitted chain in `src/why_engine/audio/voice.py`; `fit_error()` is the
regression check. Raw Piper output also peaks at 1.000 -- it clips -- which the
chain's limiter fixes.

## 11. Retention template

**Rewritten 2026-09-17.** The first finished script passed every check in the
old table and still sounded like a regulation being read aloud, because it was:
13 of its 14 segments carried runs of CFR text straight into the narration, the
longest 45 consecutive words. "Documentary narration, plain spoken English,
short sentences" is satisfied completely by reciting the Code of Federal
Regulations in short sentences.

The template is now Veritasium's formula, which is not a style choice: Muller's
doctoral work found that a polished explainer floats over what the viewer
already believes and teaches *less* than a messier one that states the
misconception first and dismantles it on screen.

    V = [ M + (Q -> D -> E -> P) ] x ( A <-> B )

| | |
|---|---|
| **M** | The misconception: the reasonable belief that turns out incomplete. Must be common, consequential, correctable and explained -- never invented for drama. |
| **Q** | One question with a contradiction in it, driving the whole film. Secondary questions per act, each resolved before the next opens. |
| **D -> E -> P** | Demonstration, then explanation (conceptual, then what follows), then where it stops applying. |
| **A** | A-plot: the mechanism, in the order a person would discover it. |
| **B** | B-plot: consequence, comparison, exception. Cut in against A, never replacing it. |

| Beat | Budget | Requirement |
|---|---|---|
| Hook | 0:00-0:15 | The contradiction, and the misconception it invites. Under 35 words. |
| Question | 0:15-0:30 | The single question. |
| Preview | 0:30-0:45 | The payoff shown, not explained. |
| Body | ~2-4 min per act | A and B alternating; return to A carrying what B taught. |
| Resolution | final ~90s | Returns to the cold open's misconception in its own words, then states the transferable principle. |

**Honesty adaptation.** The tutorial version of this formula says "most beginners
think X". This channel cannot: that is a survey result, and there is no survey in
the corpus. The misconception is stated as the viewer's own inference -- "it
looks like an oversight", "the obvious guess is cost" -- which is framing rather
than an uncited empirical claim.

**Productive vs unproductive delay.** Delay the answer only while the viewer is
learning conditions, predicting, or watching a test. Restating the promise,
withholding something simple, or asking a question the film never answers is
padding, and padding is worse than answering early.

Hard rules: no greeting, no "in this video", no subscribe request before the
resolution, no recap of what was just said, and no narration sharing more than
`MAX_RECITED_WORDS` consecutive words with a quote it cites -- enforced in
`script/retention.py`, because advice in a prompt is not a constraint.

Source for the formula: the `veritasium-tutorial` skill
(github.com/andal1bttu/veritasium-tutorial), adapted from tutorial to
documentary.

## 12. State and data model

One directory per run, `runs/YYYY-MM-DD-<slug>/`, containing every intermediate
artifact and a `verdicts.json` recording each gate's decision and reasoning.
Artifacts are never overwritten; regeneration writes a new numbered attempt.

A run is fully reconstructible and auditable after the fact. If a claim is ever
challenged publicly, the exact source, quote and locator behind it is on disk.

SQLite holds the topic backlog, published-topic embeddings for the novelty
check, per-video performance pulled back from the YouTube Analytics API, and
cost accounting per run.

## 13. Compliance posture

- Real author input is asserted through per-video primary-source research,
  verified claim grounding, and a published source list. This is materially
  different from templated regeneration, and the run artifacts are the evidence.
- Variation is enforced mechanically by the novelty score in topic_miner.
- Synthetic-content disclosure is set where YouTube requires it. Illustrations
  are stylized and non-photoreal by design, which keeps them clear of the
  realistic-depiction rules.
- Subjects are systems and history, not living people, which keeps the channel
  out of the AI-persona restrictions and away from defamation exposure.
- No footage and no catalogued music, so no Content ID surface.
- W-8BEN filed before monetization to avoid 24% backup withholding.

## 14. Cost model

Per video, approximate:

| Item | Cost |
|---|---|
| Research and script (LLM; grounding verification is local and free since 2026-09-17) | $0.50-1.50 |
| ~150 stills | $4-7 |
| TTS | $0 local, $2-5 API |
| Assembly, upload | $0 |
| **Total** | **$6-14** |

At daily cadence: $180-420/month. Months 1-4 are pre-revenue, so expect
$720-1,680 sunk before the first payout.

## 15. Metrics and kill criteria

Defined now, deliberately, and not renegotiable later.

| Checkpoint | Pass condition | If failed |
|---|---|---|
| Week 6 (~40 videos) | Median views >= 300 and at least one video above 5,000 | Format is not landing. Pivot the niche, keep the pipeline. |
| Month 4 | YPP thresholds met (1,000 subs, 4,000 watch hours) | Stop daily cadence. Reassess. |
| Month 6 | >= $500/month net | Shut it down. Do not extend. |

Primary tracked metrics: impressions CTR, average view duration and percentage,
median views per video, subscriber conversion, RPM once monetized, cost per
video, and gate rejection rates per stage.

Gate rejection rates are the health signal for the pipeline itself. A grounding
gate that never rejects is not working.

## 16. Failure modes

| Failure | Consequence | Mitigation |
|---|---|---|
| Hallucinated claim ships | Misinformation, channel trust, worst case legal | Grounding gate, strict mode on figures, systems-not-people subjects |
| Style drift across a video | Reads as slop, CTR and retention collapse | Style gate with embedding and palette checks |
| Flagged as inauthentic | Demonetization, the whole thesis dies | Source-grounded variation, published sources, novelty enforcement, run artifacts as evidence |
| Stuck in 26-view purgatory | Sunk cost, no revenue | Week 6 kill criterion |
| Regeneration loop burns budget | Cost overrun | Bounded retries, per-run cost ceiling, publish-nothing is acceptable |
| Upload API quota exhaustion | Missed days | Quota accounting in publisher, backoff |
| Source corpus fetch rot | Topics starve | Local cache, backlog depth target of 60+ qualified topics |

## 17. Out of scope for v1

- Multiple channels. The config is channel-parameterized; only one is launched.
- Shorts. Different format, different retention model, different RPM.
- Local image generation. Cost optimization, revisit after monetization.
- Community posts, memberships, sponsorships, affiliate revenue.
- Any human approval step. Explicitly excluded by the design brief.
- Any use of footage, stock video, or catalogued music, permanently.
