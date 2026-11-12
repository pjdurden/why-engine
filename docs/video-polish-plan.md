# Video polish: make it pop

Written 2026-09-17, after he watched the first finished video and the second script.

Verdict on the script: approved. Verdict on the picture: **"not colourful and very
slow."** This is the work list for that, plus the art defects from the first audit
that are still open.

Budget is raised: **$1-2 of generated assets per video**, up from the $0.21 the
first one spent. At fal's $0.003 an image that is **330-660 images**, so nothing
below is blocked on cost. We were using a tenth of what we are allowed.

---

## 1. Colour — measured, not impressionistic

Archetype mix on the approved 14.3-minute script, 177 shots:

| archetype | shots | share |
|---|---|---|
| `text_card` | 162 | **91.5%** |
| `icon_multiply` | 12 | 6.8% |
| `versus_split` | 2 | 1.1% |
| `painted_scene` | 1 | **0.6%** |

**99% white cards, 1% colour.** The reference video is ~38% full-bleed painted
scenes. We generated 8 school-bus scenes and the director used **one of them**.

### Root cause

`direct_sentence` (`art/direct.py:103`) is a cascade of keyword tests with
`text_card` as the fall-through. `painted_scene` is gated behind
`lowered & SCENE_WORDS` — a fixed vocabulary — and the new script's language is
legal and abstract, so almost nothing hits it. Every sentence that matches no
rule becomes a white card, and almost every sentence matches no rule.

### Fix

- **Invert the default.** A sentence with no diagram to make should be a picture,
  not a text card. `text_card` earns its place when there is a number, a name or a
  quote worth reading; otherwise fall through to `painted_scene`.
- **Target a mix, don't let it emerge.** Ration archetypes against a budget the
  way `plan_motion` already rations movement. The director already sees all shots
  at once, so it can allocate rather than decide sentence by sentence.

  **THE TARGET IS 50/50** (set by him, 2026-09-17):

  | half | archetypes | now | target |
  |---|---|---|---|
  | colour | `painted_scene`, `archival_plate` | **0.6%** | **50%** |
  | cards | `text_card`, `icon_multiply`, `versus_split`, `measure_compare`, `then_now`, `quantity_viz` | **99.4%** | **50%** |

  The diagram archetypes count on the card side because they are drawn on a white
  card; if he meant them to count as colour, the text-card share drops further,
  never the other way. This is a hard target, not a floor to drift under: the
  director should allocate to it and be tested against it.
- **Stop cycling 8 scenes by index.** Generate art per segment from the script's
  own `image_brief` field, which is already in the model and currently unused by
  `Assets`. ~17 segments, or go further and brief per shot: 177 images is $0.53.
- **Give the white cards colour.** Even a kept text card can sit on a tinted
  ground from the palette instead of the same cream every time.

## 2. Pace — where the 14 minutes actually go

- Average shot is **4.9s** across 177 shots. The reference holds a shot far less
  than that on cards.
- `GAP_SECONDS = 0.38` after every sentence is **1.1 minutes** of silence in a
  14.3-minute video. Dropping to ~0.22 returns ~30 seconds and tightens every cut.
- Piper `length_scale` is 0.96. Slightly faster delivery compounds across 14 min.
- Motion is rationed by `MOVE_EVERY` and most shots are static. He has said before
  he wants **more motion than the reference actually has** — build to his
  preference, not to the measurement.
- **Cut within a sentence.** Shot boundaries are narration-chunk boundaries today,
  one shot per chunk. A long sentence could carry two or three cuts without
  touching the alignment invariant, since the timing map only needs the shot
  lengths to sum to the segment length.

## 3. Art defects still open from the first audit

- **`belt.png` is a satchel.** FLUX drew the wrong object. Because the topic ends
  in "Belts?" it resolves as the topic icon and is therefore the default on most
  cards. `icon_is_isolated` passes it — isolation is not identity. Needs an
  identity check, or a human pass over a contact sheet, which is cheap.
- **Icons do not match the art direction.** Scenes come back on-style (flat
  colour, heavy black outline); icons come back as glossy gradient clipart. FLUX
  honours the direction on scene prompts and ignores it on isolated-object
  prompts. `bus` and `window` landed; `car`, `truck`, `child` did not. Likely fix:
  generate icons in the scene style and cut them out, rather than prompting for an
  isolated object.
- **Text cards waste the frame.** Headline and caption are the same sentence, the
  headline truncated mid-clause, most of the frame empty.
- **`icon_multiply` fires on sentences that are not about counts.** "An
  evenly-distributed vertical force is applied" rendered as 1 belt = 13 belts. The
  `_countable` test needs to require a countable noun, not just a number.

## 4. Order to do it in

1. Icon style pass + identity check. Cheap, and it fixes every future video.
2. Invert the archetype default and ration the mix. This is the colour fix.
3. Per-segment scene generation from `image_brief`.
4. Pacing: gap, length_scale, motion ration, intra-sentence cuts.
5. Text card layout: split headline from caption, use the frame.

One render is ~22 minutes, so land as many of these as possible before spending it.

## Not to be confused with

The **script** is done and approved — see spec section 11 for the Veritasium
template and `script/retention.py` for the recitation limit that enforces it.
Nothing in this document is about the writing.
