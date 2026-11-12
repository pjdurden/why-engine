# Shot Composition Implementation Plan (Plan 2B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a shot archetype plus its slots into a finished 1920x1080 frame, so that Plan 2C has only to choose archetypes, narrate and assemble.

**Architecture:** Eleven archetypes measured from the reference video. Nine are composed locally in Pillow from a reusable icon and cast library plus hand-lettered text; only `painted_scene` and `archival_plate` need generating or fetching per video. Composition is free, deterministic and drift-proof, which is what makes a 238-shot video affordable.

**Tech Stack:** Python 3.10+, Pillow, numpy, pytest. Fonts: Permanent Marker (Apache 2.0) and Architects Daughter (SIL OFL), both vendored in `assets/fonts/`.

**Spec:** `docs/superpowers/specs/2026-09-16-why-engine-design.md` (sections 5, 7.6-7.13, 9, 9a)

## Global Constraints

- Python >= 3.10, Pydantic v2. Run pytest as `.venv/bin/python -m pytest`, never bare `pytest`.
- Do not import `anthropic`. Nothing under `art/` ever does.
- No test may call a paid API or touch the network.
- **Frame size is exactly 1920x1080.** Every archetype returns that size.
- **Colour comes from the palette module, never from a literal in an archetype.** Generators drift on colour; composition must not.
- **Text is hand-lettered:** `ArchitectsDaughter-Regular.ttf` for captions and labels, `PermanentMarker-Regular.ttf` for headlines and emphasis. Never a system sans.
- **Measured palette** (sampled from 20 reference frames): paper `#FDFDFD`, ink `#1A1A1A`, emphasis red `#C8262A`, earth `#AB8861`, foliage `#92A767`, warm `#CC6B2C`, amber `#F6A135`, slate `#657A73`.
- The monochrome threshold from Plan 2A is **removed** in Task 1. The displacement warp and stroke variation survive.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/why_engine/art/palette.py` | the named colour contract; the only place colours are defined |
| `src/why_engine/art/humanise.py` | MODIFIED: warp and stroke retained, threshold removed |
| `src/why_engine/art/text.py` | hand-lettered text rendering, wrapping, emphasis, underline |
| `src/why_engine/art/icons.py` | icon library manifest and build |
| `src/why_engine/art/compose.py` | canvas, slot placement, archetype dispatch |
| `src/why_engine/art/archetypes.py` | the eleven archetype renderers |

Split rationale: `palette` and `text` are pure and have no dependencies, so every other module can use them freely. `archetypes` is the only file that grows as new shot kinds are added, and it depends on everything else rather than the reverse.

---

### Task 1: Colour palette and a colour-preserving humanise

**Files:**
- Create: `src/why_engine/art/palette.py`
- Modify: `src/why_engine/art/humanise.py`
- Test: `tests/test_palette.py`
- Test: `tests/test_humanise.py` (modify)

**Interfaces:**
- Consumes: nothing
- Produces: `PAPER`, `INK`, `RED`, `EARTH`, `FOLIAGE`, `WARM`, `AMBER`, `SLATE` as `(r,g,b)` tuples; `PALETTE: dict[str, tuple]`; `nearest(rgb) -> str`; and from humanise, `humanise(img, seed=0) -> Image.Image` now preserving colour

The threshold that guaranteed monochrome is removed. It was correct for an ink-only world and is incompatible with this direction. What survives is the warp, which is what keeps the line hand-made.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_palette.py
import pytest
from PIL import Image

from why_engine.art.palette import AMBER, EARTH, INK, PALETTE, PAPER, RED, nearest


def test_every_colour_is_an_rgb_triple():
    assert all(len(v) == 3 and all(0 <= c <= 255 for c in v) for v in PALETTE.values())


def test_the_measured_colours_are_present():
    """Sampled from 20 frames of the reference video, not invented."""
    assert PAPER == (253, 253, 253)
    assert INK == (26, 26, 26)
    assert RED == (200, 38, 42)
    assert EARTH == (171, 136, 97)


def test_palette_maps_names_to_the_same_tuples():
    assert PALETTE["paper"] == PAPER
    assert PALETTE["red"] == RED


def test_nearest_snaps_an_approximate_colour_to_its_name():
    assert nearest((252, 252, 250)) == "paper"
    assert nearest((205, 40, 45)) == "red"


def test_nearest_is_exact_for_palette_members():
    for name, rgb in PALETTE.items():
        assert nearest(rgb) == name
```

```python
# append to tests/test_humanise.py
def test_humanise_preserves_colour():
    """The monochrome threshold is removed: colour is the art direction now."""
    img = Image.new("RGB", (128, 128), (200, 38, 42))
    out = humanise(img, seed=1).convert("RGB")
    px = out.getpixel((64, 64))
    assert px[0] > px[1] and px[0] > px[2], f"red was flattened to {px}"


def test_humanise_still_returns_the_input_size():
    assert humanise(Image.new("RGB", (576, 1024), (255, 255, 255))).size == (576, 1024)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_palette.py tests/test_humanise.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'why_engine.art.palette'`, and the colour test fails because the current humanise thresholds to black and white.

- [ ] **Step 3: Write minimal implementation**

```python
# src/why_engine/art/palette.py
"""The colour contract.

Sampled from twenty frames of the reference video rather than invented. Colours
live here and nowhere else: generators drift on colour, composition must not.
"""
from typing import Dict, Tuple

RGB = Tuple[int, int, int]

PAPER: RGB = (253, 253, 253)
INK: RGB = (26, 26, 26)
RED: RGB = (200, 38, 42)      # emphasis only
EARTH: RGB = (171, 136, 97)
FOLIAGE: RGB = (146, 167, 103)
WARM: RGB = (204, 107, 44)
AMBER: RGB = (246, 161, 53)
SLATE: RGB = (101, 122, 115)

PALETTE: Dict[str, RGB] = {
    "paper": PAPER, "ink": INK, "red": RED, "earth": EARTH,
    "foliage": FOLIAGE, "warm": WARM, "amber": AMBER, "slate": SLATE,
}


def nearest(rgb: RGB) -> str:
    """Name the closest palette entry by squared euclidean distance."""
    return min(PALETTE, key=lambda n: sum((a - b) ** 2 for a, b in zip(PALETTE[n], rgb)))
```

In `src/why_engine/art/humanise.py`, remove the final threshold and keep the warp. Replace the body of `humanise` after the rotation with:

```python
    cropped = rotated.crop((off_x, off_y, off_x + out_w, off_y + out_h))
    return cropped
```

and change the working image from `"L"` to `"RGB"` throughout: build `square` as `Image.new("RGB", (side, side), PAPER)`, warp each channel with the same displacement field so colours do not separate, and drop `is_monochrome` from the return path. Keep `is_monochrome` itself — Plan 2A's library verifier still imports it, and it remains a useful check for line-art assets.

Warp all three channels with one shared field:

```python
    arr = np.array(src, dtype=np.uint8)              # (WORK, WORK, 3)
    warped = arr[ys, xs]                             # same indices for every channel
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_palette.py tests/test_humanise.py -q`
Expected: PASS. Note `test_output_is_strictly_black_and_white` and `test_colour_is_removed_not_merely_reduced` from Plan 2A will now fail — they assert the removed guarantee. **Delete those two tests and say so in your report;** they encode a decision that has been reversed. Do not weaken any other test.

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
git add src/why_engine/art/palette.py src/why_engine/art/humanise.py tests/test_palette.py tests/test_humanise.py
git commit -m "feat: colour palette contract and colour-preserving humanise"
```

---

### Task 2: Hand-lettered text renderer

**Files:**
- Create: `src/why_engine/art/text.py`
- Test: `tests/test_text.py`

**Interfaces:**
- Consumes: `palette` (Task 1)
- Produces: `HEADLINE_FONT`, `CAPTION_FONT` paths; `load_font(path, size)`; `measure(text, font) -> (w, h)`; `draw_caps(draw, xy, text, font, fill, anchor="mm")`; `draw_headline(img, text, y, colour=INK, underline=False)`; `wrap_caps(text, font, max_width) -> list[str]`

Text is the primary information channel in the reference — it appears on nearly every frame. This module is used by every archetype.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_text.py
import pytest
from PIL import Image, ImageDraw

from why_engine.art.palette import INK, RED
from why_engine.art.text import (
    CAPTION_FONT, HEADLINE_FONT, draw_caps, draw_headline, load_font, measure, wrap_caps,
)


def test_both_fonts_are_vendored_and_loadable():
    """A system sans would break the hand-drawn register."""
    assert load_font(HEADLINE_FONT, 40) is not None
    assert load_font(CAPTION_FONT, 40) is not None


def test_measure_returns_positive_dimensions():
    w, h = measure("HELLO", load_font(CAPTION_FONT, 40))
    assert w > 0 and h > 0


def test_bigger_text_measures_wider():
    f_small, f_big = load_font(CAPTION_FONT, 20), load_font(CAPTION_FONT, 60)
    assert measure("HELLO", f_big)[0] > measure("HELLO", f_small)[0]


def test_draw_caps_uppercases_its_input():
    """The reference letters everything in caps."""
    img = Image.new("RGB", (400, 120), (255, 255, 255))
    d = ImageDraw.Draw(img)
    draw_caps(d, (200, 60), "hello", load_font(CAPTION_FONT, 48), INK)
    assert img.getcolors(200000) is not None
    assert any(c[1] != (255, 255, 255) for c in img.getcolors(200000))


def test_wrap_caps_splits_on_width():
    f = load_font(CAPTION_FONT, 40)
    lines = wrap_caps("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG", f, 300)
    assert len(lines) > 1
    assert all(measure(l, f)[0] <= 300 for l in lines)


def test_wrap_caps_keeps_a_short_string_on_one_line():
    f = load_font(CAPTION_FONT, 40)
    assert len(wrap_caps("SHORT", f, 800)) == 1


def test_wrap_caps_does_not_lose_words():
    f = load_font(CAPTION_FONT, 40)
    src = "ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT"
    assert " ".join(wrap_caps(src, f, 240)).split() == src.split()


def test_draw_headline_marks_the_canvas_in_the_given_colour():
    img = Image.new("RGB", (900, 300), (253, 253, 253))
    draw_headline(img, "SOMEONE TRIED", 80, colour=RED)
    assert any(abs(c[1][0] - RED[0]) < 40 and c[1][1] < 90 for c in img.getcolors(300000))


def test_draw_headline_underline_adds_ink_below_the_text():
    plain = Image.new("RGB", (900, 300), (253, 253, 253))
    lined = Image.new("RGB", (900, 300), (253, 253, 253))
    draw_headline(plain, "HIPPO WINS", 80, colour=RED, underline=False)
    draw_headline(lined, "HIPPO WINS", 80, colour=RED, underline=True)
    non_paper = lambda im: sum(n for n, c in im.getcolors(300000) if c != (253, 253, 253))
    assert non_paper(lined) > non_paper(plain)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_text.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'why_engine.art.text'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/why_engine/art/text.py
"""Hand-lettered text.

Text appears on nearly every frame of the reference and carries most of the
information. Two faces, two roles, matching how the reference uses them:
Architects Daughter for captions and labels, Permanent Marker for headlines and
emphasis. A system sans would break the hand-drawn register.
"""
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

from why_engine.art.palette import INK, RGB

FONT_DIR = Path(__file__).resolve().parents[3] / "assets" / "fonts"
HEADLINE_FONT = FONT_DIR / "PermanentMarker-Regular.ttf"
CAPTION_FONT = FONT_DIR / "ArchitectsDaughter-Regular.ttf"


def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def measure(text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    box = font.getbbox(text.upper())
    return box[2] - box[0], box[3] - box[1]


def draw_caps(draw: ImageDraw.ImageDraw, xy, text: str, font, fill: RGB, anchor: str = "mm") -> None:
    draw.text(xy, text.upper(), font=font, fill=fill, anchor=anchor)


def wrap_caps(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    words, lines, current = text.upper().split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if current and measure(trial, font)[0] > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def draw_headline(img: Image.Image, text: str, y: int, colour: RGB = INK,
                  size: int = 96, underline: bool = False) -> int:
    """Centred headline. Returns the y below the text, so callers can stack."""
    draw = ImageDraw.Draw(img)
    font = load_font(HEADLINE_FONT, size)
    w, h = measure(text, font)
    cx = img.width // 2
    draw_caps(draw, (cx, y), text, font, colour, anchor="mm")
    if underline:
        half = w // 2 + 12
        draw.line([(cx - half, y + h // 2 + 16), (cx + half, y + h // 2 + 16)],
                  fill=colour, width=7)
    return y + h // 2 + 30
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_text.py -q`
Expected: PASS, 9 passed

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
git add src/why_engine/art/text.py tests/test_text.py
git commit -m "feat: hand-lettered text rendering with wrapping and emphasis"
```

---

### Task 3: Icon library

**Files:**
- Create: `src/why_engine/art/icons.py`
- Test: `tests/test_icons.py`

**Interfaces:**
- Consumes: `Asset`, `build_library`, `verify_library` (Plan 2A `library.py`), `FalClient`, `humanise`
- Produces: `ICON_STYLE_SUFFIX`; `ICONS: List[Asset]`; `build_icons(client, out_dir) -> List[Path]`; `icon_path(name, out_dir) -> Path`

Icons are what make 238 shots affordable: generated once, recombined into hundreds of layouts, amortised across every video. Distinct from the cast library, which holds the two recurring characters.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_icons.py
import io
import numpy as np
import pytest
from PIL import Image

from why_engine.art.fal import PORTRAIT, FalClient
from why_engine.art.icons import ICON_STYLE_SUFFIX, ICONS, build_icons, icon_path


def _png_bytes():
    a = np.full((64, 64, 3), 255, dtype=np.uint8)
    a[10:50, 10:50] = (146, 167, 103)
    buf = io.BytesIO()
    Image.fromarray(a, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _client():
    def transport(*, url, headers, payload=None, download=False):
        return _png_bytes() if download else {"images": [{"url": "https://fal.media/x.png"}]}
    return FalClient("k", transport=transport)


def test_the_icon_set_covers_the_archetypes_that_need_it():
    names = {i.name for i in ICONS}
    for required in ("icon_check", "icon_cross", "icon_arrow_right", "icon_cow", "icon_map_world"):
        assert required in names, f"missing {required}"


def test_icon_names_are_unique():
    names = [i.name for i in ICONS]
    assert len(names) == len(set(names))


def test_the_icon_style_suffix_asks_for_flat_colour_and_bold_outline():
    low = ICON_STYLE_SUFFIX.lower()
    assert "flat" in low
    assert "outline" in low
    assert "white background" in low


def test_build_icons_writes_one_png_per_icon(tmp_path, monkeypatch):
    monkeypatch.setattr("why_engine.art.humanise.WORK", 192)
    subset = ICONS[:3]
    paths = build_icons(_client(), tmp_path, icons=subset)
    assert len(paths) == 3
    assert all(p.exists() for p in paths)


def test_built_icons_keep_their_colour(tmp_path, monkeypatch):
    """The monochrome threshold is gone; icons are flat colour now."""
    monkeypatch.setattr("why_engine.art.humanise.WORK", 192)
    p = build_icons(_client(), tmp_path, icons=ICONS[:1])[0]
    a = np.array(Image.open(p).convert("RGB"), dtype=np.int16)
    spread = (a.max(axis=2) - a.min(axis=2)).max()
    assert spread > 20, "icon came out greyscale"


def test_build_icons_skips_what_already_exists(tmp_path, monkeypatch):
    monkeypatch.setattr("why_engine.art.humanise.WORK", 192)
    client = _client()
    build_icons(client, tmp_path, icons=ICONS[:1])
    first = client.image_count
    build_icons(client, tmp_path, icons=ICONS[:1])
    assert client.image_count == first


def test_icon_path_resolves_by_name(tmp_path):
    assert icon_path("icon_check", tmp_path) == tmp_path / "icon_check.png"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_icons.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'why_engine.art.icons'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/why_engine/art/icons.py
"""The reusable icon set.

Generated once and recombined into hundreds of layouts. This is what makes a
238-shot video cost roughly sixty generations rather than two hundred and
thirty-eight.
"""
from pathlib import Path
from typing import List, Optional, Sequence

from why_engine.art.fal import PORTRAIT, FalClient
from why_engine.art.library import Asset, build_library

ICON_STYLE_SUFFIX = (
    ", single centred icon, flat colour fill, bold black outline, simple cartoon, "
    "no text, no shadow, plain white background"
)


def _icon(name: str, prompt: str) -> Asset:
    return Asset(name=f"icon_{name}", kind="prop", prompt=prompt + ICON_STYLE_SUFFIX, size=PORTRAIT)


ICONS: List[Asset] = [
    _icon("check", "a thick green tick mark"),
    _icon("cross", "a thick red cross mark"),
    _icon("arrow_right", "a bold red arrow pointing right"),
    _icon("question", "a bold red question mark"),
    _icon("cow", "a cartoon cow standing side on"),
    _icon("pig", "a cartoon pig standing side on"),
    _icon("dog", "a cartoon dog sitting"),
    _icon("lion", "a cartoon lion standing side on"),
    _icon("crocodile", "a cartoon crocodile side on"),
    _icon("hippo", "a cartoon hippopotamus standing side on"),
    _icon("fish", "a cartoon fish side on"),
    _icon("map_world", "a simple flat world map"),
    _icon("map_pin", "a red map pin"),
    _icon("grass", "a tuft of green grass"),
    _icon("sack", "a tied cloth sack of grain"),
    _icon("coin", "a stack of gold coins"),
    _icon("document", "a sheet of paper with writing lines"),
    _icon("clock", "a simple round clock face"),
    _icon("factory", "a small factory with a chimney"),
    _icon("house", "a simple suburban house"),
]


def icon_path(name: str, out_dir: Path) -> Path:
    return Path(out_dir) / f"{name}.png"


def build_icons(client: FalClient, out_dir: Path,
                icons: Optional[Sequence[Asset]] = None) -> List[Path]:
    return build_library(list(icons if icons is not None else ICONS), client, Path(out_dir))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_icons.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
git add src/why_engine/art/icons.py tests/test_icons.py
git commit -m "feat: reusable icon library for composed archetypes"
```

---

### Task 4: Composer core

**Files:**
- Create: `src/why_engine/art/compose.py`
- Test: `tests/test_compose.py`

**Interfaces:**
- Consumes: `palette` (Task 1), `text` (Task 2)
- Produces: `FRAME_W = 1920`, `FRAME_H = 1080`; `canvas(bg=PAPER) -> Image.Image`; `paste_icon(canvas, path, cx, cy, height) -> None`; `row(canvas, paths, cy, height, gap) -> None`; `grid(canvas, paths, cx, cy, cols, height, gap) -> None`

Every archetype builds on these. Keeping placement here rather than in each archetype is what stops eleven renderers each inventing their own spacing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compose.py
import numpy as np
import pytest
from PIL import Image

from why_engine.art.compose import FRAME_H, FRAME_W, canvas, grid, paste_icon, row
from why_engine.art.palette import PAPER


@pytest.fixture
def icon(tmp_path):
    p = tmp_path / "i.png"
    a = np.full((80, 80, 3), 255, dtype=np.uint8)
    a[10:70, 10:70] = (200, 38, 42)
    Image.fromarray(a, "RGB").save(p)
    return p


def _ink(img):
    a = np.array(img.convert("RGB"), dtype=np.int16)
    return int((np.abs(a - np.array(PAPER)).sum(axis=2) > 40).sum())


def test_canvas_is_exactly_frame_sized():
    assert canvas().size == (FRAME_W, FRAME_H)


def test_frame_is_1920_by_1080():
    assert (FRAME_W, FRAME_H) == (1920, 1080)


def test_canvas_starts_as_paper():
    assert canvas().getpixel((5, 5)) == PAPER


def test_paste_icon_marks_the_canvas(icon):
    c = canvas()
    before = _ink(c)
    paste_icon(c, icon, 960, 540, 300)
    assert _ink(c) > before


def test_paste_icon_centres_on_the_given_point(icon):
    c = canvas()
    paste_icon(c, icon, 480, 300, 200)
    a = np.array(c.convert("RGB"), dtype=np.int16)
    mask = np.abs(a - np.array(PAPER)).sum(axis=2) > 40
    ys, xs = np.nonzero(mask)
    assert abs(int(xs.mean()) - 480) < 25
    assert abs(int(ys.mean()) - 300) < 25


def test_row_places_every_icon(icon):
    c = canvas()
    row(c, [icon] * 4, cy=540, height=180, gap=40)
    a = np.array(c.convert("RGB"), dtype=np.int16)
    mask = np.abs(a - np.array(PAPER)).sum(axis=2) > 40
    xs = np.nonzero(mask)[1]
    assert xs.max() - xs.min() > 500, "icons were not spread across a row"


def test_row_is_horizontally_centred(icon):
    c = canvas()
    row(c, [icon] * 3, cy=540, height=180, gap=40)
    a = np.array(c.convert("RGB"), dtype=np.int16)
    xs = np.nonzero(np.abs(a - np.array(PAPER)).sum(axis=2) > 40)[1]
    assert abs(int((xs.min() + xs.max()) / 2) - FRAME_W // 2) < 30


def test_grid_wraps_onto_multiple_lines(icon):
    c = canvas()
    grid(c, [icon] * 8, cx=960, cy=540, cols=4, height=120, gap=30)
    a = np.array(c.convert("RGB"), dtype=np.int16)
    ys = np.nonzero(np.abs(a - np.array(PAPER)).sum(axis=2) > 40)[0]
    assert ys.max() - ys.min() > 150, "grid did not wrap to a second row"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_compose.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'why_engine.art.compose'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/why_engine/art/compose.py
"""Canvas and placement primitives shared by every archetype."""
from pathlib import Path
from typing import Sequence

from PIL import Image

from why_engine.art.palette import PAPER, RGB

FRAME_W, FRAME_H = 1920, 1080


def canvas(bg: RGB = PAPER) -> Image.Image:
    return Image.new("RGB", (FRAME_W, FRAME_H), bg)


def _scaled(path: Path, height: int) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    w = max(1, int(img.width * height / img.height))
    return img.resize((w, height), Image.LANCZOS)


def paste_icon(surface: Image.Image, path: Path, cx: int, cy: int, height: int) -> None:
    """Place an icon centred on (cx, cy), dropping near-white to transparent so it
    sits on any background."""
    icon = _scaled(Path(path), height)
    px = icon.load()
    for y in range(icon.height):
        for x in range(icon.width):
            r, g, b, a = px[x, y]
            if r > 238 and g > 238 and b > 238:
                px[x, y] = (r, g, b, 0)
    surface.paste(icon, (int(cx - icon.width / 2), int(cy - icon.height / 2)), icon)


def row(surface: Image.Image, paths: Sequence[Path], cy: int, height: int, gap: int = 40) -> None:
    widths = [_scaled(Path(p), height).width for p in paths]
    total = sum(widths) + gap * (len(paths) - 1)
    x = (FRAME_W - total) // 2
    for p, w in zip(paths, widths):
        paste_icon(surface, p, x + w // 2, cy, height)
        x += w + gap


def grid(surface: Image.Image, paths: Sequence[Path], cx: int, cy: int,
         cols: int, height: int, gap: int = 30) -> None:
    rows = (len(paths) + cols - 1) // cols
    cell_w = _scaled(Path(paths[0]), height).width + gap
    cell_h = height + gap
    top = cy - (rows * cell_h) // 2 + cell_h // 2
    for i, p in enumerate(paths):
        r, c = divmod(i, cols)
        n_in_row = min(cols, len(paths) - r * cols)
        left = cx - (n_in_row * cell_w) // 2 + cell_w // 2
        paste_icon(surface, p, left + c * cell_w, top + r * cell_h, height)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_compose.py -q`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
git add src/why_engine/art/compose.py tests/test_compose.py
git commit -m "feat: canvas and placement primitives for shot composition"
```

---

### Task 5: Archetypes, first group

**Files:**
- Create: `src/why_engine/art/archetypes.py`
- Test: `tests/test_archetypes.py`

**Interfaces:**
- Consumes: `compose`, `text`, `palette`, `icons`
- Produces: `Shot(BaseModel)` with `archetype: str`, `slots: dict`; `render(shot, icon_dir) -> Image.Image`; `ARCHETYPES: dict[str, Callable]`; renderers `text_card`, `icon_multiply`, `ranked_row`, `measure_compare`

Each renderer takes the canvas and a slots dict and returns a frame. `render` dispatches and is the only entry point Plan 2C will use.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_archetypes.py
import numpy as np
import pytest
from PIL import Image

from why_engine.art.archetypes import ARCHETYPES, Shot, render
from why_engine.art.compose import FRAME_H, FRAME_W
from why_engine.art.palette import PAPER, RED


@pytest.fixture
def icon_dir(tmp_path):
    for name in ("icon_hippo", "icon_pig", "icon_lion", "icon_cow", "icon_check"):
        a = np.full((80, 80, 3), 255, dtype=np.uint8)
        a[12:68, 12:68] = (146, 167, 103)
        Image.fromarray(a, "RGB").save(tmp_path / f"{name}.png")
    return tmp_path


def _ink(img):
    a = np.array(img.convert("RGB"), dtype=np.int16)
    return int((np.abs(a - np.array(PAPER)).sum(axis=2) > 40).sum())


def _has_red(img):
    a = np.array(img.convert("RGB"), dtype=np.int16)
    return bool(((np.abs(a - np.array(RED)).sum(axis=2)) < 90).any())


def test_every_registered_archetype_renders_a_full_frame(icon_dir):
    slots = {
        "text_card": {"headline": "SOMEONE TRIED", "subhead": "IT WAS NOT FOR LACK OF TRYING"},
        "icon_multiply": {"subject": "icon_hippo", "icon": "icon_pig", "count": 12,
                          "caption": "MORE MEAT THAN 30 PIGS"},
        "ranked_row": {"subject": "icon_hippo", "rank": 1,
                       "others": ["icon_lion", "icon_cow"], "caption": "MOST DANGEROUS"},
        "measure_compare": {"left": "icon_hippo", "left_value": "30 MPH",
                            "right": "icon_cow", "right_value": "28 MPH", "verdict": "HIPPO WINS"},
    }
    for name, s in slots.items():
        img = render(Shot(archetype=name, slots=s), icon_dir)
        assert img.size == (FRAME_W, FRAME_H), name
        assert _ink(img) > 4000, f"{name} rendered a near-empty frame"


def test_text_card_puts_the_headline_in_emphasis_red(icon_dir):
    img = render(Shot(archetype="text_card",
                      slots={"headline": "SOMEONE TRIED", "subhead": "SUB"}), icon_dir)
    assert _has_red(img)


def test_icon_multiply_draws_the_requested_count(icon_dir):
    few = render(Shot(archetype="icon_multiply", slots={"subject": "icon_hippo",
                 "icon": "icon_pig", "count": 3, "caption": "C"}), icon_dir)
    many = render(Shot(archetype="icon_multiply", slots={"subject": "icon_hippo",
                  "icon": "icon_pig", "count": 18, "caption": "C"}), icon_dir)
    assert _ink(many) > _ink(few) * 1.5


def test_ranked_row_marks_the_rank_in_red(icon_dir):
    img = render(Shot(archetype="ranked_row", slots={"subject": "icon_hippo", "rank": 1,
                 "others": ["icon_lion"], "caption": "MOST DANGEROUS"}), icon_dir)
    assert _has_red(img)


def test_measure_compare_shows_both_values(icon_dir):
    img = render(Shot(archetype="measure_compare", slots={"left": "icon_hippo",
                 "left_value": "30 MPH", "right": "icon_cow", "right_value": "28 MPH",
                 "verdict": "HIPPO WINS"}), icon_dir)
    assert _ink(img) > 8000


def test_an_unknown_archetype_raises():
    with pytest.raises(KeyError):
        render(Shot(archetype="does_not_exist", slots={}), "/tmp")


def test_archetypes_registry_matches_the_renderers():
    for name in ("text_card", "icon_multiply", "ranked_row", "measure_compare"):
        assert name in ARCHETYPES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_archetypes.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'why_engine.art.archetypes'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/why_engine/art/archetypes.py
"""The shot archetypes, measured from the reference video.

Nine of the eleven are composed locally from icons and text at zero marginal
cost. Only painted_scene and archival_plate need generating or fetching.
"""
from pathlib import Path
from typing import Callable, Dict, List

from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from why_engine.art.compose import FRAME_H, FRAME_W, canvas, grid, paste_icon, row
from why_engine.art.palette import INK, PAPER, RED
from why_engine.art.text import CAPTION_FONT, draw_caps, draw_headline, load_font


class Shot(BaseModel):
    archetype: str = Field(min_length=1)
    slots: Dict = Field(default_factory=dict)


def _icon(icon_dir, name: str) -> Path:
    return Path(icon_dir) / f"{name}.png"


def _caption(img: Image.Image, text: str, y: int, size: int = 62, colour=INK) -> None:
    draw = ImageDraw.Draw(img)
    draw_caps(draw, (FRAME_W // 2, y), text, load_font(CAPTION_FONT, size), colour)


def text_card(slots, icon_dir) -> Image.Image:
    img = canvas()
    draw_headline(img, slots["headline"], 420, colour=RED, size=120, underline=True)
    if slots.get("subhead"):
        _caption(img, slots["subhead"], 560, size=54)
    return img


def icon_multiply(slots, icon_dir) -> Image.Image:
    img = canvas()
    paste_icon(img, _icon(icon_dir, slots["subject"]), 420, 460, 320)
    draw = ImageDraw.Draw(img)
    draw_caps(draw, (700, 460), "=", load_font(CAPTION_FONT, 130), INK)
    grid(img, [_icon(icon_dir, slots["icon"])] * int(slots["count"]),
         cx=1290, cy=460, cols=6, height=110, gap=18)
    _caption(img, slots["caption"], 860)
    return img


def ranked_row(slots, icon_dir) -> Image.Image:
    img = canvas()
    _caption(img, slots["caption"], 250, size=70, colour=RED)
    paste_icon(img, _icon(icon_dir, slots["subject"]), 560, 580, 300)
    draw = ImageDraw.Draw(img)
    draw.ellipse([760, 500, 920, 660], fill=RED)
    draw_caps(draw, (840, 580), str(slots["rank"]), load_font(CAPTION_FONT, 110), PAPER)
    others = [_icon(icon_dir, n) for n in slots["others"]]
    x = 1080
    for p in others:
        paste_icon(img, p, x, 580, 230)
        x += 260
    return img


def measure_compare(slots, icon_dir) -> Image.Image:
    img = canvas()
    paste_icon(img, _icon(icon_dir, slots["left"]), 520, 450, 300)
    paste_icon(img, _icon(icon_dir, slots["right"]), 1400, 450, 300)
    draw = ImageDraw.Draw(img)
    draw.line([(160, 690), (1760, 690)], fill=INK, width=8)
    draw.rectangle([380, 720, 660, 800], fill=RED)
    draw_caps(draw, (520, 760), slots["left_value"], load_font(CAPTION_FONT, 56), PAPER)
    draw_caps(draw, (1400, 760), slots["right_value"], load_font(CAPTION_FONT, 56), INK)
    draw_headline(img, slots["verdict"], 920, colour=RED, size=86)
    return img


ARCHETYPES: Dict[str, Callable] = {
    "text_card": text_card,
    "icon_multiply": icon_multiply,
    "ranked_row": ranked_row,
    "measure_compare": measure_compare,
}


def render(shot: Shot, icon_dir) -> Image.Image:
    return ARCHETYPES[shot.archetype](shot.slots, icon_dir)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_archetypes.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
git add src/why_engine/art/archetypes.py tests/test_archetypes.py
git commit -m "feat: first four shot archetypes"
```

---

### Task 6: Archetypes, second group

**Files:**
- Modify: `src/why_engine/art/archetypes.py`
- Test: `tests/test_archetypes.py`

**Interfaces:**
- Consumes: everything from Task 5
- Produces: renderers `then_now`, `quantity_viz`, `numbered_grid`, `versus_split`, `map_annotate`, `archival_plate`, all registered in `ARCHETYPES`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_archetypes.py
def test_the_second_group_renders_full_frames(icon_dir, tmp_path):
    photo = tmp_path / "photo.png"
    a = np.random.default_rng(3).integers(60, 200, (300, 500, 3), dtype=np.uint8)
    Image.fromarray(a, "RGB").save(photo)
    slots = {
        "then_now": {"then": "icon_cow", "now": "icon_hippo",
                     "then_label": "THEN.", "now_label": "NOW."},
        "quantity_viz": {"many": 40, "few": 4,
                         "many_label": "DOMESTICATED", "few_label": "FAILED"},
        "numbered_grid": {"cells": ["STRESS", "CALM", "HEART", "MIND", "PANIC", "RELAXED"],
                          "highlight": 4},
        "versus_split": {"left": "icon_hippo", "right": "icon_cow", "winner": "right"},
        "map_annotate": {"map": "icon_map_world", "label": "ASIA"},
        "archival_plate": {"image": str(photo), "caption": "US FISH COMMISSION, 1877"},
    }
    for name, s in slots.items():
        img = render(Shot(archetype=name, slots=s), icon_dir)
        assert img.size == (FRAME_W, FRAME_H), name
        assert _ink(img) > 4000, f"{name} rendered a near-empty frame"


def test_quantity_viz_draws_more_marks_for_the_larger_cluster(icon_dir):
    img = render(Shot(archetype="quantity_viz", slots={"many": 60, "few": 3,
                 "many_label": "A", "few_label": "B"}), icon_dir)
    a = np.array(img.convert("RGB"), dtype=np.int16)
    mask = np.abs(a - np.array(PAPER)).sum(axis=2) > 40
    left, right = mask[:, : FRAME_W // 2].sum(), mask[:, FRAME_W // 2 :].sum()
    assert left > right, "the larger cluster should carry more ink"


def test_numbered_grid_highlights_exactly_one_cell(icon_dir):
    plain = render(Shot(archetype="numbered_grid",
                   slots={"cells": ["A"] * 6, "highlight": None}), icon_dir)
    marked = render(Shot(archetype="numbered_grid",
                    slots={"cells": ["A"] * 6, "highlight": 2}), icon_dir)
    assert _has_red(marked) and not _has_red(plain)


def test_versus_split_marks_the_winner_side(icon_dir):
    img = render(Shot(archetype="versus_split", slots={"left": "icon_hippo",
                 "right": "icon_cow", "winner": "right"}), icon_dir)
    assert _has_red(img)


def test_archival_plate_keeps_the_photograph_tonal(icon_dir, tmp_path):
    """The photo is the only thing on screen carrying continuous tone."""
    photo = tmp_path / "p.png"
    a = np.random.default_rng(5).integers(40, 210, (300, 500, 3), dtype=np.uint8)
    Image.fromarray(a, "RGB").save(photo)
    img = render(Shot(archetype="archival_plate",
                 slots={"image": str(photo), "caption": "PLATE"}), icon_dir)
    arr = np.array(img.convert("L"))
    mid = ((arr > 70) & (arr < 190)).sum()
    assert mid > 20000, "the photograph lost its midtones"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_archetypes.py -q`
Expected: FAIL with `KeyError: 'then_now'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/why_engine/art/archetypes.py`:

```python
def then_now(slots, icon_dir) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    for i, (key, label, cx) in enumerate(
        [("then", slots["then_label"], 530), ("now", slots["now_label"], 1390)]
    ):
        draw.rectangle([cx - 380, 300, cx + 380, 860], outline=INK, width=8)
        paste_icon(img, _icon(icon_dir, slots[key]), cx, 600, 300)
        draw_caps(draw, (cx, 230), label, load_font(CAPTION_FONT, 72), INK)
    draw.polygon([(940, 570), (980, 570), (980, 550), (1030, 585),
                  (980, 620), (980, 600), (940, 600)], fill=INK)
    return img


def quantity_viz(slots, icon_dir) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    rng = __import__("random").Random(7)
    for n, cx, colour, label in [
        (int(slots["many"]), 520, (110, 168, 92), slots["many_label"]),
        (int(slots["few"]), 1400, (170, 170, 170), slots["few_label"]),
    ]:
        for _ in range(n):
            x = cx + rng.randint(-300, 300)
            y = 520 + rng.randint(-250, 250)
            draw.ellipse([x - 22, y - 22, x + 22, y + 22], fill=colour)
        draw_caps(draw, (cx, 880), label, load_font(CAPTION_FONT, 62), INK)
    return img


def numbered_grid(slots, icon_dir) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    cells: List[str] = list(slots["cells"])
    highlight = slots.get("highlight")
    cw, ch, gap = 520, 300, 40
    for i, label in enumerate(cells[:6]):
        r, c = divmod(i, 3)
        x = 240 + c * (cw + gap)
        y = 260 + r * (ch + gap)
        outline = RED if highlight is not None and i == int(highlight) else INK
        draw.rectangle([x, y, x + cw, y + ch], outline=outline, width=10)
        draw_caps(draw, (x + 40, y + 46), str(i + 1), load_font(CAPTION_FONT, 48), INK, anchor="lm")
        draw_caps(draw, (x + cw // 2, y + ch // 2 + 20), label,
                  load_font(CAPTION_FONT, 52), INK)
    return img


def versus_split(slots, icon_dir) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    draw.line([(960, 180), (960, 900)], fill=INK, width=8)
    paste_icon(img, _icon(icon_dir, slots["left"]), 520, 520, 360)
    paste_icon(img, _icon(icon_dir, slots["right"]), 1400, 520, 360)
    draw_headline(img, "VS", 520, colour=RED, size=110)
    win_cx = 520 if slots["winner"] == "left" else 1400
    draw_caps(draw, (win_cx, 880), "WINNER", load_font(CAPTION_FONT, 64), RED)
    return img


def map_annotate(slots, icon_dir) -> Image.Image:
    img = canvas()
    paste_icon(img, _icon(icon_dir, slots["map"]), 960, 500, 620)
    draw = ImageDraw.Draw(img)
    draw.line([(1300, 780), (1080, 560)], fill=RED, width=10)
    draw.polygon([(1080, 560), (1120, 588), (1112, 548)], fill=RED)
    draw_caps(draw, (1340, 820), slots["label"], load_font(CAPTION_FONT, 62), RED)
    return img


def archival_plate(slots, icon_dir) -> Image.Image:
    """A real photograph, framed as if taped in. It is the only element on screen
    carrying continuous tone, which is what makes it read as real."""
    img = canvas()
    photo = Image.open(slots["image"]).convert("RGB")
    h = 660
    w = max(1, int(photo.width * h / photo.height))
    photo = photo.resize((w, h), Image.LANCZOS)
    framed = Image.new("RGB", (w + 36, h + 36), (252, 252, 250))
    framed.paste(photo, (18, 18))
    rot = framed.rotate(-2.5, expand=True, fillcolor=PAPER, resample=Image.BICUBIC)
    img.paste(rot, (int(960 - rot.width / 2), int(480 - rot.height / 2)))
    if slots.get("caption"):
        _caption(img, slots["caption"], 940, size=50)
    return img


ARCHETYPES.update({
    "then_now": then_now,
    "quantity_viz": quantity_viz,
    "numbered_grid": numbered_grid,
    "versus_split": versus_split,
    "map_annotate": map_annotate,
    "archival_plate": archival_plate,
})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_archetypes.py -q`
Expected: PASS, 12 passed

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
git add src/why_engine/art/archetypes.py tests/test_archetypes.py
git commit -m "feat: remaining shot archetypes including archival plate"
```

---

### Task 7: Archetype contact sheet

**Files:**
- Create: `tools/archetype_sheet.py`
- Test: `tests/test_archetype_sheet.py`

**Interfaces:**
- Consumes: `ARCHETYPES`, `Shot`, `render`
- Produces: `sample_shots() -> list[Shot]`; `build_sheet(icon_dir, out_path) -> Path`

Every archetype rendered on one page, so the look can be approved in a single glance instead of ten. This is the aesthetic gate before Plan 2C builds assembly on top.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_archetype_sheet.py
import numpy as np
import pytest
from PIL import Image

import sys
sys.path.insert(0, "tools")
from archetype_sheet import build_sheet, sample_shots

from why_engine.art.archetypes import ARCHETYPES


@pytest.fixture
def icon_dir(tmp_path):
    for name in ("icon_hippo", "icon_pig", "icon_lion", "icon_cow",
                 "icon_check", "icon_map_world"):
        a = np.full((80, 80, 3), 255, dtype=np.uint8)
        a[12:68, 12:68] = (146, 167, 103)
        Image.fromarray(a, "RGB").save(tmp_path / f"{name}.png")
    return tmp_path


def test_there_is_a_sample_for_every_registered_archetype():
    """A sheet that silently skips an archetype cannot gate the look."""
    assert {s.archetype for s in sample_shots()} >= set(ARCHETYPES) - {"painted_scene"}


def test_build_sheet_writes_an_image(tmp_path, icon_dir):
    out = build_sheet(icon_dir, tmp_path / "sheet.png")
    assert out.exists()
    assert Image.open(out).width > 1000


def test_the_sheet_is_not_blank(tmp_path, icon_dir):
    out = build_sheet(icon_dir, tmp_path / "sheet.png")
    a = np.array(Image.open(out).convert("RGB"), dtype=np.int16)
    assert (np.abs(a - 253).sum(axis=2) > 40).sum() > 50000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_archetype_sheet.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'archetype_sheet'`

- [ ] **Step 3: Write minimal implementation**

```python
# tools/archetype_sheet.py
"""Render every archetype onto one page for visual approval."""
import sys
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from why_engine.art.archetypes import Shot, render  # noqa: E402
from why_engine.art.text import CAPTION_FONT, load_font  # noqa: E402


def sample_shots() -> List[Shot]:
    return [
        Shot(archetype="text_card", slots={"headline": "SOMEONE TRIED",
             "subhead": "IT WAS NOT FOR LACK OF TRYING"}),
        Shot(archetype="icon_multiply", slots={"subject": "icon_hippo", "icon": "icon_pig",
             "count": 30, "caption": "MORE MEAT THAN 30 PIGS"}),
        Shot(archetype="ranked_row", slots={"subject": "icon_hippo", "rank": 1,
             "others": ["icon_lion", "icon_cow"], "caption": "MOST DANGEROUS"}),
        Shot(archetype="measure_compare", slots={"left": "icon_hippo", "left_value": "30 MPH",
             "right": "icon_cow", "right_value": "28 MPH", "verdict": "HIPPO WINS"}),
        Shot(archetype="then_now", slots={"then": "icon_cow", "now": "icon_hippo",
             "then_label": "THEN.", "now_label": "NOW."}),
        Shot(archetype="quantity_viz", slots={"many": 60, "few": 5,
             "many_label": "DOMESTICATED", "few_label": "FAILED"}),
        Shot(archetype="numbered_grid", slots={"cells": ["STRESS", "CALM", "HEART",
             "MIND", "PANIC", "RELAXED"], "highlight": 4}),
        Shot(archetype="versus_split", slots={"left": "icon_hippo", "right": "icon_cow",
             "winner": "right"}),
        Shot(archetype="map_annotate", slots={"map": "icon_map_world", "label": "ASIA"}),
    ]


def build_sheet(icon_dir, out_path) -> Path:
    shots = sample_shots()
    cell_w, cols = 640, 3
    cell_h = int(cell_w * 1080 / 1920)
    rows = (len(shots) + cols - 1) // cols
    sheet = Image.new("RGB", (cell_w * cols, (cell_h + 26) * rows), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    font = load_font(CAPTION_FONT, 22)
    for i, shot in enumerate(shots):
        r, c = divmod(i, cols)
        y = r * (cell_h + 26)
        frame = render(shot, icon_dir).resize((cell_w, cell_h), Image.LANCZOS)
        sheet.paste(frame, (c * cell_w, y + 26))
        d.text((c * cell_w + 8, y + 4), shot.archetype.upper(), font=font, fill=(0, 0, 0))
    out_path = Path(out_path)
    sheet.save(out_path)
    return out_path


if __name__ == "__main__":
    print(build_sheet(sys.argv[1] if len(sys.argv) > 1 else "assets/icons",
                      sys.argv[2] if len(sys.argv) > 2 else "build/archetypes.png"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_archetype_sheet.py -q`
Expected: PASS, 3 passed

- [ ] **Step 5: Run the whole suite and commit**

```bash
cd ~/why-engine
.venv/bin/python -m pytest -q
git add tools/archetype_sheet.py tests/test_archetype_sheet.py
git commit -m "feat: archetype contact sheet for visual approval"
```

---

## Scope note: `painted_scene`

Spec section 9a lists eleven archetypes. This plan implements ten. `painted_scene`
— a full generated establishing scene — is deliberately excluded: it is the only
archetype that needs a per-video generation round trip, so it belongs with
`art_director` in Plan 2C, where shot selection and the generation budget live
together. The contact sheet in Task 7 therefore covers ten, and its test asserts
coverage of `set(ARCHETYPES) - {"painted_scene"}` rather than silently skipping it.

## Font licensing

Both faces are vendored in `assets/fonts/` with their licences alongside:
Architects Daughter under SIL OFL, Permanent Marker under Apache 2.0. Both permit
commercial use, which matters for a monetised channel — the same discipline the
archival licence gate applies to images.

## Definition of Done

- `.venv/bin/python -m pytest` passes with the 6 paid live tests still deselected.
- `.venv/bin/python -m why_engine.art.icons --out assets/icons` builds the icon set for under $0.10.
- `.venv/bin/python tools/archetype_sheet.py assets/icons build/archetypes.png` renders every composed archetype on one page.
- Every archetype returns exactly 1920x1080.
- No colour literal appears in `archetypes.py`; all colour comes from `palette`.
- No module under `art/` imports `anthropic`; no test touches the network.
