# Visual Assets Implementation Plan (Plan 2A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build every visual input a video needs — a reusable hand-drawn asset library, a licence-gated archival image fetcher, and the deterministic style pass both flow through — so that Plan 2B has only to compose, narrate and assemble.

**Architecture:** Two sources feed one style pipeline. Drawings come from fal.ai FLUX.1 schnell; real photographs come from public-domain archives behind a hard licence gate. Everything passes through `humanise`, which thresholds to pure black-on-white and applies a fixed hand-tremor warp, so 200 assets share one visual language by construction rather than by hope.

**Tech Stack:** Python 3.10+, Pillow, numpy, httpx2, fal.ai REST API, Wikimedia Commons API, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-why-engine-design.md` (sections 5, 7.7, 7.8, 7.9, 9)

## Global Constraints

- Python >= 3.10, Pydantic v2. Run pytest as `.venv/bin/python -m pytest`, never bare `pytest`.
- **Local image generation is permanently out of scope.** The machine's integrated Vega GPU has no ROCm compute agent. All generation is API-only.
- `anthropic` is imported only in `src/why_engine/llm.py`. Nothing in `art/` imports it.
- `anthropic` 1.x is built on `httpx2`, not `httpx`. Use `import httpx2 as httpx`.
- No test may call a paid API or the network. Network calls go behind an injectable client, exactly as `ClaudeCodeLLM` takes an injectable `runner`.
- **Archival images: public domain or CC0 only.** Verified before use, no exceptions. A copyrighted image reintroduces the Content ID exposure the original-illustrations rule exists to avoid.
- **The humanise parameters are a fixed style contract**, not per-call tuning: `WOBBLE = 5.0`, `PRESSURE = 0.35`, `BLUR_FRAC = 0.045`, `WORK = 1536`. Chosen from a measured contact sheet. Do not expose them as routine call-site arguments.
- fal.ai bills per megapixel **rounded up**. `portrait_16_9` (576x1024) and `landscape_16_9` (1024x576) are both under 1 MP and bill at $0.003. `square_hd` (1024x1024) is 1.05 MP and bills at $0.006 for less usable area. Never use `square_hd`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/why_engine/art/__init__.py` | package marker |
| `src/why_engine/art/humanise.py` | the deterministic style pass; the only place style parameters live |
| `src/why_engine/art/fal.py` | fal.ai generation client, injectable transport, spend counter |
| `src/why_engine/art/library.py` | asset manifest, prompt construction, library build and integrity check |
| `src/why_engine/art/archival.py` | Wikimedia search, licence gate, fetch with provenance |

Split rationale: `humanise` is pure and has no network; `fal` and `archival` are the two IO boundaries and fail for unrelated reasons; `library` is orchestration over all three.

---

### Task 1: The humanise style pass

**Files:**
- Create: `src/why_engine/art/__init__.py`
- Create: `src/why_engine/art/humanise.py`
- Test: `tests/test_humanise.py`

**Interfaces:**
- Consumes: nothing
- Produces: `WOBBLE`, `PRESSURE`, `BLUR_FRAC`, `WORK` constants; `humanise(img: Image.Image, seed: int = 0) -> Image.Image`; `is_monochrome(img: Image.Image) -> bool`

Every asset in the project passes through this. The threshold is a guarantee, not a tolerance: FLUX leaks faint colour (measured pink cheek blush) and prompting against it failed, so it is removed mechanically.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_humanise.py
import numpy as np
import pytest
from PIL import Image

from why_engine.art.humanise import BLUR_FRAC, PRESSURE, WOBBLE, humanise, is_monochrome


def _drawing(size=256):
    """A white canvas with a black cross: survives thresholding, shows warping."""
    a = np.full((size, size), 255, dtype=np.uint8)
    a[size // 2 - 1 : size // 2 + 1, 20 : size - 20] = 0
    a[20 : size - 20, size // 2 - 1 : size // 2 + 1] = 0
    return Image.fromarray(a, "L")


def _coloured(size=256):
    img = _drawing(size).convert("RGB")
    px = img.load()
    for x in range(40, 60):
        for y in range(40, 60):
            px[x, y] = (255, 190, 200)  # the blush FLUX leaks
    return img


def test_output_is_strictly_black_and_white():
    out = humanise(_drawing())
    assert set(np.unique(np.array(out))) <= {0, 255}


def test_colour_is_removed_not_merely_reduced():
    """Prompting FLUX not to add colour was measured and failed; this is the guarantee."""
    assert is_monochrome(humanise(_coloured())) is True


def test_is_monochrome_rejects_a_colour_image():
    assert is_monochrome(_coloured()) is False


def test_output_keeps_the_input_size():
    assert humanise(_drawing(256)).size == (256, 256)


def test_ink_survives_the_pass():
    """A warp that erases the drawing is worse than no warp."""
    before = (np.array(_drawing()) < 128).sum()
    after = (np.array(humanise(_drawing())) < 128).sum()
    assert after > before * 0.5


def test_the_same_seed_gives_a_byte_identical_result():
    """Consistent imperfection reads as one hand; varying imperfection reads worse
    than clean, so the pass must be reproducible."""
    a = np.array(humanise(_drawing(), seed=4))
    b = np.array(humanise(_drawing(), seed=4))
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    a = np.array(humanise(_drawing(), seed=1))
    b = np.array(humanise(_drawing(), seed=2))
    assert not np.array_equal(a, b)


def test_the_style_contract_matches_the_measured_setting():
    """These came off a contact sheet. Changing them changes every asset."""
    assert (WOBBLE, PRESSURE, BLUR_FRAC) == (5.0, 0.35, 0.045)


def test_an_rgb_input_is_accepted():
    assert humanise(_coloured()).mode == "L"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_humanise.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'why_engine.art'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/why_engine/art/__init__.py
```

```python
# src/why_engine/art/humanise.py
"""The deterministic style pass every asset flows through.

FLUX schnell has a strong prior toward clean polished illustration and leaks
small amounts of colour. Prompting against both was measured and failed, so the
hand-drawn character is applied here instead, identically to every asset.
"""
import numpy as np
from PIL import Image, ImageFilter

# Style contract. Measured from a contact sheet; not per-call knobs.
WOBBLE = 5.0        # displacement amplitude in output pixels
PRESSURE = 0.35     # stroke thickening; higher blobs the line
BLUR_FRAC = 0.045   # displacement smoothness as a fraction of WORK
WORK = 1536         # warp here, then downscale: what makes tremor, not noise
ROTATE_DEG = 1.2
INK_THRESHOLD = 200


def is_monochrome(img: Image.Image) -> bool:
    a = np.array(img.convert("RGB"), dtype=np.int16)
    return bool((a.max(axis=2) - a.min(axis=2)).max() == 0)


def _field(rng, shape, amp: float, blur: int) -> np.ndarray:
    n = rng.normal(0, 1, shape).astype(np.float32)
    n = (n - n.min()) / (n.max() - n.min() + 1e-9) * 255.0
    n = np.array(
        Image.fromarray(n.astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(blur)),
        dtype=np.float32,
    )
    n -= n.mean()
    return n / (np.abs(n).max() + 1e-9) * amp


def humanise(img: Image.Image, seed: int = 0) -> Image.Image:
    out_w, out_h = img.size
    src = img.convert("L").resize((WORK, WORK), Image.LANCZOS)
    ink = (np.array(src, dtype=np.float32) < INK_THRESHOLD).astype(np.float32)

    rng = np.random.default_rng(seed)
    blur = max(3, int(WORK * BLUR_FRAC))
    scale = WORK / max(out_w, out_h)

    yy, xx = np.meshgrid(np.arange(WORK), np.arange(WORK), indexing="ij")
    xs = np.clip(xx + _field(rng, (WORK, WORK), WOBBLE * scale, blur), 0, WORK - 1).astype(np.int32)
    ys = np.clip(yy + _field(rng, (WORK, WORK), WOBBLE * scale, blur), 0, WORK - 1).astype(np.int32)
    k = ink[ys, xs]

    radius = int(3 * scale) | 1
    thick = (
        np.array(
            Image.fromarray((k * 255).astype(np.uint8), "L").filter(ImageFilter.MaxFilter(radius)),
            dtype=np.float32,
        )
        / 255.0
    )
    merged = np.clip(k + thick * PRESSURE, 0, 1)

    grey = Image.fromarray(((1 - merged) * 255).astype(np.uint8), "L").resize(
        (out_w, out_h), Image.LANCZOS
    )
    rotated = grey.rotate(
        rng.uniform(-ROTATE_DEG, ROTATE_DEG), fillcolor=255, resample=Image.BICUBIC
    )
    # Threshold last so the resample and rotation cannot reintroduce grey.
    return Image.fromarray(
        np.where(np.array(rotated) < INK_THRESHOLD, 0, 255).astype(np.uint8), "L"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_humanise.py -q`
Expected: PASS, 9 passed

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
git add src/why_engine/art tests/test_humanise.py
git commit -m "feat: deterministic humanise style pass"
```

---

### Task 2: fal.ai generation client

**Files:**
- Create: `src/why_engine/art/fal.py`
- Test: `tests/test_fal.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `PORTRAIT`, `LANDSCAPE` size constants; `USD_PER_IMAGE`; `FalClient(api_key: str, transport: Callable | None = None)` with `.generate(prompt: str, image_size: str = PORTRAIT, seed: int | None = None) -> bytes`, `.image_count: int`, `.spend_usd: float`

Transport is injectable for the same reason `ClaudeCodeLLM` takes a `runner`: no test may spend money or touch the network.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fal.py
import pytest

from why_engine.art.fal import LANDSCAPE, PORTRAIT, USD_PER_IMAGE, FalClient


def _client(*responses):
    """Transport returns (json_dict, image_bytes_by_url) pairs in order."""
    calls = []

    def transport(*, url, headers, payload=None, download=False):
        calls.append({"url": url, "headers": headers, "payload": payload, "download": download})
        if download:
            return b"PNGBYTES"
        return responses[len([c for c in calls if not c["download"]]) - 1]

    c = FalClient("test-key", transport=transport)
    c.calls = calls
    return c


def test_generate_returns_the_downloaded_bytes():
    c = _client({"images": [{"url": "https://fal.media/x.jpg"}]})
    assert c.generate("a drawing") == b"PNGBYTES"


def test_the_api_key_is_sent_as_a_key_authorization_header():
    c = _client({"images": [{"url": "https://fal.media/x.jpg"}]})
    c.generate("a drawing")
    assert c.calls[0]["headers"]["Authorization"] == "Key test-key"


def test_the_prompt_is_sent_in_the_payload():
    c = _client({"images": [{"url": "https://fal.media/x.jpg"}]})
    c.generate("a stick figure")
    assert c.calls[0]["payload"]["prompt"] == "a stick figure"


def test_portrait_is_the_default_size():
    c = _client({"images": [{"url": "https://fal.media/x.jpg"}]})
    c.generate("a drawing")
    assert c.calls[0]["payload"]["image_size"] == PORTRAIT


def test_both_size_presets_stay_under_one_megapixel():
    """fal rounds megapixels UP. square_hd is 1.05 MP and bills double for less
    usable area, so only these two presets are permitted."""
    assert PORTRAIT == "portrait_16_9"
    assert LANDSCAPE == "landscape_16_9"


def test_a_seed_is_forwarded_when_given():
    c = _client({"images": [{"url": "https://fal.media/x.jpg"}]})
    c.generate("a drawing", seed=11)
    assert c.calls[0]["payload"]["seed"] == 11


def test_no_seed_key_is_sent_when_none():
    c = _client({"images": [{"url": "https://fal.media/x.jpg"}]})
    c.generate("a drawing")
    assert "seed" not in c.calls[0]["payload"]


def test_spend_is_tracked_per_image():
    c = _client(
        {"images": [{"url": "https://fal.media/a.jpg"}]},
        {"images": [{"url": "https://fal.media/b.jpg"}]},
    )
    c.generate("one")
    c.generate("two")
    assert c.image_count == 2
    assert c.spend_usd == pytest.approx(2 * USD_PER_IMAGE)


def test_a_response_without_images_raises():
    c = _client({"detail": "bad request"})
    with pytest.raises(RuntimeError, match="no image"):
        c.generate("a drawing")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_fal.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'why_engine.art.fal'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/why_engine/art/fal.py
"""fal.ai FLUX.1 schnell client.

fal bills per megapixel rounded UP, so only sub-megapixel presets are used:
portrait_16_9 (576x1024) and landscape_16_9 (1024x576) both bill at the 1 MP
tier. square_hd is 1.05 MP and would bill double for less usable area.
"""
from typing import Any, Callable, Dict, Optional

ENDPOINT = "https://fal.run/fal-ai/flux/schnell"
PORTRAIT = "portrait_16_9"
LANDSCAPE = "landscape_16_9"
USD_PER_IMAGE = 0.003
TIMEOUT_S = 120


def _http(*, url: str, headers: Dict[str, str], payload=None, download: bool = False):
    import httpx2 as httpx

    if download:
        r = httpx.get(url, timeout=TIMEOUT_S, follow_redirects=True)
        r.raise_for_status()
        return r.content
    r = httpx.post(url, headers=headers, json=payload, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


class FalClient:
    def __init__(self, api_key: str, transport: Optional[Callable[..., Any]] = None):
        if not api_key:
            raise ValueError("FAL_KEY is empty; set it in ~/why-engine/.env")
        self._key = api_key
        self._transport = transport or _http
        self.image_count = 0
        self.spend_usd = 0.0

    def generate(self, prompt: str, image_size: str = PORTRAIT, seed: Optional[int] = None) -> bytes:
        payload: Dict[str, Any] = {"prompt": prompt, "image_size": image_size, "num_images": 1}
        if seed is not None:
            payload["seed"] = seed
        body = self._transport(
            url=ENDPOINT,
            headers={"Authorization": f"Key {self._key}", "Content-Type": "application/json"},
            payload=payload,
        )
        images = body.get("images") if isinstance(body, dict) else None
        if not images:
            raise RuntimeError(f"fal returned no image: {str(body)[:200]}")
        data = self._transport(url=images[0]["url"], headers={}, download=True)
        self.image_count += 1
        self.spend_usd += USD_PER_IMAGE
        return data
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_fal.py -q`
Expected: PASS, 9 passed

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
git add src/why_engine/art/fal.py tests/test_fal.py
git commit -m "feat: fal.ai generation client with injectable transport"
```

---

### Task 3: Asset library manifest and build

**Files:**
- Create: `src/why_engine/art/library.py`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: `humanise` (Task 1), `FalClient`, `PORTRAIT`, `LANDSCAPE` (Task 2)
- Produces: `STYLE_SUFFIX: str`; `Asset(BaseModel)` with `name, kind, prompt, size`; `CAST`, `BACKGROUNDS`, `PROPS` lists; `build_prompt(asset: Asset) -> str`; `build_library(assets, client, out_dir) -> list[Path]`; `verify_library(out_dir) -> list[str]`

The style contract lives in the prompt suffix; the hand-drawn character comes from `humanise`. Both are applied to every asset, so the library is uniform by construction.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_library.py
import numpy as np
from PIL import Image

from why_engine.art.fal import LANDSCAPE, PORTRAIT, FalClient
from why_engine.art.library import (
    CAST, STYLE_SUFFIX, Asset, build_library, build_prompt, verify_library,
)


def _png_bytes(colour=(255, 190, 200)):
    import io
    a = np.full((64, 64, 3), 255, dtype=np.uint8)
    a[10:40, 10:40] = 0
    a[45:55, 45:55] = colour  # colour that humanise must remove
    buf = io.BytesIO()
    Image.fromarray(a, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _client(n):
    def transport(*, url, headers, payload=None, download=False):
        return _png_bytes() if download else {"images": [{"url": "https://fal.media/x.png"}]}
    return FalClient("k", transport=transport)


def test_build_prompt_appends_the_style_contract():
    p = build_prompt(Asset(name="a", kind="cast", prompt="a man waving", size=PORTRAIT))
    assert p.startswith("a man waving")
    assert STYLE_SUFFIX in p


def test_the_style_contract_forbids_colour_and_shading():
    low = STYLE_SUFFIX.lower()
    assert "no colour" in low or "no color" in low
    assert "no shading" in low


def test_the_cast_includes_both_characters():
    names = " ".join(a.name for a in CAST)
    assert "everyman" in names
    assert "bureaucrat" in names


def test_every_cast_asset_is_portrait():
    """Characters are taller than wide; portrait_16_9 also bills at the 1 MP tier."""
    assert all(a.size == PORTRAIT for a in CAST)


def test_build_library_writes_one_png_per_asset(tmp_path):
    assets = [Asset(name="x", kind="cast", prompt="p", size=PORTRAIT),
              Asset(name="y", kind="background", prompt="q", size=LANDSCAPE)]
    paths = build_library(assets, _client(2), tmp_path)
    assert [p.name for p in paths] == ["x.png", "y.png"]
    assert all(p.exists() for p in paths)


def test_built_assets_are_monochrome(tmp_path):
    """The generator leaks colour; the library must not contain any."""
    from why_engine.art.humanise import is_monochrome
    paths = build_library([Asset(name="x", kind="cast", prompt="p", size=PORTRAIT)], _client(1), tmp_path)
    assert is_monochrome(Image.open(paths[0])) is True


def test_build_library_skips_assets_that_already_exist(tmp_path):
    """The library is a one-time cost; rebuilding must not re-spend."""
    asset = Asset(name="x", kind="cast", prompt="p", size=PORTRAIT)
    client = _client(1)
    build_library([asset], client, tmp_path)
    first = client.image_count
    build_library([asset], client, tmp_path)
    assert client.image_count == first


def test_verify_library_passes_a_clean_directory(tmp_path):
    build_library([Asset(name="x", kind="cast", prompt="p", size=PORTRAIT)], _client(1), tmp_path)
    assert verify_library(tmp_path) == []


def test_verify_library_flags_a_colour_asset(tmp_path):
    Image.new("RGB", (32, 32), (200, 40, 40)).save(tmp_path / "bad.png")
    problems = verify_library(tmp_path)
    assert len(problems) == 1
    assert "bad.png" in problems[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_library.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'why_engine.art.library'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/why_engine/art/library.py
"""The one-time reusable asset library.

Generated once and amortised across every video. Recurring elements are the
same PNG in every shot, which is what makes the embedding-distance check from
the original design unnecessary.
"""
import io
from pathlib import Path
from typing import List, Literal, Sequence

from PIL import Image
from pydantic import BaseModel, Field

from why_engine.art.fal import LANDSCAPE, PORTRAIT, FalClient
from why_engine.art.humanise import humanise, is_monochrome

STYLE_SUFFIX = (
    ", simple hand drawn doodle, thin black ink line on plain white paper, "
    "flat straight-on view, no colour, no shading, no crosshatching, no background detail"
)


class Asset(BaseModel):
    name: str = Field(min_length=1)
    kind: Literal["cast", "background", "prop"]
    prompt: str = Field(min_length=1)
    size: str


def _cast(name: str, prompt: str) -> Asset:
    return Asset(name=name, kind="cast", prompt=prompt, size=PORTRAIT)


EVERYMAN = (
    "an ordinary adult man, medium large round head, flat hair with one cowlick, "
    "small dot eyes, tiny line nose, plain collared shirt and trousers, "
    "thin noodle arms and legs, full body"
)
BUREAUCRAT = (
    "a very tall thin bureaucrat in a plain suit and tie, small head, blank face "
    "with no visible eyes, stiff formal posture, thin noodle limbs, full body"
)

CAST: List[Asset] = [
    _cast("everyman_neutral", EVERYMAN + ", standing still, neutral expression"),
    _cast("everyman_confused", EVERYMAN + ", shrugging, confused expression"),
    _cast("everyman_pointing", EVERYMAN + ", pointing to one side"),
    _cast("everyman_surprised", EVERYMAN + ", arms raised, surprised expression"),
    _cast("everyman_thinking", EVERYMAN + ", hand on chin, thinking"),
    _cast("everyman_disgusted", EVERYMAN + ", recoiling, disgusted expression"),
    _cast("bureaucrat_neutral", BUREAUCRAT + ", standing still"),
    _cast("bureaucrat_pointing", BUREAUCRAT + ", pointing sternly"),
    _cast("bureaucrat_clipboard", BUREAUCRAT + ", holding a clipboard"),
    _cast("bureaucrat_stamping", BUREAUCRAT + ", stamping a document"),
]

BACKGROUNDS: List[Asset] = [
    Asset(name="bg_" + n, kind="background", prompt=p, size=LANDSCAPE)
    for n, p in [
        ("empty", "an empty plain room, minimal outline"),
        ("river", "a simple river bank with a few reeds"),
        ("street", "a plain suburban street with two simple houses"),
        ("office", "a plain government office with a desk and a filing cabinet"),
        ("field", "an open farm field with a simple fence"),
        ("factory", "a plain factory interior with simple machines"),
    ]
]

PROPS: List[Asset] = [
    Asset(name="prop_" + n, kind="prop", prompt=p, size=PORTRAIT)
    for n, p in [
        ("document", "a single sheet of paper with a few lines of writing"),
        ("chart", "a simple line chart going up"),
        ("money", "a plain banknote"),
        ("crate", "a plain wooden crate"),
    ]
]

ALL_ASSETS: List[Asset] = CAST + BACKGROUNDS + PROPS


def build_prompt(asset: Asset) -> str:
    return asset.prompt + STYLE_SUFFIX


def build_library(assets: Sequence[Asset], client: FalClient, out_dir: Path) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for index, asset in enumerate(assets):
        path = out_dir / f"{asset.name}.png"
        if not path.exists():  # the library is a one-time cost
            raw = client.generate(build_prompt(asset), image_size=asset.size)
            humanise(Image.open(io.BytesIO(raw)), seed=index).save(path)
        paths.append(path)
    return paths


def verify_library(out_dir: Path) -> List[str]:
    """Every asset must be pure black on white. Anything else breaks the contrast
    with archival photographs, which are the only thing allowed to carry tone."""
    problems: List[str] = []
    for path in sorted(Path(out_dir).glob("*.png")):
        if not is_monochrome(Image.open(path)):
            problems.append(f"{path.name}: not monochrome")
    return problems
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_library.py -q`
Expected: PASS, 9 passed

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
git add src/why_engine/art/library.py tests/test_library.py
git commit -m "feat: asset library manifest, build and integrity check"
```

---

### Task 4: Archival licence gate

**Files:**
- Create: `src/why_engine/art/archival.py`
- Test: `tests/test_archival.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `ACCEPTED_LICENCES: frozenset[str]`; `ArchivalImage(BaseModel)` with `title, url, licence, source_page`; `is_acceptable_licence(raw: str) -> bool`; `parse_search_response(body: dict) -> list[ArchivalImage]`

Measured against the live Wikimedia API: searching `Cyprinus carpio` returned two CC BY-SA 4.0 results and one Public domain. **The majority of results fail the gate**, so the fetcher searches broadly and filters rather than taking the top hit.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_archival.py
import pytest

from why_engine.art.archival import (
    ACCEPTED_LICENCES, ArchivalImage, is_acceptable_licence, parse_search_response,
)


def _page(title, licence, url="https://upload.wikimedia.org/x.jpg"):
    return {
        "title": title,
        "imageinfo": [{
            "url": url,
            "descriptionurl": "https://commons.wikimedia.org/wiki/" + title,
            "extmetadata": {"LicenseShortName": {"value": licence}},
        }],
    }


@pytest.mark.parametrize("licence", ["Public domain", "CC0", "PD-US", "public domain"])
def test_free_licences_are_accepted(licence):
    assert is_acceptable_licence(licence) is True


@pytest.mark.parametrize("licence", ["CC BY-SA 4.0", "CC BY 3.0", "GFDL", "Fair use", "", None])
def test_everything_else_is_rejected(licence):
    """Attribution licences are rejected too: an autonomous pipeline cannot
    reliably carry attribution, and a wrong call is a Content ID claim."""
    assert is_acceptable_licence(licence) is False


def test_cc_by_sa_is_rejected_despite_containing_cc():
    """Guards against a substring check on 'CC' letting share-alike through."""
    assert is_acceptable_licence("CC BY-SA 4.0") is False


def test_parse_keeps_only_acceptable_results():
    body = {"query": {"pages": {
        "1": _page("File:A.jpg", "CC BY-SA 4.0"),
        "2": _page("File:B.jpg", "Public domain"),
        "3": _page("File:C.jpg", "CC0"),
    }}}
    out = parse_search_response(body)
    assert sorted(i.title for i in out) == ["File:B.jpg", "File:C.jpg"]


def test_parse_records_the_licence_and_source_page():
    body = {"query": {"pages": {"1": _page("File:B.jpg", "Public domain")}}}
    image = parse_search_response(body)[0]
    assert image.licence == "Public domain"
    assert image.source_page.endswith("File:B.jpg")


def test_parse_returns_empty_when_nothing_qualifies():
    body = {"query": {"pages": {"1": _page("File:A.jpg", "CC BY-SA 4.0")}}}
    assert parse_search_response(body) == []


def test_parse_tolerates_an_empty_response():
    assert parse_search_response({}) == []


def test_parse_skips_a_page_with_no_imageinfo():
    assert parse_search_response({"query": {"pages": {"1": {"title": "File:X.jpg"}}}}) == []


def test_accepted_licences_does_not_contain_any_attribution_licence():
    assert not any("BY" in x.upper() for x in ACCEPTED_LICENCES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_archival.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'why_engine.art.archival'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/why_engine/art/archival.py
"""Real photographs for the drawn world, behind a hard licence gate.

Measured: a Wikimedia search for "Cyprinus carpio" returned 2 of 3 results under
CC BY-SA 4.0. The majority of results fail the gate, so search broadly and
filter. A copyrighted image reintroduces exactly the Content ID exposure the
original-illustrations rule exists to avoid, and this pipeline cannot dispute a
claim.
"""
from typing import List, Optional

from pydantic import BaseModel, Field

# Public domain and CC0 only. Attribution licences are excluded deliberately:
# an autonomous pipeline cannot reliably carry attribution, and the downside of
# a wrong call is a claim on a monetised video.
ACCEPTED_LICENCES = frozenset({"public domain", "cc0", "pd"})


class ArchivalImage(BaseModel):
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    licence: str = Field(min_length=1)
    source_page: str = ""


def is_acceptable_licence(raw: Optional[str]) -> bool:
    if not raw:
        return False
    text = raw.strip().lower()
    if text in ACCEPTED_LICENCES:
        return True
    # "PD-US", "PD-old-70" and similar are public-domain tags; "CC BY..." is not.
    return text.startswith("pd-") or text.startswith("public domain")


def parse_search_response(body: dict) -> List[ArchivalImage]:
    pages = ((body or {}).get("query") or {}).get("pages") or {}
    out: List[ArchivalImage] = []
    for page in pages.values():
        info = (page.get("imageinfo") or [None])[0]
        if not info:
            continue
        licence = (info.get("extmetadata") or {}).get("LicenseShortName", {}).get("value")
        if not is_acceptable_licence(licence):
            continue
        out.append(ArchivalImage(
            title=page.get("title", ""),
            url=info.get("url", ""),
            licence=licence,
            source_page=info.get("descriptionurl", ""),
        ))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_archival.py -q`
Expected: PASS, 17 passed

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
git add src/why_engine/art/archival.py tests/test_archival.py
git commit -m "feat: archival licence gate rejecting all non-free licences"
```

---

### Task 5: Archival search and fetch

**Files:**
- Modify: `src/why_engine/art/archival.py`
- Test: `tests/test_archival.py`

**Interfaces:**
- Consumes: `ArchivalImage`, `parse_search_response` (Task 4)
- Produces: `COMMONS_ENDPOINT`, `USER_AGENT`; `search_terms(query: str) -> dict` returning API params; `ArchivalFetcher(transport: Callable | None = None)` with `.search(query: str, limit: int = 10) -> list[ArchivalImage]` and `.fetch(image: ArchivalImage, out_path: Path) -> Path`

`fetch` writes a sidecar `.json` recording title, URL, licence and source page, so a published frame's provenance is reconstructible from disk exactly as a claim's is.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_archival.py
import json

from why_engine.art.archival import USER_AGENT, ArchivalFetcher, search_terms


def _fetcher(*responses):
    calls = []

    def transport(*, url, params=None, headers=None, download=False):
        calls.append({"url": url, "params": params, "headers": headers, "download": download})
        if download:
            return b"JPEGBYTES"
        return responses[len([c for c in calls if not c["download"]]) - 1]

    f = ArchivalFetcher(transport=transport)
    f.calls = calls
    return f


def test_search_terms_restricts_to_the_file_namespace():
    """Namespace 6 is File:. Without it the search returns articles, not images."""
    assert search_terms("carp")["gsrnamespace"] == 6


def test_search_terms_requests_the_licence_metadata():
    params = search_terms("carp")
    assert "extmetadata" in params["iiprop"]
    assert "LicenseShortName" in params["iiextmetadatafilter"]


def test_search_returns_only_licence_cleared_images():
    body = {"query": {"pages": {
        "1": _page("File:A.jpg", "CC BY-SA 4.0"),
        "2": _page("File:B.jpg", "Public domain"),
    }}}
    out = _fetcher(body).search("carp")
    assert [i.title for i in out] == ["File:B.jpg"]


def test_search_sends_a_descriptive_user_agent():
    """Wikimedia rate-limits anonymous clients without one."""
    f = _fetcher({"query": {"pages": {}}})
    f.search("carp")
    assert USER_AGENT in f.calls[0]["headers"]["User-Agent"]


def test_fetch_writes_the_image_bytes(tmp_path):
    f = _fetcher()
    image = ArchivalImage(title="File:B.jpg", url="https://u/x.jpg", licence="Public domain")
    out = f.fetch(image, tmp_path / "b.jpg")
    assert out.read_bytes() == b"JPEGBYTES"


def test_fetch_writes_a_provenance_sidecar(tmp_path):
    """A published frame's provenance must be reconstructible from disk."""
    f = _fetcher()
    image = ArchivalImage(title="File:B.jpg", url="https://u/x.jpg",
                          licence="Public domain", source_page="https://commons/B")
    f.fetch(image, tmp_path / "b.jpg")
    side = json.loads((tmp_path / "b.jpg.json").read_text())
    assert side["licence"] == "Public domain"
    assert side["title"] == "File:B.jpg"
    assert side["source_page"] == "https://commons/B"


def test_fetch_refuses_an_image_that_did_not_clear_the_gate(tmp_path):
    """Defence in depth: the gate runs at search time AND at fetch time."""
    bad = ArchivalImage(title="File:A.jpg", url="https://u/a.jpg", licence="CC BY-SA 4.0")
    with pytest.raises(ValueError, match="licence"):
        _fetcher().fetch(bad, tmp_path / "a.jpg")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_archival.py -q`
Expected: FAIL, `ImportError: cannot import name 'ArchivalFetcher'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/why_engine/art/archival.py`:

```python
import json
from pathlib import Path
from typing import Any, Callable, Dict

COMMONS_ENDPOINT = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "why-engine/0.1 (research; contact via repository)"
TIMEOUT_S = 60


def search_terms(query: str, limit: int = 10) -> Dict[str, Any]:
    return {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,  # File: namespace; without it the search returns articles
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiextmetadatafilter": "LicenseShortName|UsageTerms|Artist",
    }


def _http(*, url: str, params=None, headers=None, download: bool = False):
    import httpx2 as httpx

    if download:
        r = httpx.get(url, headers=headers, timeout=TIMEOUT_S, follow_redirects=True)
        r.raise_for_status()
        return r.content
    r = httpx.get(url, params=params, headers=headers, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


class ArchivalFetcher:
    def __init__(self, transport: Optional[Callable[..., Any]] = None):
        self._transport = transport or _http

    def search(self, query: str, limit: int = 10) -> List[ArchivalImage]:
        body = self._transport(
            url=COMMONS_ENDPOINT,
            params=search_terms(query, limit),
            headers={"User-Agent": USER_AGENT},
        )
        return parse_search_response(body)

    def fetch(self, image: ArchivalImage, out_path: Path) -> Path:
        # The gate runs again here: search and fetch are separate entry points and
        # a caller could construct an ArchivalImage directly.
        if not is_acceptable_licence(image.licence):
            raise ValueError(f"refusing {image.title}: licence {image.licence!r} is not free")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = self._transport(url=image.url, headers={"User-Agent": USER_AGENT}, download=True)
        out_path.write_bytes(data)
        out_path.with_suffix(out_path.suffix + ".json").write_text(
            json.dumps(image.model_dump(), indent=2), encoding="utf-8"
        )
        return out_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_archival.py -q`
Expected: PASS, 24 passed (17 from Task 4 plus 7 new)

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
git add src/why_engine/art/archival.py tests/test_archival.py
git commit -m "feat: archival search and fetch with provenance sidecars"
```

---

### Task 6: Library build CLI

**Files:**
- Modify: `src/why_engine/art/library.py`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: everything above
- Produces: `main(argv: list[str] | None = None) -> int`

Reads `FAL_KEY` from the environment, builds any missing assets, verifies the result, reports spend. Exit 0 on a clean library, 1 if verification finds a problem.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_library.py
from why_engine.art.library import main


def test_main_builds_and_verifies_a_library(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.setattr("why_engine.art.library.FalClient",
                        lambda key, transport=None: _client(1))
    code = main(["--out", str(tmp_path), "--only", "everyman_neutral"])
    assert code == 0
    assert (tmp_path / "everyman_neutral.png").exists()
    assert "spend" in capsys.readouterr().out.lower()


def test_main_exits_one_without_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    assert main(["--out", str(tmp_path)]) == 1


def test_main_exits_one_when_verification_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.setattr("why_engine.art.library.FalClient",
                        lambda key, transport=None: _client(1))
    Image.new("RGB", (16, 16), (200, 40, 40)).save(tmp_path / "contaminated.png")
    assert main(["--out", str(tmp_path), "--only", "everyman_neutral"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_library.py -q`
Expected: FAIL, `ImportError: cannot import name 'main'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/why_engine/art/library.py`:

```python
import argparse
import os
import sys
from typing import Optional


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="why-engine-library")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--only", action="append", default=None,
                        help="build only the named asset; repeatable")
    args = parser.parse_args(argv)

    key = os.environ.get("FAL_KEY", "")
    if not key:
        print("FAL_KEY is not set; put it in ~/why-engine/.env", file=sys.stderr)
        return 1

    assets = ALL_ASSETS
    if args.only:
        wanted = set(args.only)
        assets = [a for a in ALL_ASSETS if a.name in wanted]

    client = FalClient(key)
    build_library(assets, client, args.out)
    print(f"built {len(assets)} asset(s); spend ${client.spend_usd:.3f}")

    problems = verify_library(args.out)
    if problems:
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_library.py -q`
Expected: PASS, 12 passed

- [ ] **Step 5: Run the whole suite and commit**

```bash
cd ~/why-engine
.venv/bin/python -m pytest -q
git add src/why_engine/art/library.py tests/test_library.py
git commit -m "feat: library build CLI with spend reporting and verification gate"
```

Expected: 147 existing plus this plan's tests, 6 deselected.

---

## Scope note: spec section 7.8

Spec 7.8 covers the 15-20 per-video topic-specific drawings. This plan builds the
*mechanism* those need — `FalClient` and `humanise` — but not the decision of what
to draw, which belongs to `art_director` in Plan 2B. That is a deliberate split,
not an omission: shot planning needs the script and the timing map, neither of
which exists until 2B.

## Definition of Done

- `.venv/bin/python -m pytest` passes with the live tests still deselected.
- `.venv/bin/python -m why_engine.art.library --out assets/library` builds the full library, reports spend under $0.30, and exits 0.
- Every file in `assets/library/` is pure black on white, verified by `verify_library`.
- `ArchivalFetcher.search` returns only public-domain or CC0 images, and `fetch` refuses anything else even when handed it directly.
- Every fetched archival image has a provenance sidecar naming its licence and source page.
- No module under `art/` imports `anthropic`; no test touches the network or spends money.
