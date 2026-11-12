import pytest
import numpy as np
from PIL import Image

from why_engine.art.fal import LANDSCAPE, PORTRAIT, FalClient
from why_engine.art.library import (
    CAST, STYLE_SUFFIX, Asset, build_library, build_prompt, main, verify_library,
)


@pytest.fixture(autouse=True)
def _fast_warp(monkeypatch):
    """These tests exercise build orchestration, not tremor quality. Warping at
    the production 1536 costs ~15s per test and proves nothing here.
    tests/test_humanise.py keeps the real constant."""
    monkeypatch.setattr("why_engine.art.humanise.WORK", 192)


def _png_bytes(colour=(255, 190, 200)):
    import io
    a = np.full((64, 64, 3), 255, dtype=np.uint8)
    a[10:40, 10:40] = 0
    a[45:55, 45:55] = colour  # colour that humanise must remove
    buf = io.BytesIO()
    Image.fromarray(a, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _client():
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
    paths = build_library(assets, _client(), tmp_path)
    assert [p.name for p in paths] == ["x.png", "y.png"]
    assert all(p.exists() for p in paths)


def test_built_assets_are_monochrome(tmp_path):
    """The generator leaks colour; the library must not contain any."""
    from why_engine.art.humanise import is_monochrome
    paths = build_library([Asset(name="x", kind="cast", prompt="p", size=PORTRAIT)], _client(), tmp_path)
    assert is_monochrome(Image.open(paths[0])) is True


def test_build_library_skips_assets_that_already_exist(tmp_path):
    """The library is a one-time cost; rebuilding must not re-spend."""
    asset = Asset(name="x", kind="cast", prompt="p", size=PORTRAIT)
    client = _client()
    build_library([asset], client, tmp_path)
    first = client.image_count
    build_library([asset], client, tmp_path)
    assert client.image_count == first


def test_verify_library_passes_a_clean_directory(tmp_path):
    build_library([Asset(name="x", kind="cast", prompt="p", size=PORTRAIT)], _client(), tmp_path)
    assert verify_library(tmp_path) == []


def test_verify_library_flags_a_colour_asset(tmp_path):
    Image.new("RGB", (32, 32), (200, 40, 40)).save(tmp_path / "bad.png")
    problems = verify_library(tmp_path)
    assert len(problems) == 1
    assert "bad.png" in problems[0]


def test_verify_library_reports_a_corrupt_file_instead_of_crashing(tmp_path):
    """A build killed mid-save leaves a truncated png. The gate must report it."""
    (tmp_path / "truncated.png").write_bytes(b"")
    problems = verify_library(tmp_path)
    assert len(problems) == 1
    assert "truncated.png" in problems[0]
    assert "corrupt" in problems[0] or "unreadable" in problems[0]


def test_build_library_regenerates_a_corrupt_asset(tmp_path):
    """Existence-only skipping would leave it stuck forever."""
    asset = Asset(name="x", kind="cast", prompt="p", size=PORTRAIT)
    (tmp_path / "x.png").write_bytes(b"")
    client = _client()
    build_library([asset], client, tmp_path)
    assert client.image_count == 1
    assert verify_library(tmp_path) == []


def test_build_library_still_skips_a_readable_asset(tmp_path):
    """The self-healing guard must not defeat the one-time-cost guarantee."""
    asset = Asset(name="x", kind="cast", prompt="p", size=PORTRAIT)
    client = _client()
    build_library([asset], client, tmp_path)
    first = client.image_count
    build_library([asset], client, tmp_path)
    assert client.image_count == first


def test_the_manifest_has_the_expected_shape_and_no_duplicate_names():
    """A duplicate name would make two assets write to the same path, silently
    losing one. Guards the cast expansion planned for later."""
    from why_engine.art.library import ALL_ASSETS, BACKGROUNDS, PROPS
    assert (len(CAST), len(BACKGROUNDS), len(PROPS)) == (10, 6, 4)
    names = [a.name for a in ALL_ASSETS]
    assert len(names) == len(set(names))


def test_backgrounds_are_landscape_and_props_are_portrait():
    from why_engine.art.library import BACKGROUNDS, PROPS
    assert all(a.size == LANDSCAPE for a in BACKGROUNDS)
    assert all(a.size == PORTRAIT for a in PROPS)


def test_main_builds_and_verifies_a_library(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("why_engine.art.humanise.WORK", 192)
    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.setattr("why_engine.art.library.FalClient",
                        lambda key, transport=None: _client())
    code = main(["--out", str(tmp_path), "--only", "everyman_neutral"])
    assert code == 0
    assert (tmp_path / "everyman_neutral.png").exists()
    assert "spend" in capsys.readouterr().out.lower()


def test_main_exits_one_without_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    assert main(["--out", str(tmp_path)]) == 1


def test_main_exits_one_when_verification_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("why_engine.art.humanise.WORK", 192)
    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.setattr("why_engine.art.library.FalClient",
                        lambda key, transport=None: _client())
    Image.new("RGB", (16, 16), (200, 40, 40)).save(tmp_path / "contaminated.png")
    assert main(["--out", str(tmp_path), "--only", "everyman_neutral"]) == 1


def test_verify_library_flags_a_solid_black_asset(tmp_path):
    """Solid black is perfectly monochrome, so the colour check alone cannot see it."""
    Image.new("L", (64, 64), 0).save(tmp_path / "solid.png")
    problems = verify_library(tmp_path)
    assert len(problems) == 1 and "solid" in problems[0].lower()


def test_verify_library_flags_a_blank_asset(tmp_path):
    Image.new("L", (64, 64), 255).save(tmp_path / "blank.png")
    assert len(verify_library(tmp_path)) == 1


def test_main_rejects_an_unknown_asset_name(tmp_path, monkeypatch):
    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.setattr("why_engine.art.library.FalClient",
                        lambda key, transport=None: _client())
    assert main(["--out", str(tmp_path), "--only", "everyman_neutrl"]) == 1


def test_is_readable_png_rejects_a_truncated_file(tmp_path, monkeypatch):
    """A build killed mid-save leaves a valid header with truncated data. That
    passes Image.open() and only fails at .load(), which is why the helper forces
    a decode. A zero-byte fixture never reaches that line."""
    monkeypatch.setattr("why_engine.art.humanise.WORK", 192)
    from why_engine.art.humanise import humanise
    from why_engine.art.library import _is_readable_png

    good = tmp_path / "good.png"
    humanise(Image.new("L", (32, 32), 255)).save(good)
    blob = good.read_bytes()

    truncated = tmp_path / "truncated.png"
    truncated.write_bytes(blob[: len(blob) * 2 // 3])
    assert _is_readable_png(good) is True
    assert _is_readable_png(truncated) is False
