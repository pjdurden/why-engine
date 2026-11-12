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


def test_a_non_permitted_size_is_rejected():
    """square_hd is 1.05 MP and bills at double rate, so accepting it would make
    USD_PER_IMAGE silently wrong."""
    c = _client({"images": [{"url": "https://fal.media/x.jpg"}]})
    with pytest.raises(ValueError, match="not permitted"):
        c.generate("a drawing", image_size="square_hd")


def test_a_rejected_size_costs_nothing():
    c = _client({"images": [{"url": "https://fal.media/x.jpg"}]})
    with pytest.raises(ValueError):
        c.generate("a drawing", image_size="square_hd")
    assert c.image_count == 0
    assert c.spend_usd == 0.0
    assert c.calls == []


def test_a_malformed_image_element_raises_a_clear_error():
    """An element present but missing its url must not degrade to KeyError."""
    c = _client({"images": [{}]})
    with pytest.raises(RuntimeError, match="no image"):
        c.generate("a drawing")


def test_a_failed_download_is_not_billed():
    """fal does not bill for server errors, so neither do we."""
    def transport(*, url, headers, payload=None, download=False):
        if download:
            raise RuntimeError("download failed")
        return {"images": [{"url": "https://fal.media/x.jpg"}]}

    c = FalClient("test-key", transport=transport)
    with pytest.raises(RuntimeError, match="download failed"):
        c.generate("a drawing")
    assert c.image_count == 0
    assert c.spend_usd == 0.0
