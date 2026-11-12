import pytest

from why_engine.art.archival import (
    ACCEPTED_LICENCES, ArchivalImage, _is_raster_image, is_acceptable_licence,
    parse_search_response,
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


@pytest.mark.parametrize("licence", [
    "Public domain in the United States but not in its country of origin",
    "Public domain outside the EU",
])
def test_caveated_public_domain_claims_are_rejected(licence):
    """Both name a jurisdiction where the work is still protected. A bare US-only
    claim is NOT in this list: it is what the accepted PD-US tag already means,
    and treating the two differently was a defect in this test."""
    assert is_acceptable_licence(licence) is False


@pytest.mark.parametrize("licence", ["PD-US", "PD-old-70", "PD-ineligible"])
def test_bounded_pd_tags_are_still_accepted(licence):
    assert is_acceptable_licence(licence) is True


def test_the_model_itself_refuses_a_non_free_licence():
    """Defence in depth: the gate must not depend on one call site."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ArchivalImage(title="File:A.jpg", url="https://u/a.jpg", licence="CC BY-SA 4.0", source_page="https://u/page")


@pytest.mark.parametrize("bad", [123, [], {}, 4.5])
def test_a_non_string_licence_is_rejected_without_raising(bad):
    assert is_acceptable_licence(bad) is False


def test_a_page_without_a_description_url_is_skipped():
    """No provenance, no image: the licence claim must stay verifiable."""
    page = {"title": "File:B.jpg", "imageinfo": [{
        "url": "https://u/b.jpg",
        "extmetadata": {"LicenseShortName": {"value": "Public domain"}},
    }]}
    assert parse_search_response({"query": {"pages": {"1": page}}}) == []


def test_a_page_missing_its_title_or_url_is_skipped():
    """Malformed pages must be skipped, not crash the whole response."""
    base = {"extmetadata": {"LicenseShortName": {"value": "Public domain"}},
            "descriptionurl": "https://commons.wikimedia.org/wiki/File:B.jpg"}
    no_url = {"title": "File:B.jpg", "imageinfo": [dict(base)]}
    no_title = {"imageinfo": [dict(base, url="https://u/b.jpg")]}
    assert parse_search_response({"query": {"pages": {"1": no_url}}}) == []
    assert parse_search_response({"query": {"pages": {"2": no_title}}}) == []


@pytest.mark.parametrize("url", [
    "https://upload.wikimedia.org/x/Common_carp.jpg",
    "https://upload.wikimedia.org/x/Common_carp.jpg?utm_source=commons.wikimedia.org",
    "https://upload.wikimedia.org/x/A.PNG",
])
def test_raster_urls_are_accepted(url):
    assert _is_raster_image(url) is True


@pytest.mark.parametrize("url", [
    "https://upload.wikimedia.org/x/scan.pdf",
    "https://upload.wikimedia.org/x/scan.pdf?utm_source=commons.wikimedia.org",
    "https://upload.wikimedia.org/x/book.djvu",
    "https://upload.wikimedia.org/x/clip.ogv",
    "https://upload.wikimedia.org/x/noextension",
])
def test_non_image_media_is_rejected(url):
    """The File: namespace is not an image namespace. An 81MB PDF downloaded as
    archival.jpg is what this prevents."""
    assert _is_raster_image(url) is False


def test_a_pdf_result_is_skipped_by_the_parser():
    page = {"title": "File:Scan.pdf", "imageinfo": [{
        "url": "https://upload.wikimedia.org/x/scan.pdf",
        "descriptionurl": "https://commons.wikimedia.org/wiki/File:Scan.pdf",
        "extmetadata": {"LicenseShortName": {"value": "Public domain"}},
    }]}
    assert parse_search_response({"query": {"pages": {"1": page}}}) == []


import json
from pathlib import Path

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


def test_the_user_agent_is_contactable():
    """Wikimedia's UA policy requires a URL or email. A UA naming no way to
    reach anyone is treated as anonymous and returns 403 -- verified live."""
    assert "http" in USER_AGENT or "@" in USER_AGENT


def test_fetch_writes_the_image_bytes(tmp_path):
    f = _fetcher()
    image = ArchivalImage(title="File:B.jpg", url="https://u/x.jpg",
                          licence="Public domain", source_page="https://commons/B")
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
    """Defence in depth: the gate runs at search time AND at fetch time.

    model_construct bypasses ArchivalImage's own field_validator, simulating a
    caller that built the object directly rather than through normal
    validation -- the exact case fetch()'s redundant check exists for.
    """
    bad = ArchivalImage.model_construct(title="File:A.jpg", url="https://u/a.jpg",
                                         licence="CC BY-SA 4.0", source_page="https://commons/A")
    with pytest.raises(ValueError, match="licence"):
        _fetcher().fetch(bad, tmp_path / "a.jpg")


def test_a_failed_image_write_still_leaves_provenance(tmp_path, monkeypatch):
    """Sidecar first: an orphan .json is inert, an orphan image is not. An image
    with no sidecar is indistinguishable from a properly sourced one."""
    f = _fetcher()
    image = ArchivalImage(title="File:B.jpg", url="https://u/b.jpg",
                          licence="Public domain", source_page="https://commons/B")
    target = tmp_path / "b.jpg"

    real_write_bytes = Path.write_bytes
    def boom(self, data):
        if self == target:
            raise OSError("disk full")
        return real_write_bytes(self, data)
    monkeypatch.setattr(Path, "write_bytes", boom)

    with pytest.raises(OSError):
        f.fetch(image, target)
    assert not target.exists()
    assert (tmp_path / "b.jpg.json").exists()


def test_the_download_call_also_sends_the_user_agent(tmp_path):
    """Wikimedia rate-limits anonymous clients on downloads too, not just search."""
    f = _fetcher()
    image = ArchivalImage(title="File:B.jpg", url="https://u/b.jpg",
                          licence="Public domain", source_page="https://commons/B")
    f.fetch(image, tmp_path / "b.jpg")
    download_calls = [c for c in f.calls if c["download"]]
    assert download_calls
    assert USER_AGENT in download_calls[0]["headers"]["User-Agent"]


def test_search_requests_a_bounded_thumbnail():
    """Commons hosts 200-400MB TIFFs; the full original is never needed for a
    1920x1080 frame."""
    assert search_terms("carp")["iiurlwidth"] == 1920


def test_the_thumbnail_url_is_preferred_when_present():
    page = {"title": "File:B.jpg", "imageinfo": [{
        "url": "https://upload.wikimedia.org/x/B.jpg",
        "thumburl": "https://upload.wikimedia.org/thumb/B.jpg/1920px-B.jpg",
        "descriptionurl": "https://commons.wikimedia.org/wiki/File:B.jpg",
        "extmetadata": {"LicenseShortName": {"value": "Public domain"}},
    }]}
    assert "1920px" in parse_search_response({"query": {"pages": {"1": page}}})[0].url


def test_a_pdf_is_still_rejected_even_when_a_thumburl_exists():
    """thumburl is always rendered to .jpg, so the media check must read the
    original url or every PDF would slip through."""
    page = {"title": "File:S.pdf", "imageinfo": [{
        "url": "https://upload.wikimedia.org/x/S.pdf",
        "thumburl": "https://upload.wikimedia.org/thumb/S.pdf/1920px-S.pdf.jpg",
        "descriptionurl": "https://commons.wikimedia.org/wiki/File:S.pdf",
        "extmetadata": {"LicenseShortName": {"value": "Public domain"}},
    }]}
    assert parse_search_response({"query": {"pages": {"1": page}}}) == []


@pytest.mark.parametrize("licence", [
    "Public domain in the United States",
    "PD-US",
    "PD-US-not-renewed",
])
def test_us_only_public_domain_is_accepted_consistently(licence):
    """PD-US and its prose form mean the same thing; the gate must not accept one
    and reject the other."""
    assert is_acceptable_licence(licence) is True


@pytest.mark.parametrize("licence", [
    "Public domain in the United States but not in its country of origin",
    "Public domain outside the EU",
    "pd-us-but-not-elsewhere",
    "Public domain except in Germany",
])
def test_claims_flagging_surviving_foreign_copyright_are_rejected(licence):
    """A claim naming a jurisdiction where the work is still protected is the
    real hazard, whatever form it takes."""
    assert is_acceptable_licence(licence) is False


def test_results_come_back_in_relevance_order():
    """pageid order is not search ranking; MediaWiki supplies `index` for this."""
    def page(title, index):
        return {"title": title, "index": index, "imageinfo": [{
            "url": f"https://u/{title}.jpg",
            "descriptionurl": f"https://commons/{title}",
            "extmetadata": {"LicenseShortName": {"value": "Public domain"}}}]}
    body = {"query": {"pages": {"900": page("third", 3), "100": page("first", 1),
                                "500": page("second", 2)}}}
    assert [i.title for i in parse_search_response(body)] == ["first", "second", "third"]
