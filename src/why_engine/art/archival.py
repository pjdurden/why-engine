"""Real photographs for the drawn world, behind a hard licence gate.

Measured: a Wikimedia search for "Cyprinus carpio" returned 2 of 3 results under
CC BY-SA 4.0. The majority of results fail the gate, so search broadly and
filter. A copyrighted image reintroduces exactly the Content ID exposure the
original-illustrations rule exists to avoid, and this pipeline cannot dispute a
claim.
"""
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator

# Public domain and CC0 only. Attribution licences are excluded deliberately:
# an autonomous pipeline cannot reliably carry attribution, and the downside of
# a wrong call is a claim on a monetised video.
ACCEPTED_LICENCES = frozenset({"public domain", "cc0", "pd"})

# Bounded pattern for public-domain tags: "PD-US", "PD-old-70", "PD-ineligible"
# Prevents prefix matches from accepting caveated claims like "PD-US but not in EU"
_PD_TAG = re.compile(r"^pd-[a-z0-9-]+$")

# Phrases that explicitly assert surviving copyright somewhere. These are the
# real hazard: a bare US-only claim is acceptable for this corpus, but a claim
# that names a jurisdiction where the work is still protected is not. Matched
# against a hyphen-normalised copy of the text (see below) so a tag-form
# caveat like "pd-us-but-not-elsewhere" is caught the same as the prose form
# "but not in its country of origin" -- "but not in" alone misses the tag
# form because it drops the "in".
_FOREIGN_COPYRIGHT_CAVEATS = (
    "but not",
    "not in its country of origin",
    "outside the",
    "except in",
)

_US_PD_PROSE = ("public domain in the united states",)


def is_acceptable_licence(raw: Optional[str]) -> bool:
    """Public domain and CC0 only, and only unconditionally.

    A prefix match would accept "Public domain in the United States but not in
    its country of origin", which states the work is still copyrighted abroad.
    Exact membership plus a bounded tag pattern closes that hole.
    Non-string inputs are rejected without raising.

    US-only public domain is accepted consistently: "PD-US" and its prose form
    "Public domain in the United States" mean the same thing (17 USC 105 places
    US federal works in the public domain outright, and this corpus is
    predominantly such works). What is rejected is a claim that explicitly
    names a jurisdiction where the work is still protected -- that caveat check
    runs first, before either accept path.
    """
    if not isinstance(raw, str) or not raw.strip():
        return False
    text = " ".join(raw.strip().lower().split())
    # Hyphens carry the same "joins words" role as spaces in tag-form claims
    # (e.g. "pd-us-but-not-elsewhere"), so the caveat check normalises them to
    # spaces. The bounded-tag accept path below still matches on the
    # hyphenated `text`, unaffected by this.
    caveat_text = text.replace("-", " ")
    if any(c in caveat_text for c in _FOREIGN_COPYRIGHT_CAVEATS):
        return False
    if text.startswith(_US_PD_PROSE):
        return True
    return text in ACCEPTED_LICENCES or bool(_PD_TAG.match(text))


class ArchivalImage(BaseModel):
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    licence: str = Field(min_length=1)
    source_page: str = Field(min_length=1)

    @field_validator("licence")
    @classmethod
    def _licence_must_be_free(cls, v: str) -> str:
        if not is_acceptable_licence(v):
            raise ValueError(f"licence {v!r} is not public domain or CC0")
        return v


RASTER_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".webp"})


def _is_raster_image(url: str) -> bool:
    """The File: namespace holds PDFs, DjVu, audio and video too. Wikimedia also
    appends a ?utm_source= query, so the path must be isolated before the suffix
    is read."""
    return PurePosixPath(urlsplit(url).path.lower()).suffix in RASTER_SUFFIXES


def parse_search_response(body: dict) -> List[ArchivalImage]:
    pages_by_id = ((body or {}).get("query") or {}).get("pages") or {}
    # pageid order is not search ranking; MediaWiki supplies `index` for that.
    # Sort by it so index 0 in the returned list is the best match, not an
    # arbitrary one.
    pages = sorted(pages_by_id.values(), key=lambda p: p.get("index", 1_000_000))
    out: List[ArchivalImage] = []
    for page in pages:
        info = (page.get("imageinfo") or [None])[0]
        if not info:
            continue
        licence = (info.get("extmetadata") or {}).get("LicenseShortName", {}).get("value")
        if not is_acceptable_licence(licence):
            continue
        source_page = info.get("descriptionurl")
        if not source_page:
            continue  # no provenance, no image
        title = page.get("title")
        original_url = info.get("url")
        if not title or not original_url:
            continue
        # The media-type check must read the original url: thumburl is always
        # rendered to .jpg by Wikimedia's thumbnailer, so checking it would let
        # a PDF or DjVu original slip through disguised as an image.
        if not _is_raster_image(original_url):
            continue
        # Prefer the bounded thumbnail for the actual download: Commons hosts
        # 200-400MB TIFF originals and a 1920px frame is never larger than needed.
        url = info.get("thumburl") or original_url
        out.append(ArchivalImage(
            title=title,
            url=url,
            licence=licence,
            source_page=source_page,
        ))
    return out


COMMONS_ENDPOINT = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "why-engine/0.1 (https://github.com/pjdurden/why-engine; research)"
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
        # Bounded thumbnail: Commons routinely hosts 200-400MB TIFF originals,
        # and a 1920x1080 frame never needs more than this.
        "iiurlwidth": 1920,
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
        # Provenance first: a sidecar with no image is inert and self-evident,
        # whereas an image with no sidecar looks properly sourced and could be
        # composited into a published video.
        out_path.with_suffix(out_path.suffix + ".json").write_text(
            json.dumps(image.model_dump(), indent=2), encoding="utf-8"
        )
        out_path.write_bytes(data)
        return out_path
