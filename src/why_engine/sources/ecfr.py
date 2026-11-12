"""Primary-source acquisition from the Electronic Code of Federal Regulations.

eCFR refuses programmatic access to its web pages and says so in the response
body rather than in the status code: a plain fetch returns 200 with "Your
request has been flagged as potentially automated" as the document. That text
would have been chunked into passages and quoted on screen as federal law, and
nothing downstream could have told the difference. Acquisition therefore goes
through the versioner API, which serves XML, and anything that does not look
like a section is rejected here rather than allowed into the corpus.

CFR text is a US Government work and carries no copyright, which is the whole
reason this channel quotes it directly.
"""
from __future__ import annotations

import re
from datetime import date as _date
from typing import List, Optional
from xml.etree import ElementTree

from why_engine.models import Passage
from why_engine.sources.fetch import chunk_text

API = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml"

#: Blocks are served with a 200, so they have to be recognised by their text.
_BLOCK_MARKERS = ("flagged as potentially automated", "Request Access Due to")


def cfr_url(title: int, section: str) -> str:
    """The page a viewer can open, for the source list under the video."""
    return f"https://www.ecfr.gov/current/title-{title}/section-{section}"


def section_text(xml: str) -> str:
    """Flatten one section's XML into quotable prose.

    Nested tags matter: "S1. <I>Scope.</I> This standard..." has to come back
    whole, because a quote that silently drops a word fails the verbatim
    provenance check and looks like the model invented it.
    """
    for marker in _BLOCK_MARKERS:
        if marker in xml:
            raise ValueError("eCFR returned an anti-scraping page, not a section")
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise ValueError(f"not parseable as eCFR XML: {exc}") from exc

    blocks = []
    for node in root.iter():
        if node.tag not in ("HEAD", "P"):
            continue
        # itertext() rather than .text: the words inside <I> and <E> are part
        # of the sentence, not decoration.
        words = " ".join("".join(node.itertext()).split())
        if words:
            blocks.append(words)
    if not blocks:
        raise ValueError("eCFR XML contained no headings or paragraphs")
    return re.sub(r"\s+", " ", " ".join(blocks)).strip()


def fetch_cfr_section(
    title: int,
    section: str,
    *,
    heading: str,
    subtitle: Optional[str] = None,
    chapter: Optional[str] = None,
    part: Optional[str] = None,
    on: Optional[str] = None,
    max_chars: int = 1200,
) -> List[Passage]:
    """Network IO, kept out of `section_text` so the parsing stays testable."""
    import httpx2 as httpx

    params = {k: v for k, v in (("subtitle", subtitle), ("chapter", chapter),
                                ("part", part), ("section", section)) if v}
    response = httpx.get(
        API.format(date=on or _date.today().isoformat(), title=title),
        params=params, timeout=90.0,
        # eCFR asks for a contactable agent rather than an anonymous one.
        headers={"User-Agent": "why-engine/0.1 (research; +https://github.com/pjdurden/why-engine)"},
    )
    response.raise_for_status()
    return chunk_text(section_text(response.text), source_id=f"cfr-{section}",
                      url=cfr_url(title, section), title=heading, max_chars=max_chars)
