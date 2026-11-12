"""eCFR acquisition. The network call is not tested; the parsing is.

eCFR blocks scraping of its web pages outright -- a plain fetch returns "Your
request has been flagged as potentially automated" as the document body, which
would have entered the corpus as a passage and been quoted as law. The API is
the only supported route, and it returns XML rather than HTML.
"""
import pytest

from why_engine.sources.ecfr import cfr_url, section_text

SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<DIV8 N="571.222" TYPE="SECTION">
<HEAD>§ 571.222 Standard No. 222; School bus passenger seating.</HEAD>
<P>S1. <I>Scope.</I> This standard establishes occupant protection requirements.</P>
<P>S5.1.2 Each school bus passenger seat shall be equipped with a restraining barrier.</P>
</DIV8>"""


def test_section_text_keeps_the_words_inside_nested_tags():
    """'Scope.' sits in an <I> inside a <P>. Dropping nested text would silently
    shorten quotes, and a shortened quote fails the verbatim provenance check
    for reasons that look like a model error."""
    assert "S1. Scope. This standard establishes occupant protection requirements." \
        in section_text(SAMPLE)


def test_section_text_separates_paragraphs():
    """Two paragraphs must not run together into a sentence that exists in no
    source document."""
    text = section_text(SAMPLE)
    assert "requirements. S5.1.2" in text or "requirements.\nS5.1.2" in text


def test_section_text_carries_the_section_heading():
    assert "571.222" in section_text(SAMPLE)


def test_section_text_normalises_whitespace():
    assert "  " not in section_text(SAMPLE)


def test_section_text_rejects_an_anti_scraping_page():
    """The failure this module exists to prevent: a block page entering the
    corpus as quotable law."""
    with pytest.raises(ValueError):
        section_text("Your request has been flagged as potentially automated.")


def test_cfr_url_is_the_citable_public_page_not_the_api():
    """The source list under the video has to point somewhere a viewer can read."""
    url = cfr_url(49, "571.222")
    assert url.startswith("https://www.ecfr.gov/current/title-49")
    assert "571.222" in url
    assert "/api/" not in url
