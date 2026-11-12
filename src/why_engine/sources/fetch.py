from typing import Iterable, List, Optional

from selectolax.parser import HTMLParser

from why_engine.models import Passage


def html_to_text(html: str) -> str:
    tree = HTMLParser(html)
    for tag in tree.css("script, style, nav, footer"):
        tag.decompose()
    body = tree.body if tree.body is not None else tree.root
    return " ".join(body.text(separator=" ").split())


def chunk_text(text: str, source_id: str, url: str, title: str, max_chars: int = 1200) -> List[Passage]:
    passages: List[Passage] = []
    index = 0
    for start in range(0, len(text), max_chars):
        chunk = text[start:start + max_chars].strip()
        if not chunk:
            continue
        passages.append(
            Passage(source_id=source_id, locator=f"c{index}", text=chunk, url=url, title=title)
        )
        index += 1
    return passages


def pages_to_text(pages: Iterable[Optional[str]]) -> str:
    """Join per-page extracted text, normalising whitespace and tolerating pages
    that extract to None. Kept separate from fetch_document so the parsing is
    testable without a PDF fixture."""
    return " ".join(" ".join((page or "").split()) for page in pages)


def fetch_document(url: str, source_id: str, title: str) -> List[Passage]:
    """Network IO. Kept free of parsing logic so the parsing above stays testable."""
    import httpx2 as httpx

    response = httpx.get(url, follow_redirects=True, timeout=60.0,
                         headers={"User-Agent": "why-engine/0.1 (research)"})
    response.raise_for_status()

    if url.lower().endswith(".pdf") or "application/pdf" in response.headers.get("content-type", ""):
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(response.content))
        text = pages_to_text(page.extract_text() for page in reader.pages)
    else:
        text = html_to_text(response.text)

    return chunk_text(text, source_id=source_id, url=url, title=title)
