from why_engine.sources.corpus import Corpus
from why_engine.sources.fetch import chunk_text, html_to_text, pages_to_text
from why_engine.models import Passage


def _p(source_id="s", locator="c1", text="hello"):
    return Passage(source_id=source_id, locator=locator, text=text, url="https://a.gov", title="T")


def test_html_to_text_strips_tags_and_script_content():
    text = html_to_text("<html><body><script>var x=1;</script><p>Real content.</p></body></html>")
    assert "Real content." in text
    assert "var x" not in text


def test_chunk_text_assigns_sequential_locators():
    passages = chunk_text("a" * 2500, source_id="doc", url="https://a.gov", title="T", max_chars=1000)
    assert [p.locator for p in passages] == ["c0", "c1", "c2"]


def test_chunk_text_respects_max_chars():
    passages = chunk_text("b" * 2500, source_id="doc", url="https://a.gov", title="T", max_chars=1000)
    assert all(len(p.text) <= 1000 for p in passages)


def test_chunk_text_drops_whitespace_only_chunks():
    assert chunk_text("   \n\n  ", source_id="doc", url="https://a.gov", title="T") == []


def test_corpus_roundtrips_through_jsonl(tmp_path):
    corpus = Corpus(passages=[_p(locator="c0"), _p(locator="c1")])
    path = tmp_path / "corpus.jsonl"
    corpus.save(path)
    assert Corpus.load(path).passages == corpus.passages


def test_corpus_roundtrips_text_containing_a_unicode_line_separator(tmp_path):
    """U+2028 is unescaped by JSON but splits under str.splitlines(); save and
    load must use the same separator or real fetched text corrupts on reload."""
    p = Passage(source_id="s", locator="c0", text="before after",
                url="https://a.gov", title="T")
    path = tmp_path / "corpus.jsonl"
    Corpus(passages=[p]).save(path)
    assert Corpus.load(path).passages == [p]


def test_corpus_roundtrips_text_containing_a_newline(tmp_path):
    p = Passage(source_id="s", locator="c0", text="line one\nline two",
                url="https://a.gov", title="T")
    path = tmp_path / "corpus.jsonl"
    Corpus(passages=[p]).save(path)
    assert Corpus.load(path).passages == [p]


def test_corpus_get_finds_a_passage_by_source_and_locator():
    corpus = Corpus(passages=[_p(locator="c0", text="first"), _p(locator="c1", text="second")])
    assert corpus.get("s", "c1").text == "second"


def test_corpus_get_returns_none_for_a_missing_passage():
    assert Corpus(passages=[_p()]).get("s", "nope") is None


def test_prompt_block_is_deterministic_regardless_of_input_order():
    a = Corpus(passages=[_p(locator="c0"), _p(locator="c1")])
    b = Corpus(passages=[_p(locator="c1"), _p(locator="c0")])
    assert a.as_prompt_block() == b.as_prompt_block()


def test_prompt_block_labels_each_passage_with_its_provenance():
    block = Corpus(passages=[_p(source_id="gao-24", locator="c7", text="body text")]).as_prompt_block()
    assert "gao-24" in block
    assert "c7" in block
    assert "body text" in block


def test_prompt_block_is_deterministic_even_with_duplicate_locators():
    """Nothing enforces locator uniqueness, so the sort key must be total."""
    a = Passage(source_id="s", locator="c0", text="alpha", url="u", title="T")
    b = Passage(source_id="s", locator="c0", text="beta", url="u", title="T")
    assert Corpus(passages=[a, b]).as_prompt_block() == Corpus(passages=[b, a]).as_prompt_block()


def test_chunk_text_skips_interior_whitespace_without_gapping_locators():
    """A gap here would silently misalign every citation pointing at a later passage."""
    passages = chunk_text("AAAAA" + "     " + "BBBBB", source_id="doc", url="https://a.gov",
                           title="T", max_chars=5)
    assert [p.locator for p in passages] == ["c0", "c1"]


def test_pages_to_text_joins_pages_and_normalises_whitespace():
    assert pages_to_text(["A  page\n\nhere", "and   another"]) == "A page here and another"


def test_pages_to_text_tolerates_pages_that_extract_to_none():
    """pypdf returns None for pages it cannot extract; those must not crash."""
    assert pages_to_text(["real text", None, "more text"]) == "real text  more text"
