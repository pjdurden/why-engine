import json

import pytest

from why_engine.config import Settings
from why_engine.llm import StubLLM
from why_engine.models import Claim, Passage, Script, Segment
from why_engine.run import main, produce_script, produce_video
from why_engine.script.retention import check_retention
from why_engine.sources.corpus import Corpus
from why_engine.topics.ledger import PublishedTopics

QUOTE = "In 1877 the Commission distributed carp."


def _corpus():
    return Corpus(passages=[Passage(source_id="gao-24", locator="c0", text=QUOTE,
                                    url="https://gao.gov/x", title="T")])


def _script(n_body_words=2400, claim_text="The Commission distributed carp in 1877."):
    """Body is split across five acts to satisfy the per-act ceiling. The single
    claim lives in segment 1, which test_fabricated_provenance mutates."""
    per_act = max(1, n_body_words // 5)
    # Punctuated on purpose: an unbroken block of words is a single sentence to
    # the splitter, and the synthesiser sizes its tensors off sentence length.
    filler = " ".join(" ".join(["word"] * 9) + "." for _ in range(max(1, per_act // 9)))
    body = [
        Segment(segment_id=i, act=i, narration=filler, claims=[], image_brief="a fish")
        for i in range(1, 6)
    ]
    body[0].claims = [Claim(text=claim_text, source_id="gao-24", locator="c0", quote=QUOTE)]
    return Script(topic="Why don't Americans eat carp?", segments=[
        Segment(segment_id=0, act=0, narration="Americans refuse to eat carp.",
                claims=[], image_brief="a river"),
        *body,
        Segment(segment_id=6, act=6, narration="That is why Americans refuse to eat carp.",
                claims=[], image_brief="a plate"),
    ])


def test_a_clean_script_is_returned_and_persisted(tmp_path):
    llm = StubLLM([_script()])
    result = produce_script("Why don't Americans eat carp?", _corpus(), llm, Settings(), tmp_path)
    assert result is not None
    assert (tmp_path / "script_attempt_1.json").exists()
    assert (tmp_path / "verdicts.json").exists()


def test_produce_script_snapshots_the_corpus_into_the_run_dir(tmp_path):
    """The run directory must be self-contained: resolving source_id to a URL
    later must not depend on the external corpus.jsonl, which spec section 15
    expects to rot."""
    llm = StubLLM([_script()])
    corpus = _corpus()
    produce_script("Why don't Americans eat carp?", corpus, llm, Settings(), tmp_path)
    snapshot_path = tmp_path / "corpus_snapshot.jsonl"
    assert snapshot_path.exists()
    assert Corpus.load(snapshot_path).passages == corpus.passages


def test_an_invented_figure_forces_regeneration(tmp_path):
    """The gate's remaining job, end to end: the quote is real and verbatim in
    the corpus, but the claim asserts a number the quote does not contain."""
    bad = _script(claim_text="The Commission distributed 40,000 carp in 1877.")
    llm = StubLLM([bad, _script()])
    result = produce_script("t", _corpus(), llm, Settings(), tmp_path)
    assert result is not None
    assert (tmp_path / "script_attempt_2.json").exists()


def test_a_persistent_invented_figure_publishes_nothing(tmp_path):
    """Publishing nothing is an acceptable outcome. Publishing the figure is not."""
    bad = _script(claim_text="The Commission distributed 40,000 carp in 1877.")
    llm = StubLLM([bad, bad])
    assert produce_script("t", _corpus(), llm, Settings(max_regenerations=2), tmp_path) is None


def test_the_missing_tokens_are_fed_back_into_the_next_write_call(tmp_path):
    """A rewrite told only 'rejected' repeats the mistake."""
    bad = _script(claim_text="The Commission distributed 40,000 carp in 1877.")
    llm = StubLLM([bad, _script()])
    produce_script("t", _corpus(), llm, Settings(), tmp_path)
    assert "40000" in llm.calls[1]["user"]


def test_fabricated_provenance_forces_regeneration(tmp_path):
    bad = _script()
    bad.segments[1].claims[0].source_id = "invented-source"
    llm = StubLLM([bad, _script()])
    result = produce_script("t", _corpus(), llm, Settings(), tmp_path)
    assert result is not None
    assert (tmp_path / "script_attempt_2.json").exists()


def test_retention_violation_forces_regeneration(tmp_path):
    llm = StubLLM([_script(n_body_words=50), _script()])
    assert produce_script("t", _corpus(), llm, Settings(), tmp_path) is not None


def test_retention_failure_feeds_the_error_back_into_the_next_write_call(tmp_path):
    """Attempts 2 and 3 must not be a byte-identical retry of attempt 1."""
    llm = StubLLM([_script(n_body_words=50), _script()])
    produce_script("t", _corpus(), llm, Settings(), tmp_path)
    retention_errors = check_retention(_script(n_body_words=50), Settings())
    assert retention_errors
    for reason in retention_errors:
        assert reason in llm.calls[1]["user"]


def test_exhausting_regenerations_returns_none(tmp_path):
    settings = Settings(max_regenerations=2)
    llm = StubLLM([_script(n_body_words=50), _script(n_body_words=50)])
    assert produce_script("t", _corpus(), llm, settings, tmp_path) is None


def test_verdicts_file_records_every_attempt_and_its_outcome(tmp_path):
    llm = StubLLM([_script(n_body_words=50), _script()])
    produce_script("t", _corpus(), llm, Settings(), tmp_path)
    verdicts = json.loads((tmp_path / "verdicts.json").read_text())
    assert len(verdicts["attempts"]) == 2
    assert verdicts["attempts"][0]["stage"] == "retention"
    assert verdicts["final"] == "accepted"


def test_attempt_files_are_never_overwritten(tmp_path):
    llm = StubLLM([_script(n_body_words=50), _script()])
    produce_script("t", _corpus(), llm, Settings(), tmp_path)
    assert (tmp_path / "script_attempt_1.json").exists()
    assert (tmp_path / "script_attempt_2.json").exists()


def test_an_exception_mid_run_still_writes_verdicts_before_surfacing(tmp_path):
    """Any anthropic exception from write_script or verify_claim must not leave
    the run with no audit record at all."""
    class _RaisingLLM:
        def parse(self, *, system, user, output_format, effort="high", cache_system=False):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        produce_script("t", _corpus(), _RaisingLLM(), Settings(), tmp_path)

    verdicts = json.loads((tmp_path / "verdicts.json").read_text())
    assert verdicts["final"] == "errored"
    assert not (tmp_path / "script.json").exists()


def test_main_returns_zero_and_prints_cost_on_success(tmp_path, monkeypatch, capsys):
    class _FakeLLM:
        total_cost_usd = 1.23

        def __init__(self, settings):
            pass

        def parse(self, *, system, user, output_format, effort="high", cache_system=False):
            return _script() if output_format is Script else EntailmentResult(
                verdict="SUPPORTED", reasoning="r")

    monkeypatch.setattr("why_engine.run.AnthropicLLM", _FakeLLM)
    corpus_path = tmp_path / "corpus.jsonl"
    _corpus().save(corpus_path)
    code = main(["--topic", "t", "--corpus", str(corpus_path), "--script-only",
                 "--run-dir", str(tmp_path / "run"), "--ledger", str(tmp_path / "l.jsonl")])
    assert code == 0
    assert "cost:" in capsys.readouterr().out


def test_main_returns_one_when_no_script_is_produced(tmp_path, monkeypatch):
    """Publishing nothing is an acceptable outcome, signalled by exit code, not
    by an exception."""
    class _FakeLLM:
        total_cost_usd = 0.0

        def __init__(self, settings):
            pass

        def parse(self, *, system, user, output_format, effort="high", cache_system=False):
            return _script(n_body_words=50) if output_format is Script else EntailmentResult(
                verdict="SUPPORTED", reasoning="r")

    monkeypatch.setattr("why_engine.run.AnthropicLLM", _FakeLLM)
    corpus_path = tmp_path / "corpus.jsonl"
    _corpus().save(corpus_path)
    code = main(["--topic", "t", "--corpus", str(corpus_path),
                 "--run-dir", str(tmp_path / "run"), "--ledger", str(tmp_path / "l.jsonl")])
    assert code == 1


def test_produce_video_renders_a_playable_file(tmp_path):
    """The render half of the orchestrator had never actually executed: the one
    run that reached it OOM-killed the machine before it returned.

    Deliberately a tiny script rather than the gate-passing one: this asserts
    the wiring from Script to finished file, and a 2,400-word fixture would
    spend minutes in the synthesiser proving nothing extra.
    """
    script = Script(topic="Why don't Americans eat carp?", segments=[
        Segment(segment_id=0, act=0, claims=[], image_brief="a river",
                narration="Americans refuse to eat carp. The fish is everywhere."),
    ])
    build = produce_video(script, tmp_path, music=False)
    assert build.video.exists() and build.video.stat().st_size > 0
    assert build.duration > 0
    assert len(build.shots) == 2


def _fake_llm_returning(script):
    class _FakeLLM:
        total_cost_usd = 1.23

        def __init__(self, settings):
            _FakeLLM.constructed = True

        def parse(self, *, system, user, output_format, effort="high", cache_system=False):
            return script
    _FakeLLM.constructed = False
    return _FakeLLM


def test_a_topic_already_covered_is_refused_before_anything_is_spent(tmp_path, monkeypatch):
    """The dedupe has to run before the first token, not after the script comes
    back. Exit 2 rather than 1: nothing failed, the caller just needs a
    different question."""
    ledger = tmp_path / "topics.jsonl"
    PublishedTopics.load(ledger).record("Why are American school buses yellow?")
    fake = _fake_llm_returning(_script())
    monkeypatch.setattr("why_engine.run.AnthropicLLM", fake)
    corpus_path = tmp_path / "corpus.jsonl"
    _corpus().save(corpus_path)

    code = main(["--topic", "Why is the American school bus yellow?",
                 "--corpus", str(corpus_path), "--run-dir", str(tmp_path / "run"),
                 "--ledger", str(ledger), "--script-only"])

    assert code == 2
    assert fake.constructed is False


def test_a_fresh_topic_is_recorded_so_tomorrow_will_not_repeat_it(tmp_path, monkeypatch):
    ledger = tmp_path / "topics.jsonl"
    monkeypatch.setattr("why_engine.run.AnthropicLLM", _fake_llm_returning(_script()))
    corpus_path = tmp_path / "corpus.jsonl"
    _corpus().save(corpus_path)

    main(["--topic", "Why are American school buses yellow?", "--corpus", str(corpus_path),
          "--run-dir", str(tmp_path / "run"), "--ledger", str(ledger), "--script-only"])

    assert PublishedTopics.load(ledger).questions == ["Why are American school buses yellow?"]


def test_a_topic_that_produced_nothing_is_not_recorded(tmp_path, monkeypatch):
    """Recording a topic the run failed to make would burn it permanently."""
    ledger = tmp_path / "topics.jsonl"
    monkeypatch.setattr("why_engine.run.AnthropicLLM",
                        _fake_llm_returning(_script(n_body_words=50)))
    corpus_path = tmp_path / "corpus.jsonl"
    _corpus().save(corpus_path)

    code = main(["--topic", "Why are American school buses yellow?", "--corpus", str(corpus_path),
                 "--run-dir", str(tmp_path / "run"), "--ledger", str(ledger), "--script-only"])

    assert code == 1
    assert PublishedTopics.load(ledger).questions == []


def test_the_llm_backend_is_selectable(tmp_path, monkeypatch):
    """ANTHROPIC_API_KEY is empty on this machine and the pipeline is designed to
    run on headless Claude Code, so the backend cannot be hardcoded to the API
    client the way it was."""
    built = {}

    class _FakeCC:
        total_cost_usd = 0.0

        def __init__(self, settings, **kwargs):
            built["kwargs"] = kwargs

        def parse(self, *, system, user, output_format, effort="high", cache_system=False):
            return _script()

    monkeypatch.setattr("why_engine.run.ClaudeCodeLLM", _FakeCC)
    corpus_path = tmp_path / "corpus.jsonl"
    _corpus().save(corpus_path)

    code = main(["--topic", "t", "--corpus", str(corpus_path), "--script-only",
                 "--llm", "claude-code", "--run-dir", str(tmp_path / "run"),
                 "--ledger", str(tmp_path / "l.jsonl")])

    assert code == 0
    assert built  # the Claude Code client was the one constructed


def test_the_api_client_is_still_the_default(tmp_path, monkeypatch):
    monkeypatch.setattr("why_engine.run.AnthropicLLM", _fake_llm_returning(_script()))
    corpus_path = tmp_path / "corpus.jsonl"
    _corpus().save(corpus_path)
    assert main(["--topic", "t", "--corpus", str(corpus_path), "--script-only",
                 "--run-dir", str(tmp_path / "run"),
                 "--ledger", str(tmp_path / "l.jsonl")]) == 0


def test_feedback_accumulates_across_attempts(tmp_path):
    """Observed on the school-bus run: attempt 1 failed on a non-verbatim quote,
    attempt 2 fixed it and failed on a different one, attempt 3 fixed that and
    blew the act-length ceiling it had been meeting all along. Each rewrite was
    told only about its predecessor's mistake, so it kept trading one for
    another until the attempts ran out."""
    bad_quote = _script()
    bad_quote.segments[1].claims[0].quote = "a quote that is in no passage at all"
    too_short = _script(n_body_words=50)
    llm = StubLLM([bad_quote, too_short, _script()])

    produce_script("t", _corpus(), llm, Settings(max_regenerations=3), tmp_path)

    third_prompt = llm.calls[2]["user"]
    assert "not verbatim" in third_prompt      # attempt 1's mistake, still remembered
    assert "runtime too short" in third_prompt  # attempt 2's mistake


def test_a_topic_whose_render_fails_is_not_recorded(tmp_path, monkeypatch):
    """Hit for real: the script was accepted, the renderer then died because
    piper was not on PATH, and the topic had already been written to the ledger.
    A retry would have been refused as a duplicate of a video that does not
    exist. Nothing is recorded until something has actually been made."""
    ledger = tmp_path / "topics.jsonl"
    monkeypatch.setattr("why_engine.run.AnthropicLLM", _fake_llm_returning(_script()))

    def _explode(*a, **k):
        raise FileNotFoundError("piper")
    monkeypatch.setattr("why_engine.run.produce_video", _explode)

    corpus_path = tmp_path / "corpus.jsonl"
    _corpus().save(corpus_path)

    with pytest.raises(FileNotFoundError):
        main(["--topic", "Why are American school buses yellow?", "--corpus", str(corpus_path),
              "--run-dir", str(tmp_path / "run"), "--ledger", str(ledger)])

    assert PublishedTopics.load(ledger).questions == []
