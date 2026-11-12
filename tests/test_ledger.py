"""The published-topic ledger.

`mine.py` has scored novelty against a `published` list since the start, but
nothing ever supplied one: no caller imported it and no run recorded what it
made. Dedupe that cannot outlive the process is not dedupe. This is the file
that remembers.
"""
from why_engine.topics.ledger import PublishedTopics


def test_a_fresh_ledger_is_empty(tmp_path):
    assert PublishedTopics.load(tmp_path / "topics.jsonl").questions == []


def test_a_recorded_topic_survives_a_reload(tmp_path):
    """The whole point: tomorrow's run is a different process."""
    path = tmp_path / "topics.jsonl"
    PublishedTopics.load(path).record("Why are American school buses yellow?")
    assert PublishedTopics.load(path).questions == ["Why are American school buses yellow?"]


def test_recording_appends_rather_than_replacing(tmp_path):
    path = tmp_path / "topics.jsonl"
    ledger = PublishedTopics.load(path)
    ledger.record("Why are American school buses yellow?")
    ledger.record("Why is American bread sweet?")
    assert len(PublishedTopics.load(path).questions) == 2


def test_an_exact_repeat_is_a_duplicate(tmp_path):
    ledger = PublishedTopics.load(tmp_path / "topics.jsonl")
    ledger.record("Why are American school buses yellow?")
    assert ledger.is_duplicate("Why are American school buses yellow?")


def test_a_rewording_is_a_duplicate(tmp_path):
    """The failure this is built to stop is making the same video twice under a
    different title, months apart, having forgotten."""
    ledger = PublishedTopics.load(tmp_path / "topics.jsonl")
    ledger.record("Why are American school buses yellow?")
    assert ledger.is_duplicate("Why is the American school bus yellow?")


def test_a_different_topic_is_not_a_duplicate(tmp_path):
    ledger = PublishedTopics.load(tmp_path / "topics.jsonl")
    ledger.record("Why are American school buses yellow?")
    assert not ledger.is_duplicate("Why is American bread so sweet?")


def test_the_ledger_records_when_a_topic_ran(tmp_path):
    """Kill criteria in spec section 15 are measured per week and per month, so
    the archive has to carry dates, not just questions."""
    path = tmp_path / "topics.jsonl"
    PublishedTopics.load(path).record("Why are American school buses yellow?")
    assert PublishedTopics.load(path).entries[0].published_on


def test_a_missing_parent_directory_is_created(tmp_path):
    """First run on a clean checkout must not fail on a missing state/ dir."""
    path = tmp_path / "state" / "topics.jsonl"
    PublishedTopics.load(path).record("Why are American school buses yellow?")
    assert path.exists()
