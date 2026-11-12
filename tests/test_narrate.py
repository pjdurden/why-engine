"""Narration timing. The alignment guarantee is what these tests protect."""
import shutil
import struct
import wave

import pytest

from why_engine.audio.narrate import (
    GAP_SECONDS, MAX_SYNTH_CHARS, Segment, narrate, shot_durations,
    split_sentences,
)


def test_splits_on_sentence_ends():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_does_not_split_on_abbreviations():
    """'St. Louis' is one sentence; splitting it would desync every later shot."""
    assert split_sentences("They shipped to St. Louis first.") == \
        ["They shipped to St. Louis first."]


def test_does_not_split_on_a_middle_initial():
    assert len(split_sentences("Spencer F. Baird led it.")) == 1


def test_drops_empty_fragments():
    assert split_sentences("  One.   \n\n  Two.  ") == ["One.", "Two."]


def test_rejects_empty_text():
    assert split_sentences("") == []


def test_no_chunk_exceeds_the_synthesis_limit():
    """An unpunctuated block is one 'sentence' to the splitter, and Piper sizes
    its tensors off input length: 480 words at once allocated 12GB and was
    OOM-killed, taking the whole session with it. The splitter is the only
    place that can bound what reaches the synthesiser."""
    block = " ".join(["word"] * 480)
    assert all(len(c) <= MAX_SYNTH_CHARS for c in split_sentences(block))


def test_a_long_sentence_breaks_at_clause_punctuation():
    """Breaking mid-clause is audible. Commas and semicolons are where a
    speaker would pause anyway, so they cost nothing."""
    left = "word " * 60
    right = "other " * 60
    chunks = split_sentences(f"{left.strip()}, {right.strip()}.")
    assert len(chunks) > 1
    assert chunks[0].endswith(",")


def test_a_short_sentence_is_never_broken():
    """The bound must not touch normal narration."""
    one = "Americans refuse to eat carp."
    assert split_sentences(one) == [one]


def test_a_single_token_over_the_limit_is_rejected():
    """No word boundary to break on means no safe way to synthesise it.
    Refusing is the designed outcome; allocating 12GB is not."""
    with pytest.raises(ValueError):
        split_sentences("w" * (MAX_SYNTH_CHARS + 1))


def test_shot_durations_include_the_trailing_gap():
    """The pause belongs to the outgoing shot -- cutting on the last syllable
    reads as a mistake."""
    segs = [Segment(0, "a", None, 2.0, 0.0), Segment(1, "b", None, 3.0, 2.38)]
    assert shot_durations(segs, gap=0.38) == [2.38, 3.38]


def test_shot_durations_sum_to_the_track_length():
    segs = [Segment(i, "x", None, 1.5, i * 1.88) for i in range(4)]
    assert sum(shot_durations(segs, gap=0.38)) == pytest.approx(segs[-1].end + 0.38)


def test_segment_end_follows_from_start_and_duration():
    assert Segment(0, "x", None, 2.5, 10.0).end == 12.5


def test_segments_are_contiguous_with_one_gap_each():
    """Shot N must begin exactly where shot N-1 ends. This is the guarantee."""
    durations = [1.2, 2.4, 0.9]
    cursor, segs = 0.0, []
    for i, d in enumerate(durations):
        segs.append(Segment(i, "x", None, d, cursor))
        cursor += d + GAP_SECONDS
    starts = [s.start for s in segs]
    cuts = [0.0]
    for d in shot_durations(segs):
        cuts.append(cuts[-1] + d)
    assert starts == pytest.approx(cuts[:-1])


@pytest.mark.skipif(shutil.which("piper") is None or shutil.which("ffmpeg") is None,
                    reason="needs piper and ffmpeg")
def test_narrate_measures_duration_from_the_rendered_audio(tmp_path):
    """Durations must come from the audio, never from a word-count estimate."""
    model = "assets/voices/en_US-ryan-high.onnx"
    combined, segs = narrate("Yes. No.", model, tmp_path)
    assert len(segs) == 2
    for s in segs:
        with wave.open(str(s.wav)) as w:
            assert s.duration == pytest.approx(w.getnframes() / w.getframerate())
    with wave.open(str(combined)) as w:
        total = w.getnframes() / w.getframerate()
    assert total == pytest.approx(sum(shot_durations(segs)), abs=0.05)
