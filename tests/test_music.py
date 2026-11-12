"""Procedural bed. Level targets are measured from the reference; see music.py."""
import wave

import numpy as np
import pytest

from why_engine.audio.music import (
    DUCK_DB, OPEN_DB, SAMPLE_RATE, Window, bed, duck_envelope, write_bed,
    windows_from_segments,
)

SR = SAMPLE_RATE
db = lambda x: 10.0 ** (x / 20.0)


def test_bed_has_the_requested_length():
    assert len(bed(3.0)) == int(3.0 * SR)


def test_bed_is_deterministic_for_a_seed():
    assert np.array_equal(bed(2.0, seed=7), bed(2.0, seed=7))


def test_different_seeds_give_different_beds():
    assert not np.array_equal(bed(2.0, seed=1), bed(2.0, seed=2))


def test_bed_is_normalised_and_never_clips():
    b = bed(6.0, seed=3)
    assert np.abs(b).max() == pytest.approx(1.0, abs=1e-6)


def test_bed_is_not_silent():
    assert np.sqrt((bed(4.0, seed=4) ** 2).mean()) > 0.05


def test_bed_energy_is_low_frequency():
    """A bed sits under narration; energy up where the voice lives fights it."""
    b = bed(6.0, seed=5)
    spec = np.abs(np.fft.rfft(b)) ** 2
    freqs = np.fft.rfftfreq(len(b), 1 / SR)
    below_300 = spec[freqs < 300].sum() / spec.sum()
    assert below_300 > 0.80


def test_duck_envelope_is_quieter_under_speech():
    env = duck_envelope(10.0, [Window(3, 6)])
    assert env[int(4.5 * SR)] < env[int(1.0 * SR)]


def test_duck_depth_matches_the_measured_level():
    env = duck_envelope(12.0, [Window(4, 9)])
    ratio = env[int(6.5 * SR)] / env[int(1.5 * SR)]
    assert 20 * np.log10(ratio) == pytest.approx(DUCK_DB - OPEN_DB, abs=1.0)


def test_envelope_does_not_fade_at_the_edges():
    """Zero-padded smoothing silently fades the bed out at the start and end."""
    env = duck_envelope(8.0, [Window(3, 5)])
    assert env[0] == pytest.approx(db(OPEN_DB), rel=0.05)
    assert env[-1] == pytest.approx(db(OPEN_DB), rel=0.05)


def test_envelope_ramps_rather_than_stepping():
    """A hard step clicks."""
    env = duck_envelope(8.0, [Window(4, 6)])
    assert np.abs(np.diff(env)).max() < 0.002


def test_envelope_handles_back_to_back_windows():
    env = duck_envelope(9.0, [Window(1, 3), Window(3, 5)])
    assert env[int(2.0 * SR)] == pytest.approx(env[int(4.0 * SR)], rel=0.10)


def test_envelope_ignores_windows_past_the_end():
    env = duck_envelope(4.0, [Window(10, 12)])
    assert len(env) == int(4.0 * SR)
    assert env.min() == pytest.approx(db(OPEN_DB), rel=0.05)


def test_envelope_clamps_a_window_that_overruns():
    env = duck_envelope(4.0, [Window(3, 99)])
    assert len(env) == int(4.0 * SR)
    assert env[-1] < env[0]


def test_no_windows_leaves_the_bed_open_throughout():
    env = duck_envelope(5.0, [])
    assert env.min() == pytest.approx(db(OPEN_DB), rel=0.05)


def test_write_bed_produces_mono_16bit_at_the_right_rate(tmp_path):
    p = write_bed(tmp_path / "bed.wav", 3.0, [Window(1, 2)])
    with wave.open(str(p)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SR)
        assert w.getnframes() == pytest.approx(int(3.0 * SR), rel=0.01)


def test_written_bed_never_clips(tmp_path):
    p = write_bed(tmp_path / "bed.wav", 4.0, [Window(1, 2)], seed=9)
    with wave.open(str(p)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    assert np.abs(x).max() < 32767


def test_windows_from_segments_reads_the_timing_map():
    class S:
        def __init__(self, start, end):
            self.start, self.end = start, end
    ws = windows_from_segments([S(0.0, 1.5), S(1.9, 3.0)])
    assert [(w.start, w.end) for w in ws] == [(0.0, 1.5), (1.9, 3.0)]


def test_the_bed_is_generated_at_single_precision():
    """Arrays here are sized to the whole video: a 21.7-minute bed is 28.7
    million samples, 230MB per float64 array, and write_bed chained enough of
    them to peak at 1.18GB. It was killed for memory three times at this step.
    The output is 16-bit PCM, so float32 carries far more precision than
    survives the write."""
    import numpy as np
    assert bed(0.2, seed=0).dtype == np.float32


def test_the_duck_envelope_is_single_precision_too():
    import numpy as np
    assert duck_envelope(0.2, []).dtype == np.float32
