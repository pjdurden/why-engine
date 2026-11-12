"""The voice chain is fitted to a measured target, so the tests assert the fit."""
import math
import shutil
import struct
import wave

import pytest

from why_engine.audio.voice import BAND_TARGET, BANDS, CHAIN, band_profile, fit_error, voice

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def tone_wav(path, freqs, seconds=2.0, sr=22050):
    """Equal-amplitude tones, one per frequency -- a flat probe across bands."""
    n = int(sr * seconds)
    frames = bytearray()
    for i in range(n):
        v = sum(math.sin(2 * math.pi * f * i / sr) for f in freqs) / len(freqs)
        frames += struct.pack("<h", int(v * 20000))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))
    return path


def test_band_target_matches_the_band_edges():
    assert len(BAND_TARGET) == len(BANDS)
    assert sum(BAND_TARGET) == pytest.approx(100.0, abs=0.1)


def test_band_target_peaks_where_the_reference_peaks():
    """The reference's dominant band is 160-320 Hz -- that is the whole finding."""
    assert BANDS[BAND_TARGET.index(max(BAND_TARGET))] == (160, 320)


def test_band_profile_sums_to_one_hundred(tmp_path):
    profile = band_profile(tone_wav(tmp_path / "probe.wav", [250, 500, 1000]))
    assert sum(profile) == pytest.approx(100.0, abs=0.5)


def test_band_profile_finds_the_band_a_tone_sits_in(tmp_path):
    profile = band_profile(tone_wav(tmp_path / "one.wav", [250]))
    assert BANDS[profile.index(max(profile))] == (160, 320)


def test_chain_boosts_the_low_mids_relative_to_the_mids(tmp_path):
    """250 Hz is the chest band; 900 Hz is the throat band it was losing to."""
    src = tone_wav(tmp_path / "src.wav", [250, 900])
    before = band_profile(src)
    after = band_profile(voice(src, tmp_path / "out.wav"))
    i_low, i_mid = BANDS.index((160, 320)), BANDS.index((640, 1280))
    assert after[i_low] / after[i_mid] > 3.0 * (before[i_low] / before[i_mid])


def test_chain_attenuates_the_top_end(tmp_path):
    src = tone_wav(tmp_path / "src.wav", [250, 4000])
    before = band_profile(src)
    after = band_profile(voice(src, tmp_path / "out.wav"))
    i_top = BANDS.index((2560, 5120))
    assert after[i_top] < before[i_top]


def test_chain_output_is_mono_22050(tmp_path):
    out = voice(tone_wav(tmp_path / "src.wav", [250]), tmp_path / "out.wav")
    with wave.open(str(out)) as w:
        assert (w.getnchannels(), w.getframerate()) == (1, 22050)


def test_chain_preserves_duration(tmp_path):
    """asetrate speeds the audio up; atempo must put the speed back."""
    src = tone_wav(tmp_path / "src.wav", [250], seconds=3.0)
    out = voice(src, tmp_path / "out.wav")
    with wave.open(str(src)) as a, wave.open(str(out)) as b:
        assert b.getnframes() / b.getframerate() == pytest.approx(
            a.getnframes() / a.getframerate(), rel=0.02)


def test_chain_does_not_clip(tmp_path):
    """The raw Piper narration peaked at 1.000; the limiter exists to stop that."""
    import numpy as np
    out = voice(tone_wav(tmp_path / "src.wav", [250, 500]), tmp_path / "out.wav")
    with wave.open(str(out)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    assert np.abs(x).max() / 32768.0 < 0.99


def test_fit_error_rewards_the_reference_profile(tmp_path):
    """A probe concentrated in the reference's peak band must beat one an octave up."""
    on_target = fit_error(tone_wav(tmp_path / "a.wav", [200, 250, 300]))
    octave_up = fit_error(tone_wav(tmp_path / "b.wav", [400, 500, 600]))
    assert on_target < octave_up


def test_chain_is_a_single_ffmpeg_filter_string():
    assert "," in CHAIN and "\n" not in CHAIN


# --- pitch targeting ---------------------------------------------------------

from why_engine.audio.voice import (  # noqa: E402
    REF_F0_HZ, median_f0, pitch_chain, voice_at_reference_pitch,
)


def test_reference_f0_is_the_measured_value():
    assert REF_F0_HZ == pytest.approx(170.9, abs=0.1)


@pytest.mark.parametrize("hz", [110.0, 170.0, 240.0])
def test_median_f0_recovers_a_known_tone(tmp_path, hz):
    """A pure tone's F0 is its frequency; the estimator must find it."""
    path = tone_wav(tmp_path / f"t{int(hz)}.wav", [hz], seconds=1.5)
    assert median_f0(path) == pytest.approx(hz, rel=0.05)


def test_median_f0_separates_a_low_voice_from_a_high_one(tmp_path):
    """This is the check that tells an unlabelled community model's sex."""
    low = median_f0(tone_wav(tmp_path / "low.wav", [120], seconds=1.5))
    high = median_f0(tone_wav(tmp_path / "high.wav", [215], seconds=1.5))
    assert high > low * 1.5


def test_median_f0_rejects_audio_too_short_to_measure(tmp_path):
    with pytest.raises(ValueError, match="too short"):
        median_f0(tone_wav(tmp_path / "tiny.wav", [200], seconds=0.02))


def test_pitch_chain_ratio_of_one_is_a_no_op_pair():
    chain = pitch_chain(1.0)
    assert "asetrate=22050*1.0000" in chain
    assert "atempo=1.0000" in chain


def test_pitch_chain_tempo_inverts_the_resample():
    """Duration must survive the shift, or shot alignment breaks."""
    chain = pitch_chain(0.90)
    rate = float(chain.split("asetrate=22050*")[1].split(",")[0])
    tempo = float(chain.split("atempo=")[1])
    assert rate * tempo == pytest.approx(1.0, abs=1e-3)


def test_voice_at_reference_pitch_moves_a_high_voice_down(tmp_path):
    src = tone_wav(tmp_path / "high.wav", [230, 460], seconds=2.0)
    dst, measured = voice_at_reference_pitch(src, tmp_path / "out.wav")
    assert measured > REF_F0_HZ
    assert median_f0(dst) < measured


def test_voice_at_reference_pitch_reports_the_source_pitch(tmp_path):
    src = tone_wav(tmp_path / "src.wav", [200], seconds=2.0)
    _, measured = voice_at_reference_pitch(src, tmp_path / "out.wav")
    assert measured == pytest.approx(median_f0(src), rel=0.02)


def test_voice_at_reference_pitch_preserves_duration(tmp_path):
    src = tone_wav(tmp_path / "src.wav", [210], seconds=3.0)
    dst, _ = voice_at_reference_pitch(src, tmp_path / "out.wav")
    with wave.open(str(src)) as a, wave.open(str(dst)) as b:
        assert b.getnframes() / b.getframerate() == pytest.approx(
            a.getnframes() / a.getframerate(), rel=0.03)


def test_voice_at_reference_pitch_clamps_extreme_shifts(tmp_path):
    """A voice far from the target is moved toward it, not onto it: the ratio
    that would be needed makes speech sound synthetic."""
    src = tone_wav(tmp_path / "verylow.wav", [80], seconds=2.0)
    dst, measured = voice_at_reference_pitch(src, tmp_path / "out.wav")
    assert median_f0(dst) < REF_F0_HZ  # clamped at 1.35x, cannot reach 171
