"""Sentence-aligned narration, and the timing map the composer cuts against.

The composer used to pick shot durations by hand and lay a single narration
track over the top. Nothing held the two together, so the picture ran ahead of
the voice and the drift accumulated across the video -- the further in, the
worse the mismatch.

The fix is to stop guessing. Synthesise each sentence as its own clip, measure
what it actually came out at, and let that measurement set the shot's length.
Alignment then holds by construction rather than by tuning, and it cannot drift:
segment N of the audio starts exactly where shot N does, because the same
number produced both.

Gaps between sentences are part of the design, not slack. The reference video
has speech active only ~65% of the time.
"""
from __future__ import annotations

import re
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from why_engine.audio.voice import CHAIN

#: Silence inserted after each sentence. The reference leaves room to breathe.
GAP_SECONDS = 0.38

#: Piper prosody, chosen by sweeping noise against measured pitch spread.
#: These lift relative modulation from 0.165 at Piper's defaults to 0.270.
NOISE_SCALE = 0.85
NOISE_W_SCALE = 1.25

#: Longest text handed to Piper in one call. Piper sizes its tensors off input
#: length, and it does so without a ceiling: a single unpunctuated 480-word
#: block asked for 32GB of address space, took 12GB resident, and was
#: OOM-killed along with everything else sharing the machine. Nothing upstream
#: guarantees the model writes in sentences, so the bound has to live here.
MAX_SYNTH_CHARS = 300

#: Abbreviations that must not end a sentence when splitting. Each lookbehind
#: includes the period: they are evaluated at the position just after the
#: punctuation, so the text ending there is "Mr." and not "Mr".
_ABBREV = r"(?<!\bMr\.)(?<!\bMrs\.)(?<!\bDr\.)(?<!\bSt\.)(?<!\bNo\.)(?<!\b[A-Z]\.)"
_SPLIT = re.compile(rf"(?<=[.!?]){_ABBREV}\s+")

#: Where an over-long sentence may be broken. These are places a speaker would
#: pause anyway, so a cut here is inaudible; a cut mid-clause is not.
_CLAUSE = re.compile(r"(?<=[,;:\u2014])\s+")


@dataclass(frozen=True)
class Prosody:
    """How one sentence should be spoken relative to the others."""

    length_scale: float
    pitch_ratio: float


def prosody(sentence: str, index: int) -> Prosody:
    """Per-sentence pace and pitch, from cheap signals in the text.

    Piper's own noise parameters only buy so much: sweeping them lifts relative
    pitch spread from 0.165 to 0.270 against the reference's 0.453, because
    they add jitter rather than intent. The rest of the reference's range comes
    from sentences differing FROM EACH OTHER -- a short line landing low and
    heavy, a long one running lighter and quicker -- and that is controllable
    here, where each sentence is already synthesised on its own.

    Deterministic: the same sentence at the same index always speaks the same
    way, so a re-render is identical and the timing map stays reproducible.
    """
    words = sentence.split()
    length, pitch = 1.0, 1.0

    if len(words) <= 4:
        # A short line is a beat. Slow it and drop it; rushing it throws it away.
        length, pitch = length * 1.10, pitch * 0.96
    elif len(words) >= 16:
        length, pitch = length * 0.96, pitch * 1.02

    if any(c.isdigit() for c in sentence):
        length *= 1.06  # let the figure land

    if sentence.rstrip().endswith("?"):
        pitch *= 1.05

    # A small alternating tilt so consecutive similar sentences do not sit on
    # exactly the same note, which is what reads as robotic.
    pitch *= 1.0 + (0.018 if index % 2 == 0 else -0.018)
    return Prosody(round(length, 3), round(pitch, 4))


@dataclass(frozen=True)
class Segment:
    """One sentence, its audio, and where it lands in the finished track."""

    index: int
    text: str
    wav: Path
    duration: float
    start: float
    prosody: Prosody | None = None

    @property
    def end(self) -> float:
        return self.start + self.duration


def _pack(pieces: list[str], limit: int = MAX_SYNTH_CHARS) -> list[str]:
    """Greedily rejoin `pieces` into the fewest runs no longer than `limit`.

    A piece that exceeds the limit on its own is emitted whole; the caller
    decides what to do about it, because the right answer depends on whether
    there is a finer boundary left to try.
    """
    runs: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}" if current else piece
        if current and len(candidate) > limit:
            runs.append(current)
            current = piece
        else:
            current = candidate
    if current:
        runs.append(current)
    return runs


def _bound(sentence: str) -> list[str]:
    """Break one sentence down until every part is safe to synthesise.

    Clause punctuation first, word boundaries only if that was not enough. A
    word longer than the whole budget has no boundary left, so it is refused:
    publishing nothing is an acceptable outcome here, and handing it to Piper
    is not.
    """
    if len(sentence) <= MAX_SYNTH_CHARS:
        return [sentence]
    chunks: list[str] = []
    for clause in _pack(_CLAUSE.split(sentence)):
        if len(clause) <= MAX_SYNTH_CHARS:
            chunks.append(clause)
            continue
        for word in clause.split():
            if len(word) > MAX_SYNTH_CHARS:
                raise ValueError(
                    f"unsplittable {len(word)}-character token, over the "
                    f"{MAX_SYNTH_CHARS}-character synthesis limit: {word[:40]}...")
        chunks.extend(_pack(clause.split()))
    return chunks


def split_sentences(text: str) -> list[str]:
    """Split narration into sentences without breaking on abbreviations.

    Sentences too long to synthesise are broken further, so every element of
    the result is something Piper can be handed safely. The composer splits
    through this same function, so picture and voice stay in step.
    """
    sentences = [s.strip() for s in _SPLIT.split(text.strip()) if s.strip()]
    return [chunk for s in sentences for chunk in _bound(s)]


def _retune(chain: str, ratio: float) -> str:
    """Multiply the chain's existing asetrate/atempo pair by `ratio`.

    Editing the pair already in the chain rather than prepending a second one
    matters: two resamplers in series compound their tempo corrections, and the
    duration error lands straight in the timing map.
    """
    import re as _re

    def scale_rate(m):
        return f"asetrate={m.group(1)}*{float(m.group(2)) * ratio:.4f}"

    def scale_tempo(m):
        return f"atempo={float(m.group(1)) / ratio:.4f}"

    chain = _re.sub(r"asetrate=(\d+)\*([\d.]+)", scale_rate, chain, count=1)
    return _re.sub(r"atempo=([\d.]+)", scale_tempo, chain, count=1)


def _duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def narrate(
    text: str,
    model: Path | str,
    out_dir: Path | str,
    gap: float = GAP_SECONDS,
    chain: str = CHAIN,
    modulate: bool = True,
) -> tuple[Path, list[Segment]]:
    """Synthesise `text` one sentence at a time and concatenate with gaps.

    Returns the combined WAV and the timing map. Each Segment's `duration` is
    measured from the rendered audio, never estimated from word count, so a
    shot cut to it is aligned by definition.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("narrate() got no sentences")

    segments, cursor = [], 0.0
    for i, sentence in enumerate(sentences):
        raw = out_dir / f"seg{i:02d}_raw.wav"
        wav = out_dir / f"seg{i:02d}.wav"
        pro = prosody(sentence, i) if modulate else Prosody(1.0, 1.0)
        cmd = ["piper", "-m", str(model), "-f", str(raw),
               "--length-scale", str(pro.length_scale)]
        if modulate:
            cmd += ["--noise-scale", str(NOISE_SCALE), "--noise-w-scale", str(NOISE_W_SCALE)]
        subprocess.run(cmd, input=sentence, text=True, check=True, capture_output=True)
        # The sentence's own pitch tilt rides on top of the global chain.
        af = chain if pro.pitch_ratio == 1.0 else _retune(chain, pro.pitch_ratio)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(raw), "-af", af,
                        "-ar", "22050", "-ac", "1", "-y", str(wav)], check=True)
        d = _duration(wav)
        segments.append(Segment(i, sentence, wav, d, cursor, pro))
        cursor += d + gap

    combined = out_dir / "narration.wav"
    listing = out_dir / "concat.txt"
    silence = out_dir / "gap.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    f"anullsrc=r=22050:cl=mono:d={gap}", "-ar", "22050", "-ac", "1",
                    "-y", str(silence)], check=True)
    parts = []
    for s in segments:
        parts += [f"file '{s.wav.resolve()}'", f"file '{silence.resolve()}'"]
    listing.write_text("\n".join(parts) + "\n")
    subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-c", "copy", "-y", str(combined)], check=True)
    return combined, segments


def shot_durations(segments: list[Segment], gap: float = GAP_SECONDS) -> list[float]:
    """Shot length per segment: its speech plus the pause that follows it.

    The gap belongs to the outgoing shot. Cutting on the last syllable reads as
    a mistake; letting the shot hold through the pause reads as a beat.
    """
    return [s.duration + gap for s in segments]
