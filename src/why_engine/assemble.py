"""Assembler: a verified Script becomes a finished video.

Spec S7.13. This is the seam the pipeline was missing -- the script half and
the picture half both worked and nothing joined them.

The order matters and is not arbitrary. Narration is rendered FIRST, because
its measured per-sentence durations are what every later stage cuts against:
shot lengths, motion curves, the whip's budget, and the music duck all derive
from the same timing map. Anything that picks a duration independently
reintroduces the drift this module exists to prevent.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from why_engine.art.compose import H, W, compose
from why_engine.art.direct import Shot, direct
from why_engine.art.motion import PRESETS, Zoomer, shot_moves, zoom_curve
from why_engine.art.transitions import WHIP_WINDOW_SECONDS, whip_sequence
from why_engine.audio.music import Window, windows_from_segments, write_bed
from why_engine.audio.narrate import narrate, shot_durations, split_sentences
from why_engine.audio.voice import CHAIN, NARRATION_F0_HZ, median_f0, pitch_chain
from why_engine.models import Script

FPS = 30
#: Encoder settings that preserve held frames. x264's psycho-visual defaults
#: re-dither them, taking a composition measured at 81.8% bit-identical frames
#: down to 21.6% -- which reads as each shot settling after its cut.
X264_PARAMS = "psy-rd=0.0:0.0,aq-mode=0,no-mbtree=1,bframes=0,ip-ratio=1.0"

#: Shots long enough to earn a move get one this often. The reference moves 19%
#: of its shots; this is higher by choice, not by measurement.
MOVE_EVERY = 2


@dataclass
class Build:
    video: Path
    narration: Path
    music: Path
    shots: list[Shot]
    duration: float


def _probe_pitch(model: str, work: Path) -> str:
    """Measure the voice once and build the chain that lands it on target.

    CHAIN's filter frequencies were fitted against a 167 Hz source, so a model
    sitting elsewhere needs moving onto that pitch before the fixed EQ runs.
    """
    probe = work / "probe.wav"
    subprocess.run(["piper", "-m", model, "-f", str(probe)],
                   input="The commission shipped them to twenty five states.",
                   text=True, check=True, capture_output=True)
    measured = median_f0(probe)
    ratio = max(min(NARRATION_F0_HZ / measured, 1.35), 0.70)
    return f"{pitch_chain(ratio)},{CHAIN.split('atempo=1.0417,', 1)[1]}"


def plan_motion(shots: list[Shot], durations: list[float]) -> list[str | None]:
    """Which preset each shot carries, or None to hold still.

    A shot under the static threshold is never given a move: the gate in
    zoom_curve would drop it anyway, and pretending otherwise makes the plan
    disagree with the render.
    """
    plan: list[str | None] = []
    movable = 0
    for shot, duration in zip(shots, durations):
        if not shot_moves(duration):
            plan.append(None)
            continue
        movable += 1
        if movable % MOVE_EVERY:
            plan.append(None)
            continue
        # Scenes take the long push; cards settle outward from the cut.
        plan.append("gentle_push" if shot.archetype == "painted_scene" else "pull_out")
    return plan


def plan_whips(durations: list[float], budget: int = 1) -> set[int]:
    """Indices that a whip lands ON.

    All of the reference's whips fall inside its opening window, so candidates
    are restricted to shots starting before it closes.
    """
    starts, t = [], 0.0
    for d in durations:
        starts.append(t)
        t += d
    candidates = [i for i, s in enumerate(starts)
                  if 0 < i and s < WHIP_WINDOW_SECONDS]
    if not candidates:
        return set()
    step = max(len(candidates) // max(budget, 1), 1)
    return set(candidates[::step][:budget])


def assemble(script: Script, out_dir: Path | str, assets_for, voice_model: str,
             seed: int = 0, music: bool = True) -> Build:
    """Render `script` to a finished video.

    `assets_for(shot)` returns the asset paths for one shot; it is injected so
    asset resolution -- which needs the library, the scene pool and the
    archival fetcher -- stays out of the timing logic.
    """
    out_dir = Path(out_dir)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)

    chain = _probe_pitch(voice_model, out_dir / "audio")
    text = " ".join(s.narration for s in script.segments)
    narration, segments = narrate(text, voice_model, out_dir / "audio", chain=chain)
    durations = shot_durations(segments)

    shots = direct(script, split_sentences)
    if len(shots) != len(durations):
        raise ValueError(
            f"{len(shots)} shots against {len(durations)} narration segments: the "
            f"director and the narrator disagree on the sentence split"
        )

    motion = plan_motion(shots, durations)
    whips = plan_whips(durations)

    # Frame boundaries from CUMULATIVE time. Rounding each shot independently
    # lets half a frame of error accumulate per shot, which is a quarter second
    # of drift across a long script.
    edges, t = [0], 0.0
    for d in durations:
        t += d
        edges.append(int(round(t * FPS)))

    duration = edges[-1] / FPS
    bed_path = out_dir / "music.wav"
    if music:
        write_bed(bed_path, duration, windows_from_segments(segments), seed=seed)

    video = out_dir / "video.mp4"
    _encode(_frames(shots, assets_for, edges, motion, whips), edges[-1],
            narration, bed_path if music else None, video)
    return Build(video, narration, bed_path, shots, duration)


def _frames(shots, assets_for, edges, motion, whips):
    """Yield the video one frame at a time.

    This used to build a list of every frame before encoding. A twenty-minute
    script at 1920x1080 and 30fps is 36,540 frames at 6.2MB raw; holds share an
    object, but any shot carrying a move does not, and the first real render was
    killed for memory here. Nothing downstream needed the list: the encoder
    always consumed it in order, once.
    """
    previous: Image.Image | None = None
    for i, shot in enumerate(shots):
        image = compose(shot, assets_for(shot))
        n = edges[i + 1] - edges[i]
        if i in whips and previous is not None:
            whip = whip_sequence(previous, image, fps=FPS, size=(W, H))
            yield from whip
            # Spent from the incoming shot's budget, never added on top:
            # appending pushes every later shot out of sync by its length.
            n -= len(whip)
        curve = zoom_curve(n / FPS, FPS, motion[i])[:n]
        curve += [curve[-1]] * (n - len(curve))
        yield from Zoomer(image, (W, H)).shot(curve)
        previous = image


def count_frames(frames, expected: int, sink) -> int:
    """Pass every frame to `sink`, then insist the count was what the timing map
    bought. The check used to happen before encoding, when the whole video sat
    in a list; streaming moves it after, so it can no longer prevent a bad
    encode -- only refuse to call one finished."""
    seen = 0
    for frame in frames:
        sink(frame)
        seen += 1
    if seen != expected:
        raise ValueError(f"rendered {seen} frames, timing map wants {expected}")
    return seen


def _encode(frames, expected: int, narration: Path, bed: Path | None, out: Path) -> None:
    """Pipe raw frames to ffmpeg, mixing narration over the bed."""
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-framerate", str(FPS), "-i", "-", "-i", str(narration)]
    if bed is not None:
        cmd += ["-i", str(bed),
                "-filter_complex", "[1:a][2:a]amix=inputs=2:duration=first:normalize=0[a]",
                "-map", "0:v", "-map", "[a]"]
    cmd += ["-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
            "-x264-params", X264_PARAMS, "-c:a", "aac", "-b:a", "128k", str(out)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    state = {"last": None, "payload": None}

    def write(f):
        if f is not state["last"]:     # holds repeat the same object; convert once
            state["payload"], state["last"] = f.convert("RGB").tobytes(), f
        proc.stdin.write(state["payload"])

    try:
        count_frames(frames, expected, write)
    except BrokenPipeError:
        pass
    finally:
        # A frame-count mismatch raises out of count_frames; without this the
        # encoder is left holding an open pipe and never reaped.
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        code = proc.wait()
    if code != 0:
        raise RuntimeError("ffmpeg failed while encoding the video")
