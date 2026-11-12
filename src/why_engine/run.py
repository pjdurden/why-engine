import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from why_engine.assemble import assemble
from why_engine.assets import Assets
from why_engine.config import Settings
from why_engine.grounding.gate import run_gate
from why_engine.llm import AnthropicLLM, ClaudeCodeLLM, LLMClient
from why_engine.models import Script
from why_engine.script.retention import check_retention
from why_engine.script.write import validate_provenance, write_script
from why_engine.sources.corpus import Corpus
from why_engine.topics.ledger import DEFAULT_LEDGER, PublishedTopics


def produce_script(
    topic: str,
    corpus: Corpus,
    llm: LLMClient,
    settings: Settings,
    run_dir: Path,
) -> Optional[Script]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    corpus.save(run_dir / "corpus_snapshot.jsonl")
    attempts: List[Dict[str, Any]] = []
    final = "exhausted"
    accepted: Optional[Script] = None
    # Accumulated, not replaced. Told only about its predecessor's mistake, each
    # rewrite fixed that one and broke a rule it had been meeting -- trading one
    # rejection for another until the attempts ran out.
    feedback: List[str] = []

    try:
        for attempt in range(1, settings.max_regenerations + 1):
            script = write_script(topic, corpus, llm, feedback=feedback or None)
            (run_dir / f"script_attempt_{attempt}.json").write_text(
                script.model_dump_json(indent=2), encoding="utf-8"
            )

            provenance_errors = validate_provenance(script, corpus)
            if provenance_errors:
                attempts.append({"attempt": attempt, "stage": "provenance", "errors": provenance_errors})
                feedback += provenance_errors
                continue

            retention_errors = check_retention(script, settings)
            if retention_errors:
                attempts.append({"attempt": attempt, "stage": "retention", "errors": retention_errors})
                feedback += retention_errors
                continue

            gate = run_gate(script, settings)
            attempts.append({
                "attempt": attempt, "stage": "grounding", "passed": gate.passed,
                "failed_indices": gate.failed_indices,
                "claim_verdicts": [v.model_dump() for v in gate.claim_verdicts],
            })

            if not gate.passed:
                # Rejection is whole-script. Dropping the offending claim would
                # delete its citation while leaving the narration sentence that
                # asserts it in place: the video would still say the unverified
                # thing, with its evidence quietly stripped.
                feedback += _gate_feedback(gate.claim_verdicts)
                continue

            accepted = script
            final = "accepted"
            break
    except Exception as exc:
        final = "errored"
        attempts.append({"stage": "error", "type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        (run_dir / "verdicts.json").write_text(
            json.dumps({"topic": topic, "attempts": attempts, "final": final}, indent=2),
            encoding="utf-8",
        )
        if accepted is not None:
            (run_dir / "script.json").write_text(accepted.model_dump_json(indent=2), encoding="utf-8")

    return accepted


def _gate_feedback(claim_verdicts) -> List[str]:
    """Readable rejection reasons for the failing claims, fed back to the next
    write attempt so a regenerated script does not repeat the same mistakes."""
    return [
        f"claim {v.claim_index} in segment {v.segment_id} failed the verbatim "
        f"check: tokens {v.missing_tokens} are not present in the quote"
        for v in claim_verdicts if not v.strict_ok
    ]


DEFAULT_VOICE = "assets/voices/en_US-ryan-high.onnx"


def produce_video(script: Script, run_dir: Path, voice: str = DEFAULT_VOICE,
                  music: bool = True, scenes: Optional[Path] = None):
    """Render an ACCEPTED script to a finished video.

    Only ever called with a script the gate passed. The gate is the moat, and
    an unverified script reaching the renderer would turn a pipeline that
    publishes nothing when it cannot verify into one that publishes anyway.
    """
    # Scenes are cycled by shot index with no regard to content, so a shared
    # pool puts the previous video's pictures in this one.
    assets = Assets(topic=script.topic, **({"scenes": scenes} if scenes else {}))
    return assemble(script, run_dir / "video", assets.for_shot, voice, music=music)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="why-engine")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--script-only", action="store_true",
                        help="stop after the gate, render nothing")
    parser.add_argument("--no-music", action="store_true")
    parser.add_argument("--scenes", type=Path, default=None,
                        help="this topic's scene directory, e.g. assets/scenes/school-bus")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER, type=Path,
                        help="archive of topics already covered")
    parser.add_argument("--llm", choices=("api", "claude-code"), default="api",
                        help="'claude-code' runs on the headless CLI and needs no "
                             "ANTHROPIC_API_KEY; billing is the subscription")
    parser.add_argument("--timeout", type=int, default=900,
                        help="seconds allowed per claude-code call; a full script "
                             "at high effort does not fit the 300s default")
    args = parser.parse_args(argv)

    # Before the first token. A repeat is not a failure, it is a topic the
    # caller needs to replace, so it gets its own exit code.
    ledger = PublishedTopics.load(args.ledger)
    if ledger.is_duplicate(args.topic):
        print(f"topic already covered: {args.topic!r}", file=sys.stderr)
        return 2

    settings = Settings.from_env()
    llm = (ClaudeCodeLLM(settings, timeout_s=args.timeout)
           if args.llm == "claude-code" else AnthropicLLM(settings))
    script = produce_script(args.topic, Corpus.load(args.corpus), llm, settings, args.run_dir)

    print(f"cost: ${llm.total_cost_usd:.2f}")
    if script is None:
        print("no script produced; publishing nothing", file=sys.stderr)
        return 1
    print(f"script accepted: {args.run_dir / 'script.json'}")
    # Recorded only once the run has actually made its deliverable. Recording at
    # acceptance burned a topic when the renderer then failed, and the retry was
    # refused as a duplicate of a video that did not exist.
    if args.script_only:
        ledger.record(args.topic)
        return 0

    build = produce_video(script, args.run_dir, args.voice, music=not args.no_music,
                          scenes=args.scenes)
    print(f"video: {build.video}  ({build.duration:.1f}s, {len(build.shots)} shots)")
    ledger.record(args.topic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
