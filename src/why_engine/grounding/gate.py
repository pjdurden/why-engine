"""The grounding gate: local checks only, no model call.

An adversarial entailment pass used to run here, one max-effort call per claim,
judging whether a quote supported the claim it was cited for. It was cut. What
it caught that these checks do not is paraphrase that drifts from its source;
what it cost was the bulk of the per-video budget and a second failure mode
(the judge itself) on an unattended daily run.

What remains still cannot be talked past. `validate_provenance` has already
established that the cited passage exists and that the quote appears verbatim
inside it, so the quote is real. This module then requires every hard fact in
the claim -- every number, date and proper noun -- to appear verbatim in that
quote. An invented figure cannot survive both, and an invented figure is the
failure that would actually sink the channel.
"""
from typing import List

from why_engine.config import Settings
from why_engine.grounding.strict import check_verbatim
from why_engine.models import ClaimVerdict, GateResult, Script


def run_gate(script: Script, settings: Settings) -> GateResult:
    flat = script.all_claims()
    if not flat:
        # A script that asserts nothing has nothing to ground, and nothing to
        # say. Treated as a failure so the topic is rewritten, not published.
        return GateResult(passed=False)

    verdicts: List[ClaimVerdict] = []
    for index, (segment_id, claim) in enumerate(flat):
        missing = check_verbatim(claim)
        verdicts.append(ClaimVerdict(
            segment_id=segment_id,
            claim_index=index,
            strict_ok=not missing,
            missing_tokens=missing,
        ))

    # Every failure is reported, not just the first: the next write attempt gets
    # one list of everything it got wrong rather than discovering them one run
    # at a time.
    failed = [v.claim_index for v in verdicts if not v.strict_ok]
    return GateResult(passed=not failed, claim_verdicts=verdicts, failed_indices=failed)
