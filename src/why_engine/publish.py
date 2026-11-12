"""What goes under the video.

Spec section 13 rests the compliance argument on per-video primary-source
research, verified claim grounding, and a published source list. The first two
are enforced upstream; this is the third, and until now nothing built it.

Every source the script actually cited, once each, in the order the video first
leans on it -- so a viewer following along meets them in the order they were
used, and two runs of the same script produce the same description.
"""
from __future__ import annotations

from why_engine.models import Script
from why_engine.sources.corpus import Corpus

HEADER = "Sources, in the order this video uses them. All are US Government works."


def source_list(script: Script, corpus: Corpus) -> str:
    lines, seen = [], set()
    for _, claim in script.all_claims():
        if claim.source_id in seen:
            continue
        passage = corpus.get(claim.source_id, claim.locator)
        if passage is None:
            continue  # provenance runs before this and rejects these anyway
        seen.add(claim.source_id)
        lines.append(f"{len(lines) + 1}. {passage.title}\n   {passage.url}")
    return HEADER + "\n\n" + "\n".join(lines) if lines else HEADER
