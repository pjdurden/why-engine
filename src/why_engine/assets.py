"""Asset resolution: which picture files a shot should be built from.

Kept apart from the assembler so timing logic never has to know about the
library, the scene pool, or the archival fetcher -- and so a run with no
assets at all still produces a video of plain cards rather than failing.

Subject matching is deliberately dumb. It looks for a library icon whose name
appears in the sentence. A model call here would be a per-shot cost on a daily
unattended pipeline, and it would be unverifiable: nothing downstream can tell
that the wrong animal was drawn.
"""
from __future__ import annotations

import re
from pathlib import Path

from why_engine.art.direct import Shot

DEFAULT_ICONS = Path("assets/icons")
DEFAULT_LIBRARY = Path("assets/library")
DEFAULT_SCENES = Path("assets/scenes")

#: Cast pairings for a versus split: who stands on each side.
VERSUS_CAST = (("everyman_disgusted", "THE PUBLIC"), ("bureaucrat_stamping", "THE AGENCY"))


class Assets:
    """Resolves shot assets from whatever is on disk."""

    def __init__(self, icons: Path = DEFAULT_ICONS, library: Path = DEFAULT_LIBRARY,
                 scenes: Path = DEFAULT_SCENES, topic: str = ""):
        # A sentence often does not name its own subject -- "more protein than
        # twelve pigs" never says carp -- so the topic carries it. Without this
        # both sides of a comparison resolve to whatever the sentence DID name.
        self.topic = topic
        self.icons = Path(icons)
        self.library = Path(library)
        self.scenes = sorted(Path(scenes).glob("*.png")) if Path(scenes).exists() else []
        self._icon_names = ({p.stem: p for p in self.icons.glob("*.png")}
                            if self.icons.exists() else {})

    def icon_for(self, text: str) -> Path | None:
        """The library icon whose name appears in `text`, longest name first.

        Longest first so a sentence mentioning both a 'carp' and a 'carp pond'
        resolves to the more specific asset rather than whichever hashed first.
        """
        words = set(re.findall(r"[a-z]+", text.lower()))
        for name in sorted(self._icon_names, key=len, reverse=True):
            stem = name.lower()
            if stem in words or stem.rstrip("s") in words or f"{stem}s" in words:
                return self._icon_names[name]
        return None

    def _cast(self, name: str) -> Path | None:
        path = self.library / f"{name}.png"
        return path if path.exists() else None

    def for_shot(self, shot: Shot) -> dict:
        """Asset paths for one shot. Missing entries are fine; the composer
        falls back rather than raising."""
        subject = self.icon_for(shot.sentence) or self.icon_for(self.topic)
        topic_icon = self.icon_for(self.topic)
        assets: dict = {}

        if shot.archetype == "icon_multiply":
            # The counted thing is usually the LAST noun mentioned ("more than
            # twelve pigs"), and the subject the first, so resolve them apart.
            head, _, tail = shot.sentence.partition("than")
            counted = self.icon_for(tail)
            assets["subject_icon"] = str(self.icon_for(head) or topic_icon or subject or "")
            assets["count_icon"] = str(counted or subject or "")
        elif shot.archetype == "measure_compare":
            parts = re.split(r"\band\b|\bversus\b|\bagainst\b", shot.sentence, maxsplit=1)
            left = self.icon_for(parts[0])
            right = self.icon_for(parts[1]) if len(parts) > 1 else None
            assets["left_icon"] = str(left or subject or "")
            assets["right_icon"] = str(right or topic_icon or left or "")
        elif shot.archetype == "versus_split":
            (left, left_label), (right, right_label) = VERSUS_CAST
            assets["left_cast"] = str(self._cast(left) or "")
            assets["right_cast"] = str(self._cast(right) or "")
            assets["left_label"], assets["right_label"] = left_label, right_label
        elif shot.archetype == "painted_scene" and self.scenes:
            # Deterministic by shot index so a re-render picks the same scene.
            assets["scene"] = str(self.scenes[shot.index % len(self.scenes)])
        elif shot.archetype == "archival_plate":
            assets["archival_caption"] = shot.claim_source.upper() or "ARCHIVAL PLATE"
            assets["archival_licence"] = "PUBLIC DOMAIN"

        if subject:
            assets["default_icon"] = str(subject)
        return assets
