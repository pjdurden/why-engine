import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from why_engine.models import Passage


class Corpus(BaseModel):
    passages: List[Passage]

    def save(self, path: Path) -> None:
        Path(path).write_text(
            "\n".join(p.model_dump_json() for p in self.passages) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "Corpus":
        lines = Path(path).read_text(encoding="utf-8").split("\n")
        return cls(passages=[Passage(**json.loads(line)) for line in lines if line.strip()])

    def get(self, source_id: str, locator: str) -> Optional[Passage]:
        for p in self.passages:
            if p.source_id == source_id and p.locator == locator:
                return p
        return None

    def as_prompt_block(self) -> str:
        """Deterministic. This string is the cached prompt prefix; any instability
        in ordering or formatting silently destroys the prompt cache."""
        ordered = sorted(self.passages, key=lambda p: (p.source_id, p.locator, p.text))
        return "\n\n".join(
            f"[source_id={p.source_id} locator={p.locator} title={p.title}]\n{p.text}"
            for p in ordered
        )
