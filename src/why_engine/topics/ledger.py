"""What the channel has already covered.

`mine.py` has always scored novelty against a `published` list, but nothing
supplied one: no caller imported it, and no run wrote down what it made. The
gate was real and permanently empty. This is the missing half.

JSONL rather than SQLite because the access pattern is append-once-a-day and
read-all: the file is the archive, it diffs cleanly in git, and it stays
readable when something goes wrong at 2am. Spec section 12 reserves SQLite for
the performance and cost tables, which do need queries.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List

from why_engine.topics.mine import is_duplicate

DEFAULT_LEDGER = Path("state/published_topics.jsonl")


@dataclass(frozen=True)
class PublishedTopic:
    question: str
    published_on: str


class PublishedTopics:
    """An append-only archive of the questions this channel has answered."""

    def __init__(self, path: Path, entries: List[PublishedTopic]):
        self.path = Path(path)
        self.entries = entries

    @classmethod
    def load(cls, path: Path | str = DEFAULT_LEDGER) -> "PublishedTopics":
        path = Path(path)
        if not path.exists():
            return cls(path, [])
        entries = [
            PublishedTopic(**json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(path, entries)

    @property
    def questions(self) -> List[str]:
        return [e.question for e in self.entries]

    def is_duplicate(self, question: str) -> bool:
        return is_duplicate(question, self.questions)

    def record(self, question: str, on: date | None = None) -> None:
        entry = PublishedTopic(question=question,
                               published_on=(on or date.today()).isoformat())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.__dict__) + "\n")
        self.entries.append(entry)
