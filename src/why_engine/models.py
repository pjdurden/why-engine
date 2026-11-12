from typing import List, Tuple

from pydantic import BaseModel, Field


class Passage(BaseModel):
    source_id: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    text: str = Field(min_length=1)
    url: str
    title: str


class Claim(BaseModel):
    text: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    locator: str = Field(min_length=1)


class Segment(BaseModel):
    segment_id: int
    act: int
    narration: str = Field(min_length=1)
    claims: List[Claim] = Field(default_factory=list)
    image_brief: str = Field(min_length=1)


class Script(BaseModel):
    topic: str = Field(min_length=1)
    segments: List[Segment]

    def all_claims(self) -> List[Tuple[int, Claim]]:
        return [(seg.segment_id, c) for seg in self.segments for c in seg.claims]

    def word_count(self) -> int:
        return sum(len(seg.narration.split()) for seg in self.segments)


class ClaimVerdict(BaseModel):
    """One claim's local verification record, kept whether it passed or failed:
    verdicts.json has to show what was checked, not only what broke."""

    segment_id: int
    claim_index: int
    strict_ok: bool
    missing_tokens: List[str] = Field(default_factory=list)


class GateResult(BaseModel):
    passed: bool
    claim_verdicts: List[ClaimVerdict] = Field(default_factory=list)
    failed_indices: List[int] = Field(default_factory=list)


class TopicCandidate(BaseModel):
    question: str = Field(min_length=1)
    demand: float = Field(ge=0.0, le=1.0)
    incumbent_weakness: float = Field(ge=0.0, le=1.0)
    hook_strength: float = Field(ge=0.0, le=1.0)
    source_urls: List[str] = Field(default_factory=list)
