# Grounded Script Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the stage of why-engine that turns a question into a fully source-grounded video script, where every factual claim has been independently verified against a verbatim quote from a primary document, and unverifiable scripts are rejected rather than published.

**Architecture:** A linear pipeline of pure-ish functions over typed Pydantic artifacts, persisted to a per-run directory. Every generator is paired with a gate that can reject it. The LLM is reached through one injectable client wrapper so every stage is unit-testable without network access; live API tests are marked and opt-in.

**Tech Stack:** Python 3.10+, `anthropic` SDK (Claude Opus 5), Pydantic v2, pytest, httpx2 (via the SDK), pypdf, selectolax.

**Spec:** `docs/superpowers/specs/2026-09-16-why-engine-design.md`

## Global Constraints

- Python >= 3.10 (`anthropic` 1.x requires it; system python is 3.10.12).
- Model ID is exactly `claude-opus-5`. Never append a date suffix. Never substitute a cheaper model without the user asking.
- Thinking config is `thinking={"type": "adaptive"}`. `budget_tokens` returns HTTP 400 on Opus 5 and must never appear.
- Sampling params (`temperature`, `top_p`, `top_k`) are removed on Opus 5 and return HTTP 400. Never pass them.
- Effort is set via `output_config={"effort": ...}`, never top-level.
- Structured output uses `client.messages.parse(..., output_format=<PydanticModel>)` and reads `response.parsed_output`. The `output_format` top-level parameter on `messages.create` is deprecated; do not use it there.
- Assistant prefill returns HTTP 400 on Opus 5. Never prefill.
- `anthropic` 1.x is built on `httpx2`, not `httpx`. If HTTP types are needed, `import httpx2 as httpx`.
- All LLM calls go through `src/why_engine/llm.py`. No module imports `anthropic` directly except that file and its tests.
- Every test that hits the real API is marked `@pytest.mark.live` and is excluded from the default run.
- Grounding verdicts are exactly these four strings: `SUPPORTED`, `PARTIAL`, `UNSUPPORTED`, `CONTRADICTED`.
- Publishing nothing is an acceptable outcome. Any gate may abort a run.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, deps, pytest config incl. the `live` marker |
| `src/why_engine/config.py` | Settings loaded from env, with the gate thresholds |
| `src/why_engine/models.py` | Every typed artifact: `Passage`, `Claim`, `Segment`, `Script`, `ClaimVerdict`, `GateResult`, `TopicCandidate` |
| `src/why_engine/llm.py` | `LLMClient` protocol, `AnthropicLLM` impl, `StubLLM` for tests, cost accounting |
| `src/why_engine/grounding/verify.py` | Single-claim entailment verification |
| `src/why_engine/grounding/strict.py` | Hard-fact token extraction and verbatim checking |
| `src/why_engine/grounding/gate.py` | Whole-script gate, rejection policy, batch execution |
| `src/why_engine/sources/fetch.py` | Document fetch and text normalization with provenance |
| `src/why_engine/sources/corpus.py` | Corpus persistence and passage lookup |
| `src/why_engine/script/write.py` | Retrieval-first script generation |
| `src/why_engine/script/retention.py` | Structural retention validation |
| `src/why_engine/topics/mine.py` | Topic scoring, novelty, hard source gate |
| `src/why_engine/run.py` | Orchestrator, run directory, `verdicts.json`, CLI |

Split rationale: `grounding/` has three files because verification, strict checking, and policy are rejected or changed independently. `sources/` separates network IO from persistence so the corpus is testable with no network.

---

### Task 1: Project scaffold and typed artifacts

**Files:**
- Create: `pyproject.toml`
- Create: `src/why_engine/__init__.py`
- Create: `src/why_engine/config.py`
- Create: `src/why_engine/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Passage(source_id, locator, text, url, title)`, `Claim(text, source_id, quote, locator)`, `Segment(segment_id, act, narration, claims, image_brief)`, `Script(topic, segments)` with `Script.all_claims() -> list[tuple[int, Claim]]`, `Settings` with fields `partial_fraction_limit: float`, `max_regenerations: int`, `min_runtime_s: int`, `max_runtime_s: int`, `max_act_s: int`, `words_per_second: float`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
import pytest
from pydantic import ValidationError

from why_engine.models import Claim, Passage, Script, Segment


def _claim(text="Carp were stocked by the US Fish Commission in 1877."):
    return Claim(text=text, source_id="usfc-1877", quote="In 1877 the Commission distributed carp.", locator="p.14")


def test_script_all_claims_flattens_with_segment_ids():
    script = Script(
        topic="Why don't Americans eat carp?",
        segments=[
            Segment(segment_id=0, act=0, narration="Cold open.", claims=[], image_brief="a river"),
            Segment(segment_id=1, act=1, narration="History.", claims=[_claim()], image_brief="a fish"),
        ],
    )
    flat = script.all_claims()
    assert len(flat) == 1
    assert flat[0][0] == 1
    assert flat[0][1].source_id == "usfc-1877"


def test_segment_allows_empty_claims_for_non_factual_narration():
    seg = Segment(segment_id=2, act=1, narration="But here is the strange part.", claims=[], image_brief="a pause")
    assert seg.claims == []


def test_claim_rejects_empty_quote():
    with pytest.raises(ValidationError):
        Claim(text="Something true.", source_id="x", quote="", locator="p.1")


def test_passage_rejects_empty_text():
    with pytest.raises(ValidationError):
        Passage(source_id="x", locator="p.1", text="", url="https://example.gov/a", title="A")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && python -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'why_engine'`

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[project]
name = "why-engine"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "anthropic>=1.0.0",
    "pydantic>=2.0",
    "pypdf>=4.0",
    "selectolax>=0.3",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
markers = ["live: hits the real Anthropic API, costs money, excluded by default"]
addopts = "-m 'not live'"
```

```python
# src/why_engine/__init__.py
```

```python
# src/why_engine/models.py
from typing import List, Literal, Tuple

from pydantic import BaseModel, Field

Verdict = Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"]


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
```

```python
# src/why_engine/config.py
import os

from pydantic import BaseModel


class Settings(BaseModel):
    model: str = "claude-opus-5"
    partial_fraction_limit: float = 0.15
    max_regenerations: int = 3
    min_runtime_s: int = 840
    max_runtime_s: int = 1320
    max_act_s: int = 300
    words_per_second: float = 2.6

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            partial_fraction_limit=float(os.environ.get("WE_PARTIAL_LIMIT", "0.15")),
            max_regenerations=int(os.environ.get("WE_MAX_REGEN", "3")),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && python -m venv .venv && .venv/bin/pip install -q -e ".[dev]" && .venv/bin/python -m pytest tests/test_models.py -v`
Expected: PASS, 4 passed

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
printf '.venv/\n' >> .gitignore
git add pyproject.toml src tests .gitignore
git commit -m "feat: project scaffold and typed pipeline artifacts"
```

---

### Task 2: LLM client wrapper with cost accounting

**Files:**
- Create: `src/why_engine/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `Settings` from Task 1
- Produces: `LLMClient` protocol with `parse(system, user, output_format, effort, cache_system) -> BaseModel`; `StubLLM(responses: list)` for tests, recording `.calls`; `AnthropicLLM(settings)` with `.total_cost_usd` and `.usage_log`; `estimate_cost(input_tokens, output_tokens, cache_read_tokens) -> float`

Opus 5 pricing used below: $5.00 per 1M input, $25.00 per 1M output, cache reads at 0.1x input.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm.py
import pytest
from pydantic import BaseModel

from why_engine.llm import StubLLM, estimate_cost


class Answer(BaseModel):
    value: str


def test_estimate_cost_uses_opus5_rates():
    # 1M input + 1M output = $5 + $25
    assert estimate_cost(1_000_000, 1_000_000, 0) == pytest.approx(30.0)


def test_estimate_cost_discounts_cache_reads_to_one_tenth():
    # 1M cache-read tokens bill at 0.1 x $5 = $0.50
    assert estimate_cost(0, 0, 1_000_000) == pytest.approx(0.5)


def test_stub_llm_returns_queued_responses_in_order():
    stub = StubLLM([Answer(value="first"), Answer(value="second")])
    assert stub.parse(system="s", user="u", output_format=Answer).value == "first"
    assert stub.parse(system="s", user="u", output_format=Answer).value == "second"


def test_stub_llm_records_calls_for_assertions():
    stub = StubLLM([Answer(value="x")])
    stub.parse(system="sys-prompt", user="user-prompt", output_format=Answer)
    assert stub.calls[0]["system"] == "sys-prompt"
    assert stub.calls[0]["user"] == "user-prompt"


def test_stub_llm_raises_when_exhausted():
    stub = StubLLM([])
    with pytest.raises(AssertionError):
        stub.parse(system="s", user="u", output_format=Answer)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'why_engine.llm'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/why_engine/llm.py
from typing import Any, Dict, List, Protocol, Type, TypeVar

from pydantic import BaseModel

from why_engine.config import Settings

T = TypeVar("T", bound=BaseModel)

INPUT_USD_PER_TOKEN = 5.00 / 1_000_000
OUTPUT_USD_PER_TOKEN = 25.00 / 1_000_000
CACHE_READ_MULTIPLIER = 0.1


def estimate_cost(input_tokens: int, output_tokens: int, cache_read_tokens: int) -> float:
    return (
        input_tokens * INPUT_USD_PER_TOKEN
        + output_tokens * OUTPUT_USD_PER_TOKEN
        + cache_read_tokens * INPUT_USD_PER_TOKEN * CACHE_READ_MULTIPLIER
    )


class LLMClient(Protocol):
    def parse(
        self,
        *,
        system: str,
        user: str,
        output_format: Type[T],
        effort: str = "high",
        cache_system: bool = False,
    ) -> T: ...


class StubLLM:
    """Test double. Returns queued responses and records every call."""

    def __init__(self, responses: List[BaseModel]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def parse(self, *, system, user, output_format, effort="high", cache_system=False):
        self.calls.append(
            {"system": system, "user": user, "output_format": output_format,
             "effort": effort, "cache_system": cache_system}
        )
        assert self._responses, "StubLLM exhausted: more calls than queued responses"
        return self._responses.pop(0)


class AnthropicLLM:
    """Real client. The only place in the package that imports `anthropic`."""

    def __init__(self, settings: Settings):
        import anthropic

        self._client = anthropic.Anthropic()
        self._settings = settings
        self.total_cost_usd = 0.0
        self.usage_log: List[Dict[str, Any]] = []

    def parse(self, *, system, user, output_format, effort="high", cache_system=False):
        system_arg: Any
        if cache_system:
            system_arg = [{"type": "text", "text": system,
                           "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
        else:
            system_arg = system

        response = self._client.messages.parse(
            model=self._settings.model,
            max_tokens=16000,
            system=system_arg,
            messages=[{"role": "user", "content": user}],
            output_format=output_format,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
        )
        self._record(response)
        return response.parsed_output

    def _record(self, response) -> None:
        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cost = estimate_cost(usage.input_tokens, usage.output_tokens, cache_read)
        self.total_cost_usd += cost
        self.usage_log.append(
            {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
             "cache_read_input_tokens": cache_read, "cost_usd": cost}
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_llm.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
git add src/why_engine/llm.py tests/test_llm.py
git commit -m "feat: injectable LLM client with Opus 5 cost accounting"
```

---

### Task 3: Single-claim entailment verification

**Files:**
- Create: `src/why_engine/grounding/__init__.py`
- Create: `src/why_engine/grounding/verify.py`
- Test: `tests/test_verify.py`

**Interfaces:**
- Consumes: `Claim` (Task 1), `LLMClient`/`StubLLM` (Task 2)
- Produces: `EntailmentResult(verdict: Verdict, reasoning: str)`; `verify_claim(claim: Claim, llm: LLMClient) -> EntailmentResult`; `VERIFY_SYSTEM: str`

The verification prompt must never see the script or the generation reasoning. It receives only the claim and its quote. This is the whole point: a model asked to check its own work in the same context agrees with itself.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_verify.py
import pytest

from why_engine.grounding.verify import VERIFY_SYSTEM, EntailmentResult, verify_claim
from why_engine.llm import StubLLM
from why_engine.models import Claim


def _claim(text, quote):
    return Claim(text=text, source_id="usfc-1877", quote=quote, locator="p.14")


def test_verify_claim_returns_the_models_verdict():
    stub = StubLLM([EntailmentResult(verdict="SUPPORTED", reasoning="quote states it directly")])
    c = _claim("Carp were distributed in 1877.", "In 1877 the Commission distributed carp.")
    assert verify_claim(c, stub).verdict == "SUPPORTED"


def test_verify_prompt_contains_claim_and_quote_only():
    stub = StubLLM([EntailmentResult(verdict="SUPPORTED", reasoning="ok")])
    c = _claim("Carp were distributed in 1877.", "In 1877 the Commission distributed carp.")
    verify_claim(c, stub)
    user = stub.calls[0]["user"]
    assert "Carp were distributed in 1877." in user
    assert "In 1877 the Commission distributed carp." in user


def test_verify_prompt_does_not_leak_source_id_or_locator():
    """Provenance is bookkeeping. Leaking it invites the model to trust the label
    instead of reading the quote."""
    stub = StubLLM([EntailmentResult(verdict="SUPPORTED", reasoning="ok")])
    verify_claim(_claim("A.", "B."), stub)
    user = stub.calls[0]["user"]
    assert "usfc-1877" not in user
    assert "p.14" not in user


def test_verify_uses_max_effort_because_the_gate_is_the_moat():
    stub = StubLLM([EntailmentResult(verdict="SUPPORTED", reasoning="ok")])
    verify_claim(_claim("A.", "B."), stub)
    assert stub.calls[0]["effort"] == "max"


def test_verify_system_prompt_enumerates_all_four_verdicts():
    for v in ("SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"):
        assert v in VERIFY_SYSTEM


@pytest.mark.parametrize("verdict", ["SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"])
def test_entailment_result_accepts_each_valid_verdict(verdict):
    assert EntailmentResult(verdict=verdict, reasoning="r").verdict == verdict


def test_entailment_result_rejects_an_invented_verdict():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EntailmentResult(verdict="MOSTLY_TRUE", reasoning="r")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_verify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'why_engine.grounding'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/why_engine/grounding/__init__.py
```

```python
# src/why_engine/grounding/verify.py
from pydantic import BaseModel

from why_engine.llm import LLMClient
from why_engine.models import Claim, Verdict

VERIFY_SYSTEM = """You are a fact-checking gate. You judge one thing only: whether \
the supplied QUOTE supports the supplied CLAIM.

You have no outside knowledge. Even if you believe the claim is true in the world, \
if the quote does not support it, it is not supported.

Answer with exactly one verdict:
- SUPPORTED: the quote states or directly entails the whole claim.
- PARTIAL: the quote supports some of the claim but not all of it.
- UNSUPPORTED: the quote does not support the claim. This includes quotes that are \
about the same topic but do not actually establish what the claim asserts.
- CONTRADICTED: the quote asserts something incompatible with the claim.

A quote that is merely topically related is UNSUPPORTED, not PARTIAL."""


class EntailmentResult(BaseModel):
    verdict: Verdict
    reasoning: str


def verify_claim(claim: Claim, llm: LLMClient) -> EntailmentResult:
    user = f"CLAIM:\n{claim.text}\n\nQUOTE:\n{claim.quote}"
    return llm.parse(
        system=VERIFY_SYSTEM,
        user=user,
        output_format=EntailmentResult,
        effort="max",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_verify.py -v`
Expected: PASS, 10 passed

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
git add src/why_engine/grounding tests/test_verify.py
git commit -m "feat: single-claim entailment verification with isolated context"
```

---

### Task 4: Adversarial live fixtures for the gate

**Files:**
- Create: `tests/fixtures/entailment_cases.json`
- Test: `tests/test_verify_live.py`

**Interfaces:**
- Consumes: `verify_claim` (Task 3), `AnthropicLLM` (Task 2)
- Produces: a live accuracy check. No new production code.

A gate that never rejects is not working. This task proves the gate actually discriminates, especially on the near-miss case where a quote is topically related but establishes nothing.

- [ ] **Step 1: Write the fixture file**

```json
[
  {
    "name": "exact_support",
    "claim": "The US Fish Commission distributed carp in 1877.",
    "quote": "In 1877 the Commission distributed carp to applicants in twenty-five states.",
    "expected": "SUPPORTED"
  },
  {
    "name": "topically_related_but_establishes_nothing",
    "claim": "Carp were stocked in American rivers because of a federal food-security program.",
    "quote": "Carp are a hardy freshwater fish that tolerate low oxygen and warm water.",
    "expected": "UNSUPPORTED"
  },
  {
    "name": "number_not_in_quote",
    "claim": "The program distributed carp to 298 of the 301 congressional districts.",
    "quote": "The Commission distributed carp widely across the congressional districts.",
    "expected": "UNSUPPORTED"
  },
  {
    "name": "quote_contradicts_claim",
    "claim": "Geosmin is detectable by humans only at concentrations above one part per million.",
    "quote": "Humans can detect geosmin at concentrations as low as a few parts per trillion.",
    "expected": "CONTRADICTED"
  },
  {
    "name": "supports_half_the_conjunction",
    "claim": "Tilapia farming began in the Philippines in 1988 and reached the United States by 1990.",
    "quote": "The GIFT selective-breeding program for tilapia began in the Philippines in 1988.",
    "expected": "PARTIAL"
  },
  {
    "name": "date_shifted",
    "claim": "Rudolph Hessel brought carp to the United States in 1897.",
    "quote": "Rudolph Hessel imported carp to the United States in 1877.",
    "expected": "CONTRADICTED"
  }
]
```

- [ ] **Step 2: Write the live test**

```python
# tests/test_verify_live.py
import json
import pathlib

import pytest

from why_engine.config import Settings
from why_engine.grounding.verify import verify_claim
from why_engine.llm import AnthropicLLM
from why_engine.models import Claim

CASES = json.loads((pathlib.Path(__file__).parent / "fixtures" / "entailment_cases.json").read_text())


@pytest.mark.live
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_gate_discriminates_on_adversarial_cases(case):
    llm = AnthropicLLM(Settings())
    claim = Claim(text=case["claim"], source_id="fixture", quote=case["quote"], locator="n/a")
    result = verify_claim(claim, llm)
    assert result.verdict == case["expected"], (
        f"{case['name']}: got {result.verdict}, expected {case['expected']}. "
        f"Model reasoning: {result.reasoning}"
    )
```

- [ ] **Step 3: Run the default suite to confirm live tests are excluded**

Run: `cd ~/why-engine && .venv/bin/python -m pytest -v`
Expected: PASS, and the `test_verify_live.py` cases are reported as deselected (the `addopts = -m 'not live'` from Task 1)

- [ ] **Step 4: Run the live suite against the real API**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_verify_live.py -v -m live`
Expected: PASS, 6 passed. Cost is roughly $0.05.

If `SUPPORTED` cases pass but the near-miss cases also return `SUPPORTED`, the gate is not discriminating and `VERIFY_SYSTEM` in Task 3 needs strengthening before any later task is built on it. Do not proceed past this task with a non-discriminating gate.

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
git add tests/fixtures tests/test_verify_live.py
git commit -m "test: adversarial live fixtures proving the gate discriminates"
```

---

### Task 5: Strict mode for hard facts

**Files:**
- Create: `src/why_engine/grounding/strict.py`
- Test: `tests/test_strict.py`

**Interfaces:**
- Consumes: `Claim` (Task 1)
- Produces: `hard_fact_tokens(text: str) -> list[str]`; `check_verbatim(claim: Claim) -> list[str]` returning the tokens missing from the quote

Paraphrase is acceptable for reasoning. It is not acceptable for a figure. A model that turns "a few parts per trillion" into "5 parts per trillion" has invented a number, and entailment checking alone is soft on this.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strict.py
from why_engine.grounding.strict import check_verbatim, hard_fact_tokens
from why_engine.models import Claim


def test_extracts_years():
    assert "1877" in hard_fact_tokens("The Commission acted in 1877.")


def test_extracts_plain_numbers_and_percentages():
    toks = hard_fact_tokens("Roughly 25 states and 40% of districts took part.")
    assert "25" in toks
    assert "40" in toks


def test_extracts_capitalised_proper_nouns_midsentence():
    toks = hard_fact_tokens("The program was run by Spencer Baird in Washington.")
    assert "Spencer" in toks
    assert "Baird" in toks
    assert "Washington" in toks


def test_ignores_the_leading_word_of_a_sentence():
    """Sentence-initial capitals are grammar, not proper nouns."""
    assert "Carp" not in hard_fact_tokens("Carp were distributed widely.")


def test_ignores_common_stopwords_even_when_capitalised():
    assert "The" not in hard_fact_tokens("The Commission met. The vote passed.")


def test_check_verbatim_returns_empty_when_all_tokens_present():
    c = Claim(
        text="Spencer Baird led the program in 1877.",
        source_id="s", locator="p.1",
        quote="In 1877 Spencer Baird led the Commission's carp program.",
    )
    assert check_verbatim(c) == []


def test_check_verbatim_flags_a_number_absent_from_the_quote():
    c = Claim(
        text="The program reached 298 districts.",
        source_id="s", locator="p.1",
        quote="The program reached many districts.",
    )
    assert check_verbatim(c) == ["298"]


def test_check_verbatim_flags_a_shifted_year():
    c = Claim(
        text="Hessel imported carp in 1897.",
        source_id="s", locator="p.1",
        quote="Hessel imported carp in 1877.",
    )
    assert check_verbatim(c) == ["1897"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_strict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'why_engine.grounding.strict'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/why_engine/grounding/strict.py
import re
from typing import List

from why_engine.models import Claim

_NUMBER = re.compile(r"\b\d[\d,.]*\b")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\b[A-Za-z][A-Za-z'\-]*\b")

_STOPWORDS = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "They", "He", "She",
    "But", "And", "Or", "So", "Then", "When", "While", "If", "In", "On", "At",
    "By", "For", "From", "To", "Of", "With", "As", "Its", "Their", "His", "Her",
}


def hard_fact_tokens(text: str) -> List[str]:
    tokens: List[str] = []

    for match in _NUMBER.finditer(text):
        tokens.append(match.group().rstrip(".").replace(",", ""))

    for sentence in _SENTENCE_SPLIT.split(text.strip()):
        words = _WORD.findall(sentence)
        for index, word in enumerate(words):
            if index == 0:
                continue  # sentence-initial capital is grammar, not a proper noun
            if word[0].isupper() and word not in _STOPWORDS:
                tokens.append(word)

    seen = set()
    unique = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    return unique


def check_verbatim(claim: Claim) -> List[str]:
    quote_normalised = claim.quote.replace(",", "")
    return [t for t in hard_fact_tokens(claim.text) if t not in quote_normalised]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_strict.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
git add src/why_engine/grounding/strict.py tests/test_strict.py
git commit -m "feat: strict verbatim checking for numbers, dates and proper nouns"
```

---

### Task 6: Whole-script gate and rejection policy

**Files:**
- Create: `src/why_engine/grounding/gate.py`
- Modify: `src/why_engine/models.py` (append `ClaimVerdict` and `GateResult`)
- Test: `tests/test_gate.py`

**Interfaces:**
- Consumes: `Script`, `Claim`, `Settings` (Task 1), `verify_claim` (Task 3), `check_verbatim` (Task 5)
- Produces: `ClaimVerdict(segment_id, claim_index, verdict, strict_ok, missing_tokens)`; `GateResult(passed, action, claim_verdicts, cut_indices)` where `action` is one of `accept`, `cut_claims`, `regenerate`, `abort`; `run_gate(script, llm, settings) -> GateResult`

Policy, exactly as specified:
- any `CONTRADICTED` -> `abort`, immediately, no regeneration
- `UNSUPPORTED` claims -> cut them (`cut_claims`)
- a claim that is `SUPPORTED` but fails strict verbatim -> treated as `UNSUPPORTED` and cut
- `PARTIAL` fraction above `settings.partial_fraction_limit` -> `regenerate`
- otherwise -> `accept`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gate.py
from why_engine.config import Settings
from why_engine.grounding.gate import run_gate
from why_engine.grounding.verify import EntailmentResult
from why_engine.llm import StubLLM
from why_engine.models import Claim, Script, Segment


def _script(*claims):
    return Script(
        topic="t",
        segments=[Segment(segment_id=1, act=1, narration="n", claims=list(claims), image_brief="b")],
    )


def _claim(text="Baird led it in 1877.", quote="In 1877 Baird led it."):
    return Claim(text=text, source_id="s", quote=quote, locator="p.1")


def _verdicts(*vs):
    return StubLLM([EntailmentResult(verdict=v, reasoning="r") for v in vs])


def test_all_supported_claims_are_accepted():
    result = run_gate(_script(_claim()), _verdicts("SUPPORTED"), Settings())
    assert result.passed is True
    assert result.action == "accept"


def test_a_single_contradiction_aborts_the_run():
    result = run_gate(_script(_claim(), _claim()), _verdicts("SUPPORTED", "CONTRADICTED"), Settings())
    assert result.passed is False
    assert result.action == "abort"


def test_contradiction_aborts_without_verifying_remaining_claims():
    """Abort is immediate. Spending tokens on the rest of a doomed script is waste."""
    stub = _verdicts("CONTRADICTED")  # only one response queued for two claims
    result = run_gate(_script(_claim(), _claim()), stub, Settings())
    assert result.action == "abort"
    assert len(stub.calls) == 1


def test_unsupported_claims_are_cut():
    result = run_gate(_script(_claim(), _claim()), _verdicts("SUPPORTED", "UNSUPPORTED"), Settings())
    assert result.action == "cut_claims"
    assert result.cut_indices == [1]


def test_supported_claim_failing_strict_check_is_cut():
    """The model said the quote supports it, but the quote has no 298 in it."""
    bad = _claim(text="It reached 298 districts.", quote="It reached many districts.")
    result = run_gate(_script(bad), _verdicts("SUPPORTED"), Settings())
    assert result.action == "cut_claims"
    assert result.claim_verdicts[0].strict_ok is False
    assert result.claim_verdicts[0].missing_tokens == ["298"]


def test_partial_fraction_over_the_limit_triggers_regeneration():
    settings = Settings(partial_fraction_limit=0.15)
    # 1 of 4 = 0.25 > 0.15
    result = run_gate(
        _script(_claim(), _claim(), _claim(), _claim()),
        _verdicts("PARTIAL", "SUPPORTED", "SUPPORTED", "SUPPORTED"),
        settings,
    )
    assert result.action == "regenerate"


def test_partial_fraction_under_the_limit_is_accepted():
    settings = Settings(partial_fraction_limit=0.30)
    result = run_gate(
        _script(_claim(), _claim(), _claim(), _claim()),
        _verdicts("PARTIAL", "SUPPORTED", "SUPPORTED", "SUPPORTED"),
        settings,
    )
    assert result.action == "accept"


def test_a_script_with_no_claims_at_all_is_rejected():
    """A script that asserts nothing is not a documentary, it is filler."""
    result = run_gate(_script(), StubLLM([]), Settings())
    assert result.passed is False
    assert result.action == "regenerate"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'why_engine.grounding.gate'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/why_engine/models.py`:

```python
class ClaimVerdict(BaseModel):
    segment_id: int
    claim_index: int
    verdict: Verdict
    strict_ok: bool
    missing_tokens: List[str] = Field(default_factory=list)


class GateResult(BaseModel):
    passed: bool
    action: Literal["accept", "cut_claims", "regenerate", "abort"]
    claim_verdicts: List[ClaimVerdict] = Field(default_factory=list)
    cut_indices: List[int] = Field(default_factory=list)
```

```python
# src/why_engine/grounding/gate.py
from typing import List

from why_engine.config import Settings
from why_engine.grounding.strict import check_verbatim
from why_engine.grounding.verify import verify_claim
from why_engine.llm import LLMClient
from why_engine.models import ClaimVerdict, GateResult, Script


def run_gate(script: Script, llm: LLMClient, settings: Settings) -> GateResult:
    flat = script.all_claims()
    if not flat:
        return GateResult(passed=False, action="regenerate")

    verdicts: List[ClaimVerdict] = []

    for index, (segment_id, claim) in enumerate(flat):
        entailment = verify_claim(claim, llm)
        missing = check_verbatim(claim)
        record = ClaimVerdict(
            segment_id=segment_id,
            claim_index=index,
            verdict=entailment.verdict,
            strict_ok=not missing,
            missing_tokens=missing,
        )
        verdicts.append(record)

        if entailment.verdict == "CONTRADICTED":
            # Immediate. Do not spend tokens verifying the rest of a doomed script.
            return GateResult(passed=False, action="abort", claim_verdicts=verdicts)

    cut = [
        v.claim_index
        for v in verdicts
        if v.verdict == "UNSUPPORTED" or (v.verdict == "SUPPORTED" and not v.strict_ok)
    ]

    partial_fraction = sum(1 for v in verdicts if v.verdict == "PARTIAL") / len(verdicts)
    if partial_fraction > settings.partial_fraction_limit:
        return GateResult(passed=False, action="regenerate", claim_verdicts=verdicts)

    if cut:
        return GateResult(passed=False, action="cut_claims", claim_verdicts=verdicts, cut_indices=cut)

    return GateResult(passed=True, action="accept", claim_verdicts=verdicts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_gate.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
git add src/why_engine/grounding/gate.py src/why_engine/models.py tests/test_gate.py
git commit -m "feat: whole-script grounding gate with abort-on-contradiction policy"
```

---

### Task 7: Source fetching and corpus persistence

**Files:**
- Create: `src/why_engine/sources/__init__.py`
- Create: `src/why_engine/sources/fetch.py`
- Create: `src/why_engine/sources/corpus.py`
- Test: `tests/test_corpus.py`

**Interfaces:**
- Consumes: `Passage` (Task 1)
- Produces: `chunk_text(text, source_id, url, title, max_chars=1200) -> list[Passage]`; `html_to_text(html: str) -> str`; `Corpus` with `.passages`, `.save(path)`, `.load(path)` (classmethod), `.get(source_id, locator) -> Passage | None`, `.as_prompt_block() -> str`

`as_prompt_block()` output is the exact text the script writer and the gate both cache, so it must be deterministic: passages sorted by `(source_id, locator)`, no timestamps, no UUIDs. A varying prefix silently destroys the cache.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_corpus.py
from why_engine.sources.corpus import Corpus
from why_engine.sources.fetch import chunk_text, html_to_text
from why_engine.models import Passage


def _p(source_id="s", locator="c1", text="hello"):
    return Passage(source_id=source_id, locator=locator, text=text, url="https://a.gov", title="T")


def test_html_to_text_strips_tags_and_script_content():
    text = html_to_text("<html><body><script>var x=1;</script><p>Real content.</p></body></html>")
    assert "Real content." in text
    assert "var x" not in text


def test_chunk_text_assigns_sequential_locators():
    passages = chunk_text("a" * 2500, source_id="doc", url="https://a.gov", title="T", max_chars=1000)
    assert [p.locator for p in passages] == ["c0", "c1", "c2"]


def test_chunk_text_respects_max_chars():
    passages = chunk_text("b" * 2500, source_id="doc", url="https://a.gov", title="T", max_chars=1000)
    assert all(len(p.text) <= 1000 for p in passages)


def test_chunk_text_drops_whitespace_only_chunks():
    assert chunk_text("   \n\n  ", source_id="doc", url="https://a.gov", title="T") == []


def test_corpus_roundtrips_through_jsonl(tmp_path):
    corpus = Corpus(passages=[_p(locator="c0"), _p(locator="c1")])
    path = tmp_path / "corpus.jsonl"
    corpus.save(path)
    assert Corpus.load(path).passages == corpus.passages


def test_corpus_get_finds_a_passage_by_source_and_locator():
    corpus = Corpus(passages=[_p(locator="c0", text="first"), _p(locator="c1", text="second")])
    assert corpus.get("s", "c1").text == "second"


def test_corpus_get_returns_none_for_a_missing_passage():
    assert Corpus(passages=[_p()]).get("s", "nope") is None


def test_prompt_block_is_deterministic_regardless_of_input_order():
    a = Corpus(passages=[_p(locator="c0"), _p(locator="c1")])
    b = Corpus(passages=[_p(locator="c1"), _p(locator="c0")])
    assert a.as_prompt_block() == b.as_prompt_block()


def test_prompt_block_labels_each_passage_with_its_provenance():
    block = Corpus(passages=[_p(source_id="gao-24", locator="c7", text="body text")]).as_prompt_block()
    assert "gao-24" in block
    assert "c7" in block
    assert "body text" in block
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'why_engine.sources'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/why_engine/sources/__init__.py
```

```python
# src/why_engine/sources/fetch.py
from typing import List

from selectolax.parser import HTMLParser

from why_engine.models import Passage


def html_to_text(html: str) -> str:
    tree = HTMLParser(html)
    for tag in tree.css("script, style, nav, footer"):
        tag.decompose()
    body = tree.body if tree.body is not None else tree.root
    return " ".join(body.text(separator=" ").split())


def chunk_text(text: str, source_id: str, url: str, title: str, max_chars: int = 1200) -> List[Passage]:
    passages: List[Passage] = []
    index = 0
    for start in range(0, len(text), max_chars):
        chunk = text[start:start + max_chars].strip()
        if not chunk:
            continue
        passages.append(
            Passage(source_id=source_id, locator=f"c{index}", text=chunk, url=url, title=title)
        )
        index += 1
    return passages


def fetch_document(url: str, source_id: str, title: str) -> List[Passage]:
    """Network IO. Kept free of parsing logic so the parsing above stays testable."""
    import httpx2 as httpx

    response = httpx.get(url, follow_redirects=True, timeout=60.0,
                         headers={"User-Agent": "why-engine/0.1 (research)"})
    response.raise_for_status()

    if url.lower().endswith(".pdf") or "application/pdf" in response.headers.get("content-type", ""):
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(response.content))
        text = " ".join(" ".join((page.extract_text() or "").split()) for page in reader.pages)
    else:
        text = html_to_text(response.text)

    return chunk_text(text, source_id=source_id, url=url, title=title)
```

```python
# src/why_engine/sources/corpus.py
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
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return cls(passages=[Passage(**json.loads(line)) for line in lines if line.strip()])

    def get(self, source_id: str, locator: str) -> Optional[Passage]:
        for p in self.passages:
            if p.source_id == source_id and p.locator == locator:
                return p
        return None

    def as_prompt_block(self) -> str:
        """Deterministic. This string is the cached prompt prefix; any instability
        in ordering or formatting silently destroys the prompt cache."""
        ordered = sorted(self.passages, key=lambda p: (p.source_id, p.locator))
        return "\n\n".join(
            f"[source_id={p.source_id} locator={p.locator} title={p.title}]\n{p.text}"
            for p in ordered
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_corpus.py -v`
Expected: PASS, 9 passed

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
git add src/why_engine/sources tests/test_corpus.py
git commit -m "feat: source fetching and deterministic corpus persistence"
```

---

### Task 8: Retrieval-first script writer

**Files:**
- Create: `src/why_engine/script/__init__.py`
- Create: `src/why_engine/script/write.py`
- Test: `tests/test_write.py`

**Interfaces:**
- Consumes: `Script`, `Segment` (Task 1), `LLMClient` (Task 2), `Corpus` (Task 7)
- Produces: `write_script(topic: str, corpus: Corpus, llm: LLMClient) -> Script`; `validate_provenance(script: Script, corpus: Corpus) -> list[str]` returning human-readable provenance errors; `WRITE_SYSTEM: str`

The writer must cite only passages that exist. `validate_provenance` catches a fabricated `source_id`/`locator` pair before the expensive gate ever runs, and catches a quote the model paraphrased rather than copied.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_write.py
from why_engine.models import Claim, Passage, Script, Segment
from why_engine.llm import StubLLM
from why_engine.script.write import WRITE_SYSTEM, validate_provenance, write_script
from why_engine.sources.corpus import Corpus


def _corpus():
    return Corpus(passages=[
        Passage(source_id="gao-24", locator="c0", text="In 1877 the Commission distributed carp.",
                url="https://gao.gov/x", title="Carp Report"),
    ])


def _script_with(claim):
    return Script(topic="t", segments=[
        Segment(segment_id=0, act=1, narration="n", claims=[claim], image_brief="b")])


def test_write_script_returns_the_parsed_script():
    expected = _script_with(Claim(text="Carp were distributed in 1877.", source_id="gao-24",
                                  locator="c0", quote="In 1877 the Commission distributed carp."))
    stub = StubLLM([expected])
    assert write_script("t", _corpus(), stub).topic == "t"


def test_corpus_is_sent_as_the_cached_system_prefix():
    stub = StubLLM([_script_with(Claim(text="a", source_id="gao-24", locator="c0", quote="q"))])
    write_script("t", _corpus(), stub)
    assert stub.calls[0]["cache_system"] is True
    assert "In 1877 the Commission distributed carp." in stub.calls[0]["system"]


def test_topic_goes_in_the_user_turn_not_the_cached_prefix():
    """Volatile content after the cache breakpoint, or the cache never hits."""
    stub = StubLLM([_script_with(Claim(text="a", source_id="gao-24", locator="c0", quote="q"))])
    write_script("Why do we eat tilapia?", _corpus(), stub)
    assert "Why do we eat tilapia?" in stub.calls[0]["user"]
    assert "Why do we eat tilapia?" not in stub.calls[0]["system"]


def test_write_system_forbids_outside_knowledge():
    assert "only" in WRITE_SYSTEM.lower()
    assert "passage" in WRITE_SYSTEM.lower()


def test_validate_provenance_passes_for_a_real_passage_and_verbatim_quote():
    claim = Claim(text="Carp were distributed in 1877.", source_id="gao-24", locator="c0",
                  quote="In 1877 the Commission distributed carp.")
    assert validate_provenance(_script_with(claim), _corpus()) == []


def test_validate_provenance_flags_an_invented_source_id():
    claim = Claim(text="x", source_id="does-not-exist", locator="c0", quote="In 1877 the Commission distributed carp.")
    errors = validate_provenance(_script_with(claim), _corpus())
    assert len(errors) == 1
    assert "does-not-exist" in errors[0]


def test_validate_provenance_flags_a_quote_not_present_in_the_passage():
    """The model paraphrased instead of copying. The quote must be verbatim."""
    claim = Claim(text="x", source_id="gao-24", locator="c0",
                  quote="The Commission handed out carp in 1877.")
    errors = validate_provenance(_script_with(claim), _corpus())
    assert len(errors) == 1
    assert "not verbatim" in errors[0]


def test_validate_provenance_tolerates_whitespace_differences_in_the_quote():
    claim = Claim(text="x", source_id="gao-24", locator="c0",
                  quote="In 1877   the Commission\ndistributed carp.")
    assert validate_provenance(_script_with(claim), _corpus()) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_write.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'why_engine.script'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/why_engine/script/__init__.py
```

```python
# src/why_engine/script/write.py
from typing import List

from why_engine.llm import LLMClient
from why_engine.models import Script
from why_engine.sources.corpus import Corpus

WRITE_SYSTEM = """You write documentary narration for an illustrated explainer video.

You may use only the passages supplied below. You have no other knowledge. If a fact \
is not in a passage, you may not assert it.

Every factual assertion must appear as a claim carrying:
- source_id and locator copied exactly from the passage you used
- a quote copied VERBATIM from that passage, not paraphrased

Narration that asserts nothing factual (transitions, framing, rhetorical questions) \
carries an empty claims list. Use this freely: not every sentence needs a citation, \
but every fact does.

Write in plain spoken English. Short sentences. No greeting, no "in this video", \
no subscribe request."""


def _normalise(text: str) -> str:
    return " ".join(text.split())


def write_script(topic: str, corpus: Corpus, llm: LLMClient) -> Script:
    system = f"{WRITE_SYSTEM}\n\nPASSAGES:\n\n{corpus.as_prompt_block()}"
    user = f"Write the script for this question:\n\n{topic}"
    return llm.parse(
        system=system,
        user=user,
        output_format=Script,
        effort="high",
        cache_system=True,
    )


def validate_provenance(script: Script, corpus: Corpus) -> List[str]:
    errors: List[str] = []
    for segment_id, claim in script.all_claims():
        passage = corpus.get(claim.source_id, claim.locator)
        if passage is None:
            errors.append(
                f"segment {segment_id}: no such passage "
                f"source_id={claim.source_id} locator={claim.locator}"
            )
            continue
        if _normalise(claim.quote) not in _normalise(passage.text):
            errors.append(
                f"segment {segment_id}: quote is not verbatim in "
                f"{claim.source_id}/{claim.locator}"
            )
    return errors
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_write.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
git add src/why_engine/script tests/test_write.py
git commit -m "feat: retrieval-first script writer with verbatim provenance validation"
```

---

### Task 9: Retention structure checker

**Files:**
- Create: `src/why_engine/script/retention.py`
- Test: `tests/test_retention.py`

**Interfaces:**
- Consumes: `Script`, `Settings` (Task 1)
- Produces: `estimate_runtime_s(script, settings) -> float`; `check_retention(script, settings) -> list[str]` returning structural violations

Rules from the spec, section 10. Act 0 is the cold open and stakes. Acts 1..N are the body. The final segment must resolve the cold open.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retention.py
from why_engine.config import Settings
from why_engine.models import Script, Segment
from why_engine.script.retention import check_retention, estimate_runtime_s


def _seg(segment_id, act, narration):
    return Segment(segment_id=segment_id, act=act, narration=narration, claims=[], image_brief="b")


def _good_script(body_words=2400):
    """Body is split across five acts so it satisfies the per-act ceiling.
    A single 2400-word act would be ~923s, far over the 300s limit."""
    per_act = max(1, body_words // 5)
    body = [_seg(i, i, " ".join(["word"] * per_act)) for i in range(1, 6)]
    return Script(topic="Why is American bread sweet?", segments=[
        _seg(0, 0, "American bread is legally close to cake."),
        *body,
        _seg(6, 6, "And that is why American bread is legally close to cake."),
    ])


def test_estimate_runtime_uses_the_configured_speaking_rate():
    script = Script(topic="t", segments=[_seg(0, 0, " ".join(["w"] * 260))])
    assert estimate_runtime_s(script, Settings(words_per_second=2.6)) == 100.0


def test_a_well_formed_script_has_no_violations():
    assert check_retention(_good_script(), Settings()) == []


def test_missing_cold_open_is_flagged():
    script = Script(topic="t", segments=[_seg(i, i, " ".join(["w"] * 480)) for i in range(1, 6)])
    assert any("cold open" in e for e in check_retention(script, Settings()))


def test_cold_open_longer_than_fifteen_seconds_is_flagged():
    long_open = " ".join(["word"] * 200)  # ~77s at 2.6 wps
    script = Script(topic="t", segments=[
        _seg(0, 0, long_open),
        *[_seg(i, i, " ".join(["w"] * 480)) for i in range(1, 6)],
        _seg(6, 6, "And that is why."),
    ])
    assert any("cold open" in e and "too long" in e for e in check_retention(script, Settings()))


def test_runtime_below_the_floor_is_flagged():
    assert any("too short" in e for e in check_retention(_good_script(body_words=200), Settings()))


def test_runtime_above_the_ceiling_is_flagged():
    """Asserts on the runtime message specifically: a 9000-word body also trips
    the per-act ceiling, and a bare "too long" match would pass for the wrong reason."""
    violations = check_retention(_good_script(body_words=9000), Settings())
    assert any("runtime too long" in e for e in violations)


def test_an_act_longer_than_the_ceiling_is_flagged():
    """Spec S10 budgets each act at 2-4 minutes. A 12-minute act is a lecture,
    not an act, and retention dies inside it."""
    huge_act = " ".join(["word"] * 2000)  # ~769s at 2.6 wps, over the 300s ceiling
    script = Script(topic="t", segments=[
        _seg(0, 0, "American bread is legally close to cake."),
        _seg(1, 1, huge_act),
        _seg(2, 2, "And that is why American bread is legally close to cake."),
    ])
    assert any("act 1" in e and "too long" in e for e in check_retention(script, Settings()))


def test_act_length_is_summed_across_segments_sharing_an_act():
    half = " ".join(["word"] * 500)  # ~192s each, 384s combined, over the ceiling
    script = Script(topic="t", segments=[
        _seg(0, 0, "American bread is legally close to cake."),
        _seg(1, 3, half),
        _seg(2, 3, half),
        _seg(3, 4, " ".join(["word"] * 1400)),
        _seg(4, 5, "And that is why American bread is legally close to cake."),
    ])
    assert any("act 3" in e and "too long" in e for e in check_retention(script, Settings()))


def test_banned_youtuber_filler_is_flagged():
    script = _good_script()
    script.segments[0].narration = "Hey guys, before we begin, make sure to subscribe."
    violations = check_retention(script, Settings())
    assert any("filler" in e for e in violations)


def test_final_segment_must_echo_the_cold_open():
    script = _good_script()
    script.segments[-1].narration = "Thanks for watching everyone."
    assert any("resolve" in e for e in check_retention(script, Settings()))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_retention.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'why_engine.script.retention'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/why_engine/script/retention.py
from typing import List, Set

from why_engine.config import Settings
from why_engine.models import Script

COLD_OPEN_MAX_S = 15.0

BANNED_PHRASES = (
    "before we begin",
    "make sure to subscribe",
    "hey guys",
    "in this video",
    "don't forget to like",
    "welcome back to the channel",
)

_ECHO_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "of",
    "in", "on", "at", "it", "that", "this", "why", "how", "so", "for", "with", "as",
}


def estimate_runtime_s(script: Script, settings: Settings) -> float:
    return script.word_count() / settings.words_per_second


def _content_words(text: str) -> Set[str]:
    return {w.strip(".,!?;:\"'").lower() for w in text.split()} - _ECHO_STOPWORDS


def check_retention(script: Script, settings: Settings) -> List[str]:
    violations: List[str] = []

    if not script.segments:
        return ["script has no segments"]

    first = script.segments[0]
    if first.act != 0:
        violations.append("missing cold open: first segment must be act 0")
    else:
        open_s = len(first.narration.split()) / settings.words_per_second
        if open_s > COLD_OPEN_MAX_S:
            violations.append(f"cold open too long: {open_s:.0f}s exceeds {COLD_OPEN_MAX_S:.0f}s")

    runtime = estimate_runtime_s(script, settings)
    if runtime < settings.min_runtime_s:
        violations.append(f"runtime too short: {runtime:.0f}s below {settings.min_runtime_s}s")
    if runtime > settings.max_runtime_s:
        violations.append(f"runtime too long: {runtime:.0f}s above {settings.max_runtime_s}s")

    act_words: dict = {}
    for segment in script.segments:
        if segment.act == 0:
            continue  # the cold open has its own, tighter ceiling
        act_words[segment.act] = act_words.get(segment.act, 0) + len(segment.narration.split())
    for act in sorted(act_words):
        act_s = act_words[act] / settings.words_per_second
        if act_s > settings.max_act_s:
            violations.append(
                f"act {act} too long: {act_s:.0f}s exceeds {settings.max_act_s}s"
            )

    for segment in script.segments:
        lowered = segment.narration.lower()
        for phrase in BANNED_PHRASES:
            if phrase in lowered:
                violations.append(f"segment {segment.segment_id}: banned filler phrase '{phrase}'")

    if len(script.segments) >= 2 and first.act == 0:
        shared = _content_words(first.narration) & _content_words(script.segments[-1].narration)
        if len(shared) < 2:
            violations.append("final segment does not resolve the cold open in its own words")

    return violations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_retention.py -v`
Expected: PASS, 10 passed

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
git add src/why_engine/script/retention.py tests/test_retention.py
git commit -m "feat: retention structure checker enforcing the cold-open template"
```

---

### Task 10: Topic miner with hard source gate and novelty

**Files:**
- Create: `src/why_engine/topics/__init__.py`
- Create: `src/why_engine/topics/mine.py`
- Modify: `src/why_engine/models.py` (append `TopicCandidate`)
- Test: `tests/test_mine.py`

**Interfaces:**
- Consumes: `Settings` (Task 1)
- Produces: `TopicCandidate(question, demand, incumbent_weakness, hook_strength, source_urls)`; `score_topic(candidate, published: list[str]) -> float`; `select_next_topic(candidates, published) -> TopicCandidate | None`

Source availability is a hard gate, not a weight: a candidate with no `source_urls` scores 0 and can never be selected. Novelty is a token-overlap check against already-published questions, which is what enforces the variation the inauthentic-content policy requires.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mine.py
from why_engine.models import TopicCandidate
from why_engine.topics.mine import score_topic, select_next_topic


def _cand(question="Why are American houses made of wood?", demand=0.8,
          incumbent_weakness=0.7, hook_strength=0.9, sources=("https://gao.gov/a",)):
    return TopicCandidate(question=question, demand=demand,
                          incumbent_weakness=incumbent_weakness,
                          hook_strength=hook_strength, source_urls=list(sources))


def test_a_candidate_with_no_sources_scores_zero():
    assert score_topic(_cand(sources=()), published=[]) == 0.0


def test_a_strong_candidate_scores_above_zero():
    assert score_topic(_cand(), published=[]) > 0.0


def test_an_exact_duplicate_of_a_published_question_scores_zero():
    q = "Why are American houses made of wood?"
    assert score_topic(_cand(question=q), published=[q]) == 0.0


def test_a_near_duplicate_is_penalised_below_a_fresh_topic():
    fresh = score_topic(_cand(question="Why is American bread so sweet?"),
                        published=["Why are American houses made of wood?"])
    near = score_topic(_cand(question="Why are American homes made of wood?"),
                       published=["Why are American houses made of wood?"])
    assert near < fresh


def test_higher_demand_scores_higher_all_else_equal():
    assert score_topic(_cand(demand=0.9), []) > score_topic(_cand(demand=0.2), [])


def test_weaker_incumbents_score_higher_all_else_equal():
    assert score_topic(_cand(incumbent_weakness=0.9), []) > score_topic(_cand(incumbent_weakness=0.1), [])


def test_select_next_topic_returns_the_highest_scoring_candidate():
    weak = _cand(question="Why is bread sweet?", demand=0.1, incumbent_weakness=0.1, hook_strength=0.1)
    strong = _cand(question="Why are roads so wide?", demand=0.9, incumbent_weakness=0.9, hook_strength=0.9)
    assert select_next_topic([weak, strong], published=[]).question == "Why are roads so wide?"


def test_select_next_topic_returns_none_when_every_candidate_is_gated_out():
    """No source, no topic. Returning None is correct: the day publishes nothing."""
    assert select_next_topic([_cand(sources=()), _cand(sources=())], published=[]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_mine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'why_engine.topics'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/why_engine/models.py`:

```python
class TopicCandidate(BaseModel):
    question: str = Field(min_length=1)
    demand: float = Field(ge=0.0, le=1.0)
    incumbent_weakness: float = Field(ge=0.0, le=1.0)
    hook_strength: float = Field(ge=0.0, le=1.0)
    source_urls: List[str] = Field(default_factory=list)
```

```python
# src/why_engine/topics/__init__.py
```

```python
# src/why_engine/topics/mine.py
from typing import List, Optional, Set

from why_engine.models import TopicCandidate

DEMAND_WEIGHT = 0.4
INCUMBENT_WEIGHT = 0.35
HOOK_WEIGHT = 0.25

NOVELTY_FLOOR = 0.35  # Jaccard overlap at or above this is treated as a duplicate

_STOPWORDS = {"why", "the", "a", "an", "are", "is", "do", "does", "we", "so", "of", "in", "to"}


def _tokens(question: str) -> Set[str]:
    return {w.strip("?.,!").lower() for w in question.split()} - _STOPWORDS


def _novelty(question: str, published: List[str]) -> float:
    """1.0 means wholly novel, 0.0 means a duplicate of something already published."""
    candidate = _tokens(question)
    if not candidate:
        return 0.0
    worst = 0.0
    for prior in published:
        prior_tokens = _tokens(prior)
        union = candidate | prior_tokens
        if not union:
            continue
        worst = max(worst, len(candidate & prior_tokens) / len(union))
    if worst >= NOVELTY_FLOOR:
        return 0.0
    return 1.0 - worst


def score_topic(candidate: TopicCandidate, published: List[str]) -> float:
    if not candidate.source_urls:
        return 0.0  # hard gate: no source, no topic

    novelty = _novelty(candidate.question, published)
    if novelty == 0.0:
        return 0.0

    base = (
        DEMAND_WEIGHT * candidate.demand
        + INCUMBENT_WEIGHT * candidate.incumbent_weakness
        + HOOK_WEIGHT * candidate.hook_strength
    )
    return base * novelty


def select_next_topic(candidates: List[TopicCandidate], published: List[str]) -> Optional[TopicCandidate]:
    scored = [(score_topic(c, published), c) for c in candidates]
    viable = [(s, c) for s, c in scored if s > 0.0]
    if not viable:
        return None
    return max(viable, key=lambda pair: pair[0])[1]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_mine.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
cd ~/why-engine
git add src/why_engine/topics src/why_engine/models.py tests/test_mine.py
git commit -m "feat: topic miner with hard source gate and novelty enforcement"
```

---

### Task 11: Orchestrator, run artifacts and CLI

**Files:**
- Create: `src/why_engine/run.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: everything above
- Produces: `produce_script(topic, corpus, llm, settings, run_dir) -> Script | None`; `main(argv) -> int` CLI entry point

Loop: write -> provenance check -> retention check -> gate. `cut_claims` removes the offending claims and re-runs the retention check (cutting can push runtime under the floor). `regenerate` retries up to `settings.max_regenerations`. `abort` stops immediately. Exhausting retries returns `None`. Every attempt is written to a numbered file; nothing is overwritten.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run.py
import json

from why_engine.config import Settings
from why_engine.grounding.verify import EntailmentResult
from why_engine.llm import StubLLM
from why_engine.models import Claim, Passage, Script, Segment
from why_engine.run import produce_script
from why_engine.sources.corpus import Corpus

QUOTE = "In 1877 the Commission distributed carp."


def _corpus():
    return Corpus(passages=[Passage(source_id="gao-24", locator="c0", text=QUOTE,
                                    url="https://gao.gov/x", title="T")])


def _script(n_body_words=2400, claim_text="The Commission distributed carp in 1877."):
    """Body is split across five acts to satisfy the per-act ceiling. The single
    claim lives in segment 1, which test_fabricated_provenance mutates."""
    per_act = max(1, n_body_words // 5)
    body = [
        Segment(segment_id=i, act=i, narration=" ".join(["word"] * per_act),
                claims=[], image_brief="a fish")
        for i in range(1, 6)
    ]
    body[0].claims = [Claim(text=claim_text, source_id="gao-24", locator="c0", quote=QUOTE)]
    return Script(topic="Why don't Americans eat carp?", segments=[
        Segment(segment_id=0, act=0, narration="Americans refuse to eat carp.",
                claims=[], image_brief="a river"),
        *body,
        Segment(segment_id=6, act=6, narration="That is why Americans refuse to eat carp.",
                claims=[], image_brief="a plate"),
    ])


def test_a_clean_script_is_returned_and_persisted(tmp_path):
    llm = StubLLM([_script(), EntailmentResult(verdict="SUPPORTED", reasoning="r")])
    result = produce_script("Why don't Americans eat carp?", _corpus(), llm, Settings(), tmp_path)
    assert result is not None
    assert (tmp_path / "script_attempt_1.json").exists()
    assert (tmp_path / "verdicts.json").exists()


def test_contradiction_aborts_and_returns_none(tmp_path):
    llm = StubLLM([_script(), EntailmentResult(verdict="CONTRADICTED", reasoning="r")])
    assert produce_script("t", _corpus(), llm, Settings(), tmp_path) is None


def test_abort_does_not_consume_regeneration_attempts(tmp_path):
    """Abort is terminal. One script written, one claim verified, then stop."""
    llm = StubLLM([_script(), EntailmentResult(verdict="CONTRADICTED", reasoning="r")])
    produce_script("t", _corpus(), llm, Settings(), tmp_path)
    assert not (tmp_path / "script_attempt_2.json").exists()


def test_fabricated_provenance_forces_regeneration(tmp_path):
    bad = _script()
    bad.segments[1].claims[0].source_id = "invented-source"
    llm = StubLLM([bad, _script(), EntailmentResult(verdict="SUPPORTED", reasoning="r")])
    result = produce_script("t", _corpus(), llm, Settings(), tmp_path)
    assert result is not None
    assert (tmp_path / "script_attempt_2.json").exists()


def test_retention_violation_forces_regeneration(tmp_path):
    llm = StubLLM([_script(n_body_words=50), _script(),
                   EntailmentResult(verdict="SUPPORTED", reasoning="r")])
    assert produce_script("t", _corpus(), llm, Settings(), tmp_path) is not None


def test_exhausting_regenerations_returns_none(tmp_path):
    settings = Settings(max_regenerations=2)
    llm = StubLLM([_script(n_body_words=50), _script(n_body_words=50)])
    assert produce_script("t", _corpus(), llm, settings, tmp_path) is None


def test_verdicts_file_records_every_attempt_and_its_outcome(tmp_path):
    llm = StubLLM([_script(n_body_words=50), _script(),
                   EntailmentResult(verdict="SUPPORTED", reasoning="r")])
    produce_script("t", _corpus(), llm, Settings(), tmp_path)
    verdicts = json.loads((tmp_path / "verdicts.json").read_text())
    assert len(verdicts["attempts"]) == 2
    assert verdicts["attempts"][0]["stage"] == "retention"
    assert verdicts["final"] == "accepted"


def test_attempt_files_are_never_overwritten(tmp_path):
    llm = StubLLM([_script(n_body_words=50), _script(),
                   EntailmentResult(verdict="SUPPORTED", reasoning="r")])
    produce_script("t", _corpus(), llm, Settings(), tmp_path)
    assert (tmp_path / "script_attempt_1.json").exists()
    assert (tmp_path / "script_attempt_2.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'why_engine.run'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/why_engine/run.py
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from why_engine.config import Settings
from why_engine.grounding.gate import run_gate
from why_engine.llm import AnthropicLLM, LLMClient
from why_engine.models import Script
from why_engine.script.retention import check_retention
from why_engine.script.write import validate_provenance, write_script
from why_engine.sources.corpus import Corpus


def _cut_claims(script: Script, cut_indices: List[int]) -> Script:
    keep = set()
    index = 0
    for segment in script.segments:
        for _ in segment.claims:
            if index not in cut_indices:
                keep.add(index)
            index += 1
    trimmed = script.model_copy(deep=True)
    index = 0
    for segment in trimmed.segments:
        surviving = []
        for claim in segment.claims:
            if index in keep:
                surviving.append(claim)
            index += 1
        segment.claims = surviving
    return trimmed


def produce_script(
    topic: str,
    corpus: Corpus,
    llm: LLMClient,
    settings: Settings,
    run_dir: Path,
) -> Optional[Script]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    attempts: List[Dict[str, Any]] = []
    final = "exhausted"
    accepted: Optional[Script] = None

    for attempt in range(1, settings.max_regenerations + 1):
        script = write_script(topic, corpus, llm)
        (run_dir / f"script_attempt_{attempt}.json").write_text(
            script.model_dump_json(indent=2), encoding="utf-8"
        )

        provenance_errors = validate_provenance(script, corpus)
        if provenance_errors:
            attempts.append({"attempt": attempt, "stage": "provenance", "errors": provenance_errors})
            continue

        retention_errors = check_retention(script, settings)
        if retention_errors:
            attempts.append({"attempt": attempt, "stage": "retention", "errors": retention_errors})
            continue

        gate = run_gate(script, llm, settings)
        attempts.append({
            "attempt": attempt, "stage": "grounding", "action": gate.action,
            "claim_verdicts": [v.model_dump() for v in gate.claim_verdicts],
        })

        if gate.action == "abort":
            final = "aborted"
            break

        if gate.action == "regenerate":
            continue

        if gate.action == "cut_claims":
            script = _cut_claims(script, gate.cut_indices)
            if check_retention(script, settings):
                attempts.append({"attempt": attempt, "stage": "retention_after_cut",
                                 "errors": check_retention(script, settings)})
                continue

        accepted = script
        final = "accepted"
        break

    (run_dir / "verdicts.json").write_text(
        json.dumps({"topic": topic, "attempts": attempts, "final": final}, indent=2),
        encoding="utf-8",
    )
    if accepted is not None:
        (run_dir / "script.json").write_text(accepted.model_dump_json(indent=2), encoding="utf-8")
    return accepted


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="why-engine")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    llm = AnthropicLLM(settings)
    script = produce_script(args.topic, Corpus.load(args.corpus), llm, settings, args.run_dir)

    print(f"cost: ${llm.total_cost_usd:.2f}")
    if script is None:
        print("no script produced; publishing nothing", file=sys.stderr)
        return 1
    print(f"script accepted: {args.run_dir / 'script.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/why-engine && .venv/bin/python -m pytest tests/test_run.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Run the whole suite and commit**

```bash
cd ~/why-engine
.venv/bin/python -m pytest -v
git add src/why_engine/run.py tests/test_run.py
git commit -m "feat: pipeline orchestrator with auditable run artifacts and CLI"
```

Expected: 78 passed, 6 deselected

---

## Deviations from the spec

Spec section 10 lists "an act ending without a hook" as a retention rejection
rule. It is not implemented in this plan. Every other S10 rule (missing cold
open, per-act ceiling, unresolved callback, runtime bounds) is mechanical and is
enforced in Task 9; hook detection is a judgement call that needs its own LLM
call and its own adversarial fixture set to be worth trusting. Adding an
unvalidated hook checker would produce a gate that rejects arbitrarily, which is
worse than no gate. It is deferred to a dedicated task once there are real
scripts to calibrate against.

## Definition of Done

- `.venv/bin/python -m pytest` passes with zero failures and the live tests deselected.
- `.venv/bin/python -m pytest -m live` passes all six adversarial entailment cases.
- `python -m why_engine.run --topic "..." --corpus corpus.jsonl --run-dir runs/test` produces either a `script.json` whose every claim is verified, or exit code 1 and no script.
- `runs/<id>/verdicts.json` records every attempt, every gate decision, and every claim verdict.
- No module except `llm.py` imports `anthropic`.
