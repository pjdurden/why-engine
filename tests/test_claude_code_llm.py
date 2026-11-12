import pytest
from pydantic import BaseModel

from why_engine.config import Settings
from why_engine.llm import ClaudeCodeLLM, extract_json


class Answer(BaseModel):
    value: str


def _llm(*responses, model="opus"):
    """ClaudeCodeLLM with a scripted runner standing in for the claude subprocess."""
    calls = []

    def runner(*, model, system, user, timeout_s):
        calls.append({"model": model, "system": system, "user": user})
        return responses[len(calls) - 1]

    llm = ClaudeCodeLLM(Settings(), model=model, runner=runner)
    llm.calls = calls
    return llm


# --- extract_json ------------------------------------------------------------

def test_extract_json_passes_through_bare_json():
    assert extract_json('{"value": "x"}') == '{"value": "x"}'


def test_extract_json_strips_markdown_fences():
    assert extract_json('```json\n{"value": "x"}\n```') == '{"value": "x"}'


def test_extract_json_ignores_prose_around_the_object():
    assert extract_json('Sure! Here you go:\n{"value": "x"}\nHope that helps.') == '{"value": "x"}'


def test_extract_json_keeps_nested_braces_intact():
    assert extract_json('noise {"a": {"b": 1}} noise') == '{"a": {"b": 1}}'


def test_extract_json_raises_when_there_is_no_object():
    with pytest.raises(ValueError):
        extract_json("no json here at all")


# --- ClaudeCodeLLM -----------------------------------------------------------

def test_parse_returns_a_validated_model():
    llm = _llm('{"value": "hello"}')
    assert llm.parse(system="s", user="u", output_format=Answer).value == "hello"


def test_parse_appends_the_json_schema_to_the_system_prompt():
    """The CLI has no structured-output guarantee, so the schema must be in-band."""
    llm = _llm('{"value": "x"}')
    llm.parse(system="BASE PROMPT", user="u", output_format=Answer)
    system = llm.calls[0]["system"]
    assert "BASE PROMPT" in system
    assert "value" in system and "properties" in system


def test_parse_retries_once_on_unparseable_output():
    llm = _llm("not json", '{"value": "recovered"}')
    assert llm.parse(system="s", user="u", output_format=Answer).value == "recovered"
    assert len(llm.calls) == 2


def test_parse_raises_after_the_retry_also_fails():
    llm = _llm("not json", "still not json")
    with pytest.raises(ValueError):
        llm.parse(system="s", user="u", output_format=Answer)


def test_parse_counts_calls_and_reports_zero_dollar_cost():
    """Subscription-billed: the meaningful meter is calls, not dollars."""
    llm = _llm('{"value": "a"}', '{"value": "b"}')
    llm.parse(system="s", user="u", output_format=Answer)
    llm.parse(system="s", user="u", output_format=Answer)
    assert llm.call_count == 2
    assert llm.total_cost_usd == 0.0


def test_a_per_call_model_override_beats_the_default():
    """The gate runs on a cheaper tier than the writer."""
    llm = _llm('{"value": "x"}', model="opus")
    llm.parse(system="s", user="u", output_format=Answer, model="haiku")
    assert llm.calls[0]["model"] == "haiku"


def test_claude_code_llm_prefers_batched_verification():
    """run_gate reads this flag to decide whether to batch."""
    assert ClaudeCodeLLM(Settings(), runner=lambda **kw: "{}").prefers_batch is True


def test_the_constructed_model_is_the_one_actually_used():
    """Model tier is per client. A stored-but-unused tier would silently route
    the gate to the expensive model, which is exactly what this guards."""
    llm = _llm('{"value": "x"}', model="haiku")
    llm.parse(system="s", user="u", output_format=Answer)
    assert llm.calls[0]["model"] == "haiku"
