import types

import pytest
from pydantic import BaseModel

from why_engine.llm import AnthropicLLM, StubLLM, estimate_cost


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


def test_estimate_cost_bills_cache_writes_at_one_point_two_five_x():
    # 1M cache-write tokens bill at 1.25 x $5 = $6.25
    assert estimate_cost(0, 0, 0, 1_000_000) == pytest.approx(6.25)


def test_estimate_cost_backwards_compatible_with_three_positional_args():
    # Existing call sites pass only three positional args; this must not break
    result = estimate_cost(1_000_000, 1_000_000, 0)
    assert result == pytest.approx(30.0)


def test_cache_control_has_no_ttl_key():
    # The 1h TTL bills cache writes at 2.0x; CACHE_WRITE_MULTIPLIER assumes the
    # 5-minute rate (1.25x). Dropping the ttl keeps the ephemeral (5-minute) tier.
    system_arg = AnthropicLLM._build_system_arg("sys-prompt", True)
    assert system_arg[0]["cache_control"] == {"type": "ephemeral"}
    assert "ttl" not in system_arg[0]["cache_control"]


def test_record_handles_missing_cache_fields():
    # _record must handle responses with no cache_creation_input_tokens
    fake_usage = types.SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=None,
    )
    fake_response = types.SimpleNamespace(usage=fake_usage)

    llm = object.__new__(AnthropicLLM)
    llm.total_cost_usd = 0.0
    llm.usage_log = []

    llm._record(fake_response)

    # 100 input tokens at $5/M + 50 output at $25/M = $0.00175
    assert llm.total_cost_usd == pytest.approx(0.00175)
    assert len(llm.usage_log) == 1
    assert llm.usage_log[0]["cache_creation_input_tokens"] == 0
