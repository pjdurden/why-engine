import json
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional, Protocol, Type, TypeVar

from pydantic import BaseModel

from why_engine.config import Settings

T = TypeVar("T", bound=BaseModel)

INPUT_USD_PER_TOKEN = 5.00 / 1_000_000
OUTPUT_USD_PER_TOKEN = 25.00 / 1_000_000
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int = 0,
) -> float:
    return (
        input_tokens * INPUT_USD_PER_TOKEN
        + output_tokens * OUTPUT_USD_PER_TOKEN
        + cache_read_tokens * INPUT_USD_PER_TOKEN * CACHE_READ_MULTIPLIER
        + cache_write_tokens * INPUT_USD_PER_TOKEN * CACHE_WRITE_MULTIPLIER
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

    @staticmethod
    def _build_system_arg(system: str, cache_system: bool) -> Any:
        if cache_system:
            return [{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}]
        return system

    def parse(self, *, system, user, output_format, effort="high", cache_system=False):
        system_arg = self._build_system_arg(system, cache_system)

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
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cost = estimate_cost(usage.input_tokens, usage.output_tokens, cache_read, cache_write)
        self.total_cost_usd += cost
        self.usage_log.append(
            {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
             "cache_read_input_tokens": cache_read, "cache_creation_input_tokens": cache_write,
             "cost_usd": cost}
        )


# --- headless Claude Code backend -------------------------------------------
#
# Subscription-billed inference via the `claude` CLI. No API key required.
# The CLI has no structured-output guarantee, so the schema travels in-band and
# the response is validated client-side with one retry.

_JSON_INSTRUCTION = (
    "\n\nReply with ONLY a single JSON object matching this schema. "
    "No prose, no explanation, no markdown fences.\n\nSCHEMA:\n"
)


def extract_json(text: str) -> str:
    """Pull the outermost JSON object out of a model response.

    The CLI may wrap output in fences or add a sentence either side; neither is
    a failure, so recover rather than reject.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in response: {text[:120]!r}")
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"unterminated JSON object in response: {text[:120]!r}")


def _run_claude(*, model: str, system: str, user: str, timeout_s: int) -> str:
    """One headless Claude Code call, stripped of everything the pipeline does not need.

    `--system-prompt` replaces the coding preamble rather than appending to it,
    `--tools ""` drops tool definitions, and `--strict-mcp-config` keeps the
    user's MCP servers out of every call.
    """
    exe = shutil.which("claude")
    if exe is None:
        raise RuntimeError("the `claude` CLI is not on PATH")
    proc = subprocess.run(
        [exe, "-p", "--model", model, "--system-prompt", system,
         "--tools", "", "--strict-mcp-config", user],
        capture_output=True, text=True, timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr[:300]}")
    return proc.stdout


class ClaudeCodeLLM:
    """LLMClient backed by headless Claude Code instead of the API.

    Model tier is chosen per client, not per call: construct one instance for
    script writing and another for the gate. Measured on the six adversarial
    fixtures, haiku scored 6/6 and opus 5/6, so the gate should run on haiku.
    """

    prefers_batch = True

    def __init__(
        self,
        settings: Settings,
        model: str = "opus",
        timeout_s: int = 300,
        runner: Optional[Callable[..., str]] = None,
    ):
        self._settings = settings
        self._model = model
        self._timeout_s = timeout_s
        self._runner = runner or _run_claude
        self.call_count = 0
        self.total_cost_usd = 0.0  # subscription-billed; calls are the real meter
        self.usage_log: List[Dict[str, Any]] = []

    def parse(
        self,
        *,
        system: str,
        user: str,
        output_format: Type[T],
        effort: str = "high",
        cache_system: bool = False,
        model: Optional[str] = None,
    ) -> T:
        schema = json.dumps(output_format.model_json_schema(), sort_keys=True)
        full_system = f"{system}{_JSON_INSTRUCTION}{schema}"
        chosen = model or self._model

        last_error: Optional[Exception] = None
        for _ in range(2):
            raw = self._runner(
                model=chosen, system=full_system, user=user, timeout_s=self._timeout_s
            )
            self.call_count += 1
            self.usage_log.append({"model": chosen, "chars_out": len(raw)})
            try:
                return output_format.model_validate_json(extract_json(raw))
            except Exception as exc:  # malformed output is recoverable; retry once
                last_error = exc
        raise ValueError(
            f"claude returned unparseable output twice for {output_format.__name__}: {last_error}"
        )
