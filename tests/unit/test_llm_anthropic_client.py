"""AnthropicLLMClient — spec §7.7.

Exercises the real client's request-shaping and response-handling logic
against a stub standing in for the network-calling SDK object — no live API
access, ever, in a unit test (see ``docs/llm-determinism.md``; the actual
network path is covered only by the opt-in ``live_llm`` marker with a real
key).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from atrader.core.clock import SimulatedClock
from atrader.core.errors import LLMRefusalError, LLMSchemaError
from atrader.strategy.llm.client import AnthropicLLMClient, LLMRequest, LLMTradingResponse


def make_request(**overrides: object) -> LLMRequest:
    defaults: dict[str, object] = {
        "system_prompt": "system",
        "user_content": "user",
        "model": "claude-opus-5",
        "effort": "medium",
        "max_tokens": 1000,
    }
    return LLMRequest(**{**defaults, **overrides})  # type: ignore[arg-type]


@dataclass
class _StubUsage:
    input_tokens: int = 10
    output_tokens: int = 5
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


@dataclass
class _StubStopDetails:
    category: str | None
    explanation: str | None


@dataclass
class _StubResponse:
    stop_reason: str
    model: str = "claude-opus-5"
    parsed_output_value: LLMTradingResponse | None = None
    usage: _StubUsage = field(default_factory=_StubUsage)
    stop_details: _StubStopDetails | None = None
    _request_id: str = "req_stub"
    accessed_parsed_output: bool = field(default=False, init=False)

    @property
    def parsed_output(self) -> LLMTradingResponse | None:
        self.accessed_parsed_output = True
        return self.parsed_output_value


class _StubMessagesEndpoint:
    def __init__(self, response: _StubResponse, *, calls: list[dict[str, Any]]) -> None:
        self._response = response
        self._calls = calls

    def parse(self, **kwargs: Any) -> _StubResponse:
        self._calls.append(kwargs)
        return self._response


class _StubBetaNamespace:
    def __init__(self, messages: _StubMessagesEndpoint) -> None:
        self.messages = messages


class _StubAnthropic:
    def __init__(self, response: _StubResponse) -> None:
        self.calls: list[dict[str, Any]] = []
        self.beta_calls: list[dict[str, Any]] = []
        self.messages = _StubMessagesEndpoint(response, calls=self.calls)
        self.beta = _StubBetaNamespace(_StubMessagesEndpoint(response, calls=self.beta_calls))


def make_client(
    response: _StubResponse, *, enable_refusal_fallback: bool = False
) -> tuple[AnthropicLLMClient, _StubAnthropic]:
    client = AnthropicLLMClient(
        api_key="test-key",
        clock=SimulatedClock(start_ns=0),
        enable_refusal_fallback=enable_refusal_fallback,
    )
    stub = _StubAnthropic(response)
    client._client = stub  # type: ignore[assignment]
    return client, stub


class TestSuccess:
    def test_builds_a_response_from_parsed_output(self) -> None:
        decisions = LLMTradingResponse(decisions=())
        stub_response = _StubResponse(stop_reason="end_turn", parsed_output_value=decisions)
        client, _ = make_client(stub_response)
        result = client.complete(make_request())
        assert result.decisions is decisions
        assert result.model == "claude-opus-5"
        assert result.stop_reason == "end_turn"
        assert result.request_id == "req_stub"

    def test_uses_the_non_beta_endpoint_by_default(self) -> None:
        stub_response = _StubResponse(
            stop_reason="end_turn", parsed_output_value=LLMTradingResponse(decisions=())
        )
        client, stub = make_client(stub_response)
        client.complete(make_request())
        assert len(stub.calls) == 1
        assert len(stub.beta_calls) == 0

    def test_effort_is_forwarded_via_output_config(self) -> None:
        stub_response = _StubResponse(
            stop_reason="end_turn", parsed_output_value=LLMTradingResponse(decisions=())
        )
        client, stub = make_client(stub_response)
        client.complete(make_request(effort="high"))
        assert stub.calls[0]["output_config"]["effort"] == "high"

    def test_never_sends_temperature_top_p_or_seed(self) -> None:
        stub_response = _StubResponse(
            stop_reason="end_turn", parsed_output_value=LLMTradingResponse(decisions=())
        )
        client, stub = make_client(stub_response)
        client.complete(make_request())
        sent = stub.calls[0]
        assert "temperature" not in sent
        assert "top_p" not in sent
        assert "seed" not in sent

    def test_system_prompt_is_cached_via_cache_control(self) -> None:
        stub_response = _StubResponse(
            stop_reason="end_turn", parsed_output_value=LLMTradingResponse(decisions=())
        )
        client, stub = make_client(stub_response)
        client.complete(make_request(system_prompt="fixed system prompt"))
        system = stub.calls[0]["system"]
        assert system[0]["text"] == "fixed system prompt"
        assert system[0]["cache_control"] == {"type": "ephemeral"}


class TestRefusalFallback:
    def test_enabling_it_routes_through_the_beta_endpoint_with_default_fallback(self) -> None:
        stub_response = _StubResponse(
            stop_reason="end_turn", parsed_output_value=LLMTradingResponse(decisions=())
        )
        client, stub = make_client(stub_response, enable_refusal_fallback=True)
        client.complete(make_request())
        assert len(stub.beta_calls) == 1
        assert len(stub.calls) == 0
        assert stub.beta_calls[0]["fallbacks"] == "default"
        assert "server-side-fallback-2026-07-01" in stub.beta_calls[0]["betas"]


class TestRefusal:
    def test_a_refusal_is_raised_with_category_and_explanation(self) -> None:
        stub_response = _StubResponse(
            stop_reason="refusal",
            stop_details=_StubStopDetails(category="cyber", explanation="declined"),
        )
        client, _ = make_client(stub_response)
        with pytest.raises(LLMRefusalError) as exc_info:
            client.complete(make_request())
        assert exc_info.value.category == "cyber"
        assert exc_info.value.explanation == "declined"

    def test_refusal_is_detected_without_ever_reading_parsed_output(self) -> None:
        # Spec §7.7: stop_reason == "refusal" must be checked *before* content
        # is read. Proven structurally, not assumed: the stub records whether
        # its parsed_output property was ever touched.
        stub_response = _StubResponse(
            stop_reason="refusal",
            stop_details=_StubStopDetails(category=None, explanation=None),
            parsed_output_value=LLMTradingResponse(decisions=()),
        )
        client, _ = make_client(stub_response)
        with pytest.raises(LLMRefusalError):
            client.complete(make_request())
        assert stub_response.accessed_parsed_output is False


class TestMissingParsedOutput:
    def test_no_parsed_output_raises_a_schema_error(self) -> None:
        stub_response = _StubResponse(stop_reason="max_tokens", parsed_output_value=None)
        client, _ = make_client(stub_response)
        with pytest.raises(LLMSchemaError):
            client.complete(make_request())


class TestSdkValidationFailure:
    def test_a_validation_error_from_parse_itself_becomes_a_schema_error(self) -> None:
        # Simulates the SDK's own structured-output parsing failing before it
        # ever returns a response object — e.g. output truncated mid-JSON at
        # max_tokens. `messages.parse()` raises pydantic.ValidationError in
        # that case; this must not escape as a raw pydantic error.
        import pydantic

        class _RaisingMessagesEndpoint:
            def parse(self, **kwargs: Any) -> _StubResponse:
                try:
                    pydantic.TypeAdapter(int).validate_python("not an int")
                except pydantic.ValidationError as exc:
                    raise exc
                raise AssertionError("expected validate_python to raise")

        client = AnthropicLLMClient(
            api_key="test-key", clock=SimulatedClock(start_ns=0), enable_refusal_fallback=False
        )
        client._client = _StubAnthropic(_StubResponse(stop_reason="max_tokens"))  # type: ignore[assignment]
        client._client.messages = _RaisingMessagesEndpoint()  # type: ignore[assignment]

        with pytest.raises(LLMSchemaError):
            client.complete(make_request())
