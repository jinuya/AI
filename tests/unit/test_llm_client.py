"""LLM client — spec §7.7. RecordedLLMClient and FakeLLMClient, no network."""

from __future__ import annotations

import pytest

from atrader.core.errors import LLMError
from atrader.strategy.llm.client import (
    FakeLLMClient,
    LLMRequest,
    LLMResponse,
    LLMTradingDecision,
    LLMTradingResponse,
    LLMUsage,
    RecordedLLMClient,
)


def make_request(**overrides: object) -> LLMRequest:
    defaults: dict[str, object] = {
        "system_prompt": "system",
        "user_content": "user",
        "model": "claude-opus-5",
        "effort": "medium",
        "max_tokens": 1000,
    }
    return LLMRequest(**{**defaults, **overrides})  # type: ignore[arg-type]


def make_response(*, symbol: str = "AAPL", action: str = "hold") -> LLMResponse:
    return LLMResponse(
        decisions=LLMTradingResponse(
            decisions=(
                LLMTradingDecision(
                    symbol=symbol, action=action, confidence="0.9", rationale="test"
                ),
            )
        ),
        model="claude-opus-5",
        stop_reason="end_turn",
        usage=LLMUsage(input_tokens=10, output_tokens=5),
        latency_ms=12,
        request_id="req_1",
        raw_text="{}",
    )


class TestLLMRequestCacheKey:
    def test_identical_requests_hash_identically(self) -> None:
        assert make_request().cache_key() == make_request().cache_key()

    def test_a_different_field_changes_the_hash(self) -> None:
        assert make_request().cache_key() != make_request(user_content="different").cache_key()

    def test_effort_is_part_of_the_hash(self) -> None:
        assert make_request().cache_key() != make_request(effort="high").cache_key()


class TestFakeLLMClient:
    def test_returns_queued_responses_in_order(self) -> None:
        first = make_response(symbol="AAPL")
        second = make_response(symbol="MSFT")
        client = FakeLLMClient(responses=[first, second])
        assert client.complete(make_request()) is first
        assert client.complete(make_request()) is second

    def test_falls_back_to_default_once_the_queue_is_empty(self) -> None:
        default = make_response(symbol="DEFAULT")
        client = FakeLLMClient(default=default)
        assert client.complete(make_request()) is default
        assert client.complete(make_request()) is default

    def test_raises_without_a_queue_or_default(self) -> None:
        client = FakeLLMClient()
        with pytest.raises(LLMError):
            client.complete(make_request())

    def test_raises_configured_exception_every_call(self) -> None:
        client = FakeLLMClient(raises=ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            client.complete(make_request())

    def test_records_every_request(self) -> None:
        client = FakeLLMClient(default=make_response())
        request_a = make_request(user_content="first")
        request_b = make_request(user_content="second")
        client.complete(request_a)
        client.complete(request_b)
        assert client.requests == [request_a, request_b]


class TestRecordedLLMClientReplayMode:
    def test_replays_a_matching_recording(self) -> None:
        request = make_request()
        response = make_response()
        client = RecordedLLMClient(recordings={request.cache_key(): response})
        assert client.complete(request) is response

    def test_raises_on_a_miss_with_no_upstream(self) -> None:
        client = RecordedLLMClient()
        with pytest.raises(LLMError, match="no recording"):
            client.complete(make_request())


class TestRecordedLLMClientRecordMode:
    def test_calls_upstream_once_and_caches(self) -> None:
        response = make_response()
        upstream = FakeLLMClient(default=response)
        client = RecordedLLMClient(upstream=upstream)
        request = make_request()

        first = client.complete(request)
        second = client.complete(request)

        assert first is response
        assert second is response
        assert len(upstream.requests) == 1, (
            "the second call must be served from cache, not upstream"
        )

    def test_invokes_on_record_exactly_once_per_new_request(self) -> None:
        recorded: list[tuple[str, LLMResponse]] = []
        upstream = FakeLLMClient(default=make_response())
        client = RecordedLLMClient(
            upstream=upstream, on_record=lambda k, r: recorded.append((k, r))
        )

        request = make_request()
        client.complete(request)
        client.complete(request)

        assert len(recorded) == 1
        assert recorded[0][0] == request.cache_key()

    def test_len_reports_the_number_of_distinct_recordings(self) -> None:
        upstream = FakeLLMClient(default=make_response())
        client = RecordedLLMClient(upstream=upstream)
        client.complete(make_request(user_content="a"))
        client.complete(make_request(user_content="b"))
        client.complete(make_request(user_content="a"))  # cache hit, not a new recording
        assert len(client) == 2


class TestRecordedLLMClientJsonlRoundtrip:
    def test_recordings_survive_a_write_read_cycle(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "recordings.jsonl"
        response = LLMResponse(
            decisions=LLMTradingResponse(
                decisions=(
                    LLMTradingDecision(
                        symbol="AAPL",
                        action="buy",
                        quantity="10",
                        confidence="0.75",
                        rationale="golden cross",
                        stop_loss="95.5",
                        take_profit="110.25",
                    ),
                )
            ),
            model="claude-opus-5",
            stop_reason="end_turn",
            usage=LLMUsage(
                input_tokens=100,
                output_tokens=20,
                cache_read_input_tokens=80,
                cache_creation_input_tokens=0,
            ),
            latency_ms=345,
            request_id="req_abc",
            raw_text='{"decisions": []}',
        )
        request = make_request()
        original = RecordedLLMClient(recordings={request.cache_key(): response})

        original.to_jsonl(path)
        reloaded = RecordedLLMClient.from_jsonl(path)

        replayed = reloaded.complete(request)
        assert replayed == response

    def test_reloaded_client_still_needs_an_upstream_for_a_miss(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "empty.jsonl"
        RecordedLLMClient().to_jsonl(path)
        reloaded = RecordedLLMClient.from_jsonl(path)
        with pytest.raises(LLMError):
            reloaded.complete(make_request())

    def test_blank_lines_in_the_file_are_skipped(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "with_blanks.jsonl"
        response = make_response()
        request = make_request()
        RecordedLLMClient(recordings={request.cache_key(): response}).to_jsonl(path)
        with_blank_lines = path.read_text(encoding="utf-8") + "\n\n"
        path.write_text(with_blank_lines, encoding="utf-8")

        reloaded = RecordedLLMClient.from_jsonl(path)
        assert len(reloaded) == 1
        assert reloaded.complete(request) == response
