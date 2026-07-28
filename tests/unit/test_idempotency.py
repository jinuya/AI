"""Safe submission and retry — spec §FR-EXE-04.

    응답 없음 = 재조회 우선, 존재하지 않을 때만 재전송. 무조건 재전송 금지.

The scenario this module exists to get right: a submit times out, and the
question is never "did it fail" but "do I know whether it landed". Every test
here is really asking one of two things — did a lost-response case resolve by
querying rather than resending, or did an unresolvable one lock rather than
guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pytest

from atrader.brokers.models import OrderAck, OrderState
from atrader.core.clock import SimulatedClock
from atrader.core.errors import (
    AmbiguousBrokerError,
    PermanentBrokerError,
    TransientBrokerError,
)
from atrader.core.models import OrderRequest
from atrader.core.rng import SeededRng
from atrader.core.types import OrderStatus, OrderType, Side, TimeInForce
from atrader.execution.idempotency import (
    SubmissionOutcome,
    SubmitPolicy,
    submit_with_retry,
)


def make_request(**overrides: Any) -> OrderRequest:
    defaults: dict[str, Any] = {
        "client_order_id": "coid-001",
        "symbol": "AAPL",
        "side": Side.BUY,
        "order_type": OrderType.LIMIT,
        "quantity": Decimal("100"),
        "limit_price": Decimal("187.50"),
        "time_in_force": TimeInForce.DAY,
    }
    return OrderRequest(**{**defaults, **overrides})


def make_state(request: OrderRequest, **overrides: Any) -> OrderState:
    defaults: dict[str, Any] = {
        "client_order_id": request.client_order_id,
        "broker_order_id": "broker-1",
        "symbol": request.symbol,
        "side": request.side,
        "status": OrderStatus.NEW,
        "quantity": request.quantity,
    }
    return OrderState(**{**defaults, **overrides})


@dataclass
class ScriptedBroker:
    """A minimal stand-in that plays a fixed script of submit/query outcomes.

    Deliberately not the paper broker: this module's job is to test the retry
    *policy*, and a scripted double makes each scenario's premise explicit in
    the test itself rather than buried in simulator configuration.

    ``query_result`` may be a single value/exception (returned every time) or a
    list consumed one entry per call, holding on the last entry once exhausted —
    which is what lets a test say "the first two queries find nothing, the
    third finds it".
    """

    submit_script: list[Any] = field(default_factory=list)
    query_result: OrderState | Exception | list[Any] | None = None
    submit_calls: int = field(default=0, init=False)
    query_calls: int = field(default=0, init=False)

    async def submit_order(self, request: OrderRequest) -> OrderAck:
        outcome = self.submit_script[self.submit_calls]
        self.submit_calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, OrderAck)
        return outcome

    async def get_order(self, client_order_id: str) -> OrderState | None:
        self.query_calls += 1
        if isinstance(self.query_result, list):
            index = min(self.query_calls - 1, len(self.query_result) - 1)
            outcome = self.query_result[index]
        else:
            outcome = self.query_result
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    # Unused BrokerAdapter methods — not exercised by submit_with_retry.
    async def cancel_order(self, client_order_id: str) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def modify_order(self, client_order_id: str, mods: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


def accepted_ack(request: OrderRequest) -> OrderAck:
    return OrderAck(
        client_order_id=request.client_order_id,
        broker_order_id="broker-1",
        accepted=True,
        status=OrderStatus.NEW,
    )


def rejected_ack(request: OrderRequest, reason: str = "invalid symbol") -> OrderAck:
    return OrderAck(
        client_order_id=request.client_order_id,
        broker_order_id=None,
        accepted=False,
        status=OrderStatus.REJECTED,
        reason=reason,
    )


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(start_ns=1_000_000)


@pytest.fixture
def no_delay(clock: SimulatedClock) -> Any:
    """A sleep that advances the simulated clock instead of the real one."""

    async def sleep(seconds: float) -> None:
        clock.advance_seconds(seconds)

    return sleep


class TestHappyPath:
    async def test_immediate_acceptance(self, clock: SimulatedClock, no_delay: Any) -> None:
        request = make_request()
        broker = ScriptedBroker(submit_script=[accepted_ack(request)])
        result = await submit_with_retry(broker, request, clock=clock, sleep=no_delay)  # type: ignore[arg-type]
        assert result.outcome is SubmissionOutcome.ACCEPTED
        assert result.is_live
        assert result.attempts == 1
        assert broker.query_calls == 0  # no need to ever ask

    async def test_immediate_rejection_does_not_retry(
        self, clock: SimulatedClock, no_delay: Any
    ) -> None:
        request = make_request()
        broker = ScriptedBroker(submit_script=[rejected_ack(request, "insufficient funds")])
        result = await submit_with_retry(broker, request, clock=clock, sleep=no_delay)  # type: ignore[arg-type]
        assert result.outcome is SubmissionOutcome.REJECTED
        assert not result.is_live
        assert result.reason == "insufficient funds"
        assert broker.submit_calls == 1


class TestPermanentFailure:
    async def test_permanent_error_rejects_without_retry(
        self, clock: SimulatedClock, no_delay: Any
    ) -> None:
        request = make_request()
        broker = ScriptedBroker(submit_script=[PermanentBrokerError("400 invalid symbol")])
        result = await submit_with_retry(broker, request, clock=clock, sleep=no_delay)  # type: ignore[arg-type]
        assert result.outcome is SubmissionOutcome.REJECTED
        assert broker.submit_calls == 1
        assert broker.query_calls == 0  # nothing to look up — it never went out


class TestAmbiguityResolvedByQuery:
    """The critical path: an unanswered submit that a query settles."""

    async def test_timeout_then_the_order_is_found_by_query(
        self, clock: SimulatedClock, no_delay: Any
    ) -> None:
        request = make_request()
        broker = ScriptedBroker(
            submit_script=[TimeoutError("no response")],
            query_result=make_state(request, filled_quantity=Decimal(0)),
        )
        result = await submit_with_retry(broker, request, clock=clock, sleep=no_delay)  # type: ignore[arg-type]
        assert result.outcome is SubmissionOutcome.ALREADY_EXISTS
        assert result.is_live
        assert broker.submit_calls == 1  # never resent
        assert broker.query_calls == 1

    async def test_ambiguous_broker_error_then_the_order_is_found(
        self, clock: SimulatedClock, no_delay: Any
    ) -> None:
        request = make_request()
        broker = ScriptedBroker(
            submit_script=[
                AmbiguousBrokerError("connection reset", client_order_id=request.client_order_id)
            ],
            query_result=make_state(request),
        )
        result = await submit_with_retry(broker, request, clock=clock, sleep=no_delay)  # type: ignore[arg-type]
        assert result.outcome is SubmissionOutcome.ALREADY_EXISTS
        assert broker.submit_calls == 1

    async def test_partially_filled_state_is_adopted_from_the_query(
        self, clock: SimulatedClock, no_delay: Any
    ) -> None:
        request = make_request()
        broker = ScriptedBroker(
            submit_script=[TimeoutError()],
            query_result=make_state(
                request, status=OrderStatus.PARTIALLY_FILLED, filled_quantity=Decimal("30")
            ),
        )
        result = await submit_with_retry(broker, request, clock=clock, sleep=no_delay)  # type: ignore[arg-type]
        assert result.broker_state is not None
        assert result.broker_state.filled_quantity == Decimal("30")


class TestAmbiguityUnresolved:
    """Neither confirmed nor refuted: the case that must never resend blind."""

    async def test_ambiguous_error_and_query_finds_nothing_locks_rather_than_resends(
        self, clock: SimulatedClock, no_delay: Any
    ) -> None:
        request = make_request()
        broker = ScriptedBroker(
            submit_script=[
                AmbiguousBrokerError("no response", client_order_id=request.client_order_id)
            ],
            query_result=None,
        )
        result = await submit_with_retry(broker, request, clock=clock, sleep=no_delay)  # type: ignore[arg-type]
        assert result.outcome is SubmissionOutcome.AMBIGUOUS
        assert result.needs_human
        assert not result.is_live
        assert broker.submit_calls == 1  # the whole point: never a second submit

    async def test_ambiguous_error_and_a_failing_query_also_locks(
        self, clock: SimulatedClock, no_delay: Any
    ) -> None:
        request = make_request()
        broker = ScriptedBroker(
            submit_script=[
                AmbiguousBrokerError("no response", client_order_id=request.client_order_id)
            ],
            query_result=RuntimeError("query endpoint also down"),
        )
        result = await submit_with_retry(broker, request, clock=clock, sleep=no_delay)  # type: ignore[arg-type]
        assert result.outcome is SubmissionOutcome.AMBIGUOUS
        assert broker.submit_calls == 1


class TestTransientRetry:
    async def test_transient_error_is_retried_and_then_succeeds(
        self, clock: SimulatedClock, no_delay: Any
    ) -> None:
        request = make_request()
        broker = ScriptedBroker(
            submit_script=[
                TransientBrokerError("503"),
                TransientBrokerError("503"),
                accepted_ack(request),
            ],
            query_result=None,  # each transient attempt queries and finds nothing
        )
        result = await submit_with_retry(broker, request, clock=clock, sleep=no_delay)  # type: ignore[arg-type]
        assert result.outcome is SubmissionOutcome.ACCEPTED
        assert result.attempts == 3
        assert broker.query_calls == 2

    async def test_transient_retries_use_backoff_with_jitter(
        self, clock: SimulatedClock, no_delay: Any
    ) -> None:
        request = make_request()
        broker = ScriptedBroker(
            submit_script=[TransientBrokerError("503"), accepted_ack(request)],
            query_result=None,
        )
        start = clock.now_ns()
        await submit_with_retry(
            broker,
            request,
            clock=clock,
            policy=SubmitPolicy(base_delay_seconds=0.1, jitter_pct=0.0),
            rng=SeededRng(seed=1),
            sleep=no_delay,  # type: ignore[arg-type]
        )
        # The fake sleep advances the simulated clock, so a real delay happened.
        assert clock.now_ns() > start

    async def test_retries_exhausted_and_final_query_finds_it(
        self, clock: SimulatedClock, no_delay: Any
    ) -> None:
        request = make_request()
        policy = SubmitPolicy(max_attempts=2, base_delay_seconds=0.01)
        broker = ScriptedBroker(
            submit_script=[TransientBrokerError("503"), TransientBrokerError("503")],
            # The in-loop query after each transient failure finds nothing; only
            # the final post-retry lookup — the third call — finds the order.
            query_result=[None, None, make_state(request)],
        )
        result = await submit_with_retry(
            broker, request, clock=clock, policy=policy, sleep=no_delay
        )  # type: ignore[arg-type]
        assert result.outcome is SubmissionOutcome.ALREADY_EXISTS
        assert result.attempts == 2
        assert broker.query_calls == 3

    async def test_retries_exhausted_and_nothing_found_is_ambiguous(
        self, clock: SimulatedClock, no_delay: Any
    ) -> None:
        request = make_request()
        policy = SubmitPolicy(max_attempts=2, base_delay_seconds=0.01)
        broker = ScriptedBroker(
            submit_script=[TransientBrokerError("503"), TransientBrokerError("503")],
            query_result=None,
        )
        result = await submit_with_retry(
            broker, request, clock=clock, policy=policy, sleep=no_delay
        )  # type: ignore[arg-type]
        assert result.outcome is SubmissionOutcome.AMBIGUOUS
        assert result.needs_human
        assert "2 attempts failed" in result.reason


class TestSubmitPolicyDelay:
    def test_delay_grows_with_attempt_and_caps_at_max(self) -> None:
        policy = SubmitPolicy(base_delay_seconds=1.0, max_delay_seconds=4.0, jitter_pct=0.0)
        rng = SeededRng(seed=0)
        assert policy.delay_for(1, rng) == pytest.approx(1.0)
        assert policy.delay_for(2, rng) == pytest.approx(2.0)
        assert policy.delay_for(3, rng) == pytest.approx(4.0)
        assert policy.delay_for(10, rng) == pytest.approx(4.0)  # capped

    def test_delay_never_goes_negative_even_with_jitter(self) -> None:
        policy = SubmitPolicy(base_delay_seconds=0.01, jitter_pct=1000.0)
        rng = SeededRng(seed=0)
        for attempt in range(1, 6):
            assert policy.delay_for(attempt, rng) >= 0.0
