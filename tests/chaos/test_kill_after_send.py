"""Chaos: kill immediately after order send (acceptance criterion #4).

    "주문 전송 직후 kill" 필수 — spec §8.3.

The scenario: an order is on the wire — sent, not yet acknowledged — and the
process is killed (or the connection drops) at that exact instant. What must
never happen is a duplicate order at the venue when the process (or a fresh
one) comes back and, not knowing whether the original went through, decides
what to do next.

These tests drive that instant using :class:`~atrader.brokers.faults.FaultyBroker`
with :data:`~atrader.brokers.faults.Fault.AMBIGUOUS` — the fault that actually
lands the order at the inner broker and *then* raises, which is exactly what
"killed after send, before ack" looks like from the caller's perspective — and
assert that recovery goes through :mod:`atrader.execution.idempotency`'s
query-first rule rather than a blind resend.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest

from atrader.audit.logger import AuditLogger
from atrader.brokers.faults import Fault, FaultInjector, FaultyBroker, ScriptedFaults
from atrader.brokers.paper import PaperBroker, PaperBrokerConfig
from atrader.core.clock import SimulatedClock
from atrader.core.ids import DeterministicIdGenerator
from atrader.core.models import Order
from atrader.core.types import OrderStatus, OrderType, Side
from atrader.execution.deadman import DeadManSwitch
from atrader.execution.idempotency import SubmissionOutcome
from atrader.execution.oms import OrderManager
from atrader.storage.memory import InMemoryStorage

BASE_NS = 1_700_000_000_000_000_000


def make_order(**overrides: Any) -> Order:
    defaults: dict[str, Any] = {
        "order_id": uuid4(),
        "client_order_id": str(uuid4()),
        "strategy_id": "sma_crossover",
        "symbol": "AAPL",
        "side": Side.BUY,
        "order_type": OrderType.LIMIT,
        "quantity": Decimal("100"),
        "limit_price": Decimal("190.00"),
        "status": OrderStatus.PENDING_NEW,
        "risk_check_id": uuid4(),
        "created_at_ns": BASE_NS,
        "updated_at_ns": BASE_NS,
    }
    return Order(**{**defaults, **overrides})


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(start_ns=BASE_NS)


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


@pytest.fixture
def audit(storage: InMemoryStorage, clock: SimulatedClock) -> AuditLogger:
    return AuditLogger(storage.audit, clock)


def make_faulty_paper(clock: SimulatedClock, *faults: str) -> FaultyBroker:
    inner = PaperBroker(
        clock,
        config=PaperBrokerConfig(
            latency_ns=0, partial_fill_probability=0.0, reject_probability=0.0
        ),
        prices={"AAPL": Decimal("190.00")},
    )
    return FaultyBroker(inner=inner, injector=FaultInjector(scripted=ScriptedFaults.of(*faults)))


class TestSingleOrderKilledAfterSend:
    """The order was sent; the response never came back. What happens next
    must never be "send it again and hope"."""

    async def test_the_order_lands_exactly_once_at_the_broker(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        broker = make_faulty_paper(clock, Fault.AMBIGUOUS)
        oms = OrderManager(
            broker=broker,
            orders=storage.orders,
            fills=storage.fills,
            clock=clock,
            ids=DeterministicIdGenerator(clock, seed=1),
            audit=audit,
        )
        order = make_order()

        report = await oms.submit(order)

        # The query (the fault only applies to the submit call, not the
        # follow-up query) found the order — it resolved as ALREADY_EXISTS,
        # not as a fresh, second submission.
        assert report.result.outcome is SubmissionOutcome.ALREADY_EXISTS
        assert report.is_live
        # Ask the underlying broker directly: exactly one order, not two.
        open_orders = await broker.inner.get_open_orders()
        assert len(open_orders) == 1
        assert broker.injector.faults_fired == {Fault.AMBIGUOUS: 1}

    async def test_when_even_the_query_is_unreachable_the_symbol_locks_instead_of_guessing(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        # Kill was so total that the recovery query fails too — this is the
        # one case where the system is honestly not sure. It must lock and
        # call a human (spec §6.4), never resend blind.
        broker = make_faulty_paper(clock, Fault.AMBIGUOUS, Fault.TRANSIENT)
        oms = OrderManager(
            broker=broker,
            orders=storage.orders,
            fills=storage.fills,
            clock=clock,
            ids=DeterministicIdGenerator(clock, seed=1),
            audit=audit,
        )
        order = make_order()

        report = await oms.submit(order)

        assert report.result.outcome is SubmissionOutcome.AMBIGUOUS
        assert report.result.needs_human
        assert oms.locked_symbols == frozenset({"AAPL"})
        # Critically: exactly one submission reached the broker. A second,
        # blind resend would double the position; the lock is what prevents it.
        open_orders = await broker.inner.get_open_orders()
        assert len(open_orders) == 1

        # And the lock actually holds: a second attempt at the same symbol
        # never reaches the broker at all.
        second_report = await oms.submit(make_order(symbol="AAPL"))
        assert not second_report.is_live
        assert "locked" in second_report.result.reason
        open_orders_after = await broker.inner.get_open_orders()
        assert len(open_orders_after) == 1  # still just the one


class TestDeadManSwitchDuringTheSameOutage:
    """The kill also means no heartbeat — the dead-man switch is the backstop
    that cancels whatever is left working once the timeout elapses."""

    async def test_orders_submitted_before_the_kill_are_swept_by_the_deadman(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        # A clean broker this time: the point here is the watchdog, not the
        # submission ambiguity, so keep those two concerns separate.
        broker = PaperBroker(
            clock,
            config=PaperBrokerConfig(latency_ns=0, partial_fill_probability=0.0),
            prices={"AAPL": Decimal("190.00")},
        )
        oms = OrderManager(
            broker=broker,
            orders=storage.orders,
            fills=storage.fills,
            clock=clock,
            ids=DeterministicIdGenerator(clock, seed=1),
            audit=audit,
        )
        switch = DeadManSwitch(
            cancel_all=oms.cancel_all, clock=clock, audit=audit, timeout_seconds=60
        )
        switch.beat()

        await oms.submit(make_order(symbol="AAPL"))
        await oms.submit(make_order(symbol="MSFT", limit_price=Decimal("400")))
        assert len(storage.orders.open_orders()) == 2

        # The process is "killed" here: no more heartbeats arrive.
        clock.advance_seconds(61)
        report = await switch.check()

        assert report is not None
        assert len(report.canceled_order_ids) == 2
        assert storage.orders.open_orders() == []

    async def test_a_heartbeat_that_keeps_arriving_prevents_the_sweep(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        broker = PaperBroker(
            clock, config=PaperBrokerConfig(latency_ns=0), prices={"AAPL": Decimal("190.00")}
        )
        oms = OrderManager(
            broker=broker,
            orders=storage.orders,
            fills=storage.fills,
            clock=clock,
            ids=DeterministicIdGenerator(clock, seed=1),
            audit=audit,
        )
        switch = DeadManSwitch(
            cancel_all=oms.cancel_all, clock=clock, audit=audit, timeout_seconds=60
        )
        switch.beat()
        await oms.submit(make_order())

        for _ in range(5):
            clock.advance_seconds(30)
            switch.beat()  # the process is alive and well between kills
            assert await switch.check() is None

        assert len(storage.orders.open_orders()) == 1


class TestAmbiguousSubmitThenDeadmanSweep:
    """The combined scenario: kill right after send, recovery locks the
    symbol, and — because a locked symbol is not the same as a resolved one —
    the dead-man switch is still what actually gets the order off the book if
    nobody clears the lock in time."""

    async def test_a_locked_order_is_still_swept_by_the_deadman(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        broker = make_faulty_paper(clock, Fault.AMBIGUOUS)
        oms = OrderManager(
            broker=broker,
            orders=storage.orders,
            fills=storage.fills,
            clock=clock,
            ids=DeterministicIdGenerator(clock, seed=1),
            audit=audit,
        )
        switch = DeadManSwitch(
            cancel_all=oms.cancel_all, clock=clock, audit=audit, timeout_seconds=60
        )
        switch.beat()

        report = await oms.submit(make_order())
        assert report.result.outcome is SubmissionOutcome.ALREADY_EXISTS  # resolved, order is live

        clock.advance_seconds(61)
        trigger = await switch.check()
        assert trigger is not None
        assert len(trigger.canceled_order_ids) == 1
        assert storage.orders.open_orders() == []
