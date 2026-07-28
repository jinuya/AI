"""Chaos: shuffled and redelivered broker events (spec §4.3, §8.3).

The event stream is at-least-once and makes no ordering promise. This module
proves the OMS consumer copes with both failure modes at once — duplication
and reordering — driven through the real
:class:`~atrader.brokers.paper.PaperBroker` fault-injection surface
(:meth:`~atrader.brokers.paper.PaperBroker.shuffle_pending_events`) and
:class:`~atrader.brokers.faults.FaultyBroker`'s corrupted stream, rather than
hand-built event lists — the guarantee that matters is that the *real*
delivery path can misbehave this way and the consumer still ends up correct.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest

from atrader.audit.logger import AuditLogger
from atrader.brokers.faults import FaultInjector, FaultProfile, FaultyBroker
from atrader.brokers.paper import PaperBroker, PaperBrokerConfig
from atrader.core.clock import SimulatedClock
from atrader.core.ids import DeterministicIdGenerator
from atrader.core.models import Order
from atrader.core.rng import SeededRng
from atrader.core.types import OrderStatus, OrderType, Side
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


class TestShuffledPaperBrokerEvents:
    async def test_a_fill_delivered_before_its_own_accept_still_ends_up_correct(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        broker = PaperBroker(
            clock,
            config=PaperBrokerConfig(
                latency_ns=0, partial_fill_probability=0.0, queue_ahead_multiple=Decimal(0)
            ),
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
        report = await oms.submit(make_order(quantity=Decimal("10")))
        broker.advance_market("AAPL", Decimal("189.00"), Decimal("10"))  # fills fully

        # The queue now holds [ORDER_ACCEPTED, FILL]. Shuffle it — a real
        # multi-connection feed can and does deliver these out of order.
        broker.shuffle_pending_events()
        for event in broker.pending_events():
            oms.apply_event(event)

        final = storage.orders.get(report.order.order_id)
        assert final is not None
        assert final.status is OrderStatus.FILLED  # correct regardless of delivery order
        assert final.filled_quantity == Decimal("10")

    async def test_many_orders_shuffled_together_all_converge(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        broker = PaperBroker(
            clock,
            config=PaperBrokerConfig(
                latency_ns=0, partial_fill_probability=0.0, queue_ahead_multiple=Decimal(0)
            ),
            prices={"AAPL": Decimal("190.00"), "MSFT": Decimal("400.00")},
            rng=SeededRng(seed=2),
        )
        oms = OrderManager(
            broker=broker,
            orders=storage.orders,
            fills=storage.fills,
            clock=clock,
            ids=DeterministicIdGenerator(clock, seed=1),
            audit=audit,
        )
        reports = []
        for i in range(6):
            symbol = "AAPL" if i % 2 == 0 else "MSFT"
            price = Decimal("190.00") if symbol == "AAPL" else Decimal("400.00")
            reports.append(
                await oms.submit(
                    make_order(symbol=symbol, limit_price=price, quantity=Decimal("5"))
                )
            )
        broker.advance_market("AAPL", Decimal("189.00"), Decimal("100"))
        broker.advance_market("MSFT", Decimal("399.00"), Decimal("100"))

        broker.shuffle_pending_events()
        for event in broker.pending_events():
            oms.apply_event(event)

        for report in reports:
            final = storage.orders.get(report.order.order_id)
            assert final is not None
            assert final.status is OrderStatus.FILLED
            assert final.filled_quantity == Decimal("5")


class TestFaultyBrokerCorruptedStream:
    async def test_duplicated_fill_events_do_not_double_the_position(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        inner = PaperBroker(
            clock,
            config=PaperBrokerConfig(
                latency_ns=0, partial_fill_probability=0.0, queue_ahead_multiple=Decimal(0)
            ),
            prices={"AAPL": Decimal("190.00")},
        )
        faulty = FaultyBroker(
            inner=inner,
            injector=FaultInjector(profile=FaultProfile(duplicate_event_pct=Decimal(100))),
        )
        oms = OrderManager(
            broker=faulty,
            orders=storage.orders,
            fills=storage.fills,
            clock=clock,
            ids=DeterministicIdGenerator(clock, seed=1),
            audit=audit,
        )
        report = await oms.submit(make_order(quantity=Decimal("10")))
        inner.advance_market("AAPL", Decimal("189.00"), Decimal("10"))

        events = [e async for e in faulty.stream_updates()]  # every event now appears twice
        assert len(events) == 4  # ACCEPTED x2, FILL x2

        for event in events:
            oms.apply_event(event)

        final = storage.orders.get(report.order.order_id)
        assert final is not None
        assert final.filled_quantity == Decimal("10")  # not 20
        assert len(storage.fills.for_order(report.order.order_id)) == 1

    async def test_dropped_accept_event_does_not_block_a_later_fill(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        # If the ORDER_ACCEPTED event is lost entirely, the order is still
        # locally NEW (set synchronously by OrderManager.submit) — the FILL
        # event, when it does arrive, must still apply cleanly.
        inner = PaperBroker(
            clock,
            config=PaperBrokerConfig(
                latency_ns=0, partial_fill_probability=0.0, queue_ahead_multiple=Decimal(0)
            ),
            prices={"AAPL": Decimal("190.00")},
        )
        faulty = FaultyBroker(
            inner=inner,
            injector=FaultInjector(
                scripted=None,
                profile=FaultProfile(),
            ),
        )
        oms = OrderManager(
            broker=faulty,
            orders=storage.orders,
            fills=storage.fills,
            clock=clock,
            ids=DeterministicIdGenerator(clock, seed=1),
            audit=audit,
        )
        report = await oms.submit(make_order(quantity=Decimal("10")))
        inner.advance_market("AAPL", Decimal("189.00"), Decimal("10"))

        # Simulate the dropped ACCEPTED by only applying the FILL event.
        events = list(inner.pending_events())
        fill_only = [e for e in events if e.event_type.value == "fill"]
        for event in fill_only:
            oms.apply_event(event)

        final = storage.orders.get(report.order.order_id)
        assert final is not None
        assert final.status is OrderStatus.FILLED
        assert final.filled_quantity == Decimal("10")
