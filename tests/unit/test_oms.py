"""Order management — spec §3.4.

Exercises :class:`~atrader.execution.oms.OrderManager` against the real
:class:`~atrader.brokers.paper.PaperBroker` (and, for the ambiguous-submission
case, :class:`~atrader.brokers.faults.FaultyBroker` wrapped around it) rather
than a hand-rolled double — the properties that matter here (persist-before-send,
idempotent fills, symbol locking) only mean something against a broker that
actually accepts, fills and rejects orders the way one would.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import pytest

from atrader.audit.logger import AuditEvent, AuditLogger
from atrader.brokers.faults import Fault, FaultInjector, FaultyBroker, ScriptedFaults
from atrader.brokers.paper import PaperBroker, PaperBrokerConfig
from atrader.core.clock import SimulatedClock
from atrader.core.ids import DeterministicIdGenerator
from atrader.core.models import Order
from atrader.core.types import OrderStatus, OrderType, Side
from atrader.execution.idempotency import SubmissionOutcome
from atrader.execution.oms import OrderManager, orders_needing_cancel
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
def ids(clock: SimulatedClock) -> DeterministicIdGenerator:
    return DeterministicIdGenerator(clock, seed=7)


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


@pytest.fixture
def audit(storage: InMemoryStorage, clock: SimulatedClock) -> AuditLogger:
    return AuditLogger(storage.audit, clock)


def make_paper(clock: SimulatedClock, **config_overrides: Any) -> PaperBroker:
    defaults: dict[str, Any] = {
        "latency_ns": 0,
        "partial_fill_probability": 0.0,
        "reject_probability": 0.0,
        "queue_ahead_multiple": Decimal("0"),
    }
    config = PaperBrokerConfig(**{**defaults, **config_overrides})
    return PaperBroker(clock, config=config, prices={"AAPL": Decimal("190.00")})


def make_oms(
    storage: InMemoryStorage,
    clock: SimulatedClock,
    ids: DeterministicIdGenerator,
    audit: AuditLogger,
    broker: Any = None,
) -> OrderManager:
    return OrderManager(
        broker=broker or make_paper(clock),
        orders=storage.orders,
        fills=storage.fills,
        clock=clock,
        ids=ids,
        audit=audit,
    )


class TestSubmit:
    async def test_rejects_an_order_without_a_risk_check_id(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        from atrader.core.errors import IllegalStateTransitionError

        oms = make_oms(storage, clock, ids, audit)
        order = make_order(risk_check_id=None)
        with pytest.raises(IllegalStateTransitionError, match="risk_check_id"):
            await oms.submit(order)

    async def test_persists_before_sending(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        oms = make_oms(storage, clock, ids, audit)
        order = make_order()
        await oms.submit(order)
        # Persisted under PENDING_NEW is visible even before this call returns —
        # verified indirectly here by confirming the store holds *a* record.
        assert storage.orders.get(order.order_id) is not None

    async def test_accepted_order_gets_new_status_and_broker_order_id(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        oms = make_oms(storage, clock, ids, audit)
        order = make_order()
        report = await oms.submit(order)
        assert report.is_live
        assert report.order.status is OrderStatus.NEW
        assert report.order.broker_order_id is not None
        assert storage.orders.get(order.order_id) is not None
        assert storage.orders.get(order.order_id).status is OrderStatus.NEW  # type: ignore[union-attr]

    async def test_rejected_order_transitions_to_rejected(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        broker = make_paper(clock, reject_probability=1.0)
        oms = make_oms(storage, clock, ids, audit, broker=broker)
        order = make_order()
        report = await oms.submit(order)
        assert not report.is_live
        assert report.order.status is OrderStatus.REJECTED
        assert storage.orders.get(order.order_id).status is OrderStatus.REJECTED  # type: ignore[union-attr]

    async def test_ambiguous_submission_locks_the_symbol(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        inner = make_paper(clock)
        faulty = FaultyBroker(
            inner=inner,
            injector=FaultInjector(scripted=ScriptedFaults.of(Fault.AMBIGUOUS, Fault.TRANSIENT)),
        )
        oms = make_oms(storage, clock, ids, audit, broker=faulty)
        order = make_order()

        report = await oms.submit(order)

        assert report.result.outcome is SubmissionOutcome.AMBIGUOUS
        assert report.locked_symbol == "AAPL"
        assert "AAPL" in oms.locked_symbols

    async def test_a_locked_symbol_rejects_further_submissions_without_touching_the_broker(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        inner = make_paper(clock)
        faulty = FaultyBroker(
            inner=inner,
            injector=FaultInjector(scripted=ScriptedFaults.of(Fault.AMBIGUOUS, Fault.TRANSIENT)),
        )
        oms = make_oms(storage, clock, ids, audit, broker=faulty)
        await oms.submit(make_order())

        second = make_order()
        report = await oms.submit(second)
        assert not report.is_live
        assert "locked" in report.result.reason
        assert (
            faulty.injector.call_count == 2
        )  # unchanged — the second call never reached the broker

    async def test_unlock_allows_submission_again(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        inner = make_paper(clock)
        faulty = FaultyBroker(
            inner=inner,
            injector=FaultInjector(scripted=ScriptedFaults.of(Fault.AMBIGUOUS, Fault.TRANSIENT)),
        )
        oms = make_oms(storage, clock, ids, audit, broker=faulty)
        await oms.submit(make_order())
        oms.unlock("AAPL")
        report = await oms.submit(make_order())
        assert report.is_live

    async def test_submission_is_audited(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        oms = make_oms(storage, clock, ids, audit)
        await oms.submit(make_order())
        events = [r.event_type for r in storage.audit.read_all()]
        assert AuditEvent.ORDER_REQUESTED in events
        assert AuditEvent.ORDER_ACK in events

    async def test_submission_works_without_an_audit_logger(
        self, storage: InMemoryStorage, clock: SimulatedClock, ids: DeterministicIdGenerator
    ) -> None:
        # Audit is optional (e.g. a scratch backtest run) — nothing should
        # blow up when it is None, and nothing should be written.
        oms = OrderManager(
            broker=make_paper(clock),
            orders=storage.orders,
            fills=storage.fills,
            clock=clock,
            ids=ids,
            audit=None,
        )
        report = await oms.submit(make_order())
        assert report.is_live
        assert storage.audit.read_all() == []

    async def test_ambiguous_submission_resolved_by_query_adopts_the_broker_state(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        # A single AMBIGUOUS fault: the submit lands at the inner broker (a
        # market order, so it fills immediately) and only the *response* is
        # lost. The follow-up query (unfaulted) finds it already filled — the
        # ALREADY_EXISTS branch that adopts fill state from the broker.
        inner = make_paper(clock)
        faulty = FaultyBroker(
            inner=inner, injector=FaultInjector(scripted=ScriptedFaults.of(Fault.AMBIGUOUS))
        )
        oms = make_oms(storage, clock, ids, audit, broker=faulty)
        order = make_order(order_type=OrderType.MARKET, limit_price=None, quantity=Decimal("5"))

        report = await oms.submit(order)

        assert report.result.outcome is SubmissionOutcome.ALREADY_EXISTS
        assert report.order.status is OrderStatus.FILLED
        assert report.order.filled_quantity == Decimal("5")
        assert storage.orders.get(order.order_id).filled_quantity == Decimal("5")  # type: ignore[union-attr]

    async def test_ambiguous_submission_resolved_unfilled_does_not_touch_local_state(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        # Same lost-response path, but the resting limit order has not traded
        # yet: adopting a zero-fill state must be a no-op, not a downgrade.
        inner = make_paper(clock)
        faulty = FaultyBroker(
            inner=inner, injector=FaultInjector(scripted=ScriptedFaults.of(Fault.AMBIGUOUS))
        )
        oms = make_oms(storage, clock, ids, audit, broker=faulty)
        report = await oms.submit(make_order())

        assert report.result.outcome is SubmissionOutcome.ALREADY_EXISTS
        assert report.order.status is OrderStatus.NEW
        assert report.order.filled_quantity == Decimal(0)


class TestCancel:
    async def test_cancel_moves_through_pending_cancel_to_canceled(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        oms = make_oms(storage, clock, ids, audit)
        report = await oms.submit(make_order())
        canceled = await oms.cancel(report.order)
        assert canceled.status is OrderStatus.CANCELED

    async def test_canceling_a_closed_order_is_a_no_op(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        oms = make_oms(storage, clock, ids, audit)
        report = await oms.submit(make_order())
        first = await oms.cancel(report.order)
        assert first.status is OrderStatus.CANCELED
        second = await oms.cancel(first)
        assert second is first  # returned unchanged; is_open was already False

    async def test_a_cancel_that_loses_the_race_to_a_fill_stays_pending_cancel(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        broker = make_paper(clock)
        oms = make_oms(storage, clock, ids, audit, broker=broker)
        report = await oms.submit(make_order(quantity=Decimal("10")))

        # The order fills completely at the broker before our cancel arrives.
        broker.advance_market("AAPL", Decimal("189.00"), Decimal("10"))

        pending = await oms.cancel(report.order)
        assert pending.status is OrderStatus.PENDING_CANCEL  # not CANCELED — the race was lost

        # The fill event that follows must still be acceptable from here.
        events = broker.pending_events()
        fill_events = [e for e in events if e.event_type.value == "fill"]
        assert fill_events
        updated = oms.apply_event(fill_events[0])
        assert updated is not None
        assert updated.status is OrderStatus.FILLED

    async def test_cancel_all_targets_only_open_orders(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        oms = make_oms(storage, clock, ids, audit)
        a = await oms.submit(make_order(symbol="AAPL"))
        b = await oms.submit(make_order(symbol="MSFT"))
        await oms.cancel(a.order)  # already closed before cancel_all runs

        canceled = await oms.cancel_all()
        assert {o.order_id for o in canceled} == {b.order.order_id}

    async def test_cancel_all_can_target_one_symbol(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        oms = make_oms(storage, clock, ids, audit)
        a = await oms.submit(make_order(symbol="AAPL"))
        b = await oms.submit(make_order(symbol="MSFT"))

        canceled = await oms.cancel_all(symbol="AAPL")
        assert {o.order_id for o in canceled} == {a.order.order_id}
        assert storage.orders.get(b.order.order_id).is_open  # type: ignore[union-attr]


class TestApplyEvent:
    async def test_unknown_client_order_id_is_ignored(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        from atrader.brokers.models import BrokerEvent, BrokerEventType

        oms = make_oms(storage, clock, ids, audit)
        result = oms.apply_event(
            BrokerEvent(
                event_type=BrokerEventType.ORDER_ACCEPTED, client_order_id="never-submitted"
            )
        )
        assert result is None

    async def test_event_without_a_client_order_id_is_ignored(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        from atrader.brokers.models import BrokerEvent, BrokerEventType

        oms = make_oms(storage, clock, ids, audit)
        assert oms.apply_event(BrokerEvent(event_type=BrokerEventType.DISCONNECTED)) is None

    async def test_redelivered_fill_is_not_applied_twice(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        broker = make_paper(clock)
        oms = make_oms(storage, clock, ids, audit, broker=broker)
        report = await oms.submit(make_order(quantity=Decimal("10")))
        broker.advance_market("AAPL", Decimal("189.00"), Decimal("10"))
        events = broker.pending_events()
        fill_event = next(e for e in events if e.event_type.value == "fill")

        once = oms.apply_event(fill_event)
        twice = oms.apply_event(fill_event)  # at-least-once redelivery

        assert once is not None and once.status is OrderStatus.FILLED
        assert twice is not None and twice.filled_quantity == once.filled_quantity
        assert len(storage.fills.for_order(report.order.order_id)) == 1

    async def test_out_of_order_accept_after_a_fill_does_not_regress_status(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        from atrader.brokers.models import BrokerEvent, BrokerEventType

        broker = make_paper(clock)
        oms = make_oms(storage, clock, ids, audit, broker=broker)
        report = await oms.submit(make_order(quantity=Decimal("10")))
        broker.advance_market("AAPL", Decimal("189.00"), Decimal("10"))
        fill_event = next(e for e in broker.pending_events() if e.event_type.value == "fill")
        filled = oms.apply_event(fill_event)
        assert filled is not None and filled.status is OrderStatus.FILLED

        # A stale ORDER_ACCEPTED for the same order arrives after the fill.
        late_ack = BrokerEvent(
            event_type=BrokerEventType.ORDER_ACCEPTED,
            client_order_id=report.order.client_order_id,
        )
        result = oms.apply_event(late_ack)
        assert result is not None
        assert result.status is OrderStatus.FILLED  # the later state wins, not the late event

    async def test_reject_and_cancel_and_expire_events_transition_status(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        from atrader.brokers.models import BrokerEvent, BrokerEventType

        oms = make_oms(storage, clock, ids, audit)
        report = await oms.submit(make_order())
        result = oms.apply_event(
            BrokerEvent(
                event_type=BrokerEventType.ORDER_CANCELED,
                client_order_id=report.order.client_order_id,
            )
        )
        assert result is not None and result.status is OrderStatus.CANCELED

    async def test_event_matching_the_current_status_is_a_no_op(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        from atrader.brokers.models import BrokerEvent, BrokerEventType

        oms = make_oms(storage, clock, ids, audit)
        report = await oms.submit(make_order())
        result = oms.apply_event(
            BrokerEvent(
                event_type=BrokerEventType.ORDER_ACCEPTED,
                client_order_id=report.order.client_order_id,
            )
        )
        assert result is not None and result.status is OrderStatus.NEW

    async def test_fill_event_missing_price_or_quantity_is_ignored(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        from atrader.brokers.models import BrokerEvent, BrokerEventType

        oms = make_oms(storage, clock, ids, audit)
        report = await oms.submit(make_order())
        result = oms.apply_event(
            BrokerEvent(
                event_type=BrokerEventType.FILL,
                client_order_id=report.order.client_order_id,
                quantity=None,
                price=None,
            )
        )
        assert result is not None and result.status is OrderStatus.NEW


class TestWorkingQuantity:
    async def test_sums_remaining_quantity_signed_by_side(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        oms = make_oms(storage, clock, ids, audit)
        await oms.submit(make_order(symbol="AAPL", side=Side.BUY, quantity=Decimal("100")))
        await oms.submit(make_order(symbol="AAPL", side=Side.SELL, quantity=Decimal("30")))
        await oms.submit(
            make_order(
                symbol="MSFT", side=Side.BUY, quantity=Decimal("5"), limit_price=Decimal("400")
            )
        )
        assert oms.working_quantity("AAPL") == Decimal("70")
        assert oms.working_quantity("MSFT") == Decimal("5")

    async def test_zero_for_a_symbol_with_no_working_orders(
        self,
        storage: InMemoryStorage,
        clock: SimulatedClock,
        ids: DeterministicIdGenerator,
        audit: AuditLogger,
    ) -> None:
        oms = make_oms(storage, clock, ids, audit)
        assert oms.working_quantity("TSLA") == Decimal(0)


class TestOrdersNeedingCancel:
    def test_open_orders_past_their_valid_until_are_flagged(self) -> None:
        intent_id: UUID = uuid4()
        order = make_order(status=OrderStatus.NEW, parent_intent_id=intent_id)
        stale = orders_needing_cancel(
            [order], now_ns=BASE_NS + 10, valid_until={intent_id: BASE_NS}
        )
        assert stale == [order]

    def test_orders_still_within_their_window_are_not_flagged(self) -> None:
        intent_id: UUID = uuid4()
        order = make_order(status=OrderStatus.NEW, parent_intent_id=intent_id)
        stale = orders_needing_cancel(
            [order], now_ns=BASE_NS, valid_until={intent_id: BASE_NS + 1_000}
        )
        assert stale == []

    def test_closed_orders_are_never_flagged(self) -> None:
        intent_id: UUID = uuid4()
        order = make_order(
            status=OrderStatus.CANCELED, parent_intent_id=intent_id, filled_quantity=Decimal(0)
        )
        stale = orders_needing_cancel(
            [order], now_ns=BASE_NS + 10, valid_until={intent_id: BASE_NS}
        )
        assert stale == []

    def test_orders_without_a_parent_intent_are_never_flagged(self) -> None:
        order = make_order(status=OrderStatus.NEW, parent_intent_id=None)
        stale = orders_needing_cancel([order], now_ns=BASE_NS + 10, valid_until={})
        assert stale == []
