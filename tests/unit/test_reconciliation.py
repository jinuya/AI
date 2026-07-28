"""Reconciliation — spec §FR-EXE-05.

브로커의 포지션·미체결주문·현금 잔고를 로컬 상태와 대조한다.
불일치 발견 시 즉시 신규 주문을 중단하고, 자동 수정은 금지한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest

from atrader.audit.logger import AuditEvent, AuditLogger
from atrader.brokers.models import OrderState
from atrader.core.clock import NS_PER_SECOND, SimulatedClock
from atrader.core.models import AccountState, Order, Position
from atrader.core.types import OrderStatus, OrderType, Side
from atrader.execution.reconciliation import (
    Break,
    BreakKind,
    Reconciler,
    expected_positions_from_fills,
)
from atrader.storage.memory import InMemoryStorage

BASE_NS = 1_700_000_000 * NS_PER_SECOND


@dataclass
class FakeBroker:
    """Reports whatever a test configures — the broker side of a comparison."""

    positions: list[Position] = field(default_factory=list)
    open_orders: list[OrderState] = field(default_factory=list)
    account: AccountState = field(default_factory=AccountState)

    async def get_positions(self) -> list[Position]:
        return self.positions

    async def get_open_orders(self) -> list[OrderState]:
        return self.open_orders

    async def get_account(self) -> AccountState:
        return self.account


def make_position(**overrides: Any) -> Position:
    defaults: dict[str, Any] = {
        "symbol": "AAPL",
        "quantity": Decimal("100"),
        "avg_price": Decimal("190"),
    }
    return Position(**{**defaults, **overrides})


def make_local_order(**overrides: Any) -> Order:
    defaults: dict[str, Any] = {
        "order_id": uuid4(),
        "client_order_id": "coid-1",
        "strategy_id": "test",
        "symbol": "AAPL",
        "side": Side.BUY,
        "order_type": OrderType.LIMIT,
        "quantity": Decimal("100"),
        "limit_price": Decimal("190"),
        "status": OrderStatus.NEW,
        "created_at_ns": BASE_NS,
        "updated_at_ns": BASE_NS,
    }
    return Order(**{**defaults, **overrides})


def make_broker_order_state(**overrides: Any) -> OrderState:
    defaults: dict[str, Any] = {
        "client_order_id": "coid-1",
        "broker_order_id": "b-1",
        "symbol": "AAPL",
        "side": Side.BUY,
        "status": OrderStatus.NEW,
        "quantity": Decimal("100"),
    }
    return OrderState(**{**defaults, **overrides})


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(start_ns=BASE_NS)


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


@pytest.fixture
def audit(storage: InMemoryStorage, clock: SimulatedClock) -> AuditLogger:
    return AuditLogger(storage.audit, clock)


def make_reconciler(
    broker: FakeBroker,
    storage: InMemoryStorage,
    clock: SimulatedClock,
    audit: AuditLogger,
    **overrides: Any,
) -> Reconciler:
    defaults: dict[str, Any] = {
        "broker": broker,
        "orders": storage.orders,
        "positions": storage.positions,
        "clock": clock,
        "audit": audit,
    }
    return Reconciler(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestCleanReconciliation:
    async def test_matching_state_is_clean(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        position = make_position()
        storage.positions.upsert(position)
        broker = FakeBroker(positions=[position])
        reconciler = make_reconciler(broker, storage, clock, audit)

        report = await reconciler.reconcile()

        assert report.is_clean
        assert not reconciler.has_active_break
        assert "clean" in report.summary()

    async def test_empty_state_on_both_sides_is_clean(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        broker = FakeBroker()
        reconciler = make_reconciler(broker, storage, clock, audit)
        report = await reconciler.reconcile()
        assert report.is_clean

    async def test_flat_local_positions_are_excluded_from_the_comparison(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        storage.positions.upsert(make_position(quantity=Decimal(0)))
        broker = FakeBroker()
        reconciler = make_reconciler(broker, storage, clock, audit)
        report = await reconciler.reconcile()
        assert report.is_clean


class TestPositionBreaks:
    async def test_position_missing_locally(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        broker = FakeBroker(positions=[make_position()])
        reconciler = make_reconciler(broker, storage, clock, audit)
        report = await reconciler.reconcile()
        assert not report.is_clean
        assert report.breaks[0].kind == BreakKind.POSITION_MISSING_LOCALLY
        assert reconciler.has_active_break

    async def test_position_missing_at_broker(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        storage.positions.upsert(make_position())
        broker = FakeBroker()
        reconciler = make_reconciler(broker, storage, clock, audit)
        report = await reconciler.reconcile()
        assert report.breaks[0].kind == BreakKind.POSITION_MISSING_AT_BROKER

    async def test_quantity_mismatch_within_tolerance_is_clean(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        storage.positions.upsert(make_position(quantity=Decimal("100")))
        broker = FakeBroker(positions=[make_position(quantity=Decimal("100.4"))])
        reconciler = make_reconciler(
            broker, storage, clock, audit, quantity_tolerance=Decimal("0.5")
        )
        report = await reconciler.reconcile()
        assert report.is_clean

    async def test_quantity_mismatch_beyond_tolerance_breaks(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        storage.positions.upsert(make_position(quantity=Decimal("100")))
        broker = FakeBroker(positions=[make_position(quantity=Decimal("150"))])
        reconciler = make_reconciler(broker, storage, clock, audit)
        report = await reconciler.reconcile()
        assert report.breaks[0].kind == BreakKind.POSITION_QUANTITY
        assert "differ by 50" in report.breaks[0].message


class TestOrderBreaks:
    async def test_order_missing_locally_looks_like_a_duplicate_submission(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        broker = FakeBroker(open_orders=[make_broker_order_state()])
        reconciler = make_reconciler(broker, storage, clock, audit)
        report = await reconciler.reconcile()
        assert report.breaks[0].kind == BreakKind.ORDER_MISSING_LOCALLY
        assert "duplicate submission" in report.breaks[0].message

    async def test_order_missing_at_broker(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        storage.orders.upsert(make_local_order())
        broker = FakeBroker()
        reconciler = make_reconciler(broker, storage, clock, audit)
        report = await reconciler.reconcile()
        assert report.breaks[0].kind == BreakKind.ORDER_MISSING_AT_BROKER

    async def test_filled_quantity_mismatch(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        storage.orders.upsert(make_local_order(filled_quantity=Decimal("10")))
        broker = FakeBroker(open_orders=[make_broker_order_state(filled_quantity=Decimal("40"))])
        reconciler = make_reconciler(broker, storage, clock, audit)
        report = await reconciler.reconcile()
        assert report.breaks[0].kind == BreakKind.ORDER_QUANTITY

    async def test_matching_open_orders_are_clean(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        storage.orders.upsert(make_local_order(filled_quantity=Decimal("10")))
        broker = FakeBroker(open_orders=[make_broker_order_state(filled_quantity=Decimal("10"))])
        reconciler = make_reconciler(broker, storage, clock, audit)
        report = await reconciler.reconcile()
        assert report.is_clean


class TestCashComparison:
    async def test_no_local_account_view_skips_the_cash_check(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        broker = FakeBroker(account=AccountState(cash=Decimal("50000")))
        reconciler = make_reconciler(broker, storage, clock, audit)  # local_account unset
        report = await reconciler.reconcile()
        assert report.is_clean

    async def test_matching_cash_is_clean(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        broker = FakeBroker(account=AccountState(cash=Decimal("50000.00"), currency="USD"))
        reconciler = make_reconciler(
            broker,
            storage,
            clock,
            audit,
            local_account=lambda: AccountState(cash=Decimal("50000.005"), currency="USD"),
        )
        report = await reconciler.reconcile()
        assert report.is_clean  # within the default 0.01 tolerance

    async def test_cash_mismatch_beyond_tolerance_breaks(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        broker = FakeBroker(account=AccountState(cash=Decimal("50000"), currency="USD"))
        reconciler = make_reconciler(
            broker,
            storage,
            clock,
            audit,
            local_account=lambda: AccountState(cash=Decimal("49000"), currency="USD"),
        )
        report = await reconciler.reconcile()
        assert report.breaks[0].kind == BreakKind.CASH
        assert "1000" in report.breaks[0].message

    async def test_currency_mismatch_breaks_instead_of_comparing_numbers(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        broker = FakeBroker(account=AccountState(cash=Decimal("50000"), currency="USD"))
        reconciler = make_reconciler(
            broker,
            storage,
            clock,
            audit,
            local_account=lambda: AccountState(cash=Decimal("50000"), currency="KRW"),
        )
        report = await reconciler.reconcile()
        assert report.breaks[0].kind == BreakKind.CURRENCY

    async def test_local_account_returning_none_skips_the_check(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        broker = FakeBroker(account=AccountState(cash=Decimal("50000")))
        reconciler = make_reconciler(broker, storage, clock, audit, local_account=lambda: None)
        report = await reconciler.reconcile()
        assert report.is_clean


class TestLatchedBreak:
    async def test_break_persists_across_a_subsequent_clean_run(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        broker = FakeBroker(positions=[make_position()])
        reconciler = make_reconciler(broker, storage, clock, audit)
        await reconciler.reconcile()
        assert reconciler.has_active_break

        # The disagreement is "fixed" by mutating the broker double directly —
        # a real fix would come from a human, but the assertion here is only
        # that a subsequent clean pass does NOT auto-clear the latch.
        broker.positions = []
        report = await reconciler.reconcile()
        assert report.is_clean
        assert reconciler.has_active_break  # still latched — only clear_break lifts it

    async def test_clear_break_requires_a_named_actor_and_is_audited(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        broker = FakeBroker(positions=[make_position()])
        reconciler = make_reconciler(broker, storage, clock, audit)
        await reconciler.reconcile()

        reconciler.clear_break(by="ops-oncall", note="confirmed broker was right, fixed the bug")

        assert not reconciler.has_active_break
        events = [r.event_type for r in storage.audit.read_all()]
        assert AuditEvent.MANUAL_INTERVENTION in events


class TestIsDue:
    def test_always_due_on_first_call(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        reconciler = make_reconciler(FakeBroker(), storage, clock, audit)
        assert reconciler.is_due()

    async def test_not_due_immediately_after_a_run(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        reconciler = make_reconciler(FakeBroker(), storage, clock, audit, interval_seconds=30)
        await reconciler.reconcile()
        assert not reconciler.is_due()

    async def test_due_again_after_the_interval_elapses(
        self, storage: InMemoryStorage, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        reconciler = make_reconciler(FakeBroker(), storage, clock, audit, interval_seconds=30)
        await reconciler.reconcile()
        clock.advance_seconds(31)
        assert reconciler.is_due()


class TestReportSummary:
    def test_break_summary_lists_every_break(self) -> None:
        from atrader.core.types import AlertLevel
        from atrader.execution.reconciliation import ReconciliationReport

        report = ReconciliationReport(
            at_ns=BASE_NS,
            breaks=(
                Break(BreakKind.CASH, "account", "100", "90", "diff"),
                Break(BreakKind.POSITION_QUANTITY, "AAPL", "10", "5"),
            ),
        )
        assert not report.is_clean
        assert report.alert_level is AlertLevel.CRITICAL
        summary = report.summary()
        assert "2 break(s)" in summary
        assert "cash on account" in summary


class TestExpectedPositionsFromFills:
    def test_nets_fills_by_symbol(self) -> None:
        fills = [("AAPL", Decimal("10")), ("AAPL", Decimal("-4")), ("MSFT", Decimal("5"))]
        assert expected_positions_from_fills(fills) == {
            "AAPL": Decimal("6"),
            "MSFT": Decimal("5"),
        }

    def test_symbols_that_net_to_zero_are_dropped(self) -> None:
        fills = [("AAPL", Decimal("10")), ("AAPL", Decimal("-10"))]
        assert expected_positions_from_fills(fills) == {}
