"""Repository contract tests.

Every test runs against **both** the in-memory and the SQL implementation. A
fake that behaves differently from the real store is worse than no fake at all —
it makes green tests that mean nothing. Parametrising the fixture is what keeps
the two honest.

The SQL cases use file-backed SQLite, so they need no external infrastructure;
the same code path runs on PostgreSQL by changing the DSN.
"""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from atrader.core.models import Fill, Order, Position, TradingIntent
from atrader.core.types import OrderStatus, OrderType, Side, TargetType
from atrader.storage.memory import InMemoryStorage
from atrader.storage.sql.repositories import SqlStorage, create_storage


@pytest.fixture(params=["memory", "sqlite"])
def storage(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Any]:
    if request.param == "memory":
        store: Any = InMemoryStorage()
        yield store
    else:
        sql: SqlStorage = create_storage(f"sqlite+pysqlite:///{tmp_path / 'test.db'}")
        yield sql
        sql.close()


def make_order(**overrides: Any) -> Order:
    defaults: dict[str, Any] = {
        "order_id": uuid4(),
        "client_order_id": f"coid-{uuid4()}",
        "strategy_id": "test",
        "symbol": "AAPL",
        "side": Side.BUY,
        "order_type": OrderType.LIMIT,
        "quantity": Decimal("100"),
        "limit_price": Decimal("187.50"),
        "created_at_ns": 1_000,
        "updated_at_ns": 1_000,
    }
    return Order(**{**defaults, **overrides})


def make_fill(order: Order, **overrides: Any) -> Fill:
    defaults: dict[str, Any] = {
        "fill_id": uuid4(),
        "order_id": order.order_id,
        "symbol": order.symbol,
        "side": order.side,
        "quantity": Decimal("50"),
        "price": Decimal("187.50"),
        "commission": Decimal("0.10"),
        "executed_at_ns": 2_000,
    }
    return Fill(**{**defaults, **overrides})


class TestOrderStore:
    def test_round_trip(self, storage: Any) -> None:
        order = make_order()
        storage.orders.upsert(order)
        assert storage.orders.get(order.order_id) == order

    def test_decimal_precision_survives_the_round_trip(self, storage: Any) -> None:
        # The whole reason for the NUMERIC(20,8) / text-on-SQLite treatment.
        order = make_order(quantity=Decimal("0.00000001"), limit_price=Decimal("12345.67891234"))
        storage.orders.upsert(order)
        loaded = storage.orders.get(order.order_id)
        assert loaded is not None
        assert loaded.quantity == Decimal("0.00000001")
        assert loaded.limit_price == Decimal("12345.67891234")

    def test_lookup_by_client_order_id(self, storage: Any) -> None:
        # This is the first step of every safe retry (spec §FR-EXE-04).
        order = make_order(client_order_id="stable-id")
        storage.orders.upsert(order)
        assert storage.orders.get_by_client_order_id("stable-id") == order

    def test_missing_lookups_return_none(self, storage: Any) -> None:
        assert storage.orders.get(uuid4()) is None
        assert storage.orders.get_by_client_order_id("nope") is None

    def test_update_replaces_in_place(self, storage: Any) -> None:
        order = make_order()
        storage.orders.upsert(order)
        filled = order.model_copy(
            update={
                "status": OrderStatus.FILLED,
                "filled_quantity": order.quantity,
                "avg_fill_price": Decimal("187.49"),
            }
        )
        storage.orders.upsert(filled)
        stored = storage.orders.get(order.order_id)
        assert stored is not None
        assert stored.status is OrderStatus.FILLED
        assert len(storage.orders.all_orders()) == 1

    def test_client_order_id_cannot_be_reused_by_another_order(self, storage: Any) -> None:
        # Reuse would defeat the idempotency guarantee outright.
        storage.orders.upsert(make_order(client_order_id="shared"))
        with pytest.raises(ValueError, match="already"):
            storage.orders.upsert(make_order(client_order_id="shared"))

    def test_open_orders_excludes_terminal_states(self, storage: Any) -> None:
        open_order = make_order(status=OrderStatus.NEW)
        partial = make_order(status=OrderStatus.PARTIALLY_FILLED, filled_quantity=Decimal("10"))
        done = make_order(status=OrderStatus.FILLED, filled_quantity=Decimal("100"))
        cancelled = make_order(status=OrderStatus.CANCELED)
        for order in (open_order, partial, done, cancelled):
            storage.orders.upsert(order)

        open_ids = {o.order_id for o in storage.orders.open_orders()}
        assert open_ids == {open_order.order_id, partial.order_id}


class TestFillStore:
    def test_round_trip(self, storage: Any) -> None:
        order = make_order()
        fill = make_fill(order)
        assert storage.fills.append(fill) is True
        assert storage.fills.get(fill.fill_id) == fill

    def test_duplicate_broker_fill_id_is_rejected(self, storage: Any) -> None:
        # The bus is at-least-once (spec §4.3). The same execution WILL arrive
        # twice, and applying it twice doubles the position.
        order = make_order()
        first = make_fill(order, broker_fill_id="exec-1")
        redelivered = make_fill(order, broker_fill_id="exec-1")

        assert storage.fills.append(first) is True
        assert storage.fills.append(redelivered) is False
        assert len(storage.fills.all_fills()) == 1

    def test_duplicate_fill_id_is_rejected(self, storage: Any) -> None:
        order = make_order()
        fill = make_fill(order)
        assert storage.fills.append(fill) is True
        assert storage.fills.append(fill) is False

    def test_fills_without_a_broker_id_are_not_deduplicated(self, storage: Any) -> None:
        # Two genuine partial fills at the same price look identical apart from
        # fill_id; they must both count.
        order = make_order()
        assert storage.fills.append(make_fill(order)) is True
        assert storage.fills.append(make_fill(order)) is True
        assert len(storage.fills.for_order(order.order_id)) == 2

    def test_for_order_filters(self, storage: Any) -> None:
        first, second = make_order(), make_order()
        storage.fills.append(make_fill(first))
        storage.fills.append(make_fill(second))
        assert len(storage.fills.for_order(first.order_id)) == 1


class TestIntentStore:
    def test_round_trip(self, storage: Any) -> None:
        intent = TradingIntent(
            intent_id=uuid4(),
            strategy_id="momentum_v3",
            symbol="AAPL",
            side=Side.BUY,
            target_type=TargetType.SHARES,
            target_value=Decimal("100"),
            limit_price=Decimal("187.50"),
            confidence=Decimal("0.72"),
            rationale="20/50 SMA golden cross, volume 1.8x",
            created_at_ns=1_000,
        )
        storage.intents.append(intent)
        assert storage.intents.get(intent.intent_id) == intent


class TestPositionStore:
    def test_round_trip_and_upsert(self, storage: Any) -> None:
        position = Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("187.50"))
        storage.positions.upsert(position)
        assert storage.positions.get("AAPL") == position

        updated = position.model_copy(update={"quantity": Decimal("150")})
        storage.positions.upsert(updated)
        assert len(storage.positions.all_positions()) == 1
        stored = storage.positions.get("AAPL")
        assert stored is not None
        assert stored.quantity == Decimal("150")

    def test_short_position_keeps_its_sign(self, storage: Any) -> None:
        storage.positions.upsert(
            Position(symbol="AAPL", quantity=Decimal("-100"), avg_price=Decimal("187.50"))
        )
        stored = storage.positions.get("AAPL")
        assert stored is not None
        assert stored.quantity == Decimal("-100")
        assert stored.is_short


class TestDurability:
    def test_sqlite_data_survives_reopening(self, tmp_path: Path) -> None:
        # A crash between writing and reading must not lose an order — that is
        # the state reconciliation cannot recover from.
        dsn = f"sqlite+pysqlite:///{tmp_path / 'durable.db'}"
        order = make_order()

        first = create_storage(dsn)
        first.orders.upsert(order)
        first.close()

        second = create_storage(dsn)
        try:
            assert second.orders.get(order.order_id) == order
        finally:
            second.close()
