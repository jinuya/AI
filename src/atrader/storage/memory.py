"""In-memory repositories.

Used by unit tests, the backtest engine and paper trading. Behaviour matches the
SQL implementation exactly — including fill deduplication — so a test that
passes here is evidence about production, not just about the fake.
"""

from __future__ import annotations

from uuid import UUID

from atrader.audit.hashchain import AuditRecord
from atrader.core.models import Fill, Order, Position, TradingIntent

__all__ = [
    "InMemoryFillStore",
    "InMemoryIntentStore",
    "InMemoryOrderStore",
    "InMemoryPositionStore",
    "InMemoryStorage",
    "MemoryAuditSink",
]


class InMemoryOrderStore:
    __slots__ = ("_by_client_id", "_orders")

    def __init__(self) -> None:
        self._orders: dict[UUID, Order] = {}
        self._by_client_id: dict[str, UUID] = {}

    def upsert(self, order: Order) -> None:
        existing = self._by_client_id.get(order.client_order_id)
        if existing is not None and existing != order.order_id:
            # client_order_id is UNIQUE in the schema (spec §4.5); two orders
            # sharing one would defeat the idempotency guarantee entirely.
            raise ValueError(
                f"client_order_id {order.client_order_id!r} is already bound to order "
                f"{existing}; it cannot be reused for {order.order_id}"
            )
        self._orders[order.order_id] = order
        self._by_client_id[order.client_order_id] = order.order_id

    def get(self, order_id: UUID) -> Order | None:
        return self._orders.get(order_id)

    def get_by_client_order_id(self, client_order_id: str) -> Order | None:
        order_id = self._by_client_id.get(client_order_id)
        return None if order_id is None else self._orders.get(order_id)

    def open_orders(self) -> list[Order]:
        return [order for order in self._orders.values() if order.is_open]

    def all_orders(self) -> list[Order]:
        return list(self._orders.values())

    def __len__(self) -> int:
        return len(self._orders)


class InMemoryFillStore:
    __slots__ = ("_broker_ids", "_fills")

    def __init__(self) -> None:
        self._fills: dict[UUID, Fill] = {}
        self._broker_ids: set[str] = set()

    def append(self, fill: Fill) -> bool:
        if fill.fill_id in self._fills:
            return False
        if fill.broker_fill_id is not None:
            if fill.broker_fill_id in self._broker_ids:
                # The same execution redelivered. Expected on an at-least-once
                # bus; applying it again would double the position.
                return False
            self._broker_ids.add(fill.broker_fill_id)
        self._fills[fill.fill_id] = fill
        return True

    def get(self, fill_id: UUID) -> Fill | None:
        return self._fills.get(fill_id)

    def for_order(self, order_id: UUID) -> list[Fill]:
        return [fill for fill in self._fills.values() if fill.order_id == order_id]

    def all_fills(self) -> list[Fill]:
        return list(self._fills.values())

    def __len__(self) -> int:
        return len(self._fills)


class InMemoryIntentStore:
    __slots__ = ("_intents",)

    def __init__(self) -> None:
        self._intents: dict[UUID, TradingIntent] = {}

    def append(self, intent: TradingIntent) -> None:
        self._intents[intent.intent_id] = intent

    def get(self, intent_id: UUID) -> TradingIntent | None:
        return self._intents.get(intent_id)

    def all_intents(self) -> list[TradingIntent]:
        return list(self._intents.values())

    def __len__(self) -> int:
        return len(self._intents)


class InMemoryPositionStore:
    __slots__ = ("_positions",)

    def __init__(self) -> None:
        self._positions: dict[str, Position] = {}

    def upsert(self, position: Position) -> None:
        self._positions[position.symbol] = position

    def get(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    def all_positions(self) -> list[Position]:
        return list(self._positions.values())

    def __len__(self) -> int:
        return len(self._positions)


class MemoryAuditSink:
    """In-memory :class:`~atrader.audit.logger.AuditSink`."""

    __slots__ = ("_records",)

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []

    def append(self, record: AuditRecord) -> None:
        self._records.append(record)

    def read_all(self) -> list[AuditRecord]:
        return list(self._records)

    def last(self) -> AuditRecord | None:
        return self._records[-1] if self._records else None

    def __len__(self) -> int:
        return len(self._records)


class InMemoryStorage:
    """Bundle of in-memory repositories satisfying :class:`~atrader.storage.protocol.Storage`."""

    __slots__ = ("_audit", "_fills", "_intents", "_orders", "_positions")

    def __init__(self) -> None:
        self._orders = InMemoryOrderStore()
        self._fills = InMemoryFillStore()
        self._intents = InMemoryIntentStore()
        self._positions = InMemoryPositionStore()
        self._audit = MemoryAuditSink()

    @property
    def orders(self) -> InMemoryOrderStore:
        return self._orders

    @property
    def fills(self) -> InMemoryFillStore:
        return self._fills

    @property
    def intents(self) -> InMemoryIntentStore:
        return self._intents

    @property
    def positions(self) -> InMemoryPositionStore:
        return self._positions

    @property
    def audit(self) -> MemoryAuditSink:
        return self._audit

    def close(self) -> None:
        """No-op. Present so callers can treat this like the SQL backend."""
