"""Repository interfaces.

Everything above this layer talks to Protocols, so the same code runs against
in-memory stores in tests and PostgreSQL in production (spec §4.4).

Two contracts here carry real weight:

* :meth:`FillStore.append` returns ``False`` when the fill was already recorded.
  The bus is at-least-once (spec §4.3), so a duplicate fill event is expected,
  not exceptional — and silently applying it twice doubles a position.
* :meth:`OrderStore.get_by_client_order_id` is what makes retry-after-timeout
  safe (spec §FR-EXE-04): before resending anything, ask whether the order
  already exists.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from atrader.core.models import Fill, Order, Position, TradingIntent

__all__ = [
    "FillStore",
    "IntentStore",
    "OrderStore",
    "PositionStore",
    "Storage",
]


@runtime_checkable
class OrderStore(Protocol):
    def upsert(self, order: Order) -> None:
        """Insert or replace by ``order_id``."""
        ...

    def get(self, order_id: UUID) -> Order | None: ...

    def get_by_client_order_id(self, client_order_id: str) -> Order | None:
        """Look up by the idempotency key. The first step of any safe retry."""
        ...

    def open_orders(self) -> list[Order]:
        """Orders that may still fill — the working set for reconciliation."""
        ...

    def all_orders(self) -> list[Order]: ...


@runtime_checkable
class FillStore(Protocol):
    def append(self, fill: Fill) -> bool:
        """Record a fill.

        Returns ``False`` if a fill with the same ``broker_fill_id`` (or
        ``fill_id``) is already stored, in which case nothing was written.
        Callers must treat that as "already applied", not as an error.
        """
        ...

    def get(self, fill_id: UUID) -> Fill | None: ...

    def for_order(self, order_id: UUID) -> list[Fill]: ...

    def all_fills(self) -> list[Fill]: ...


@runtime_checkable
class IntentStore(Protocol):
    def append(self, intent: TradingIntent) -> None: ...

    def get(self, intent_id: UUID) -> TradingIntent | None: ...

    def all_intents(self) -> list[TradingIntent]: ...


@runtime_checkable
class PositionStore(Protocol):
    def upsert(self, position: Position) -> None: ...

    def get(self, symbol: str) -> Position | None: ...

    def all_positions(self) -> list[Position]:
        """Every position, including flat ones — a flat position still carries
        realised P&L that the daily report needs."""
        ...


@runtime_checkable
class Storage(Protocol):
    """The set of repositories a running system needs."""

    @property
    def orders(self) -> OrderStore: ...

    @property
    def fills(self) -> FillStore: ...

    @property
    def intents(self) -> IntentStore: ...

    @property
    def positions(self) -> PositionStore: ...

    def close(self) -> None: ...
