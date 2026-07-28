"""Broker-facing records.

Deliberately separate from :mod:`atrader.core.models`. These describe what the
*broker* said, which is not always what we asked for — a venue may partially
reject, re-price, or report a status we do not have a local equivalent for. The
adapter is where those differences are reconciled, and having distinct types
means the difference cannot be papered over by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any

from atrader.core.types import OrderStatus, Side

__all__ = [
    "BrokerEvent",
    "BrokerEventType",
    "CancelAck",
    "OrderAck",
    "OrderModification",
    "OrderState",
]


@dataclass(frozen=True, slots=True)
class OrderAck:
    """The broker's response to a submission."""

    client_order_id: str
    broker_order_id: str | None
    accepted: bool
    status: OrderStatus
    reason: str = ""
    received_at_ns: int = 0

    @property
    def rejected(self) -> bool:
        return not self.accepted


@dataclass(frozen=True, slots=True)
class CancelAck:
    client_order_id: str
    accepted: bool
    reason: str = ""
    received_at_ns: int = 0


@dataclass(frozen=True, slots=True)
class OrderModification:
    """An amendment. Venues that cannot amend do cancel-and-replace instead."""

    quantity: Decimal | None = None
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None

    @property
    def is_empty(self) -> bool:
        return self.quantity is None and self.limit_price is None and self.stop_price is None


@dataclass(frozen=True, slots=True)
class OrderState:
    """The broker's view of an order.

    Spec §2.2: when this disagrees with local state, **this wins**. The whole
    of reconciliation (§FR-EXE-05) is a comparison against these.
    """

    client_order_id: str
    broker_order_id: str | None
    symbol: str
    side: Side
    status: OrderStatus
    quantity: Decimal
    filled_quantity: Decimal = Decimal(0)
    avg_fill_price: Decimal | None = None
    limit_price: Decimal | None = None
    updated_at_ns: int = 0

    @property
    def remaining(self) -> Decimal:
        return self.quantity - self.filled_quantity


class BrokerEventType(StrEnum):
    ORDER_ACCEPTED = "order_accepted"
    ORDER_REJECTED = "order_rejected"
    ORDER_CANCELED = "order_canceled"
    ORDER_EXPIRED = "order_expired"
    FILL = "fill"
    ACCOUNT_UPDATE = "account_update"
    DISCONNECTED = "disconnected"
    RECONNECTED = "reconnected"


@dataclass(frozen=True, slots=True)
class BrokerEvent:
    """One update from the broker's stream.

    ``broker_fill_id`` is what makes fill handling idempotent: the stream is
    at-least-once (spec §4.3), so the same execution arrives more than once and
    applying it twice would double the position.
    """

    event_type: BrokerEventType
    client_order_id: str | None = None
    broker_order_id: str | None = None
    broker_fill_id: str | None = None
    symbol: str | None = None
    side: Side | None = None
    quantity: Decimal | None = None
    price: Decimal | None = None
    commission: Decimal = Decimal(0)
    tax: Decimal = Decimal(0)
    status: OrderStatus | None = None
    reason: str = ""
    at_ns: int = 0
    raw: dict[str, Any] = field(default_factory=dict)
    """The unmodified venue payload. Spec §9.3 requires the original response to
    be retained — a normalised copy is not enough when the question is whether
    our parsing was wrong."""
