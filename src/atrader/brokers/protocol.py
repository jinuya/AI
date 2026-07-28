"""Broker adapter interface — spec §6.1.

    모든 브로커는 이 인터페이스 뒤에 숨긴다. 전략 코드는 어떤 브로커를 쓰는지 몰라야 한다.

:class:`BrokerCapabilities` exists because venues differ in what they accept, and
the spec is explicit about why it matters:

    전략이 지원 안 되는 주문을 내면 어댑터가 즉시 거부해야지, 브로커까지 갔다가
    거부당하면 시간만 낭비다.

The wasted round trip is the smaller cost. The larger one is that a rejection
arriving asynchronously, a second later, has to be reconciled against an order
the system already believes is working — whereas a synchronous
:class:`~atrader.core.errors.UnsupportedByBrokerError` never creates that state
at all.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol, runtime_checkable

from atrader.brokers.models import (
    BrokerEvent,
    CancelAck,
    OrderAck,
    OrderModification,
    OrderState,
)
from atrader.core.models import AccountState, OrderRequest, Position
from atrader.core.types import OrderType, TimeInForce

__all__ = ["BrokerAdapter", "BrokerCapabilities"]


@dataclass(frozen=True, slots=True)
class BrokerCapabilities:
    """What this venue actually supports."""

    name: str
    order_types: frozenset[OrderType] = field(
        default_factory=lambda: frozenset({OrderType.MARKET, OrderType.LIMIT})
    )
    time_in_force: frozenset[TimeInForce] = field(
        default_factory=lambda: frozenset({TimeInForce.DAY, TimeInForce.GTC})
    )
    supports_short_selling: bool = False
    supports_modify: bool = False
    """Many venues require cancel-and-replace instead of an in-place amend."""
    supports_fractional_shares: bool = False
    supports_bracket_orders: bool = False
    supports_stop_at_broker: bool = True
    """Spec §7.4 prefers a stop resting at the broker: a local-only stop offers
    no protection once our process dies, which is when it is needed most."""
    max_orders_per_minute: int = 200
    min_order_notional: Decimal = Decimal(0)

    def rejects(self, request: OrderRequest) -> str | None:
        """Why this request cannot be sent, or ``None`` if it can."""
        if request.order_type not in self.order_types:
            return (
                f"{self.name} does not support {request.order_type.value} orders "
                f"(supports {sorted(t.value for t in self.order_types)})"
            )
        if request.time_in_force not in self.time_in_force:
            return (
                f"{self.name} does not support {request.time_in_force.value} "
                f"(supports {sorted(t.value for t in self.time_in_force)})"
            )
        if not self.supports_fractional_shares and request.quantity % 1 != 0:
            return f"{self.name} does not support fractional shares ({request.quantity})"
        if self.min_order_notional > 0 and request.limit_price is not None:
            notional = request.quantity * request.limit_price
            if notional < self.min_order_notional:
                return (
                    f"order notional {notional} is below {self.name}'s minimum "
                    f"{self.min_order_notional}"
                )
        return None


@runtime_checkable
class BrokerAdapter(Protocol):
    """The only interface the execution layer knows about."""

    @property
    def name(self) -> str: ...

    @property
    def capabilities(self) -> BrokerCapabilities: ...

    async def submit_order(self, request: OrderRequest) -> OrderAck: ...

    async def cancel_order(self, client_order_id: str) -> CancelAck: ...

    async def modify_order(self, client_order_id: str, mods: OrderModification) -> OrderAck: ...

    async def get_order(self, client_order_id: str) -> OrderState | None:
        """Look up one order.

        The single most important method for correctness: after an unanswered
        submit, this is what tells us whether the order exists before we
        consider resending (spec §FR-EXE-04). ``None`` means the broker has
        never seen it.
        """
        ...

    async def get_open_orders(self) -> list[OrderState]: ...

    async def get_positions(self) -> list[Position]: ...

    async def get_account(self) -> AccountState: ...

    def stream_updates(self) -> AsyncIterator[BrokerEvent]: ...

    async def close(self) -> None: ...
