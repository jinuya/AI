"""Shared domain records.

These types are the vocabulary the whole system speaks, so they live in
``core``: strategies produce a :class:`TradingIntent`, the risk engine turns one
into an :class:`Order`, the broker returns :class:`Fill` objects, the portfolio
maintains :class:`Position` objects. Putting them here keeps the dependency
graph acyclic — every layer can name them without importing a sibling.

Everything is **frozen**. State changes produce a new record via
:meth:`~pydantic.BaseModel.model_copy`, which means an order's history is a
sequence of values rather than a mutated object whose past is gone. That is what
makes the audit log (spec §9.3) and deterministic replay possible.

Money and quantities are ``Decimal`` throughout (spec §4.4). Timestamps are UTC
nanosecond ints (spec §FR-MD-02); the spec's JSON examples show ISO-8601 strings,
and :meth:`TradingIntent.to_spec_dict` renders that shape for wire formats.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atrader.core.clock import ns_to_datetime
from atrader.core.types import (
    OrderStatus,
    OrderType,
    Side,
    TargetType,
    TimeInForce,
    Urgency,
)

__all__ = [
    "AccountState",
    "Fill",
    "Order",
    "OrderRequest",
    "Position",
    "TradingIntent",
]

PositiveQty = Annotated[Decimal, Field(gt=Decimal(0))]
NonNegative = Annotated[Decimal, Field(ge=Decimal(0))]


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Strategy output (spec §FR-STR-03)
# ---------------------------------------------------------------------------


class TradingIntent(_Record):
    """What a strategy emits. Deliberately *not* an order.

    The strategy says what it wants; the execution engine decides how to get it
    (spec §FR-STR-03). Keeping those separate is what allows the risk engine to
    reduce a size, the router to choose an algo, and the netting layer to cancel
    two opposing intents against each other before either reaches the market.
    """

    intent_id: UUID
    strategy_id: str = Field(min_length=1, max_length=64)
    symbol: str = Field(min_length=1, max_length=32)
    side: Side
    target_type: TargetType = TargetType.SHARES
    target_value: Decimal
    urgency: Urgency = Urgency.NORMAL
    limit_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    valid_until_ns: int | None = None
    confidence: Annotated[Decimal, Field(ge=Decimal(0), le=Decimal(1))] = Decimal(1)
    rationale: str = ""
    created_at_ns: int = 0

    @model_validator(mode="after")
    def _validate(self) -> TradingIntent:
        if self.target_type is TargetType.TARGET_WEIGHT:
            if not Decimal(-1) <= self.target_value <= Decimal(1):
                raise ValueError(
                    f"TARGET_WEIGHT target_value must be a fraction in [-1, 1], "
                    f"got {self.target_value}. (0.1 means 10%, not 10.)"
                )
        elif self.target_value <= 0:
            raise ValueError(
                f"{self.target_type} target_value must be positive, got {self.target_value}. "
                "Direction is carried by `side`, not by the sign of the size."
            )

        for name, price in (
            ("limit_price", self.limit_price),
            ("stop_loss", self.stop_loss),
            ("take_profit", self.take_profit),
        ):
            if price is not None and price <= 0:
                raise ValueError(f"{name} must be positive, got {price}")

        # A stop above entry on a long is a stop that fires immediately.
        if self.stop_loss is not None and self.limit_price is not None:
            if self.side is Side.BUY and self.stop_loss >= self.limit_price:
                raise ValueError(
                    f"BUY stop_loss ({self.stop_loss}) must be below the limit price "
                    f"({self.limit_price}) or it triggers on entry"
                )
            if self.side is Side.SELL and self.stop_loss <= self.limit_price:
                raise ValueError(
                    f"SELL stop_loss ({self.stop_loss}) must be above the limit price "
                    f"({self.limit_price}) or it triggers on entry"
                )
        return self

    def is_expired(self, now_ns: int) -> bool:
        return self.valid_until_ns is not None and now_ns >= self.valid_until_ns

    def to_spec_dict(self) -> dict[str, Any]:
        """Render the §FR-STR-03 JSON shape, with ISO-8601 timestamps."""
        return {
            "intent_id": str(self.intent_id),
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "target_type": self.target_type.value,
            "target_value": str(self.target_value),
            "urgency": self.urgency.value,
            "limit_price": None if self.limit_price is None else str(self.limit_price),
            "time_in_force": self.time_in_force.value,
            "stop_loss": None if self.stop_loss is None else str(self.stop_loss),
            "take_profit": None if self.take_profit is None else str(self.take_profit),
            "valid_until": (
                None
                if self.valid_until_ns is None
                else ns_to_datetime(self.valid_until_ns).isoformat()
            ),
            "confidence": str(self.confidence),
            "rationale": self.rationale,
            "created_at": ns_to_datetime(self.created_at_ns).isoformat(),
        }


# ---------------------------------------------------------------------------
# Orders (spec §4.5, §FR-EXE-01)
# ---------------------------------------------------------------------------


class OrderRequest(_Record):
    """What is handed to a broker adapter.

    ``client_order_id`` is generated by us and carried through every retry — it
    is the whole idempotency mechanism (spec §FR-EXE-04).
    """

    client_order_id: str = Field(min_length=1, max_length=64)
    symbol: str = Field(min_length=1, max_length=32)
    side: Side
    order_type: OrderType
    quantity: PositiveQty
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.DAY

    @model_validator(mode="after")
    def _prices_match_the_order_type(self) -> OrderRequest:
        needs_limit = self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT)
        needs_stop = self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT)
        if needs_limit and self.limit_price is None:
            raise ValueError(f"{self.order_type} requires a limit_price")
        if needs_stop and self.stop_price is None:
            raise ValueError(f"{self.order_type} requires a stop_price")
        # Both directions, symmetrically. A stray price on an order type that
        # does not use it is not harmless decoration: this request goes
        # straight to a broker adapter, and a broker that honours the extra
        # field executes something other than what the risk engine approved.
        if not needs_limit and self.limit_price is not None:
            raise ValueError(f"{self.order_type} must not carry a limit_price")
        if not needs_stop and self.stop_price is not None:
            raise ValueError(f"{self.order_type} must not carry a stop_price")
        for name, price in (("limit_price", self.limit_price), ("stop_price", self.stop_price)):
            if price is not None and price <= 0:
                raise ValueError(f"{name} must be positive, got {price}")
        return self


class Order(_Record):
    """Our record of an order. Mirrors the ``orders`` table in spec §4.5."""

    order_id: UUID
    client_order_id: str = Field(min_length=1, max_length=64)
    broker_order_id: str | None = None
    parent_intent_id: UUID | None = None
    strategy_id: str = Field(min_length=1, max_length=64)
    symbol: str = Field(min_length=1, max_length=32)
    side: Side
    order_type: OrderType
    quantity: PositiveQty
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING_NEW
    filled_quantity: NonNegative = Decimal(0)
    avg_fill_price: Decimal | None = None
    risk_check_id: UUID | None = None
    """Which risk evaluation let this through. Spec §4.5 makes this NOT NULL for
    live orders — it is the audit link back to the decision."""
    reduce_only: bool = False
    """True when the order can only shrink exposure. Liquidation orders are
    exempt from the loss-limit checks (spec §7.4) and this is the flag that says
    so — a stop must never be blocked by the limit it exists to enforce."""
    created_at_ns: int = 0
    updated_at_ns: int = 0

    @model_validator(mode="after")
    def _fills_cannot_exceed_the_order(self) -> Order:
        if self.filled_quantity > self.quantity:
            raise ValueError(
                f"filled_quantity ({self.filled_quantity}) exceeds order quantity "
                f"({self.quantity}) — an over-fill means a duplicate fill was applied"
            )
        if self.status is OrderStatus.FILLED and self.filled_quantity != self.quantity:
            raise ValueError(
                f"status FILLED but filled_quantity ({self.filled_quantity}) != quantity "
                f"({self.quantity})"
            )
        return self

    @property
    def remaining_quantity(self) -> Decimal:
        return self.quantity - self.filled_quantity

    @property
    def is_open(self) -> bool:
        return self.status.is_open

    def to_request(self) -> OrderRequest:
        return OrderRequest(
            client_order_id=self.client_order_id,
            symbol=self.symbol,
            side=self.side,
            order_type=self.order_type,
            quantity=self.quantity,
            limit_price=self.limit_price,
            stop_price=self.stop_price,
            time_in_force=self.time_in_force,
        )


class Fill(_Record):
    """An execution. Mirrors the ``fills`` table in spec §4.5.

    ``broker_fill_id`` is UNIQUE in the schema: the bus is at-least-once
    (spec §4.3), so the same fill event *will* arrive twice, and the position
    must not double.
    """

    fill_id: UUID
    order_id: UUID
    broker_fill_id: str | None = None
    symbol: str = Field(min_length=1, max_length=32)
    side: Side
    quantity: PositiveQty
    price: PositiveQty
    commission: NonNegative = Decimal(0)
    tax: NonNegative = Decimal(0)
    executed_at_ns: int = 0

    @property
    def notional(self) -> Decimal:
        return self.quantity * self.price

    @property
    def total_cost(self) -> Decimal:
        """Notional plus fees, signed by direction — what actually moves cash."""
        gross = self.notional * self.side.sign
        return gross + self.commission + self.tax


# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------


class Position(_Record):
    """A holding. ``quantity`` is signed: negative means short."""

    symbol: str = Field(min_length=1, max_length=32)
    quantity: Decimal = Decimal(0)
    avg_price: NonNegative = Decimal(0)
    realized_pnl: Decimal = Decimal(0)
    unrealized_pnl: Decimal = Decimal(0)
    last_price: Decimal | None = None
    opened_at_ns: int | None = None
    updated_at_ns: int = 0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def market_value(self) -> Decimal:
        """Signed mark-to-market value. Falls back to cost when unpriced."""
        price = self.last_price if self.last_price is not None else self.avg_price
        return self.quantity * price

    @property
    def exposure(self) -> Decimal:
        """Absolute market value — what concentration limits measure."""
        return abs(self.market_value)

    def side_to_close(self) -> Side | None:
        if self.quantity > 0:
            return Side.SELL
        if self.quantity < 0:
            return Side.BUY
        return None


class AccountState(_Record):
    """Account-level snapshot, as reported by the broker.

    Spec §2.2: the broker is the source of truth. When this disagrees with local
    state, this wins — and reconciliation stops trading until a human looks
    (spec §FR-EXE-05).
    """

    cash: Decimal = Decimal(0)
    equity: Decimal = Decimal(0)
    buying_power: Decimal = Decimal(0)
    maintenance_margin: NonNegative = Decimal(0)
    currency: str = "USD"
    as_of_ns: int = 0

    @property
    def margin_ratio(self) -> Decimal | None:
        """Equity as a multiple of maintenance margin.

        Spec §FR-PF-04: below 1.2 warn, below 1.1 switch to reducing only.
        ``None`` when no margin is required, which is the cash-trading case.
        """
        if self.maintenance_margin == 0:
            return None
        return self.equity / self.maintenance_margin
