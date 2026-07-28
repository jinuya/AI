"""Simulated broker.

Runs paper trading, integration tests and chaos tests without any venue
connection. What it models is chosen by what actually goes wrong in production:

* **Queue position.** A resting limit order does not fill because the price
  touched it — it fills when enough volume traded through. Assuming otherwise is
  the single most common way a backtest flatters a strategy (spec §8.1).
* **Partial fills.** A 10,000 share order arriving as one fill is the exception,
  not the rule, and code that only ever sees complete fills breaks on the first
  real partial.
* **Latency.** Acks and fills do not arrive instantly, which is what makes a
  cancel able to lose its race with a fill.
* **Out-of-order events.** Real streams deliver a fill before the ack that
  precedes it.

Everything is driven by an injected clock and RNG, so a paper session replays
exactly (spec §2.2).
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from decimal import Decimal

from atrader.brokers.models import (
    BrokerEvent,
    BrokerEventType,
    CancelAck,
    OrderAck,
    OrderModification,
    OrderState,
)
from atrader.brokers.protocol import BrokerCapabilities
from atrader.core.clock import Clock
from atrader.core.errors import PermanentBrokerError, UnsupportedByBrokerError
from atrader.core.models import AccountState, OrderRequest, Position
from atrader.core.money import ZERO, quantize
from atrader.core.rng import Rng, SeededRng
from atrader.core.types import OrderStatus, OrderType, Side, TimeInForce

__all__ = ["PaperBroker", "PaperBrokerConfig", "RestingOrder"]


@dataclass(frozen=True, slots=True)
class PaperBrokerConfig:
    """Knobs for how realistic — and how hostile — the simulation is."""

    latency_ns: int = 1_000_000
    """Ack latency. Non-zero so cancel/fill races are reachable in tests."""
    partial_fill_probability: float = 0.3
    max_partial_fraction: float = 0.6
    commission_bps: Decimal = Decimal("0.5")
    reject_probability: float = 0.0
    queue_ahead_multiple: Decimal = Decimal("1.0")
    """Volume that must trade at a price level before our resting order fills,
    as a multiple of the order size. 0 makes fills optimistic."""
    allow_short: bool = True
    starting_cash: Decimal = Decimal("100000")


@dataclass(slots=True)
class RestingOrder:
    """A working order and its simulated queue position."""

    request: OrderRequest
    broker_order_id: str
    status: OrderStatus
    filled_quantity: Decimal = ZERO
    avg_fill_price: Decimal | None = None
    volume_seen: Decimal = ZERO
    """Volume traded at or through our price since we joined the queue."""
    accepted_at_ns: int = 0

    @property
    def remaining(self) -> Decimal:
        return self.request.quantity - self.filled_quantity

    def to_state(self, updated_at_ns: int) -> OrderState:
        return OrderState(
            client_order_id=self.request.client_order_id,
            broker_order_id=self.broker_order_id,
            symbol=self.request.symbol,
            side=self.request.side,
            status=self.status,
            quantity=self.request.quantity,
            filled_quantity=self.filled_quantity,
            avg_fill_price=self.avg_fill_price,
            limit_price=self.request.limit_price,
            updated_at_ns=updated_at_ns,
        )


class PaperBroker:
    """In-process broker simulator."""

    __slots__ = (
        "_cash",
        "_clock",
        "_config",
        "_events",
        "_fill_counter",
        "_order_counter",
        "_orders",
        "_positions",
        "_prices",
        "_rng",
    )

    def __init__(
        self,
        clock: Clock,
        *,
        config: PaperBrokerConfig | None = None,
        rng: Rng | None = None,
        prices: dict[str, Decimal] | None = None,
    ) -> None:
        self._clock = clock
        self._config = config or PaperBrokerConfig()
        self._rng = rng or SeededRng(seed=0)
        self._orders: dict[str, RestingOrder] = {}
        self._positions: dict[str, Position] = {}
        self._prices: dict[str, Decimal] = dict(prices or {})
        self._events: deque[BrokerEvent] = deque()
        self._order_counter = 0
        self._fill_counter = 0
        self._cash = self._config.starting_cash

    # ------------------------------------------------------------------
    # BrokerAdapter
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "paper"

    @property
    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            name="paper",
            order_types=frozenset(
                {OrderType.MARKET, OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT}
            ),
            time_in_force=frozenset(
                {TimeInForce.DAY, TimeInForce.GTC, TimeInForce.IOC, TimeInForce.FOK}
            ),
            supports_short_selling=self._config.allow_short,
            supports_modify=True,
            supports_stop_at_broker=True,
        )

    async def submit_order(self, request: OrderRequest) -> OrderAck:
        now = self._clock.now_ns()

        rejection = self.capabilities.rejects(request)
        if rejection is not None:
            # Fail here rather than at the venue: a synchronous rejection never
            # creates an order the system believes is working.
            raise UnsupportedByBrokerError(rejection)

        existing = self._orders.get(request.client_order_id)
        if existing is not None:
            # The idempotency guarantee (spec §FR-EXE-04). A resend of the same
            # client_order_id returns the original order rather than creating a
            # second one — which is exactly what a retry after a lost response
            # must do.
            return OrderAck(
                client_order_id=request.client_order_id,
                broker_order_id=existing.broker_order_id,
                accepted=existing.status is not OrderStatus.REJECTED,
                status=existing.status,
                reason="duplicate client_order_id; returning the existing order",
                received_at_ns=now,
            )

        if self._rng.uniform(0, 1) < self._config.reject_probability:
            self._orders[request.client_order_id] = RestingOrder(
                request=request,
                broker_order_id=self._next_order_id(),
                status=OrderStatus.REJECTED,
                accepted_at_ns=now,
            )
            ack = OrderAck(
                client_order_id=request.client_order_id,
                broker_order_id=None,
                accepted=False,
                status=OrderStatus.REJECTED,
                reason="simulated rejection",
                received_at_ns=now,
            )
            self._emit(
                BrokerEventType.ORDER_REJECTED,
                client_order_id=request.client_order_id,
                reason=ack.reason,
            )
            return ack

        if not self._config.allow_short and request.side is Side.SELL:
            held = self._positions.get(request.symbol)
            if held is None or held.quantity < request.quantity:
                raise PermanentBrokerError(
                    f"short selling is disabled and there is no long position in "
                    f"{request.symbol} to sell"
                )

        resting = RestingOrder(
            request=request,
            broker_order_id=self._next_order_id(),
            status=OrderStatus.NEW,
            accepted_at_ns=now + self._config.latency_ns,
        )
        self._orders[request.client_order_id] = resting
        self._emit(
            BrokerEventType.ORDER_ACCEPTED,
            client_order_id=request.client_order_id,
            broker_order_id=resting.broker_order_id,
            status=OrderStatus.NEW,
        )

        if request.order_type is OrderType.MARKET:
            self._fill(resting, resting.remaining, self._market_price(request.symbol))

        return OrderAck(
            client_order_id=request.client_order_id,
            broker_order_id=resting.broker_order_id,
            accepted=True,
            status=resting.status,
            received_at_ns=now,
        )

    async def cancel_order(self, client_order_id: str) -> CancelAck:
        now = self._clock.now_ns()
        resting = self._orders.get(client_order_id)
        if resting is None:
            return CancelAck(client_order_id, False, "unknown order", now)
        if resting.status.is_terminal:
            # Losing the race is normal, not an error. Reporting it as one would
            # make an ordinary event look like a failure in the logs.
            return CancelAck(
                client_order_id,
                False,
                f"already {resting.status.value}; the cancel lost the race",
                now,
            )
        resting.status = OrderStatus.CANCELED
        self._emit(
            BrokerEventType.ORDER_CANCELED,
            client_order_id=client_order_id,
            status=OrderStatus.CANCELED,
        )
        return CancelAck(client_order_id, True, "", now)

    async def modify_order(self, client_order_id: str, mods: OrderModification) -> OrderAck:
        resting = self._orders.get(client_order_id)
        if resting is None:
            raise PermanentBrokerError(f"unknown order {client_order_id}")
        if resting.status.is_terminal:
            raise PermanentBrokerError(f"cannot modify a {resting.status.value} order")

        updates: dict[str, object] = {}
        if mods.quantity is not None:
            if mods.quantity < resting.filled_quantity:
                raise PermanentBrokerError(
                    f"cannot reduce quantity to {mods.quantity} below the "
                    f"{resting.filled_quantity} already filled"
                )
            updates["quantity"] = mods.quantity
        if mods.limit_price is not None:
            updates["limit_price"] = mods.limit_price
        if mods.stop_price is not None:
            updates["stop_price"] = mods.stop_price
        if updates:
            resting.request = resting.request.model_copy(update=updates)
            # Amending loses queue priority at every venue that allows it.
            resting.volume_seen = ZERO

        return OrderAck(
            client_order_id=client_order_id,
            broker_order_id=resting.broker_order_id,
            accepted=True,
            status=resting.status,
            received_at_ns=self._clock.now_ns(),
        )

    async def get_order(self, client_order_id: str) -> OrderState | None:
        resting = self._orders.get(client_order_id)
        return None if resting is None else resting.to_state(self._clock.now_ns())

    async def get_open_orders(self) -> list[OrderState]:
        now = self._clock.now_ns()
        return [order.to_state(now) for order in self._orders.values() if order.status.is_open]

    async def get_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if not p.is_flat]

    async def get_account(self) -> AccountState:
        equity = self._cash + sum((p.market_value for p in self._positions.values()), ZERO)
        return AccountState(
            cash=quantize(self._cash),
            equity=quantize(equity),
            buying_power=quantize(max(ZERO, self._cash)),
            as_of_ns=self._clock.now_ns(),
        )

    async def _iterate(self) -> AsyncIterator[BrokerEvent]:
        while self._events:
            yield self._events.popleft()

    def stream_updates(self) -> AsyncIterator[BrokerEvent]:
        return self._iterate()

    async def close(self) -> None:
        self._events.clear()

    # ------------------------------------------------------------------
    # Simulation control
    # ------------------------------------------------------------------

    def set_price(self, symbol: str, price: Decimal) -> None:
        self._prices[symbol] = price
        for position in list(self._positions.values()):
            if position.symbol == symbol:
                self._positions[symbol] = position.model_copy(update={"last_price": price})

    def advance_market(self, symbol: str, price: Decimal, volume: Decimal) -> list[BrokerEvent]:
        """Feed a trade print in and fill whatever it reaches.

        This is the queue-position model: an order fills when enough volume has
        traded *through* its price, not merely because the price touched it.
        """
        self.set_price(symbol, price)
        events: list[BrokerEvent] = []

        for resting in list(self._orders.values()):
            if resting.request.symbol != symbol or not resting.status.is_open:
                continue
            if not self._is_marketable(resting, price):
                continue

            resting.volume_seen += volume
            queue_ahead = resting.request.quantity * self._config.queue_ahead_multiple
            if resting.volume_seen < queue_ahead:
                continue  # still behind other orders at this level

            available = resting.volume_seen - queue_ahead
            fill_quantity = min(resting.remaining, available)
            if self._rng.uniform(0, 1) < self._config.partial_fill_probability:
                fill_quantity = quantize(
                    fill_quantity * Decimal(str(self._config.max_partial_fraction))
                )
            fill_quantity = self._round_down(fill_quantity)
            if fill_quantity <= ZERO:
                continue

            fill_price = resting.request.limit_price or price
            events.extend(self._fill(resting, fill_quantity, fill_price))

        return events

    def pending_events(self) -> list[BrokerEvent]:
        """Drain the event queue. Used by tests and the paper runtime."""
        drained = list(self._events)
        self._events.clear()
        return drained

    def shuffle_pending_events(self) -> None:
        """Deliver the queued events out of order.

        Real streams do this — a fill can arrive before the ack that logically
        precedes it. Consumers must cope, so the chaos tests make them.
        """
        events = list(self._events)
        self._events.clear()
        while events:
            index = self._rng.choice_index(len(events))
            self._events.append(events.pop(index))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_marketable(self, resting: RestingOrder, price: Decimal) -> bool:
        limit = resting.request.limit_price
        if limit is None:
            return True
        # Conservative: the price must trade *through* the limit, not merely
        # touch it (spec §8.1's fill assumption).
        if resting.request.side is Side.BUY:
            return price < limit
        return price > limit

    def _fill(self, resting: RestingOrder, quantity: Decimal, price: Decimal) -> list[BrokerEvent]:
        quantity = min(quantity, resting.remaining)
        if quantity <= ZERO:
            return []

        previous_value = resting.filled_quantity * (resting.avg_fill_price or ZERO)
        resting.filled_quantity = quantize(resting.filled_quantity + quantity)
        resting.avg_fill_price = quantize(
            (previous_value + quantity * price) / resting.filled_quantity
        )
        resting.status = (
            OrderStatus.FILLED
            if resting.filled_quantity >= resting.request.quantity
            else OrderStatus.PARTIALLY_FILLED
        )

        commission = quantize(quantity * price * self._config.commission_bps / Decimal(10_000))
        self._apply_to_position(resting.request.symbol, resting.request.side, quantity, price)
        self._cash -= (quantity * price * resting.request.side.sign) + commission

        self._fill_counter += 1
        event = BrokerEvent(
            event_type=BrokerEventType.FILL,
            client_order_id=resting.request.client_order_id,
            broker_order_id=resting.broker_order_id,
            broker_fill_id=f"paper-fill-{self._fill_counter:08d}",
            symbol=resting.request.symbol,
            side=resting.request.side,
            quantity=quantity,
            price=price,
            commission=commission,
            status=resting.status,
            at_ns=self._clock.now_ns(),
        )
        self._events.append(event)
        return [event]

    def _apply_to_position(
        self, symbol: str, side: Side, quantity: Decimal, price: Decimal
    ) -> None:
        current = self._positions.get(symbol) or Position(symbol=symbol)
        signed = quantity * side.sign
        new_quantity = current.quantity + signed

        if current.quantity == 0 or (current.quantity > 0) == (signed > 0):
            # Opening or adding: blend the cost basis.
            total_cost = abs(current.quantity) * current.avg_price + quantity * price
            new_avg = quantize(total_cost / abs(new_quantity)) if new_quantity != 0 else ZERO
            realized = current.realized_pnl
        else:
            # Reducing or flipping: realise P&L on the closed portion.
            closed = min(abs(signed), abs(current.quantity))
            direction = Decimal(1) if current.quantity > 0 else Decimal(-1)
            realized = current.realized_pnl + quantize(
                (price - current.avg_price) * closed * direction
            )
            new_avg = price if abs(signed) > abs(current.quantity) else current.avg_price

        self._positions[symbol] = Position(
            symbol=symbol,
            quantity=quantize(new_quantity),
            avg_price=ZERO if new_quantity == 0 else new_avg,
            realized_pnl=realized,
            last_price=price,
            opened_at_ns=current.opened_at_ns or self._clock.now_ns(),
            updated_at_ns=self._clock.now_ns(),
        )

    def _market_price(self, symbol: str) -> Decimal:
        price = self._prices.get(symbol)
        if price is None:
            raise PermanentBrokerError(f"no simulated price for {symbol}")
        return price

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"paper-order-{self._order_counter:08d}"

    def _round_down(self, value: Decimal) -> Decimal:
        return value.to_integral_value(rounding="ROUND_DOWN")

    def _emit(
        self,
        event_type: BrokerEventType,
        *,
        client_order_id: str | None = None,
        broker_order_id: str | None = None,
        status: OrderStatus | None = None,
        reason: str = "",
    ) -> None:
        self._events.append(
            BrokerEvent(
                event_type=event_type,
                client_order_id=client_order_id,
                broker_order_id=broker_order_id,
                status=status,
                reason=reason,
                at_ns=self._clock.now_ns(),
            )
        )
