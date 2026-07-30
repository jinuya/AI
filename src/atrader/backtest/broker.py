"""The backtest broker — spec §2.2.

    전략 코드는 백테스트와 실거래에서 문자 그대로 동일하다. 차이는 데이터
    소스와 실행 레이어뿐이다.

:class:`BacktestBroker` implements the exact same
:class:`~atrader.brokers.protocol.BrokerAdapter` Protocol as
:class:`~atrader.brokers.paper.PaperBroker` and any live adapter, so the risk
engine and :class:`~atrader.execution.oms.OrderManager` run completely
unmodified in a backtest — the "identical code path" half of the spec
principle above is not a design goal to aim for, it is a consequence of
reusing the same Protocol.

What differs from :class:`PaperBroker` is only *how* a fill is decided.
PaperBroker reacts to individual trade prints because a paper session has a
tick stream; a backtest never has one, only OHLCV history, so
:meth:`BacktestBroker.advance_bar` reacts to a whole bar at a time via
:class:`~atrader.backtest.fill_model.ConservativeFillModel` and
:class:`~atrader.backtest.cost_model.CostModel` instead.

Nothing fills the instant an order is submitted — not even a market order.
Filling happens only inside :meth:`advance_bar`, against a bar the caller
supplies. That is what keeps the backtest look-ahead free: an order built
from bar *N*'s close can only ever be matched against bar *N+1* onward,
because the engine driving this broker calls ``advance_bar`` on later bars
only after the order has already been submitted.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from decimal import Decimal

from atrader.backtest.cost_model import CostModel, FillCost
from atrader.backtest.fill_model import ConservativeFillModel
from atrader.brokers.models import (
    BrokerEvent,
    BrokerEventType,
    CancelAck,
    OrderAck,
    OrderModification,
    OrderState,
)
from atrader.brokers.protocol import BrokerCapabilities
from atrader.config.schema import InstrumentSpec
from atrader.core.clock import Clock
from atrader.core.errors import PermanentBrokerError, UnsupportedByBrokerError
from atrader.core.models import AccountState, OrderRequest, Position
from atrader.core.money import ZERO, quantize
from atrader.core.types import OrderStatus, OrderType, Side, TimeInForce
from atrader.marketdata.models import Bar

__all__ = ["BacktestBroker"]


def _no_instrument(symbol: str) -> InstrumentSpec | None:
    return None


def _no_adv(symbol: str) -> Decimal | None:
    return None


@dataclass(slots=True)
class _RestingOrder:
    request: OrderRequest
    broker_order_id: str
    status: OrderStatus
    filled_quantity: Decimal = ZERO
    avg_fill_price: Decimal | None = None
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


@dataclass
class BacktestBroker:
    """Fills orders against historical bars instead of a live/paper tick stream."""

    clock: Clock
    fill_model: ConservativeFillModel = field(default_factory=ConservativeFillModel)
    cost_model: CostModel = field(default_factory=CostModel)
    instrument_of: Callable[[str], InstrumentSpec | None] = field(default=_no_instrument)
    adv_of: Callable[[str], Decimal | None] = field(default=_no_adv)
    starting_cash: Decimal = Decimal("100000")

    _orders: dict[str, _RestingOrder] = field(default_factory=dict)
    _positions: dict[str, Position] = field(default_factory=dict)
    _events: deque[BrokerEvent] = field(default_factory=deque)
    _order_counter: int = field(default=0, init=False)
    _fill_counter: int = field(default=0, init=False)
    _cash: Decimal = field(init=False)

    def __post_init__(self) -> None:
        self._cash = self.starting_cash

    # ------------------------------------------------------------------
    # BrokerAdapter
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "backtest"

    @property
    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            name="backtest",
            order_types=frozenset({OrderType.MARKET, OrderType.LIMIT}),
            time_in_force=frozenset({TimeInForce.DAY, TimeInForce.GTC}),
            supports_short_selling=True,
            supports_modify=True,
            supports_stop_at_broker=False,
        )

    async def submit_order(self, request: OrderRequest) -> OrderAck:
        now = self.clock.now_ns()

        rejection = self.capabilities.rejects(request)
        if rejection is not None:
            raise UnsupportedByBrokerError(rejection)

        existing = self._orders.get(request.client_order_id)
        if existing is not None:
            return OrderAck(
                client_order_id=request.client_order_id,
                broker_order_id=existing.broker_order_id,
                accepted=existing.status is not OrderStatus.REJECTED,
                status=existing.status,
                reason="duplicate client_order_id; returning the existing order",
                received_at_ns=now,
            )

        resting = _RestingOrder(
            request=request,
            broker_order_id=self._next_order_id(),
            status=OrderStatus.NEW,
            accepted_at_ns=now,
        )
        self._orders[request.client_order_id] = resting
        self._emit(
            BrokerEventType.ORDER_ACCEPTED,
            client_order_id=request.client_order_id,
            broker_order_id=resting.broker_order_id,
            status=OrderStatus.NEW,
        )
        return OrderAck(
            client_order_id=request.client_order_id,
            broker_order_id=resting.broker_order_id,
            accepted=True,
            status=resting.status,
            received_at_ns=now,
        )

    async def cancel_order(self, client_order_id: str) -> CancelAck:
        now = self.clock.now_ns()
        resting = self._orders.get(client_order_id)
        if resting is None:
            return CancelAck(client_order_id, False, "unknown order", now)
        if resting.status.is_terminal:
            return CancelAck(client_order_id, False, f"already {resting.status.value}", now)
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
        if updates:
            resting.request = resting.request.model_copy(update=updates)

        return OrderAck(
            client_order_id=client_order_id,
            broker_order_id=resting.broker_order_id,
            accepted=True,
            status=resting.status,
            received_at_ns=self.clock.now_ns(),
        )

    async def get_order(self, client_order_id: str) -> OrderState | None:
        resting = self._orders.get(client_order_id)
        return None if resting is None else resting.to_state(self.clock.now_ns())

    async def get_open_orders(self) -> list[OrderState]:
        now = self.clock.now_ns()
        return [order.to_state(now) for order in self._orders.values() if order.status.is_open]

    async def get_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if not p.is_flat]

    async def get_account(self) -> AccountState:
        equity = self._cash + sum((p.market_value for p in self._positions.values()), ZERO)
        return AccountState(
            cash=quantize(self._cash),
            equity=quantize(equity),
            buying_power=quantize(max(ZERO, self._cash)),
            as_of_ns=self.clock.now_ns(),
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

    def advance_bar(self, bar: Bar) -> list[BrokerEvent]:
        """Evaluate every resting order for ``bar.symbol`` against this bar.

        Called once per bar by :class:`~atrader.backtest.engine.BacktestEngine`,
        strictly after the bar has already been fed to features and the
        strategy has reacted to it — never before, or a fill could happen at
        a price the order was priced against in the first place.
        """
        self._mark_to_market(bar)

        events: list[BrokerEvent] = []
        for resting in list(self._orders.values()):
            if resting.request.symbol != bar.symbol or not resting.status.is_open:
                continue

            decision = self.fill_model.evaluate(
                order_type=resting.request.order_type,
                side=resting.request.side,
                limit_price=resting.request.limit_price,
                quantity=resting.remaining,
                bar=bar,
            )
            if decision is None or decision.is_empty:
                continue

            cost = self.cost_model.apply(
                side=resting.request.side,
                quantity=decision.quantity,
                reference_price=decision.price,
                adv=self.adv_of(bar.symbol),
                instrument=self.instrument_of(bar.symbol),
            )
            events.extend(self._fill(resting, decision.quantity, cost, bar.close_ts))
        return events

    def pending_events(self) -> list[BrokerEvent]:
        drained = list(self._events)
        self._events.clear()
        return drained

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fill(
        self, resting: _RestingOrder, quantity: Decimal, cost: FillCost, at_ns: int
    ) -> list[BrokerEvent]:
        quantity = min(quantity, resting.remaining)
        if quantity <= ZERO:
            return []

        price = cost.execution_price
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

        self._apply_to_position(resting.request.symbol, resting.request.side, quantity, price)
        self._cash -= (quantity * price * resting.request.side.sign) + cost.total_fees

        self._fill_counter += 1
        event = BrokerEvent(
            event_type=BrokerEventType.FILL,
            client_order_id=resting.request.client_order_id,
            broker_order_id=resting.broker_order_id,
            broker_fill_id=f"backtest-fill-{self._fill_counter:08d}",
            symbol=resting.request.symbol,
            side=resting.request.side,
            quantity=quantity,
            price=price,
            commission=cost.commission,
            tax=cost.tax,
            status=resting.status,
            at_ns=at_ns,
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
            total_cost = abs(current.quantity) * current.avg_price + quantity * price
            new_avg = quantize(total_cost / abs(new_quantity)) if new_quantity != 0 else ZERO
            realized = current.realized_pnl
        else:
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
            opened_at_ns=current.opened_at_ns or self.clock.now_ns(),
            updated_at_ns=self.clock.now_ns(),
        )

    def _mark_to_market(self, bar: Bar) -> None:
        """Re-mark this symbol's open position against the bar's close.

        ``last_price`` was only ever written at fill time, so ``get_account()``
        — and therefore the equity curve, the daily-loss input and the drawdown
        input — was frozen at the last traded price. A position could halve
        while the reported equity did not move, which is exactly the scenario
        the loss limit and the drawdown breaker exist to catch: neither could
        ever fire on an open position, and every performance metric was
        computed from a curve blind to unrealized P&L.
        """
        position = self._positions.get(bar.symbol)
        if position is None or position.is_flat or position.last_price == bar.close:
            return
        self._positions[bar.symbol] = position.model_copy(
            update={"last_price": bar.close, "updated_at_ns": self.clock.now_ns()}
        )

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"backtest-order-{self._order_counter:08d}"

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
                at_ns=self.clock.now_ns(),
            )
        )
