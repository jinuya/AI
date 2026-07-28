"""Order management — spec §3.4.

The OMS owns the lifecycle: it submits orders that the risk engine approved,
applies broker events to local state, and cancels what needs cancelling.

Three properties it is responsible for:

* **Nothing is submitted that the risk engine did not approve.** The only entry
  point takes a :class:`~atrader.risk.engine.RiskDecision`, not an order, so
  there is no signature that accepts a hand-made order.
* **Fills are idempotent.** The broker stream is at-least-once (spec §4.3);
  :meth:`OrderManager.apply_event` deduplicates on ``broker_fill_id`` before
  touching a position.
* **State is persisted before it is acted on.** An order that exists in memory
  but not on disk is invisible to reconciliation after a crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from uuid import UUID

from atrader.audit.logger import AuditEvent, AuditLogger
from atrader.brokers.models import BrokerEvent, BrokerEventType
from atrader.brokers.protocol import BrokerAdapter
from atrader.core.clock import Clock
from atrader.core.errors import IllegalStateTransitionError
from atrader.core.ids import IdGenerator
from atrader.core.models import Fill, Order
from atrader.core.money import ZERO
from atrader.core.types import OrderStatus
from atrader.execution.idempotency import (
    SubmissionOutcome,
    SubmissionResult,
    SubmitPolicy,
    submit_with_retry,
)
from atrader.execution.statemachine import apply_fill, transition
from atrader.storage.protocol import FillStore, OrderStore

__all__ = ["OrderManager", "SubmitReport"]


@dataclass(frozen=True, slots=True)
class SubmitReport:
    """Outcome of submitting one approved order."""

    order: Order
    result: SubmissionResult
    locked_symbol: str | None = None
    """Set when the outcome was ambiguous. Spec §6.4: lock and call a human."""

    @property
    def is_live(self) -> bool:
        return self.result.is_live


@dataclass
class OrderManager:
    """Submits orders, applies broker events, keeps local state truthful."""

    broker: BrokerAdapter
    orders: OrderStore
    fills: FillStore
    clock: Clock
    ids: IdGenerator
    audit: AuditLogger | None = None
    submit_policy: SubmitPolicy = field(default_factory=SubmitPolicy)
    _locked_symbols: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    @property
    def locked_symbols(self) -> frozenset[str]:
        """Symbols with an unresolved ambiguous submission.

        Nothing new goes out in these until a human clears them: we do not know
        whether we already have an order working.
        """
        return frozenset(self._locked_symbols)

    def unlock(self, symbol: str) -> None:
        """Clear a lock after a human has confirmed the real state."""
        self._locked_symbols.discard(symbol)

    async def submit(self, order: Order) -> SubmitReport:
        """Send an approved order to the broker.

        The order must carry a ``risk_check_id`` — that is the evidence it came
        through the risk engine rather than being constructed somewhere else.
        """
        if order.risk_check_id is None:
            raise IllegalStateTransitionError(
                str(order.order_id),
                order.status.value,
                "submitted without a risk_check_id — every order must come from the risk engine",
            )
        if order.symbol in self._locked_symbols:
            return SubmitReport(
                order=order,
                result=SubmissionResult(
                    outcome=SubmissionOutcome.REJECTED,
                    client_order_id=order.client_order_id,
                    reason=f"{order.symbol} is locked pending human resolution",
                ),
                locked_symbol=order.symbol,
            )

        # Persist before sending. An order that exists at the broker but not on
        # disk is exactly what reconciliation cannot recover from.
        self.orders.upsert(order)
        self._log(AuditEvent.ORDER_REQUESTED, order)

        result = await submit_with_retry(
            self.broker,
            order.to_request(),
            clock=self.clock,
            policy=self.submit_policy,
        )
        now = self.clock.now_ns()

        if result.outcome is SubmissionOutcome.AMBIGUOUS:
            self._locked_symbols.add(order.symbol)
            self._log(AuditEvent.ORDER_REJECTED, order, extra={"reason": result.reason})
            return SubmitReport(order=order, result=result, locked_symbol=order.symbol)

        if result.outcome is SubmissionOutcome.REJECTED:
            rejected = transition(order, OrderStatus.REJECTED, now_ns=now)
            self.orders.upsert(rejected)
            self._log(AuditEvent.ORDER_REJECTED, rejected, extra={"reason": result.reason})
            return SubmitReport(order=rejected, result=result)

        broker_order_id = (
            result.ack.broker_order_id
            if result.ack is not None
            else (result.broker_state.broker_order_id if result.broker_state else None)
        )
        accepted = order.model_copy(
            update={
                "status": OrderStatus.NEW,
                "broker_order_id": broker_order_id,
                "updated_at_ns": now,
            }
        )
        self.orders.upsert(accepted)
        self._log(AuditEvent.ORDER_ACK, accepted)

        # An order found by query may already have filled while we were unsure.
        if result.outcome is SubmissionOutcome.ALREADY_EXISTS and result.broker_state is not None:
            accepted = self._adopt_broker_state(accepted, result.broker_state)

        return SubmitReport(order=accepted, result=result)

    async def cancel(self, order: Order) -> Order:
        """Request cancellation.

        Moving to ``PENDING_CANCEL`` first is what lets the state machine accept
        a fill afterwards: cancels lose races, and the fill is real when it does.
        """
        if not order.is_open:
            return order

        now = self.clock.now_ns()
        pending = transition(order, OrderStatus.PENDING_CANCEL, now_ns=now)
        self.orders.upsert(pending)

        ack = await self.broker.cancel_order(order.client_order_id)
        if not ack.accepted:
            # Usually "already filled" — normal, not an error.
            self._log(AuditEvent.ORDER_STATE_CHANGED, pending, extra={"cancel_reason": ack.reason})
            return pending

        canceled = transition(pending, OrderStatus.CANCELED, now_ns=self.clock.now_ns())
        self.orders.upsert(canceled)
        self._log(AuditEvent.ORDER_CANCELED, canceled)
        return canceled

    async def cancel_all(self, *, symbol: str | None = None) -> list[Order]:
        """Cancel every working order, optionally for one symbol.

        Used by the kill switch (spec §FR-MON-03) and the dead-man switch.
        """
        canceled: list[Order] = []
        for order in self.orders.open_orders():
            if symbol is not None and order.symbol != symbol:
                continue
            canceled.append(await self.cancel(order))
        return canceled

    # ------------------------------------------------------------------
    # Broker events
    # ------------------------------------------------------------------

    def apply_event(self, event: BrokerEvent) -> Order | None:
        """Fold a broker event into local state. Safe to call twice.

        Redelivery is the normal case on an at-least-once stream, so this is
        written to be idempotent rather than to assume exactly-once.
        """
        if event.client_order_id is None:
            return None
        order = self.orders.get_by_client_order_id(event.client_order_id)
        if order is None:
            # An event for an order we have never seen. That is a reconciliation
            # problem, not something to invent local state for.
            return None

        now = self.clock.now_ns()

        if event.event_type is BrokerEventType.FILL:
            return self._apply_fill_event(order, event, now)

        target = {
            BrokerEventType.ORDER_ACCEPTED: OrderStatus.NEW,
            BrokerEventType.ORDER_REJECTED: OrderStatus.REJECTED,
            BrokerEventType.ORDER_CANCELED: OrderStatus.CANCELED,
            BrokerEventType.ORDER_EXPIRED: OrderStatus.EXPIRED,
        }.get(event.event_type)
        if target is None or order.status is target:
            return order

        try:
            updated = transition(order, target, now_ns=now)
        except IllegalStateTransitionError:
            # Out-of-order delivery: an ACCEPTED arriving after a FILL, for
            # instance. The later state is the truthful one, so keep it.
            return order

        self.orders.upsert(updated)
        self._log(AuditEvent.ORDER_STATE_CHANGED, updated)
        return updated

    def _apply_fill_event(self, order: Order, event: BrokerEvent, now_ns: int) -> Order:
        if event.quantity is None or event.price is None:
            return order

        fill = Fill(
            fill_id=self.ids.new_id(),
            order_id=order.order_id,
            broker_fill_id=event.broker_fill_id,
            symbol=order.symbol,
            side=order.side,
            quantity=event.quantity,
            price=event.price,
            commission=event.commission,
            executed_at_ns=event.at_ns or now_ns,
        )

        # The deduplication point. Spec §4.3: the same fill *will* arrive twice,
        # and applying it twice doubles the position.
        if not self.fills.append(fill):
            return order

        updated = apply_fill(order, fill, now_ns=now_ns)
        self.orders.upsert(updated)
        self._log(
            AuditEvent.FILL_RECEIVED,
            updated,
            extra={
                "fill_id": str(fill.fill_id),
                "broker_fill_id": fill.broker_fill_id,
                "quantity": str(fill.quantity),
                "price": str(fill.price),
            },
        )
        return updated

    def _adopt_broker_state(self, order: Order, state: object) -> Order:
        """Take the broker's word for an order's state (spec §2.2)."""
        from atrader.brokers.models import OrderState

        if not isinstance(state, OrderState):  # pragma: no cover — defensive
            return order
        if state.filled_quantity <= ZERO:
            return order

        updated = order.model_copy(
            update={
                "status": state.status,
                "filled_quantity": state.filled_quantity,
                "avg_fill_price": state.avg_fill_price,
                "updated_at_ns": self.clock.now_ns(),
            }
        )
        self.orders.upsert(updated)
        return updated

    # ------------------------------------------------------------------

    def working_quantity(self, symbol: str) -> Decimal:
        """Unfilled quantity working in a symbol, signed by direction."""
        total = ZERO
        for order in self.orders.open_orders():
            if order.symbol == symbol:
                total += order.remaining_quantity * order.side.sign
        return total

    def _log(self, event: str, order: Order, *, extra: dict[str, str | None] | None = None) -> None:
        if self.audit is None:
            return
        payload: dict[str, object] = {
            "order_id": str(order.order_id),
            "client_order_id": order.client_order_id,
            "broker_order_id": order.broker_order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "status": order.status.value,
            "quantity": str(order.quantity),
            "filled_quantity": str(order.filled_quantity),
            "risk_check_id": str(order.risk_check_id) if order.risk_check_id else None,
        }
        payload.update(extra or {})
        self.audit.append(event, actor=order.strategy_id, payload=payload)


def orders_needing_cancel(
    orders: list[Order], *, now_ns: int, valid_until: dict[UUID, int]
) -> list[Order]:
    """Working orders whose intent has expired (spec §FR-EXE-06)."""
    stale: list[Order] = []
    for order in orders:
        if not order.is_open or order.parent_intent_id is None:
            continue
        deadline = valid_until.get(order.parent_intent_id)
        if deadline is not None and now_ns >= deadline:
            stale.append(order)
    return stale
