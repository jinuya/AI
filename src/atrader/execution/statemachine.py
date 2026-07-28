"""Order state machine — spec §FR-EXE-01.

    상태 전이는 아래로 제한하고, **정의되지 않은 전이는 예외를 던진다.**

An order's status is the system's belief about what the exchange is doing with
it. Allowing an arbitrary transition means allowing that belief to become
incoherent — and every downstream decision (can I cancel this? how much is still
working? am I flat?) is derived from it.

One transition deserves calling out because it looks wrong and is not:
``PENDING_CANCEL -> FILLED``. Cancels lose races. You send the cancel, and
before the venue processes it the order fills. A state machine that forbade this
would throw on a perfectly ordinary Tuesday, and the handler would either crash
or — worse — swallow the fill and leave the system believing it is flat when it
is long.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Final

from atrader.core.errors import IllegalStateTransitionError
from atrader.core.models import Fill, Order
from atrader.core.money import ZERO, quantize, safe_div
from atrader.core.types import OrderStatus

__all__ = ["ALLOWED_TRANSITIONS", "apply_fill", "can_transition", "transition"]

#: The transition table. Anything not listed raises.
ALLOWED_TRANSITIONS: Final[dict[OrderStatus, frozenset[OrderStatus]]] = {
    OrderStatus.PENDING_NEW: frozenset(
        {
            OrderStatus.NEW,
            OrderStatus.REJECTED,
            # A venue can fill immediately on receipt (marketable IOC), so the
            # ack and the fill arrive together and NEW is never observed.
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            # Cancelled before the venue ever acknowledged it — happens when the
            # kill switch fires between submission and ack.
            OrderStatus.CANCELED,
        }
    ),
    OrderStatus.NEW: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.PENDING_CANCEL,
            OrderStatus.CANCELED,
            OrderStatus.EXPIRED,
            OrderStatus.REJECTED,
        }
    ),
    OrderStatus.PARTIALLY_FILLED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,  # each additional partial fill
            OrderStatus.FILLED,
            OrderStatus.PENDING_CANCEL,
            OrderStatus.CANCELED,
            OrderStatus.EXPIRED,
        }
    ),
    OrderStatus.PENDING_CANCEL: frozenset(
        {
            OrderStatus.CANCELED,
            # The cancel lost the race. Denying this transition would make an
            # ordinary event crash the OMS, or silently drop a real fill.
            OrderStatus.FILLED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.EXPIRED,
        }
    ),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
    OrderStatus.CANCELED: frozenset(),
    OrderStatus.EXPIRED: frozenset(),
}


def can_transition(current: OrderStatus, target: OrderStatus) -> bool:
    return target in ALLOWED_TRANSITIONS[current]


def transition(order: Order, target: OrderStatus, *, now_ns: int) -> Order:
    """Return a copy of *order* in the new status.

    Raises :class:`~atrader.core.errors.IllegalStateTransitionError` when the
    transition is not in the table. Silently ignoring it would let the system's
    belief about the exchange drift out of sync with reality.
    """
    if not can_transition(order.status, target):
        allowed = sorted(s.value for s in ALLOWED_TRANSITIONS[order.status])
        raise IllegalStateTransitionError(
            str(order.order_id),
            order.status.value,
            f"{target.value} (allowed: {allowed or 'none — terminal state'})",
        )
    return order.model_copy(update={"status": target, "updated_at_ns": now_ns})


def apply_fill(order: Order, fill: Fill, *, now_ns: int) -> Order:
    """Fold a fill into an order, advancing status and average price.

    The caller is responsible for deduplication — see
    :meth:`~atrader.storage.protocol.FillStore.append`, which returns ``False``
    for a redelivered fill. Applying the same execution twice here would
    over-fill the order, which the :class:`~atrader.core.models.Order` validator
    then rejects; that is a backstop, not the primary defence.
    """
    if order.status.is_terminal and order.status is not OrderStatus.FILLED:
        raise IllegalStateTransitionError(
            str(order.order_id),
            order.status.value,
            f"cannot apply a fill to a {order.status.value} order",
        )

    new_filled = order.filled_quantity + fill.quantity
    if new_filled > order.quantity:
        raise IllegalStateTransitionError(
            str(order.order_id),
            order.status.value,
            (
                f"fill of {fill.quantity} would take filled_quantity to {new_filled}, "
                f"above the order quantity {order.quantity} — this is what a duplicate "
                "fill looks like"
            ),
        )

    # Weighted average across fills, not a running mean of prices: two fills of
    # 10 and 990 shares are not equally informative about the average price.
    previous_value = order.filled_quantity * (order.avg_fill_price or ZERO)
    new_avg = quantize(safe_div(previous_value + fill.quantity * fill.price, new_filled))

    target = OrderStatus.FILLED if new_filled >= order.quantity else OrderStatus.PARTIALLY_FILLED
    if not can_transition(order.status, target):  # pragma: no cover
        # Unreachable against the current table: every non-terminal status
        # already allows both FILLED and PARTIALLY_FILLED, and the guard above
        # has already ruled out a terminal order. Kept as a backstop so a future
        # edit to ALLOWED_TRANSITIONS that removes one of those entries fails
        # loudly here instead of producing an order whose status the table no
        # longer sanctions.
        allowed = sorted(s.value for s in ALLOWED_TRANSITIONS[order.status])
        raise IllegalStateTransitionError(
            str(order.order_id), order.status.value, f"{target.value} (allowed: {allowed})"
        )

    return order.model_copy(
        update={
            "status": target,
            "filled_quantity": quantize(new_filled),
            "avg_fill_price": new_avg,
            "updated_at_ns": now_ns,
        }
    )


def remaining_after(order: Order, additional_fill: Decimal) -> Decimal:
    """Working quantity once *additional_fill* is applied."""
    return max(ZERO, order.quantity - order.filled_quantity - additional_fill)
