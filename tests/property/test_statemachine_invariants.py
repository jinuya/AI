"""Property-based order state machine invariants — spec §8.3, §FR-EXE-01.

    상태머신 불법 전이 부재.

``OrderStatus`` is small enough (8 values, 64 ordered pairs) that Hypothesis
does not need to sample it — it can exhaust every pair every run. That turns
"no illegal transition slips through" from a claim backed by a handful of
examples someone thought to write into a claim checked against the entire
space of possible transitions, every time this file runs.
"""

from __future__ import annotations

import contextlib
from decimal import Decimal
from uuid import uuid4

from hypothesis import given
from hypothesis import strategies as st

from atrader.core.errors import IllegalStateTransitionError
from atrader.core.models import Order
from atrader.core.types import OrderStatus, OrderType, Side, TimeInForce
from atrader.execution.statemachine import ALLOWED_TRANSITIONS, can_transition, transition

_STATUSES = st.sampled_from(list(OrderStatus))


def make_order(status: OrderStatus) -> Order:
    quantity = Decimal("10")
    # Order's own validator ties filled_quantity to status (FILLED must
    # equal quantity exactly) — satisfy that so construction itself never
    # fails for reasons unrelated to what this file is testing.
    filled_quantity = {
        OrderStatus.FILLED: quantity,
        OrderStatus.PARTIALLY_FILLED: Decimal("5"),
    }.get(status, Decimal(0))
    return Order(
        order_id=uuid4(),
        client_order_id=str(uuid4()),
        strategy_id="test",
        symbol="AAPL",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        quantity=quantity,
        filled_quantity=filled_quantity,
        limit_price=Decimal("100"),
        time_in_force=TimeInForce.DAY,
        status=status,
    )


class TestEveryTransitionIsEitherAllowedOrRaises:
    @given(current=_STATUSES, target=_STATUSES)
    def test_there_is_no_third_outcome(self, current: OrderStatus, target: OrderStatus) -> None:
        """For every one of the 64 (current, target) pairs: either the pair
        is in ``ALLOWED_TRANSITIONS`` and ``transition()`` returns an order
        in exactly that target status, or it is not and ``transition()``
        raises. No pair silently no-ops, and no pair produces a status
        other than the one requested."""
        order = make_order(current)
        expected_allowed = target in ALLOWED_TRANSITIONS[current]
        assert can_transition(current, target) is expected_allowed

        if expected_allowed:
            result = transition(order, target, now_ns=1)
            assert result.status is target
        else:
            try:
                transition(order, target, now_ns=1)
            except IllegalStateTransitionError:
                pass
            else:
                raise AssertionError(
                    f"transition({current} -> {target}) should have raised "
                    "IllegalStateTransitionError but returned normally"
                )

    @given(current=_STATUSES, target=_STATUSES)
    def test_the_original_order_object_is_never_mutated(
        self, current: OrderStatus, target: OrderStatus
    ) -> None:
        """Orders are frozen (spec: state changes produce a new record) —
        an attempted illegal transition must not corrupt the original
        either, since a caller that catches the exception and continues
        should see the order exactly as it was."""
        order = make_order(current)
        with contextlib.suppress(IllegalStateTransitionError):
            transition(order, target, now_ns=1)
        assert order.status is current


class TestTerminalStatesHaveNoOutbound:
    @given(current=_STATUSES)
    def test_a_status_with_an_empty_row_accepts_nothing(self, current: OrderStatus) -> None:
        if ALLOWED_TRANSITIONS[current]:
            return  # only asserting about the genuinely terminal rows
        order = make_order(current)
        for target in OrderStatus:
            try:
                transition(order, target, now_ns=1)
            except IllegalStateTransitionError:
                continue
            raise AssertionError(f"{current} is supposed to be terminal but accepted {target}")
