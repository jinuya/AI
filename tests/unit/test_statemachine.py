"""Order state machine — spec §FR-EXE-01.

    상태 전이는 아래로 제한하고, 정의되지 않은 전이는 예외를 던진다.

Coverage target for this module is 95% (spec §8.3) — it is the transition
table every other execution component trusts to reject nonsense.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest

from atrader.core.errors import IllegalStateTransitionError
from atrader.core.models import Fill, Order
from atrader.core.types import OrderStatus, OrderType, Side
from atrader.execution.statemachine import (
    ALLOWED_TRANSITIONS,
    apply_fill,
    can_transition,
    remaining_after,
    transition,
)

ALL_STATUSES = list(OrderStatus)


def make_order(**overrides: Any) -> Order:
    defaults: dict[str, Any] = {
        "order_id": uuid4(),
        "client_order_id": f"coid-{uuid4()}",
        "strategy_id": "test",
        "symbol": "AAPL",
        "side": Side.BUY,
        "order_type": OrderType.LIMIT,
        "quantity": Decimal("100"),
        "limit_price": Decimal("187.50"),
        "status": OrderStatus.PENDING_NEW,
        "created_at_ns": 1_000,
        "updated_at_ns": 1_000,
    }
    return Order(**{**defaults, **overrides})


def make_fill(order: Order, **overrides: Any) -> Fill:
    defaults: dict[str, Any] = {
        "fill_id": uuid4(),
        "order_id": order.order_id,
        "symbol": order.symbol,
        "side": order.side,
        "quantity": Decimal("50"),
        "price": Decimal("187.50"),
        "executed_at_ns": 2_000,
    }
    return Fill(**{**defaults, **overrides})


class TestTransitionTable:
    def test_every_status_has_an_entry(self) -> None:
        # A status with no entry would KeyError inside can_transition rather
        # than raising the documented IllegalStateTransitionError.
        for status in ALL_STATUSES:
            assert status in ALLOWED_TRANSITIONS

    def test_terminal_states_allow_nothing(self) -> None:
        for status in (
            OrderStatus.FILLED,
            OrderStatus.REJECTED,
            OrderStatus.CANCELED,
            OrderStatus.EXPIRED,
        ):
            assert ALLOWED_TRANSITIONS[status] == frozenset()
            assert status.is_terminal

    @pytest.mark.parametrize(
        ("start", "target"),
        [
            (OrderStatus.PENDING_NEW, OrderStatus.NEW),
            (OrderStatus.PENDING_NEW, OrderStatus.REJECTED),
            (OrderStatus.PENDING_NEW, OrderStatus.PARTIALLY_FILLED),
            (OrderStatus.PENDING_NEW, OrderStatus.FILLED),
            (OrderStatus.PENDING_NEW, OrderStatus.CANCELED),
            (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED),
            (OrderStatus.NEW, OrderStatus.FILLED),
            (OrderStatus.NEW, OrderStatus.PENDING_CANCEL),
            (OrderStatus.NEW, OrderStatus.CANCELED),
            (OrderStatus.NEW, OrderStatus.EXPIRED),
            (OrderStatus.NEW, OrderStatus.REJECTED),
            (OrderStatus.PARTIALLY_FILLED, OrderStatus.PARTIALLY_FILLED),
            (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED),
            (OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING_CANCEL),
            (OrderStatus.PARTIALLY_FILLED, OrderStatus.CANCELED),
            (OrderStatus.PARTIALLY_FILLED, OrderStatus.EXPIRED),
            (OrderStatus.PENDING_CANCEL, OrderStatus.CANCELED),
            (OrderStatus.PENDING_CANCEL, OrderStatus.FILLED),
            (OrderStatus.PENDING_CANCEL, OrderStatus.PARTIALLY_FILLED),
            (OrderStatus.PENDING_CANCEL, OrderStatus.EXPIRED),
        ],
    )
    def test_allowed_transitions_succeed(self, start: OrderStatus, target: OrderStatus) -> None:
        assert can_transition(start, target)
        # transition() only changes status and updated_at_ns — it does not touch
        # filled_quantity — so an order landing on FILLED must already carry a
        # filled_quantity equal to its quantity, or the model validator rejects
        # the result. Every other target is unaffected by filled_quantity.
        filled_quantity = Decimal("100") if target is OrderStatus.FILLED else Decimal(0)
        order = make_order(status=start, quantity=Decimal("100"), filled_quantity=filled_quantity)
        updated = transition(order, target, now_ns=5_000)
        assert updated.status is target
        assert updated.updated_at_ns == 5_000

    def test_undefined_transition_raises(self) -> None:
        order = make_order(status=OrderStatus.FILLED, filled_quantity=Decimal("100"))
        with pytest.raises(IllegalStateTransitionError) as exc_info:
            transition(order, OrderStatus.NEW, now_ns=5_000)
        assert str(order.order_id) in str(exc_info.value)
        assert "FILLED" in str(exc_info.value)

    def test_error_lists_allowed_targets(self) -> None:
        order = make_order(status=OrderStatus.NEW)
        with pytest.raises(IllegalStateTransitionError) as exc_info:
            transition(order, OrderStatus.PENDING_NEW, now_ns=5_000)
        message = str(exc_info.value)
        assert "CANCELED" in message  # one of NEW's allowed targets

    def test_terminal_state_reports_none_allowed(self) -> None:
        order = make_order(status=OrderStatus.CANCELED)
        with pytest.raises(IllegalStateTransitionError) as exc_info:
            transition(order, OrderStatus.NEW, now_ns=5_000)
        assert "none — terminal state" in str(exc_info.value)

    def test_exhaustive_matrix_matches_the_table(self) -> None:
        # Cross-check can_transition against the table itself for every pair —
        # this is what actually exercises all (25) entries for coverage.
        for start in ALL_STATUSES:
            for target in ALL_STATUSES:
                expected = target in ALLOWED_TRANSITIONS[start]
                assert can_transition(start, target) is expected


class TestPendingCancelRaceWithFill:
    """The transition that looks wrong and is not: cancels lose races."""

    def test_pending_cancel_to_filled_is_allowed(self) -> None:
        order = make_order(status=OrderStatus.PENDING_CANCEL, quantity=Decimal("100"))
        filled = transition(
            order.model_copy(update={"filled_quantity": Decimal("100")}),
            OrderStatus.FILLED,
            now_ns=9_000,
        )
        assert filled.status is OrderStatus.FILLED

    def test_pending_cancel_to_partially_filled_is_allowed(self) -> None:
        order = make_order(status=OrderStatus.PENDING_CANCEL, quantity=Decimal("100"))
        updated = transition(order, OrderStatus.PARTIALLY_FILLED, now_ns=9_000)
        assert updated.status is OrderStatus.PARTIALLY_FILLED


class TestApplyFill:
    def test_partial_fill_advances_status_and_averages_price(self) -> None:
        order = make_order(status=OrderStatus.NEW, quantity=Decimal("100"))
        fill = make_fill(order, quantity=Decimal("40"), price=Decimal("100"))
        updated = apply_fill(order, fill, now_ns=3_000)
        assert updated.status is OrderStatus.PARTIALLY_FILLED
        assert updated.filled_quantity == Decimal("40")
        assert updated.avg_fill_price == Decimal("100.00000000")
        assert updated.updated_at_ns == 3_000

    def test_second_fill_is_weighted_by_quantity_not_averaged_flat(self) -> None:
        order = make_order(status=OrderStatus.NEW, quantity=Decimal("100"))
        first = apply_fill(
            order, make_fill(order, quantity=Decimal("10"), price=Decimal("100")), now_ns=1
        )
        second = apply_fill(
            first, make_fill(order, quantity=Decimal("90"), price=Decimal("200")), now_ns=2
        )
        # 10@100 + 90@200 = 1000 + 18000 = 19000 / 100 = 190, not the flat
        # average of 150 that a naive running mean of prices would give.
        assert second.avg_fill_price == Decimal("190.00000000")
        assert second.status is OrderStatus.FILLED

    def test_fill_completing_the_order_transitions_to_filled(self) -> None:
        order = make_order(status=OrderStatus.NEW, quantity=Decimal("100"))
        updated = apply_fill(order, make_fill(order, quantity=Decimal("100")), now_ns=3_000)
        assert updated.status is OrderStatus.FILLED
        assert updated.filled_quantity == updated.quantity

    def test_fill_from_pending_cancel_still_advances(self) -> None:
        order = make_order(status=OrderStatus.PENDING_CANCEL, quantity=Decimal("100"))
        updated = apply_fill(order, make_fill(order, quantity=Decimal("100")), now_ns=3_000)
        assert updated.status is OrderStatus.FILLED

    def test_overfill_is_rejected(self) -> None:
        order = make_order(
            status=OrderStatus.PARTIALLY_FILLED,
            quantity=Decimal("100"),
            filled_quantity=Decimal("80"),
        )
        with pytest.raises(IllegalStateTransitionError) as exc_info:
            apply_fill(order, make_fill(order, quantity=Decimal("30")), now_ns=3_000)
        assert "duplicate fill" in str(exc_info.value)

    def test_fill_on_a_terminal_non_filled_order_is_rejected(self) -> None:
        order = make_order(status=OrderStatus.CANCELED, quantity=Decimal("100"))
        with pytest.raises(IllegalStateTransitionError) as exc_info:
            apply_fill(order, make_fill(order, quantity=Decimal("10")), now_ns=3_000)
        assert "cannot apply a fill to a CANCELED order" in str(exc_info.value)

    def test_fill_on_an_already_filled_order_overfills(self) -> None:
        # FILLED is terminal but the guard is specifically "terminal and not
        # FILLED" — a stray fill after completion must still be caught, by the
        # overfill check rather than the terminal-state check.
        order = make_order(
            status=OrderStatus.FILLED, quantity=Decimal("100"), filled_quantity=Decimal("100")
        )
        with pytest.raises(IllegalStateTransitionError) as exc_info:
            apply_fill(order, make_fill(order, quantity=Decimal("1")), now_ns=3_000)
        assert "duplicate fill" in str(exc_info.value)


class TestRemainingAfter:
    def test_remaining_after_a_partial_fill(self) -> None:
        order = make_order(quantity=Decimal("100"), filled_quantity=Decimal("30"))
        assert remaining_after(order, Decimal("20")) == Decimal("50")

    def test_remaining_never_goes_negative(self) -> None:
        order = make_order(quantity=Decimal("100"), filled_quantity=Decimal("90"))
        assert remaining_after(order, Decimal("50")) == Decimal(0)
