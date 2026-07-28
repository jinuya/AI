"""Property-based portfolio accounting invariants — spec §8.3.

    포지션 수량 합 == 체결 수량 합.

For any sequence of fills on one symbol, the position's signed quantity must
equal the signed sum of every fill applied to it — a BUY adds, a SELL
subtracts, and nothing else is allowed to move the number. Checked against
hundreds of generated fill sequences (varying length, side, quantity, price)
under both cost-basis methods, since the invariant has nothing to do with
which one is configured — only :class:`~atrader.portfolio.pnl.PnLReport`'s
dollar figures depend on that choice, not the share count.
"""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

from hypothesis import given, settings
from hypothesis import strategies as st

from atrader.core.clock import SimulatedClock
from atrader.core.models import Fill
from atrader.core.types import CostBasisMethod, Side
from atrader.portfolio.positions import PositionBook
from atrader.storage.memory import InMemoryPositionStore

_SIDES = st.sampled_from([Side.BUY, Side.SELL])
_QUANTITIES = st.integers(min_value=1, max_value=1000).map(Decimal)
_PRICES = st.integers(min_value=1, max_value=10_000).map(Decimal)
_METHODS = st.sampled_from([CostBasisMethod.FIFO, CostBasisMethod.AVERAGE])

_FILL_SPECS = st.lists(st.tuples(_SIDES, _QUANTITIES, _PRICES), min_size=1, max_size=30)


def _signed_total(fills: list[tuple[Side, Decimal, Decimal]]) -> Decimal:
    total = Decimal(0)
    for side, quantity, _price in fills:
        total += quantity if side is Side.BUY else -quantity
    return total


class TestPositionQuantityMatchesFillQuantitySum:
    @given(fills=_FILL_SPECS, method=_METHODS)
    @settings(max_examples=200)
    def test_signed_quantity_sum_equals_the_final_position(
        self, fills: list[tuple[Side, Decimal, Decimal]], method: CostBasisMethod
    ) -> None:
        clock = SimulatedClock(start_ns=0)
        book = PositionBook(store=InMemoryPositionStore(), clock=clock, method=method)

        for side, quantity, price in fills:
            fill = Fill(
                fill_id=uuid4(),
                order_id=uuid4(),
                symbol="AAPL",
                side=side,
                quantity=quantity,
                price=price,
            )
            book.apply_fill(fill)

        position = book.store.get("AAPL")
        expected = _signed_total(fills)
        actual = position.quantity if position is not None else Decimal(0)
        assert actual == expected, (
            f"position quantity {actual} does not match the signed sum of "
            f"applied fills {expected} under {method}"
        )

    @given(quantity=_QUANTITIES, price=_PRICES, method=_METHODS)
    def test_a_single_buy_produces_exactly_that_long_quantity(
        self, quantity: Decimal, price: Decimal, method: CostBasisMethod
    ) -> None:
        clock = SimulatedClock(start_ns=0)
        book = PositionBook(store=InMemoryPositionStore(), clock=clock, method=method)
        fill = Fill(
            fill_id=uuid4(),
            order_id=uuid4(),
            symbol="AAPL",
            side=Side.BUY,
            quantity=quantity,
            price=price,
        )
        book.apply_fill(fill)
        position = book.store.get("AAPL")
        assert position is not None
        assert position.quantity == quantity
