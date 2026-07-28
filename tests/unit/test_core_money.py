"""Decimal arithmetic — the layer that stops rounding drift reaching a balance."""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from atrader.core.money import (
    HUNDRED,
    ONE,
    QUANTUM,
    ZERO,
    D,
    MoneyError,
    as_pct,
    floor_to_lot,
    from_float,
    pct_of,
    quantize,
    round_to_tick,
    safe_div,
)


class TestD:
    def test_accepts_exact_sources(self) -> None:
        assert D("1.5") == Decimal("1.5")
        assert D(3) == Decimal(3)
        assert D(Decimal("0.1")) == Decimal("0.1")

    def test_rejects_float(self) -> None:
        with pytest.raises(MoneyError, match="float is not an exact source"):
            D(0.1)  # type: ignore[arg-type]

    def test_rejects_garbage(self) -> None:
        with pytest.raises(MoneyError, match="cannot convert"):
            D("not a number")

    def test_from_float_uses_shortest_repr(self) -> None:
        # The whole reason D() rejects floats: Decimal(0.1) is not 0.1.
        assert from_float(0.1) == Decimal("0.1")
        assert Decimal(0.1) != Decimal("0.1")  # noqa: RUF032 — that is the point


class TestQuantize:
    def test_snaps_to_eight_places(self) -> None:
        assert quantize(D("1.123456789")) == Decimal("1.12345679")

    def test_banker_rounding_is_the_default(self) -> None:
        assert quantize(D("0.000000005")) == Decimal("0.00000000")
        assert quantize(D("0.000000015")) == Decimal("0.00000002")

    def test_rejects_overflow(self) -> None:
        with pytest.raises(MoneyError, match="overflows NUMERIC"):
            quantize(D("1000000000000"))  # 13 integer digits

    def test_accepts_the_boundary(self) -> None:
        assert quantize(D("999999999999.99999999")) == Decimal("999999999999.99999999")


class TestRoundToTick:
    def test_rounds_to_nearest_by_default(self) -> None:
        assert round_to_tick(D("187.53"), D("0.05")) == Decimal("187.55")

    def test_buy_rounds_down_so_the_order_never_gets_more_aggressive(self) -> None:
        assert round_to_tick(D("187.549"), D("0.05"), side="BUY") == Decimal("187.50")

    def test_sell_rounds_up(self) -> None:
        assert round_to_tick(D("187.501"), D("0.05"), side="SELL") == Decimal("187.55")

    def test_rejects_non_positive_tick(self) -> None:
        with pytest.raises(MoneyError, match="tick_size must be positive"):
            round_to_tick(D("100"), ZERO)

    @given(
        price=st.decimals(min_value=Decimal("0.01"), max_value=Decimal("100000"), places=4),
        tick=st.sampled_from([D("0.01"), D("0.05"), D("0.1"), D("1"), D("5")]),
    )
    def test_result_is_always_a_whole_number_of_ticks(self, price: Decimal, tick: Decimal) -> None:
        result = round_to_tick(price, tick)
        assert (result / tick) % 1 == 0


class TestFloorToLot:
    def test_always_floors(self) -> None:
        # Never round a quantity up: the risk check ran against the smaller number.
        assert floor_to_lot(D("199.9"), D("100")) == Decimal("100")
        assert floor_to_lot(D("99.9"), D("100")) == ZERO

    def test_exact_multiple_is_unchanged(self) -> None:
        assert floor_to_lot(D("300"), D("100")) == Decimal("300")

    def test_rejects_non_positive_lot(self) -> None:
        with pytest.raises(MoneyError, match="lot_size must be positive"):
            floor_to_lot(D("100"), D("-1"))

    @given(
        qty=st.decimals(min_value=ZERO, max_value=Decimal("1000000"), places=2),
        lot=st.sampled_from([ONE, D("10"), D("100")]),
    )
    def test_never_exceeds_the_input(self, qty: Decimal, lot: Decimal) -> None:
        assert floor_to_lot(qty, lot) <= qty


class TestPercentages:
    def test_pct_of(self) -> None:
        assert pct_of(D("1000"), D("2")) == Decimal("20")

    def test_as_pct(self) -> None:
        assert as_pct(D("20"), D("1000")) == Decimal("2")

    def test_as_pct_of_zero_is_zero_not_an_exception(self) -> None:
        # An empty account should report 0% exposure, not blow up the risk check.
        assert as_pct(D("20"), ZERO) == ZERO

    @given(
        whole=st.decimals(min_value=Decimal("1"), max_value=Decimal("1000000"), places=2),
        percent=st.decimals(min_value=ZERO, max_value=HUNDRED, places=2),
    )
    def test_round_trips_within_one_quantum(self, whole: Decimal, percent: Decimal) -> None:
        assume(whole > ZERO)
        part = pct_of(whole, percent)
        assert abs(as_pct(part, whole) - percent) <= QUANTUM * whole


class TestSafeDiv:
    def test_divides(self) -> None:
        assert safe_div(D("10"), D("4")) == Decimal("2.5")

    def test_zero_denominator_returns_default(self) -> None:
        assert safe_div(D("10"), ZERO) == ZERO
        assert safe_div(D("10"), ZERO, default=D("-1")) == Decimal("-1")
