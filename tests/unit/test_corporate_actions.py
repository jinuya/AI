"""Corporate action adjustment — spec §FR-MD-05.

``atrader.marketdata.corporate_actions`` had no test at all until this file:
70 statements, zero executed. It is pure arithmetic with no infrastructure
requirement, so nothing was standing in the way — and arithmetic nobody has
run is arithmetic nobody has checked.

The expensive mistake this module exists to prevent is using the wrong series:
**adjusted for backtests, raw for live orders**. An adjusted price is not a
price you can be filled at. So the tests below assert not just that the
factors are right but that adjustment applies to exactly the bars before the
ex-date and to no others — an off-by-one there silently corrupts the bar a
strategy traded on.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from atrader.core.money import QUANTUM, quantize
from atrader.marketdata.corporate_actions import ActionType, AdjustmentTable, CorporateAction
from atrader.marketdata.models import Bar

DAY_NS = 86_400 * 1_000_000_000
BASE_NS = 1_700_000_000 * 1_000_000_000
EX_DATE = BASE_NS + 10 * DAY_NS


def bar(
    *,
    day: int,
    symbol: str = "AAPL",
    close: str = "100",
    volume: str = "1000",
    vwap: str | None = None,
) -> Bar:
    open_ts = BASE_NS + day * DAY_NS
    price = Decimal(close)
    return Bar(
        symbol=symbol,
        interval="1d",
        open_ts=open_ts,
        close_ts=open_ts + DAY_NS - 1,
        open=price,
        high=price * 2,
        low=price / 2,
        close=price,
        volume=Decimal(volume),
        vwap=None if vwap is None else Decimal(vwap),
        is_final=True,
    )


def split(ratio: str, *, symbol: str = "AAPL", ex_date_ns: int = EX_DATE) -> CorporateAction:
    return CorporateAction(
        symbol=symbol,
        action_type=ActionType.SPLIT,
        ex_date_ns=ex_date_ns,
        ratio=Decimal(ratio),
    )


def dividend(
    cash: str, *, reference: str | None = "100", ex_date_ns: int = EX_DATE
) -> CorporateAction:
    return CorporateAction(
        symbol="AAPL",
        action_type=ActionType.DIVIDEND,
        ex_date_ns=ex_date_ns,
        cash_amount=Decimal(cash),
        reference_price=None if reference is None else Decimal(reference),
    )


class TestSplitFactors:
    def test_a_four_for_one_split_quarters_historical_prices(self) -> None:
        assert split("4").price_factor() == Decimal("0.25")

    def test_and_quadruples_historical_volume(self) -> None:
        assert split("4").volume_factor() == Decimal(4)

    @pytest.mark.parametrize("ratio", ["2", "3", "4", "10", "0.5"])
    def test_notional_turnover_is_preserved(self, ratio: str) -> None:
        """The stated reason volume moves the opposite way: price x volume —
        the money that changed hands — must not change because the share
        count was redefined.

        Checked at the precision the system actually stores money at.
        ``price_factor`` is ``1 / ratio``, which for a 3-for-1 split is a
        repeating decimal, so the product is 0.999...9 rather than exactly
        one. Decimal carries 28 significant digits and every stored figure is
        quantized to 8 places, so that residue is ~20 orders of magnitude
        below anything representable — demanding exact equality here would be
        asserting something stricter than the system's own money type
        promises.
        """
        action = split(ratio)
        assert quantize(action.price_factor() * action.volume_factor()) == Decimal(1)

    def test_the_residue_is_far_below_the_stored_precision(self) -> None:
        """Pins the claim above rather than leaving it as a comment: the
        worst case among common ratios stays orders of magnitude under one
        quantum, so no sequence of adjustments can round a price wrongly."""
        for ratio in ("3", "6", "7", "9"):
            action = split(ratio)
            residue = abs(action.price_factor() * action.volume_factor() - Decimal(1))
            assert residue < QUANTUM / Decimal(10**10)

    def test_a_reverse_split_raises_historical_prices(self) -> None:
        """1-for-10 is expressed as ratio 0.1: ten old shares become one, so
        historical prices multiply by ten."""
        assert split("0.1").price_factor() == Decimal(10)

    @pytest.mark.parametrize("ratio", ["0", "-1"])
    def test_a_non_positive_ratio_is_refused(self, ratio: str) -> None:
        with pytest.raises(ValueError, match="split ratio must be positive"):
            split(ratio).price_factor()


class TestDividendFactors:
    def test_the_factor_is_the_standard_total_return_adjustment(self) -> None:
        """(close - D) / close. A $2 dividend off a $100 close scales history
        by 0.98."""
        assert dividend("2", reference="100").price_factor() == Decimal("0.98")

    def test_a_dividend_does_not_change_share_counts(self) -> None:
        assert dividend("2").volume_factor() == Decimal(1)

    @pytest.mark.parametrize("reference", [None, "0", "-5"])
    def test_a_missing_or_impossible_reference_price_is_refused(
        self, reference: str | None
    ) -> None:
        """Returning 1 would under-adjust the series and look like a
        successful no-op. Refusing makes the missing input visible."""
        with pytest.raises(ValueError, match="needs reference_price"):
            dividend("2", reference=reference).price_factor()


class TestMergerAndSpinoffFactors:
    """These carry an exchange ratio rather than a split factor, so it is
    applied directly — and unlike a split, a ratio of 1 is a real answer: a
    1-for-1 exchange preserves price continuity."""

    @pytest.mark.parametrize("action_type", [ActionType.MERGER, ActionType.SPINOFF])
    def test_the_exchange_ratio_is_applied_directly(self, action_type: ActionType) -> None:
        action = CorporateAction(
            symbol="AAPL", action_type=action_type, ex_date_ns=EX_DATE, ratio=Decimal("1.5")
        )
        assert action.price_factor() == Decimal("1.5")

    @pytest.mark.parametrize("action_type", [ActionType.MERGER, ActionType.SPINOFF])
    def test_a_one_for_one_exchange_needs_no_adjustment(self, action_type: ActionType) -> None:
        action = CorporateAction(symbol="AAPL", action_type=action_type, ex_date_ns=EX_DATE)
        assert action.price_factor() == Decimal(1)

    @pytest.mark.parametrize("action_type", [ActionType.MERGER, ActionType.SPINOFF])
    @pytest.mark.parametrize("ratio", ["0", "-2"])
    def test_a_non_positive_ratio_is_bad_data_not_a_no_op(
        self, action_type: ActionType, ratio: str
    ) -> None:
        """Silently substituting 1 would hide a reference-data error behind a
        series that looks adjusted — the same failure the dividend branch
        refuses to commit, and the same answer the split branch gives."""
        action = CorporateAction(
            symbol="AAPL", action_type=action_type, ex_date_ns=EX_DATE, ratio=Decimal(ratio)
        )
        with pytest.raises(ValueError, match="must be positive"):
            action.price_factor()

    @pytest.mark.parametrize("action_type", [ActionType.MERGER, ActionType.SPINOFF])
    def test_share_counts_are_left_alone(self, action_type: ActionType) -> None:
        action = CorporateAction(
            symbol="AAPL", action_type=action_type, ex_date_ns=EX_DATE, ratio=Decimal(2)
        )
        assert action.volume_factor() == Decimal(1)


class TestCumulativeFactors:
    def test_a_price_before_two_splits_is_divided_by_both(self) -> None:
        table = AdjustmentTable(
            [
                split("2", ex_date_ns=BASE_NS + 5 * DAY_NS),
                split("4", ex_date_ns=BASE_NS + 10 * DAY_NS),
            ]
        )
        before_both = BASE_NS + 1 * DAY_NS
        assert table.cumulative_price_factor("AAPL", before_both) == Decimal("0.125")

    def test_a_price_between_them_is_divided_by_only_the_later_one(self) -> None:
        table = AdjustmentTable(
            [
                split("2", ex_date_ns=BASE_NS + 5 * DAY_NS),
                split("4", ex_date_ns=BASE_NS + 10 * DAY_NS),
            ]
        )
        between = BASE_NS + 7 * DAY_NS
        assert table.cumulative_price_factor("AAPL", between) == Decimal("0.25")

    def test_a_price_after_every_action_is_untouched(self) -> None:
        table = AdjustmentTable([split("4")])
        after = EX_DATE + DAY_NS
        assert table.cumulative_price_factor("AAPL", after) == Decimal(1)

    def test_an_action_exactly_on_the_boundary_does_not_apply(self) -> None:
        """The ex-date is the first session that already reflects the action,
        so a bar closing at that instant needs no adjustment."""
        table = AdjustmentTable([split("4")])
        assert table.cumulative_price_factor("AAPL", EX_DATE) == Decimal(1)

    def test_another_symbols_actions_are_not_applied(self) -> None:
        table = AdjustmentTable([split("4", symbol="MSFT")])
        assert table.cumulative_price_factor("AAPL", BASE_NS) == Decimal(1)

    def test_an_unknown_symbol_has_no_adjustment(self) -> None:
        assert AdjustmentTable().cumulative_price_factor("NOPE", BASE_NS) == Decimal(1)


class TestTheTable:
    def test_actions_are_kept_in_ex_date_order_however_they_were_added(self) -> None:
        table = AdjustmentTable()
        table.add(split("4", ex_date_ns=BASE_NS + 10 * DAY_NS))
        table.add(split("2", ex_date_ns=BASE_NS + 5 * DAY_NS))
        assert [a.ratio for a in table.actions_for("AAPL")] == [Decimal(2), Decimal(4)]

    def test_actions_for_returns_a_copy(self) -> None:
        """A caller that mutates the returned list must not be editing the
        table's own state."""
        table = AdjustmentTable([split("4")])
        table.actions_for("AAPL").clear()
        assert len(table.actions_for("AAPL")) == 1

    def test_actions_for_an_unknown_symbol_is_empty(self) -> None:
        assert AdjustmentTable().actions_for("NOPE") == []


class TestAdjustBar:
    def test_a_bar_before_the_ex_date_is_scaled(self) -> None:
        table = AdjustmentTable([split("4")])
        adjusted = table.adjust_bar(bar(day=1, close="400", volume="1000"))
        assert adjusted.close == Decimal("100.00000000")
        assert adjusted.volume == Decimal("4000.00000000")

    def test_a_bar_on_or_after_the_ex_date_is_returned_unchanged(self) -> None:
        table = AdjustmentTable([split("4")])
        original = bar(day=11, close="100")
        assert table.adjust_bar(original) is original

    def test_a_bar_with_no_applicable_action_is_the_same_object(self) -> None:
        """Not merely equal — returning the input avoids a pointless copy and
        makes 'nothing happened' visible at the call site."""
        original = bar(day=1)
        assert AdjustmentTable().adjust_bar(original) is original

    def test_every_ohlc_field_is_scaled_by_the_same_factor(self) -> None:
        table = AdjustmentTable([split("2")])
        source = bar(day=1, close="100")
        adjusted = table.adjust_bar(source)
        for field in ("open", "high", "low", "close"):
            assert getattr(adjusted, field) == getattr(source, field) / 2

    def test_the_adjusted_bar_is_still_a_coherent_ohlc_bar(self) -> None:
        """Bar's own validator enforces low <= open/close <= high. Scaling by
        one positive factor preserves the ordering, and this asserts the
        result actually survives construction rather than assuming it."""
        table = AdjustmentTable([split("7")])
        adjusted = table.adjust_bar(bar(day=1, close="100", vwap="100"))
        assert adjusted.low <= adjusted.open <= adjusted.high
        assert adjusted.low <= adjusted.close <= adjusted.high
        assert adjusted.vwap is not None
        assert adjusted.low <= adjusted.vwap <= adjusted.high

    def test_an_absent_vwap_stays_absent(self) -> None:
        table = AdjustmentTable([split("4")])
        assert table.adjust_bar(bar(day=1, vwap=None)).vwap is None

    def test_a_dividend_scales_price_but_leaves_volume_alone(self) -> None:
        table = AdjustmentTable([dividend("2", reference="100")])
        adjusted = table.adjust_bar(bar(day=1, close="100", volume="1000"))
        assert adjusted.close == Decimal("98.00000000")
        assert adjusted.volume == Decimal("1000.00000000")

    def test_the_raw_series_is_never_touched(self) -> None:
        """What live trading must use: the exchange fills at the printed
        price, not the adjusted one."""
        table = AdjustmentTable([split("4")])
        original = bar(day=1, close="400")
        assert table.raw_bar(original) is original
        assert original.close == Decimal("400")

    def test_adjustment_does_not_mutate_the_input(self) -> None:
        table = AdjustmentTable([split("4")])
        original = bar(day=1, close="400", volume="1000")
        table.adjust_bar(original)
        assert original.close == Decimal("400")
        assert original.volume == Decimal("1000")
