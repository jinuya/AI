"""Technical indicators — pure functions over bar/close history."""

from __future__ import annotations

from decimal import Decimal

import pytest

from atrader.features.indicators import atr, ema, rsi, sma
from atrader.marketdata.models import Bar

BASE_NS = 1_700_000_000_000_000_000
ONE_MINUTE_NS = 60_000_000_000


def make_bar(*, high: str, low: str, close: str, open_: str | None = None, index: int = 0) -> Bar:
    return Bar(
        symbol="AAPL",
        interval="1d",
        open_ts=BASE_NS + index * ONE_MINUTE_NS,
        close_ts=BASE_NS + (index + 1) * ONE_MINUTE_NS,
        open=Decimal(open_ if open_ is not None else low),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        is_final=True,
    )


def decimals(*values: str) -> list[Decimal]:
    return [Decimal(v) for v in values]


class TestSma:
    def test_not_enough_history_returns_none(self) -> None:
        assert sma(decimals("1", "2"), 3) is None

    def test_exact_window_averages_all_values(self) -> None:
        assert sma(decimals("10", "20", "30"), 3) == Decimal("20")

    def test_only_the_trailing_window_is_used(self) -> None:
        assert sma(decimals("100", "10", "20", "30"), 3) == Decimal("20")

    def test_rejects_a_non_positive_period(self) -> None:
        with pytest.raises(ValueError, match="period must be positive"):
            sma(decimals("1"), 0)


class TestEma:
    def test_not_enough_history_returns_none(self) -> None:
        assert ema(decimals("1", "2"), 3) is None

    def test_seeds_with_sma_when_exactly_one_window(self) -> None:
        assert ema(decimals("10", "20", "30"), 3) == Decimal("20")

    def test_reacts_more_to_recent_values_than_sma_does(self) -> None:
        values = decimals("10", "10", "10", "10", "100")
        assert ema(values, 4) > sma(values, 4)  # type: ignore[operator]

    def test_rejects_a_non_positive_period(self) -> None:
        with pytest.raises(ValueError, match="period must be positive"):
            ema(decimals("1"), 0)


class TestAtr:
    def test_not_enough_bars_returns_none(self) -> None:
        bars = [make_bar(high="105", low="95", close="100", index=0)]
        assert atr(bars, 3) is None

    def test_rejects_a_non_positive_period(self) -> None:
        with pytest.raises(ValueError, match="period must be positive"):
            atr([make_bar(high="105", low="95", close="100", index=0)], 0)

    def test_with_no_gaps_atr_is_the_average_high_low_range(self) -> None:
        bars = [
            make_bar(high="105", low="95", close="100", index=0),
            make_bar(high="106", low="96", close="101", index=1),
            make_bar(high="104", low="94", close="99", index=2),
        ]
        # Every bar's own range is 10 and closes sit inside the next bar's
        # range, so the gap terms never dominate: ATR is just 10.
        assert atr(bars, 2) == Decimal("10")

    def test_a_gap_up_widens_true_range_beyond_the_bars_own_high_low(self) -> None:
        bars = [
            make_bar(high="105", low="100", close="104", index=0),
            make_bar(high="130", low="125", close="128", index=1),  # gapped up
        ]
        result = atr(bars, 1)
        # bar range = 130-125 = 5, but |high-prev_close| = |130-104| = 26
        assert result == Decimal("26")


class TestRsi:
    def test_not_enough_history_returns_none(self) -> None:
        assert rsi(decimals("1", "2"), 3) is None

    def test_rejects_a_non_positive_period(self) -> None:
        with pytest.raises(ValueError, match="period must be positive"):
            rsi(decimals("1"), 0)

    def test_all_gains_is_maximally_overbought(self) -> None:
        values = decimals("10", "11", "12", "13", "14")
        assert rsi(values, 4) == Decimal("100")

    def test_all_losses_is_maximally_oversold(self) -> None:
        values = decimals("14", "13", "12", "11", "10")
        assert rsi(values, 4) == Decimal("0")

    def test_mixed_moves_land_strictly_between_the_extremes(self) -> None:
        values = decimals("10", "11", "10", "11", "10", "11")
        result = rsi(values, 5)
        assert result is not None
        assert Decimal("0") < result < Decimal("100")
