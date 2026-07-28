"""Bar-level conservative fill model — spec §8.1."""

from __future__ import annotations

from decimal import Decimal

import pytest

from atrader.backtest.fill_model import ConservativeFillModel
from atrader.core.types import OrderType, Side
from atrader.marketdata.models import Bar

BASE_NS = 1_700_000_000_000_000_000


def make_bar(*, open_: str, high: str, low: str, close: str, volume: str = "10000") -> Bar:
    return Bar(
        symbol="AAPL",
        interval="1d",
        open_ts=BASE_NS,
        close_ts=BASE_NS + 1,
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal(volume),
        is_final=True,
    )


class TestMarketOrders:
    def test_fills_at_the_bars_open_not_its_close(self) -> None:
        model = ConservativeFillModel()
        bar = make_bar(open_="100", high="105", low="99", close="104")
        decision = model.evaluate(
            order_type=OrderType.MARKET,
            side=Side.BUY,
            limit_price=None,
            quantity=Decimal("10"),
            bar=bar,
        )
        assert decision is not None
        assert decision.price == Decimal("100")


class TestConservativeCrossing:
    def test_a_buy_limit_fills_when_the_low_touches_it(self) -> None:
        model = ConservativeFillModel()
        bar = make_bar(open_="100", high="105", low="95", close="102")
        decision = model.evaluate(
            order_type=OrderType.LIMIT,
            side=Side.BUY,
            limit_price=Decimal("97"),
            quantity=Decimal("10"),
            bar=bar,
        )
        assert decision is not None
        assert decision.price == Decimal("97")  # never better than requested

    def test_a_buy_limit_below_the_bars_low_does_not_fill(self) -> None:
        # The market never traded low enough to reach this bid.
        model = ConservativeFillModel()
        bar = make_bar(open_="100", high="105", low="99", close="102")
        decision = model.evaluate(
            order_type=OrderType.LIMIT,
            side=Side.BUY,
            limit_price=Decimal("90"),
            quantity=Decimal("10"),
            bar=bar,
        )
        assert decision is None

    def test_a_buy_limit_above_the_bars_range_is_immediately_marketable(self) -> None:
        # A limit above the whole traded range would have crossed the entire
        # bar — this is the buy-side equivalent of a market order.
        model = ConservativeFillModel()
        bar = make_bar(open_="100", high="105", low="99", close="102")
        decision = model.evaluate(
            order_type=OrderType.LIMIT,
            side=Side.BUY,
            limit_price=Decimal("110"),
            quantity=Decimal("10"),
            bar=bar,
        )
        assert decision is not None
        assert decision.price == Decimal("110")  # still never better than requested

    def test_a_sell_limit_fills_when_the_high_touches_it(self) -> None:
        model = ConservativeFillModel()
        bar = make_bar(open_="100", high="110", low="99", close="102")
        decision = model.evaluate(
            order_type=OrderType.LIMIT,
            side=Side.SELL,
            limit_price=Decimal("108"),
            quantity=Decimal("10"),
            bar=bar,
        )
        assert decision is not None
        assert decision.price == Decimal("108")

    def test_a_sell_limit_above_the_bars_high_does_not_fill(self) -> None:
        # The market never traded high enough to reach this ask.
        model = ConservativeFillModel()
        bar = make_bar(open_="100", high="102", low="99", close="101")
        decision = model.evaluate(
            order_type=OrderType.LIMIT,
            side=Side.SELL,
            limit_price=Decimal("200"),
            quantity=Decimal("10"),
            bar=bar,
        )
        assert decision is None

    def test_a_limit_order_with_no_limit_price_is_a_programming_error(self) -> None:
        model = ConservativeFillModel()
        bar = make_bar(open_="100", high="105", low="99", close="102")
        with pytest.raises(ValueError, match="no limit_price"):
            model.evaluate(
                order_type=OrderType.LIMIT,
                side=Side.BUY,
                limit_price=None,
                quantity=Decimal("10"),
                bar=bar,
            )


class TestParticipationCap:
    def test_a_fill_is_capped_at_the_configured_share_of_bar_volume(self) -> None:
        model = ConservativeFillModel(max_participation_pct=Decimal("10"))
        bar = make_bar(open_="100", high="105", low="99", close="102", volume="1000")
        decision = model.evaluate(
            order_type=OrderType.MARKET,
            side=Side.BUY,
            limit_price=None,
            quantity=Decimal("500"),
            bar=bar,
        )
        assert decision is not None
        assert decision.quantity == Decimal("100")  # 10% of 1000

    def test_a_small_order_fills_in_full_when_under_the_cap(self) -> None:
        model = ConservativeFillModel(max_participation_pct=Decimal("25"))
        bar = make_bar(open_="100", high="105", low="99", close="102", volume="1000")
        decision = model.evaluate(
            order_type=OrderType.MARKET,
            side=Side.BUY,
            limit_price=None,
            quantity=Decimal("50"),
            bar=bar,
        )
        assert decision is not None
        assert decision.quantity == Decimal("50")

    def test_zero_volume_means_nothing_fills(self) -> None:
        model = ConservativeFillModel()
        bar = make_bar(open_="100", high="105", low="99", close="102", volume="0")
        decision = model.evaluate(
            order_type=OrderType.MARKET,
            side=Side.BUY,
            limit_price=None,
            quantity=Decimal("10"),
            bar=bar,
        )
        assert decision is None
