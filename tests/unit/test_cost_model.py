"""Execution cost model — spec §8.1."""

from __future__ import annotations

from decimal import Decimal

from atrader.backtest.cost_model import CostModel
from atrader.config.schema import InstrumentSpec
from atrader.core.types import Side


class TestCommissionAndTax:
    def test_commission_uses_the_instrument_spec_when_given(self) -> None:
        model = CostModel()
        spec = InstrumentSpec(symbol="AAPL", commission_bps=Decimal("10"))
        assert model.commission(Decimal("10000"), spec) == Decimal("10")

    def test_commission_falls_back_to_the_default_without_an_instrument(self) -> None:
        model = CostModel(default_commission_bps=Decimal("5"))
        assert model.commission(Decimal("10000"), None) == Decimal("5")

    def test_tax_uses_the_instrument_spec_when_given(self) -> None:
        model = CostModel()
        spec = InstrumentSpec(symbol="AAPL", tax_bps=Decimal("20"))
        assert model.tax(Decimal("10000"), spec) == Decimal("20")


class TestMarketImpact:
    def test_no_adv_estimate_leaves_the_price_unchanged(self) -> None:
        model = CostModel()
        price = model.impact_adjusted_price(
            side=Side.BUY, reference_price=Decimal("100"), quantity=Decimal("1000"), adv=None
        )
        assert price == Decimal("100")

    def test_a_buy_always_gets_a_worse_higher_price(self) -> None:
        model = CostModel(impact_coefficient=Decimal("1"))
        price = model.impact_adjusted_price(
            side=Side.BUY,
            reference_price=Decimal("100"),
            quantity=Decimal("100000"),
            adv=Decimal("1000000"),
        )
        assert price > Decimal("100")

    def test_a_sell_always_gets_a_worse_lower_price(self) -> None:
        model = CostModel(impact_coefficient=Decimal("1"))
        price = model.impact_adjusted_price(
            side=Side.SELL,
            reference_price=Decimal("100"),
            quantity=Decimal("100000"),
            adv=Decimal("1000000"),
        )
        assert price < Decimal("100")

    def test_impact_grows_with_the_square_root_of_participation(self) -> None:
        model = CostModel(impact_coefficient=Decimal("1"))
        small = model.impact_adjusted_price(
            side=Side.BUY,
            reference_price=Decimal("100"),
            quantity=Decimal("10000"),
            adv=Decimal("1000000"),
        )
        # 4x the participation should move the price by roughly 2x the impact,
        # not 4x — that is the square-root law's whole point.
        large = model.impact_adjusted_price(
            side=Side.BUY,
            reference_price=Decimal("100"),
            quantity=Decimal("40000"),
            adv=Decimal("1000000"),
        )
        small_impact = small - Decimal("100")
        large_impact = large - Decimal("100")
        ratio = large_impact / small_impact
        assert Decimal("1.9") < ratio < Decimal("2.1")

    def test_zero_or_negative_adv_leaves_the_price_unchanged(self) -> None:
        model = CostModel()
        price = model.impact_adjusted_price(
            side=Side.BUY, reference_price=Decimal("100"), quantity=Decimal("100"), adv=Decimal("0")
        )
        assert price == Decimal("100")


class TestApply:
    def test_apply_bundles_impact_commission_and_tax(self) -> None:
        model = CostModel(impact_coefficient=Decimal("0"))  # isolate fees from impact
        spec = InstrumentSpec(symbol="AAPL", commission_bps=Decimal("10"), tax_bps=Decimal("5"))
        cost = model.apply(
            side=Side.BUY,
            quantity=Decimal("100"),
            reference_price=Decimal("100"),
            adv=None,
            instrument=spec,
        )
        assert cost.execution_price == Decimal("100")
        notional = Decimal("100") * Decimal("100")
        assert cost.commission == notional * Decimal("10") / Decimal(10_000)
        assert cost.tax == notional * Decimal("5") / Decimal(10_000)
        assert cost.total_fees == cost.commission + cost.tax
