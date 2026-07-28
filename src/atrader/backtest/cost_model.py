"""Execution costs — spec §8.1.

    수수료·세금·슬리피지, 그리고 제곱근 법칙 시장충격을 반영한다.

A backtest that fills every order at the quoted price and charges nothing for
it is not measuring a strategy — it is measuring a strategy plus however much
apparent edge was invented by ignoring the cost of trading. Commission and tax
are what the exchange and the tax authority bill; slippage and market impact
are what nobody bills but you pay anyway, because your own order moves the
price against you while it is being worked.

Market impact here follows the square-root law: impact grows with the square
root of participation (order size over ADV), not linearly. That functional
form is the point — it is why halving an order's size does not halve its
impact, only divides it by roughly √2, and why the ADV participation check in
the risk engine (§7.2 #7) and the split-above-1%-ADV rule (§FR-EXE-03) both
exist independently of this cost model: this model prices the impact of a
fill, it does not cap it.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from atrader.config.schema import InstrumentSpec
from atrader.core.money import ZERO, quantize
from atrader.core.types import Side

__all__ = ["CostModel", "FillCost"]

BPS = Decimal(10_000)


@dataclass(frozen=True, slots=True)
class FillCost:
    execution_price: Decimal
    """Reference price after market impact — always at least as bad for the
    trader as the reference price it started from."""
    commission: Decimal
    tax: Decimal

    @property
    def total_fees(self) -> Decimal:
        return self.commission + self.tax


@dataclass(frozen=True, slots=True)
class CostModel:
    impact_coefficient: Decimal = Decimal("0.1")
    """The ``k`` in the square-root law. Calibrated low by default — most of
    this system's reference instruments are liquid large-caps."""
    default_commission_bps: Decimal = Decimal("1.0")
    default_tax_bps: Decimal = Decimal("0")
    default_daily_volatility_pct: Decimal = Decimal("2.0")
    """Used only when no per-symbol volatility estimate is supplied."""

    def commission(self, notional: Decimal, instrument: InstrumentSpec | None) -> Decimal:
        bps = instrument.commission_bps if instrument is not None else self.default_commission_bps
        return quantize(notional * bps / BPS)

    def tax(self, notional: Decimal, instrument: InstrumentSpec | None) -> Decimal:
        bps = instrument.tax_bps if instrument is not None else self.default_tax_bps
        return quantize(notional * bps / BPS)

    def impact_adjusted_price(
        self,
        *,
        side: Side,
        reference_price: Decimal,
        quantity: Decimal,
        adv: Decimal | None,
        daily_volatility_pct: Decimal | None = None,
    ) -> Decimal:
        """Reference price adjusted for the impact of trading *quantity* now.

        Always moves against the trader: higher for a buy, lower for a sell.
        No ADV estimate means no basis for the adjustment — returns the
        reference price unchanged rather than guessing.
        """
        if adv is None or adv <= ZERO or reference_price <= ZERO:
            return reference_price
        participation = quantity / adv
        sigma_fraction = (daily_volatility_pct or self.default_daily_volatility_pct) / Decimal(100)
        impact_fraction = self.impact_coefficient * sigma_fraction * participation.sqrt()
        return quantize(reference_price * (Decimal(1) + impact_fraction * side.sign))

    def apply(
        self,
        *,
        side: Side,
        quantity: Decimal,
        reference_price: Decimal,
        adv: Decimal | None,
        instrument: InstrumentSpec | None,
        daily_volatility_pct: Decimal | None = None,
    ) -> FillCost:
        """Everything this fill costs beyond the quoted reference price."""
        execution_price = self.impact_adjusted_price(
            side=side,
            reference_price=reference_price,
            quantity=quantity,
            adv=adv,
            daily_volatility_pct=daily_volatility_pct,
        )
        notional = quantity * execution_price
        return FillCost(
            execution_price=execution_price,
            commission=self.commission(notional, instrument),
            tax=self.tax(notional, instrument),
        )
