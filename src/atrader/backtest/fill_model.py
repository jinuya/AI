"""Bar-level fill model — spec §8.1.

    가격을 통과했을 때만 체결(보수적) + 큐 포지션 옵션.

A backtest driven by OHLCV bars cannot see the order book, so it cannot know
whether a resting limit order would actually have reached the front of the
queue when the price traded through it. Two rules bound that uncertainty from
either side, and both are load-bearing — dropping either one is the standard
way a backtest flatters a strategy:

* **Conservative crossing.** A limit fills only when the bar's range trades
  *through* it, and always at the limit price itself, never at anything
  better. Assuming price improvement the data cannot prove is assuming edge
  that was never earned.
* **Participation cap.** The "queue position" mode this spec section asks
  for, in its bar-level form: even once the price crosses, a fill is capped
  at a fraction of the bar's reported volume. A resting order at a touched
  price does not fill instantly at any size — the rest of the queue was
  there too, and the remainder rolls forward to the next bar exactly like a
  real resting order would.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

from atrader.core.money import ZERO
from atrader.core.types import OrderType, Side
from atrader.marketdata.models import Bar

__all__ = ["ConservativeFillModel", "FillDecision"]


@dataclass(frozen=True, slots=True)
class FillDecision:
    quantity: Decimal
    """How much of the order fills against this bar. May be less than asked —
    the remainder stays open for the next bar."""
    price: Decimal
    """Pre-cost execution price. :mod:`atrader.backtest.cost_model` applies
    commission, tax and market impact on top of this."""

    @property
    def is_empty(self) -> bool:
        return self.quantity <= ZERO


@dataclass(frozen=True, slots=True)
class ConservativeFillModel:
    max_participation_pct: Decimal = Decimal(25)
    """Share of one bar's volume a single order may claim. Spec §8.1's queue
    proxy: real queue position cannot be reconstructed from OHLCV alone, so
    this bounds the damage a naive "fills instantly at any size" assumption
    would otherwise do to the backtest's realism."""

    def evaluate(
        self,
        *,
        order_type: OrderType,
        side: Side,
        limit_price: Decimal | None,
        quantity: Decimal,
        bar: Bar,
    ) -> FillDecision | None:
        """Decide what fills against *bar*. ``None`` means nothing does.

        Market orders execute at the bar's open — the earliest price the
        order could plausibly have reached the tape at. Filling at the
        *close* that produced the strategy's own signal would be look-ahead:
        the strategy saw that close to decide, so the fill cannot happen at
        the same instant.
        """
        if order_type is OrderType.MARKET:
            price = bar.open
        else:
            if limit_price is None:
                raise ValueError(f"{order_type} order has no limit_price to evaluate")
            crossed = (side is Side.BUY and bar.low <= limit_price) or (
                side is Side.SELL and bar.high >= limit_price
            )
            if not crossed:
                return None
            price = limit_price

        cap = self._participation_cap(bar)
        fillable = min(quantity, cap)
        if fillable <= ZERO:
            return None
        return FillDecision(quantity=fillable, price=price)

    def _participation_cap(self, bar: Bar) -> Decimal:
        if bar.volume <= ZERO:
            return ZERO
        raw = bar.volume * self.max_participation_pct / Decimal(100)
        return raw.to_integral_value(rounding=ROUND_DOWN)
