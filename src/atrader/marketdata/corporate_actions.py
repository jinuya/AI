"""Corporate action adjustment — spec §FR-MD-05.

A 4-for-1 split makes a price series drop 75% overnight. Nothing happened
economically, but an unadjusted momentum strategy sees a catastrophic move and
an unadjusted volatility estimate goes haywire.

Both series are kept:

* **raw** — what actually printed. Live trading uses this, because that is the
  price you will be filled at.
* **adjusted** — corrected for splits and dividends. Backtests use this, because
  a strategy reading history needs a continuous series.

Getting these the wrong way round is a classic and expensive error, so the two
are separate methods with names that say which is which, rather than a boolean
flag someone will eventually pass wrongly.

Adjustment factors apply to prices *before* the ex-date. A split factor also
scales volume in the opposite direction, so notional turnover stays comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from atrader.core.money import ONE, ZERO, quantize
from atrader.marketdata.models import Bar

__all__ = ["ActionType", "AdjustmentTable", "CorporateAction"]


class ActionType(StrEnum):
    SPLIT = "SPLIT"
    DIVIDEND = "DIVIDEND"
    MERGER = "MERGER"
    SPINOFF = "SPINOFF"


@dataclass(frozen=True, slots=True)
class CorporateAction:
    """One event affecting a symbol's price continuity."""

    symbol: str
    action_type: ActionType
    ex_date_ns: int
    """First session on which the price reflects the action."""
    ratio: Decimal = ONE
    """For a split: new shares per old share. 4-for-1 is ``4``."""
    cash_amount: Decimal = ZERO
    """For a dividend: cash per share."""
    reference_price: Decimal | None = None
    """Close before the ex-date. Needed to turn a cash dividend into a factor."""

    def price_factor(self) -> Decimal:
        """Multiplier applied to prices *before* the ex-date.

        Split: a 4-for-1 divides historical prices by 4.
        Dividend: historical prices are scaled by ``(close - D) / close``, the
        standard total-return adjustment.
        """
        if self.action_type is ActionType.SPLIT:
            if self.ratio <= ZERO:
                raise ValueError(f"split ratio must be positive, got {self.ratio}")
            return ONE / self.ratio
        if self.action_type is ActionType.DIVIDEND:
            if self.reference_price is None or self.reference_price <= ZERO:
                # Without the pre-ex close there is no way to compute the
                # factor. Returning 1 silently would under-adjust the series;
                # refusing makes the missing data visible.
                raise ValueError(
                    f"dividend adjustment for {self.symbol} needs reference_price "
                    "(the close before the ex-date)"
                )
            return (self.reference_price - self.cash_amount) / self.reference_price
        # Mergers and spin-offs carry an exchange ratio rather than a split
        # factor, so it is applied directly. The default of 1 is a real answer
        # here in a way it is not for a split: a 1-for-1 exchange leaves price
        # continuity intact and needs no adjustment.
        #
        # A non-positive ratio is not an answer, though — it is bad reference
        # data. Substituting 1 would hide it behind a series that looks
        # adjusted, which is the same failure the dividend branch above
        # refuses to commit.
        if self.ratio <= ZERO:
            raise ValueError(
                f"{self.action_type} ratio for {self.symbol} must be positive, got {self.ratio}"
            )
        return self.ratio

    def volume_factor(self) -> Decimal:
        """Multiplier for historical volume. Only splits change share counts."""
        if self.action_type is ActionType.SPLIT:
            return self.ratio
        return ONE


class AdjustmentTable:
    """Cumulative adjustment factors per symbol (spec §FR-MD-05)."""

    __slots__ = ("_actions",)

    def __init__(self, actions: list[CorporateAction] | None = None) -> None:
        self._actions: dict[str, list[CorporateAction]] = {}
        for action in actions or []:
            self.add(action)

    def add(self, action: CorporateAction) -> None:
        bucket = self._actions.setdefault(action.symbol, [])
        bucket.append(action)
        bucket.sort(key=lambda a: a.ex_date_ns)

    def actions_for(self, symbol: str) -> list[CorporateAction]:
        return list(self._actions.get(symbol, []))

    def cumulative_price_factor(self, symbol: str, as_of_ns: int) -> Decimal:
        """Product of every factor from actions *after* ``as_of_ns``.

        A price from before two splits must be divided by both, which is why
        this multiplies rather than picking the nearest action.
        """
        factor = ONE
        for action in self._actions.get(symbol, []):
            if action.ex_date_ns > as_of_ns:
                factor *= action.price_factor()
        return factor

    def cumulative_volume_factor(self, symbol: str, as_of_ns: int) -> Decimal:
        factor = ONE
        for action in self._actions.get(symbol, []):
            if action.ex_date_ns > as_of_ns:
                factor *= action.volume_factor()
        return factor

    def adjust_bar(self, bar: Bar) -> Bar:
        """Return the split/dividend-adjusted bar. **Backtests only.**

        Never use the result to price a live order: the exchange will fill you
        at the raw price, not the adjusted one.
        """
        price_factor = self.cumulative_price_factor(bar.symbol, bar.close_ts)
        volume_factor = self.cumulative_volume_factor(bar.symbol, bar.close_ts)
        if price_factor == ONE and volume_factor == ONE:
            return bar
        return bar.model_copy(
            update={
                "open": quantize(bar.open * price_factor),
                "high": quantize(bar.high * price_factor),
                "low": quantize(bar.low * price_factor),
                "close": quantize(bar.close * price_factor),
                "volume": quantize(bar.volume * volume_factor),
                "vwap": None if bar.vwap is None else quantize(bar.vwap * price_factor),
            }
        )

    def raw_bar(self, bar: Bar) -> Bar:
        """Identity. Present so call sites state which series they meant.

        ``raw_bar(bar)`` at a live-trading call site documents the choice; a
        bare ``bar`` leaves the next reader wondering whether adjustment was
        forgotten.
        """
        return bar
