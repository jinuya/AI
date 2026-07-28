"""Technical indicators — pure functions over bar history.

No I/O, no clock, no state: every function takes the window of data it needs
and returns a value or ``None`` when there is not yet enough history. That is
what lets :mod:`atrader.features.engine` call the same function in a live feed
and in a backtest and get the same number for the same input — determinism
here is not a nice-to-have, it is the entire reason the backtest and the live
system are allowed to share one code path (spec §2.2).

``None`` on insufficient history is deliberate, not a placeholder for zero: a
strategy that treats "not enough data yet" as "the indicator is zero" makes a
real trading decision on a number that does not exist.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from itertools import pairwise

from atrader.core.money import ZERO
from atrader.marketdata.models import Bar

__all__ = ["atr", "ema", "rsi", "sma"]


def sma(values: Sequence[Decimal], period: int) -> Decimal | None:
    """Simple moving average of the most recent *period* values."""
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    if len(values) < period:
        return None
    window = values[-period:]
    return sum(window, ZERO) / period


def ema(values: Sequence[Decimal], period: int) -> Decimal | None:
    """Exponential moving average, seeded with an SMA of the earliest values used.

    Computed fresh from a bounded trailing window every call rather than
    carrying state between calls — a point-in-time feature store (spec
    §FR-STR-05) re-derives each value from history, so there is no "previous
    EMA" to carry forward, and recomputing is what makes a backtest replay
    byte-identical regardless of where in the series it starts reading.

    The window is capped at ``20 * period`` bars: EMA weights decay
    geometrically, so bars older than that contribute a fraction of a percent
    to the result — bounding the window keeps a full backtest's recomputation
    cost linear in bar count instead of quadratic, at negligible precision cost.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    if len(values) < period:
        return None
    window = values[-(period * 20) :]
    multiplier = Decimal(2) / Decimal(period + 1)
    result = sum(window[:period], ZERO) / period
    for value in window[period:]:
        result = (value - result) * multiplier + result
    return result


def atr(bars: Sequence[Bar], period: int) -> Decimal | None:
    """Average True Range — Wilder's method, over the most recent *period* bars.

    True range folds in the previous close so a gap between sessions counts as
    range even though it falls outside any single bar's high/low. The first
    bar in the series has no predecessor, so it contributes its own high-low
    range and nothing more.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    if len(bars) < period + 1:
        return None

    true_ranges: list[Decimal] = []
    previous_close: Decimal | None = None
    for bar in bars[-(period + 1) :]:
        if previous_close is None:
            true_ranges.append(bar.range)
        else:
            true_ranges.append(
                max(bar.range, abs(bar.high - previous_close), abs(bar.low - previous_close))
            )
        previous_close = bar.close

    return sum(true_ranges[-period:], ZERO) / period


def rsi(values: Sequence[Decimal], period: int) -> Decimal | None:
    """Relative Strength Index, Wilder-smoothed, on a 0-100 scale."""
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    if len(values) < period + 1:
        return None

    window = values[-(period + 1) :]
    gains = ZERO
    losses = ZERO
    for previous, current in pairwise(window):
        change = current - previous
        if change > ZERO:
            gains += change
        else:
            losses -= change  # losses accumulated as a positive magnitude

    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == ZERO:
        return Decimal(100)  # no losses in the window: maximally overbought
    rs = avg_gain / avg_loss
    return Decimal(100) - (Decimal(100) / (Decimal(1) + rs))
