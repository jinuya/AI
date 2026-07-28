"""20/50 SMA crossover — the reference rule-based strategy (spec §13).

    거래량이 평균의 volume_multiple 배 이상일 때만 진입한다. 거래량 확인
    없는 골든크로스는 가짜 신호가 많다.

A golden cross (fast SMA crosses above slow) opens a long; a death cross
(fast crosses below slow) closes it. This strategy is intentionally simple —
its role in this system is as the *reference* implementation the rest of the
pipeline (netting, risk engine, execution, backtest) is proven against, not
as an alpha claim.

The two SMAs come from the point-in-time feature store — whatever
:class:`~atrader.features.engine.FeatureEngine` is wired up must be
configured with the ``FeatureSpec``s :func:`feature_specs_for` returns, so
the crossover is computed once, in one place, shared by anything else that
wants the same SMAs. Volume is tracked in a small internal window instead:
it is a single-purpose confirmation filter specific to this strategy, not a
general indicator worth adding to the shared feature vocabulary for one
consumer.

**Held quantity is read from ``context.positions``, never from a shadow
flag.** A strategy that remembers "I bought, so I must be long" independently
of the real position drifts the moment an intent is reduced or rejected by
the risk engine — the position is the only thing that is ever actually true.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from decimal import Decimal
from typing import Any

from atrader.core.models import TradingIntent
from atrader.core.money import ZERO
from atrader.core.types import Side, TargetType
from atrader.features.engine import FeatureSpec
from atrader.marketdata.models import Bar
from atrader.strategy.base import Strategy, StrategyContext

__all__ = ["FAST_FEATURE", "SLOW_FEATURE", "SmaCrossoverStrategy", "feature_specs_for"]

FAST_FEATURE = "sma_fast"
SLOW_FEATURE = "sma_slow"
_FEATURE_VERSION = 1


def feature_specs_for(*, fast_period: int, slow_period: int) -> tuple[FeatureSpec, ...]:
    """``FeatureSpec``s a ``FeatureEngine`` must be configured with for this
    strategy's crossover to be computable. Call once at wiring time."""
    return (
        FeatureSpec(name=FAST_FEATURE, version=_FEATURE_VERSION, kind="sma", period=fast_period),
        FeatureSpec(name=SLOW_FEATURE, version=_FEATURE_VERSION, kind="sma", period=slow_period),
    )


class SmaCrossoverStrategy(Strategy):
    def __init__(
        self,
        strategy_id: str,
        symbols: Iterable[str],
        *,
        fast_period: int = 20,
        slow_period: int = 50,
        volume_multiple: Decimal = Decimal("1.8"),
        min_bars: int = 60,
        entry_quantity: Decimal = Decimal("10"),
    ) -> None:
        super().__init__(strategy_id)
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be below slow_period ({slow_period})"
            )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.volume_multiple = volume_multiple
        self.min_bars = min_bars
        self.entry_quantity = entry_quantity
        self._symbols = frozenset(symbols)
        self._bar_count: dict[str, int] = {}
        self._volume_window: dict[str, deque[Decimal]] = {}
        self._fast_above_slow: dict[str, bool] = {}

    def on_bar(self, bar: Bar, context: StrategyContext) -> list[TradingIntent]:
        if bar.symbol not in self._symbols:
            return []

        count = self._bar_count.get(bar.symbol, 0) + 1
        self._bar_count[bar.symbol] = count

        # Average of *prior* bars only — folding the current bar's volume
        # into its own baseline would understate a genuine spike.
        window = self._volume_window.setdefault(bar.symbol, deque(maxlen=self.slow_period))
        avg_volume = (sum(window, ZERO) / len(window)) if window else None
        window.append(bar.volume)

        fast = context.feature(bar.symbol, FAST_FEATURE)
        slow = context.feature(bar.symbol, SLOW_FEATURE)
        if fast is None or slow is None or count < self.min_bars:
            return []

        currently_above = fast > slow
        previously_above = self._fast_above_slow.get(bar.symbol)
        self._fast_above_slow[bar.symbol] = currently_above
        if previously_above is None:
            return []  # first bar with both SMAs available — nothing to compare against yet

        holding = context.position_of(bar.symbol).quantity > ZERO

        if not previously_above and currently_above and not holding:
            if avg_volume is None or avg_volume <= ZERO:
                return []
            if bar.volume < avg_volume * self.volume_multiple:
                return []  # golden cross without volume confirmation — spec: skip it
            return [self._entry_intent(bar, context)]

        if previously_above and not currently_above and holding:
            return [self._exit_intent(bar, context)]

        return []

    def _entry_intent(self, bar: Bar, context: StrategyContext) -> TradingIntent:
        return TradingIntent(
            intent_id=context.ids.new_id(),
            strategy_id=self.strategy_id,
            symbol=bar.symbol,
            side=Side.BUY,
            target_type=TargetType.SHARES,
            target_value=self.entry_quantity,
            created_at_ns=context.now_ns,
            rationale=f"golden cross: {self.fast_period}-SMA above {self.slow_period}-SMA",
        )

    def _exit_intent(self, bar: Bar, context: StrategyContext) -> TradingIntent:
        held = context.position_of(bar.symbol).quantity
        return TradingIntent(
            intent_id=context.ids.new_id(),
            strategy_id=self.strategy_id,
            symbol=bar.symbol,
            side=Side.SELL,
            target_type=TargetType.SHARES,
            target_value=held,
            created_at_ns=context.now_ns,
            rationale=f"death cross: {self.fast_period}-SMA below {self.slow_period}-SMA",
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "bar_count": dict(self._bar_count),
            "fast_above_slow": dict(self._fast_above_slow),
        }
