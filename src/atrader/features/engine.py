"""Feature computation engine — bridges bar history into the point-in-time store.

The one rule that matters is in the module docstring of
:mod:`atrader.marketdata.models`: never compute a signal from a bar that has
not closed. :meth:`FeatureEngine.on_bar` enforces it directly — a forming bar
is a programming error here, not a value to degrade gracefully around,
because silently accepting one would let a look-ahead bug through exactly the
gap the point-in-time store exists to close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from atrader.features.indicators import atr, ema, rsi, sma
from atrader.features.registry import FeatureDefinition, FeatureRegistry
from atrader.features.store import FeatureStore, FeatureValue
from atrader.marketdata.models import Bar

__all__ = ["FeatureEngine", "FeatureSpec"]

_Kind = Literal["sma", "ema", "atr", "rsi"]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    version: int
    kind: _Kind
    period: int


@dataclass
class FeatureEngine:
    """Computes a configured set of indicators from closed bars and publishes
    each to the :class:`~atrader.features.store.FeatureStore`."""

    store: FeatureStore
    registry: FeatureRegistry
    specs: tuple[FeatureSpec, ...]
    max_history: int = 500
    """Bars retained per symbol. Bounds memory; generous relative to any of
    this system's indicator periods (the longest reference is a 50-bar SMA)."""
    _history: dict[str, list[Bar]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for spec in self.specs:
            self.registry.register(FeatureDefinition(spec.name, spec.version))

    def on_bar(self, bar: Bar) -> dict[str, Decimal]:
        """Feed one closed bar in. Returns the features that had enough
        history to compute — a spec with insufficient warm-up data is simply
        absent from the result, not present as ``None``."""
        if not bar.is_final:
            raise ValueError(
                f"FeatureEngine only accepts closed bars, got a forming bar for "
                f"{bar.symbol}. Trading on a forming bar's signal is exactly the "
                "look-ahead bug spec §5.2 warns about."
            )

        history = self._history.setdefault(bar.symbol, [])
        history.append(bar)
        if len(history) > self.max_history:
            del history[: len(history) - self.max_history]

        closes = [b.close for b in history]
        produced: dict[str, Decimal] = {}
        for spec in self.specs:
            value = self._compute(spec, closes, history)
            if value is None:
                continue
            self.store.publish(
                FeatureValue(
                    symbol=bar.symbol,
                    name=spec.name,
                    value=value,
                    valid_from_ns=bar.close_ts,
                    version=spec.version,
                )
            )
            produced[spec.name] = value
        return produced

    def _compute(self, spec: FeatureSpec, closes: list[Decimal], bars: list[Bar]) -> Decimal | None:
        if spec.kind == "sma":
            return sma(closes, spec.period)
        if spec.kind == "ema":
            return ema(closes, spec.period)
        if spec.kind == "atr":
            return atr(bars, spec.period)
        return rsi(closes, spec.period)
