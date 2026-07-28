"""Point-in-time feature store — spec §8.2.

Look-ahead bias is the most common way a backtest lies to you: a feature
computed from a bar that has not closed yet, or read as though it were known
before the moment it actually became knowable, lets a strategy trade on
information it would not have had live.

The fix here is structural rather than a discipline strategies have to
remember. Every value is stored with the timestamp at which it became
knowable — ``valid_from_ns``, which is the *closing* timestamp of the bar it
was computed from, never the bar's open — and :meth:`FeatureStore.asof`
refuses to return anything whose ``valid_from_ns`` is after the time being
asked about. A strategy can only ever see what would genuinely have been on
the tape by then.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from decimal import Decimal

__all__ = ["FeatureStore", "FeatureValue"]


@dataclass(frozen=True, slots=True)
class FeatureValue:
    symbol: str
    name: str
    value: Decimal
    valid_from_ns: int
    """When this value became knowable — a bar's ``close_ts``, not its ``open_ts``."""
    version: int


@dataclass
class FeatureStore:
    """In-memory point-in-time store.

    Each ``(symbol, name)`` key holds its history in non-decreasing
    ``valid_from_ns`` order, which is what lets :meth:`asof` binary-search
    instead of scanning. :meth:`publish` enforces that ordering rather than
    assuming it: an out-of-order publish would silently break every asof
    lookup after it.
    """

    _series: dict[tuple[str, str], list[FeatureValue]] = field(default_factory=dict)

    def publish(self, value: FeatureValue) -> None:
        key = (value.symbol, value.name)
        series = self._series.setdefault(key, [])
        if series and value.valid_from_ns < series[-1].valid_from_ns:
            raise ValueError(
                f"{value.symbol}/{value.name}: publishing out of order — "
                f"valid_from_ns {value.valid_from_ns} precedes the last published "
                f"{series[-1].valid_from_ns}. Features must be published in the "
                "order their source bars close."
            )
        series.append(value)

    def asof(self, symbol: str, name: str, at_ns: int) -> FeatureValue | None:
        """The most recent value knowable at or before *at_ns*.

        ``None`` when nothing was published yet, or everything published so
        far is from *after* ``at_ns`` — which is exactly what "not knowable
        yet" should return, not a stale guess.
        """
        series = self._series.get((symbol, name))
        if not series:
            return None
        index = bisect.bisect_right(series, at_ns, key=lambda v: v.valid_from_ns) - 1
        return series[index] if index >= 0 else None

    def latest(self, symbol: str, name: str) -> FeatureValue | None:
        """The most recently published value, ignoring point-in-time semantics.

        For monitoring and dashboards — never for a trading decision, which
        must always go through :meth:`asof`.
        """
        series = self._series.get((symbol, name))
        return series[-1] if series else None

    def history(self, symbol: str, name: str) -> tuple[FeatureValue, ...]:
        return tuple(self._series.get((symbol, name), ()))
