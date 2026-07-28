"""Gap detection and backfill — spec §FR-MD-04.

After a restart or a dropped connection there is a hole in the bar series. This
module finds the hole and merges a REST backfill into it.

The precedence rule from the spec is the important part:

    백필 데이터와 실시간 데이터가 겹치는 구간은 실시간을 우선한다.

Live data wins on overlap. A REST endpoint may serve a consolidated or slightly
revised bar, and quietly replacing a bar a strategy already traded on would
rewrite history — the backtest and the live run would then disagree about what
was known at that moment, which is exactly the class of bug §8.1 exists to
prevent.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise

from atrader.marketdata.aggregator import interval_to_ns
from atrader.marketdata.models import Bar

__all__ = ["BackfillPlan", "Gap", "find_gaps", "merge_bars"]


@dataclass(frozen=True, slots=True)
class Gap:
    """A missing stretch of bars."""

    symbol: str
    interval: str
    start_ns: int
    """First missing bucket's open timestamp."""
    end_ns: int
    """Last missing bucket's open timestamp."""

    @property
    def bar_count(self) -> int:
        step = interval_to_ns(self.interval)
        return (self.end_ns - self.start_ns) // step + 1


@dataclass(frozen=True, slots=True)
class BackfillPlan:
    """What needs fetching after a disconnect."""

    gaps: tuple[Gap, ...]

    @property
    def is_empty(self) -> bool:
        return not self.gaps

    @property
    def total_bars(self) -> int:
        return sum(gap.bar_count for gap in self.gaps)

    def describe(self) -> str:
        if self.is_empty:
            return "no gaps"
        return "; ".join(
            f"{gap.symbol} {gap.interval}: {gap.bar_count} bar(s) from {gap.start_ns}"
            for gap in self.gaps
        )


def find_gaps(bars: Sequence[Bar], *, expected_end_ns: int | None = None) -> BackfillPlan:
    """Find missing buckets in a series of bars for one symbol and interval.

    Bars need not be sorted. ``expected_end_ns`` lets the caller assert that the
    series runs up to *now* — without it, a feed that died an hour ago looks
    complete, because the last bar it delivered is still the last bar present.
    """
    if not bars:
        return BackfillPlan(())

    symbols = {bar.symbol for bar in bars}
    intervals = {bar.interval for bar in bars}
    if len(symbols) != 1 or len(intervals) != 1:
        raise ValueError(
            f"find_gaps handles one symbol and interval at a time, got "
            f"{sorted(symbols)} / {sorted(intervals)}"
        )

    symbol = next(iter(symbols))
    interval = next(iter(intervals))
    step = interval_to_ns(interval)

    ordered = sorted(bars, key=lambda bar: bar.open_ts)
    gaps: list[Gap] = []

    for previous, current in pairwise(ordered):
        expected = previous.open_ts + step
        if current.open_ts > expected:
            gaps.append(Gap(symbol, interval, expected, current.open_ts - step))

    if expected_end_ns is not None:
        last_open = ordered[-1].open_ts
        # The bucket containing expected_end_ns is still forming, so the last
        # bar we can legitimately expect is the one before it.
        current_bucket = expected_end_ns - (expected_end_ns % step)
        gaps.append(Gap(symbol, interval, last_open + step, current_bucket - step))

    return BackfillPlan(tuple(gap for gap in gaps if gap.end_ns >= gap.start_ns))


def merge_bars(live: Iterable[Bar], backfilled: Iterable[Bar]) -> list[Bar]:
    """Merge backfilled bars into live ones. **Live wins on overlap.**

    Spec §FR-MD-04. Replacing a bar a strategy already acted on would rewrite
    history; the backfill exists to fill holes, not to revise the record.
    """
    merged: dict[tuple[str, str, int], Bar] = {}

    for bar in backfilled:
        merged[(bar.symbol, bar.interval, bar.open_ts)] = bar
    for bar in live:
        # Unconditional overwrite: this is the precedence rule.
        merged[(bar.symbol, bar.interval, bar.open_ts)] = bar

    return sorted(merged.values(), key=lambda bar: (bar.symbol, bar.interval, bar.open_ts))
