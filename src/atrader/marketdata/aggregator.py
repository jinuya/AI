"""Tick-to-bar aggregation — spec §FR-SIG-01.

Bars are built incrementally rather than recomputed, because in a streaming
system recomputation is quadratic and the whole point of a 1-minute bar is that
it is cheap.

The important behaviour is the ``is_final`` transition. A forming bar is emitted
on every tick with ``is_final=False`` so a dashboard can show it, and exactly one
``is_final=True`` bar is emitted when the interval closes. Strategies act only on
the final one — spec §5.2 spells out why: signals computed on a forming bar keep
changing until it closes, and the backtest, which only ever sees closed bars,
would disagree with live trading.

A bar closes when a tick arrives that belongs to a *later* interval, or when
:meth:`BarAggregator.close_expired` is called on a timer. The timer matters for
illiquid names: with no ticks, the interval boundary would otherwise never be
noticed and the bar would stay open indefinitely.
"""

from __future__ import annotations

from decimal import Decimal

from atrader.core.clock import NS_PER_SECOND
from atrader.core.money import ZERO
from atrader.marketdata.models import Bar, Tick

__all__ = ["INTERVAL_NS", "BarAggregator", "BarBuilder", "interval_to_ns"]

#: Supported bar intervals, in nanoseconds (spec §5.1).
INTERVAL_NS: dict[str, int] = {
    "1s": NS_PER_SECOND,
    "1m": 60 * NS_PER_SECOND,
    "5m": 300 * NS_PER_SECOND,
    "15m": 900 * NS_PER_SECOND,
    "1h": 3_600 * NS_PER_SECOND,
    "1d": 86_400 * NS_PER_SECOND,
}


def interval_to_ns(interval: str) -> int:
    try:
        return INTERVAL_NS[interval]
    except KeyError:
        raise ValueError(
            f"unsupported bar interval {interval!r}; supported: {sorted(INTERVAL_NS)}"
        ) from None


class BarBuilder:
    """Accumulates ticks into one bar for a single symbol and interval."""

    __slots__ = (
        "_close",
        "_close_ts",
        "_high",
        "_interval",
        "_interval_ns",
        "_low",
        "_open",
        "_open_ts",
        "_source",
        "_symbol",
        "_trade_count",
        "_volume",
        "_vwap_numerator",
    )

    def __init__(
        self, symbol: str, interval: str, bucket_start_ns: int, source: str = "unknown"
    ) -> None:
        self._symbol = symbol
        self._interval = interval
        self._interval_ns = interval_to_ns(interval)
        self._open_ts = bucket_start_ns
        self._close_ts = bucket_start_ns + self._interval_ns - 1
        self._source = source
        self._open: Decimal | None = None
        self._high: Decimal | None = None
        self._low: Decimal | None = None
        self._close: Decimal | None = None
        self._volume = ZERO
        self._vwap_numerator = ZERO
        self._trade_count = 0

    @property
    def open_ts(self) -> int:
        return self._open_ts

    @property
    def has_data(self) -> bool:
        return self._open is not None

    def add(self, tick: Tick) -> None:
        """Fold a tick in. Quote-only ticks update price but not volume."""
        price = tick.last if tick.last is not None else tick.reference_price
        if price is None:
            return

        if self._open is None:
            self._open = price
            self._high = price
            self._low = price
        else:
            assert self._high is not None and self._low is not None
            self._high = max(self._high, price)
            self._low = min(self._low, price)
        self._close = price

        if tick.last is not None and tick.last_size is not None and tick.last_size > ZERO:
            self._volume += tick.last_size
            self._vwap_numerator += price * tick.last_size
            self._trade_count += 1

    def build(self, *, is_final: bool) -> Bar | None:
        """Snapshot the current state. ``None`` when no tick has landed yet."""
        if self._open is None or self._high is None or self._low is None or self._close is None:
            return None
        vwap = self._vwap_numerator / self._volume if self._volume > ZERO else None
        # VWAP can land a hair outside [low, high] from decimal division; the
        # Bar validator rejects that, so clamp rather than emit an invalid bar.
        if vwap is not None:
            vwap = min(max(vwap, self._low), self._high)
        return Bar(
            symbol=self._symbol,
            interval=self._interval,
            open_ts=self._open_ts,
            close_ts=self._close_ts,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
            vwap=vwap,
            trade_count=self._trade_count or None,
            is_final=is_final,
            source=self._source,
        )


class BarAggregator:
    """Builds bars for many symbols across several intervals at once."""

    __slots__ = ("_builders", "_intervals")

    def __init__(self, intervals: tuple[str, ...] = ("1m",)) -> None:
        if not intervals:
            raise ValueError("at least one interval is required")
        for interval in intervals:
            interval_to_ns(interval)
        self._intervals = intervals
        self._builders: dict[tuple[str, str], BarBuilder] = {}

    @property
    def intervals(self) -> tuple[str, ...]:
        return self._intervals

    @staticmethod
    def bucket_start(timestamp_ns: int, interval_ns: int) -> int:
        """Floor a timestamp to its interval boundary."""
        return timestamp_ns - (timestamp_ns % interval_ns)

    def add(self, tick: Tick) -> list[Bar]:
        """Fold a tick in and return any bars this completed.

        Returned bars are always ``is_final=True``. Use :meth:`forming` to look
        at the in-progress bar — the separation is deliberate, so a caller
        cannot accidentally treat an unfinished bar as tradable.
        """
        completed: list[Bar] = []
        for interval in self._intervals:
            interval_ns = interval_to_ns(interval)
            bucket = self.bucket_start(tick.exchange_ts, interval_ns)
            key = (tick.symbol, interval)
            builder = self._builders.get(key)

            if builder is not None and builder.open_ts != bucket:
                if builder.open_ts > bucket:
                    # Late tick for an already-closed bar. Dropping it is the
                    # honest choice: re-opening a published bar would rewrite
                    # history a strategy has already acted on.
                    continue
                finished = builder.build(is_final=True)
                if finished is not None:
                    completed.append(finished)
                builder = None

            if builder is None:
                builder = BarBuilder(tick.symbol, interval, bucket, tick.source)
                self._builders[key] = builder
            builder.add(tick)
        return completed

    def forming(self, symbol: str, interval: str) -> Bar | None:
        """The in-progress bar, marked ``is_final=False``. Display only."""
        builder = self._builders.get((symbol, interval))
        return None if builder is None else builder.build(is_final=False)

    def close_expired(self, now_ns: int) -> list[Bar]:
        """Close bars whose interval has elapsed.

        Needed because a thin symbol may receive no tick to trigger the
        boundary, and a bar that never closes never reaches the strategy.
        """
        completed: list[Bar] = []
        for key, builder in list(self._builders.items()):
            _, interval = key
            interval_ns = interval_to_ns(interval)
            if now_ns >= builder.open_ts + interval_ns:
                finished = builder.build(is_final=True)
                if finished is not None:
                    completed.append(finished)
                del self._builders[key]
        return completed

    def flush(self) -> list[Bar]:
        """Close everything. Used at end of session and end of backtest."""
        completed = [
            bar
            for builder in self._builders.values()
            if (bar := builder.build(is_final=True)) is not None
        ]
        self._builders.clear()
        return completed
