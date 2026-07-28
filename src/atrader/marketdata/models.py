"""Normalised market data records — spec §5.2.

Every venue names its fields differently and stamps time in its own units; the
adapter layer converts all of it to these two types so nothing downstream knows
which exchange it is reading (spec §FR-MD-02).

Both timestamps are kept. ``exchange_ts`` is when the venue says it happened,
``ingest_ts`` is when we received it, and the difference between them *is* the
feed lag metric (spec §9.2) — you cannot reconstruct it later if you only keep
one.

The ``is_final`` flag on :class:`Bar` matters more than it looks:

    아직 진행 중인 분봉으로 시그널을 만들면 봉이 마감될 때까지 시그널이 계속 바뀐다.
    백테스트에서는 확정된 봉만 보이므로 실거래와 결과가 달라진다.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atrader.core.types import DataQuality

__all__ = ["Bar", "Tick"]

PositivePrice = Annotated[Decimal, Field(gt=Decimal(0))]
NonNegative = Annotated[Decimal, Field(ge=Decimal(0))]


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Tick(_Record):
    """A quote and/or trade update."""

    symbol: str = Field(min_length=1, max_length=32)
    exchange_ts: int
    """UTC nanoseconds, as reported by the venue."""
    ingest_ts: int
    """UTC nanoseconds, when we received it."""
    bid: Decimal | None = None
    ask: Decimal | None = None
    bid_size: Decimal | None = None
    ask_size: Decimal | None = None
    last: Decimal | None = None
    last_size: Decimal | None = None
    seq: int | None = None
    """Venue sequence number, for gap detection."""
    source: str = "unknown"
    quality: DataQuality = DataQuality.OK

    @model_validator(mode="after")
    def _prices_are_positive(self) -> Tick:
        for name in ("bid", "ask", "last"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name in ("bid_size", "ask_size", "last_size"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative, got {value}")
        return self

    @property
    def lag_ns(self) -> int:
        """Feed latency. Negative means our clock is behind the venue's — spec
        §5.4 requires NTP and alerts above 100ms of drift, and a negative lag is
        how that shows up."""
        return self.ingest_ts - self.exchange_ts

    @property
    def is_crossed(self) -> bool:
        """Bid at or above ask. Never legitimate; means bad or stale data."""
        return self.bid is not None and self.ask is not None and self.bid >= self.ask

    @property
    def mid(self) -> Decimal | None:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid

    @property
    def reference_price(self) -> Decimal | None:
        """Best available price: mid, else last, else whichever side we have.

        Used by the risk engine's fat-finger check, so it must never guess.
        """
        mid = self.mid
        if mid is not None:
            return mid
        if self.last is not None:
            return self.last
        return self.bid if self.bid is not None else self.ask

    @property
    def is_tradable(self) -> bool:
        """Whether new orders may be placed on this symbol (spec §7.2 check 3)."""
        return self.quality is DataQuality.OK


class Bar(_Record):
    """An OHLCV aggregate."""

    symbol: str = Field(min_length=1, max_length=32)
    interval: str = Field(min_length=1, max_length=8)
    open_ts: int
    close_ts: int
    open: PositivePrice
    high: PositivePrice
    low: PositivePrice
    close: PositivePrice
    volume: NonNegative = Decimal(0)
    vwap: Decimal | None = None
    trade_count: int | None = None
    is_final: bool = False
    """False while the bar is still forming. **Never trade on a non-final bar** —
    the signal would keep changing until the bar closes, and the backtest, which
    only ever sees final bars, would disagree with live trading."""
    source: str = "unknown"

    @model_validator(mode="after")
    def _ohlc_is_coherent(self) -> Bar:
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) is below low ({self.low})")
        if not (self.low <= self.open <= self.high):
            raise ValueError(
                f"open ({self.open}) is outside [low, high] = [{self.low}, {self.high}]"
            )
        if not (self.low <= self.close <= self.high):
            raise ValueError(
                f"close ({self.close}) is outside [low, high] = [{self.low}, {self.high}]"
            )
        if self.close_ts < self.open_ts:
            raise ValueError(f"close_ts ({self.close_ts}) precedes open_ts ({self.open_ts})")
        if self.vwap is not None and not (self.low <= self.vwap <= self.high):
            raise ValueError(f"vwap ({self.vwap}) is outside the bar's range")
        return self

    @property
    def typical_price(self) -> Decimal:
        """(H + L + C) / 3 — the standard input to volume-weighted measures."""
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> Decimal:
        return self.high - self.low

    def with_final(self, *, is_final: bool = True) -> Bar:
        return self.model_copy(update={"is_final": is_final})
