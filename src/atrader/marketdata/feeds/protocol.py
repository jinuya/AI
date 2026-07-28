"""Market data feed interface.

Every venue sits behind this, so nothing downstream knows whether it is reading
Polygon, a broker's own stream, or a recorded file (spec §5.3). Swapping the
source is what makes backtest, replay and live trading share one code path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from atrader.marketdata.models import Tick

__all__ = ["FeedStatus", "MarketDataFeed"]


class FeedStatus:
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DEGRADED = "degraded"
    """Connected but not trustworthy — e.g. diverging from the backup source."""
    EXHAUSTED = "exhausted"
    """A finite feed (replay, backtest) reached the end of its data."""


@runtime_checkable
class MarketDataFeed(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def status(self) -> str: ...

    async def connect(self, symbols: Sequence[str]) -> None:
        """Connect and subscribe. Idempotent — reconnection calls this again."""
        ...

    def __aiter__(self) -> AsyncIterator[Tick]:
        """Yield normalised ticks until the feed is closed or exhausted."""
        ...

    async def close(self) -> None: ...
