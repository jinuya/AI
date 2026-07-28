"""Primary/backup feed with divergence detection — spec §5.3.

    주 소스와 백업 소스를 이원화한다. 주 소스가 죽었을 때 자동 페일오버하되,
    두 소스의 가격이 일정 이상(기본 0.5%) 벌어지면 신규 주문을 중단한다.
    어느 쪽이 맞는지 모르는 상태에서 거래하는 게 제일 위험하다.

That last sentence is the design. When two feeds disagree, the tempting move is
to pick one — the primary, the more recent, the one closer to yesterday's close.
All of those are guesses. This class does not guess: it marks the symbol
``DEGRADED``, which stops new orders on it (spec §7.2 check 3) while leaving
liquidation available.

Failover, by contrast, is unambiguous. If the primary is not producing data at
all there is nothing to disagree with, so the backup is simply used.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from atrader.config.schema import DataQualityConfig
from atrader.core.clock import NS_PER_SECOND, Clock
from atrader.core.money import ZERO, as_pct
from atrader.marketdata.feeds.protocol import FeedStatus, MarketDataFeed
from atrader.marketdata.models import Tick
from atrader.marketdata.quality import QualityMonitor

__all__ = ["DivergenceEvent", "DualSourceMonitor"]

DIVERGENCE_ISSUE = "dual_source_divergence"


@dataclass(frozen=True, slots=True)
class DivergenceEvent:
    """Two sources disagreed on a symbol's price beyond the threshold."""

    symbol: str
    primary_price: Decimal
    backup_price: Decimal
    divergence_pct: Decimal
    threshold_pct: Decimal
    at_ns: int

    @property
    def message(self) -> str:
        return (
            f"{self.symbol}: primary {self.primary_price} vs backup {self.backup_price} "
            f"= {self.divergence_pct:.3f}% apart (threshold {self.threshold_pct}%). "
            "New orders blocked — we no longer know which price is right."
        )


class DualSourceMonitor:
    """Compares two feeds and degrades symbols that disagree.

    Deliberately *not* a feed itself. It observes ticks from both sources and
    reports; the pipeline decides what to publish. Keeping it passive means the
    comparison logic can be unit-tested without any async plumbing, and a bug
    here cannot swallow market data.
    """

    __slots__ = (
        "_backup_prices",
        "_config",
        "_divergences",
        "_max_age_ns",
        "_monitor",
        "_primary_prices",
    )

    def __init__(
        self,
        quality_monitor: QualityMonitor,
        config: DataQualityConfig | None = None,
        *,
        max_comparison_age_seconds: int = 5,
    ) -> None:
        self._monitor = quality_monitor
        self._config = config or DataQualityConfig()
        self._primary_prices: dict[str, tuple[Decimal, int]] = {}
        self._backup_prices: dict[str, tuple[Decimal, int]] = {}
        self._divergences: list[DivergenceEvent] = []
        self._max_age_ns = max_comparison_age_seconds * NS_PER_SECOND

    @property
    def divergences(self) -> list[DivergenceEvent]:
        return list(self._divergences)

    def observe_primary(self, tick: Tick) -> DivergenceEvent | None:
        return self._observe(tick, self._primary_prices, self._backup_prices, primary=True)

    def observe_backup(self, tick: Tick) -> DivergenceEvent | None:
        return self._observe(tick, self._backup_prices, self._primary_prices, primary=False)

    def _observe(
        self,
        tick: Tick,
        own: dict[str, tuple[Decimal, int]],
        other: dict[str, tuple[Decimal, int]],
        *,
        primary: bool,
    ) -> DivergenceEvent | None:
        price = tick.reference_price
        if price is None or price <= ZERO:
            return None
        own[tick.symbol] = (price, tick.ingest_ts)

        counterpart = other.get(tick.symbol)
        if counterpart is None:
            return None
        other_price, other_ts = counterpart

        # Comparing against a stale quote produces false divergences during
        # normal fast moves, which would block trading for no reason.
        if abs(tick.ingest_ts - other_ts) > self._max_age_ns:
            return None

        reference = max(price, other_price)
        divergence = abs(as_pct(price - other_price, reference))
        if divergence <= self._config.dual_source_divergence_pct:
            return None

        event = DivergenceEvent(
            symbol=tick.symbol,
            primary_price=price if primary else other_price,
            backup_price=other_price if primary else price,
            divergence_pct=divergence,
            threshold_pct=self._config.dual_source_divergence_pct,
            at_ns=tick.ingest_ts,
        )
        self._divergences.append(event)
        self._monitor.force_degrade(tick.symbol, DIVERGENCE_ISSUE)
        return event

    def clear(self, symbol: str) -> None:
        """Forget a symbol's cached prices — used when a source reconnects."""
        self._primary_prices.pop(symbol, None)
        self._backup_prices.pop(symbol, None)


def choose_active_feed(
    primary: MarketDataFeed,
    backup: MarketDataFeed | None,
    *,
    clock: Clock,
    last_primary_tick_ns: int | None,
    stale_after_seconds: int = 10,
) -> tuple[MarketDataFeed, str]:
    """Pick which feed to read, with the reason.

    Failover is only for a primary that has stopped producing. A primary that is
    producing *wrong* data is the divergence case above, and switching sources
    there would just be guessing.
    """
    if backup is None:
        return primary, "no backup configured"
    if primary.status in (FeedStatus.DISCONNECTED, FeedStatus.EXHAUSTED):
        return backup, f"primary is {primary.status}"
    if last_primary_tick_ns is None:
        return primary, "primary has not produced yet"
    idle_seconds = (clock.now_ns() - last_primary_tick_ns) / NS_PER_SECOND
    if idle_seconds > stale_after_seconds:
        return backup, f"primary silent for {idle_seconds:.1f}s"
    return primary, "primary healthy"
