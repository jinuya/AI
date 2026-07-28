"""Feed quality validation — spec §FR-MD-03.

Five checks run on every tick as it arrives: sequence gaps, backwards
timestamps, crossed markets, abnormal price jumps, and staleness. A violation
marks the symbol ``DEGRADED`` (or ``STALE``), and the risk engine refuses new
orders on a symbol that is not ``OK`` (spec §7.2 check 3).

This runs *before* anything reaches a strategy, per the pipeline order in
spec §5.4: ingest → validate → normalise → store → publish. Validating after
publication would mean a strategy had already acted on the bad tick.

Degradation is sticky until the feed proves itself again — a symbol is only
restored to ``OK`` after :attr:`QualityMonitor.recovery_ticks` consecutive clean
updates. A feed that alternates good and bad ticks would otherwise flap between
tradable and blocked, which is worse than staying blocked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from atrader.config.schema import DataQualityConfig
from atrader.core.clock import NS_PER_SECOND
from atrader.core.money import ZERO, as_pct
from atrader.core.types import DataQuality
from atrader.marketdata.models import Tick

__all__ = ["QualityIssue", "QualityMonitor", "QualityReport", "SymbolQualityState"]

DEFAULT_RECOVERY_TICKS = 3


class QualityIssue:
    """Reasons a tick can be rejected. Recorded verbatim in the audit log."""

    SEQUENCE_GAP = "sequence_gap"
    SEQUENCE_REGRESSION = "sequence_regression"
    TIMESTAMP_REGRESSION = "timestamp_regression"
    CROSSED_MARKET = "crossed_market"
    PRICE_JUMP = "price_jump"
    STALE = "stale"
    EXCESSIVE_LAG = "excessive_lag"
    CLOCK_DRIFT = "clock_drift"


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Verdict on one tick."""

    symbol: str
    quality: DataQuality
    issues: tuple[str, ...] = ()
    detail: str = ""

    @property
    def is_ok(self) -> bool:
        return self.quality is DataQuality.OK


@dataclass(slots=True)
class SymbolQualityState:
    """Rolling per-symbol state the checks compare against."""

    last_seq: int | None = None
    last_exchange_ts: int | None = None
    last_ingest_ts: int | None = None
    last_price: Decimal | None = None
    quality: DataQuality = DataQuality.OK
    clean_streak: int = 0
    issue_counts: dict[str, int] = field(default_factory=dict)

    def record_issue(self, issue: str) -> None:
        self.issue_counts[issue] = self.issue_counts.get(issue, 0) + 1


class QualityMonitor:
    """Validates ticks and tracks each symbol's tradability."""

    __slots__ = ("_config", "_recovery_ticks", "_states")

    def __init__(
        self,
        config: DataQualityConfig | None = None,
        *,
        recovery_ticks: int = DEFAULT_RECOVERY_TICKS,
    ) -> None:
        self._config = config or DataQualityConfig()
        self._recovery_ticks = recovery_ticks
        self._states: dict[str, SymbolQualityState] = {}

    def state(self, symbol: str) -> SymbolQualityState:
        return self._states.setdefault(symbol, SymbolQualityState())

    def quality_of(self, symbol: str) -> DataQuality:
        """Current verdict for a symbol. Unknown symbols are treated as STALE:
        we have never seen data, so we certainly should not trade on it."""
        state = self._states.get(symbol)
        return DataQuality.STALE if state is None else state.quality

    def is_tradable(self, symbol: str) -> bool:
        return self.quality_of(symbol) is DataQuality.OK

    def check(self, tick: Tick) -> QualityReport:
        """Validate a tick and update the symbol's state."""
        state = self.state(tick.symbol)
        issues: list[str] = []
        details: list[str] = []

        # --- sequence continuity -------------------------------------------
        if tick.seq is not None and state.last_seq is not None:
            delta = tick.seq - state.last_seq
            if delta <= 0:
                issues.append(QualityIssue.SEQUENCE_REGRESSION)
                details.append(f"seq went {state.last_seq} -> {tick.seq}")
            elif delta - 1 > self._config.max_sequence_gap:
                issues.append(QualityIssue.SEQUENCE_GAP)
                details.append(
                    f"missed {delta - 1} message(s) between {state.last_seq} and {tick.seq}"
                )

        # --- timestamp monotonicity ----------------------------------------
        if state.last_exchange_ts is not None and tick.exchange_ts < state.last_exchange_ts:
            issues.append(QualityIssue.TIMESTAMP_REGRESSION)
            details.append(
                f"exchange_ts went backwards by {state.last_exchange_ts - tick.exchange_ts}ns"
            )

        # --- crossed market -------------------------------------------------
        if tick.is_crossed:
            issues.append(QualityIssue.CROSSED_MARKET)
            details.append(f"bid {tick.bid} >= ask {tick.ask}")

        # --- abnormal price jump --------------------------------------------
        price = tick.reference_price
        if price is not None and state.last_price is not None and state.last_price > ZERO:
            move = abs(as_pct(price - state.last_price, state.last_price))
            if move > self._config.price_jump_threshold_pct:
                issues.append(QualityIssue.PRICE_JUMP)
                details.append(
                    f"price moved {move:.2f}% ({state.last_price} -> {price}), "
                    f"threshold {self._config.price_jump_threshold_pct}%"
                )

        # --- feed lag --------------------------------------------------------
        lag_ms = tick.lag_ns / 1_000_000
        if lag_ms > self._config.max_feed_lag_ms:
            issues.append(QualityIssue.EXCESSIVE_LAG)
            details.append(f"feed lag {lag_ms:.0f}ms exceeds {self._config.max_feed_lag_ms}ms")
        elif lag_ms < -self._config.clock_drift_alert_ms:
            # A tick that arrives "before" it was sent means our clock is behind
            # the venue's. Spec §5.4 wants that alerted, not silently accepted.
            issues.append(QualityIssue.CLOCK_DRIFT)
            details.append(f"tick arrived {-lag_ms:.0f}ms before its exchange timestamp")

        quality = DataQuality.OK if not issues else DataQuality.DEGRADED
        for issue in issues:
            state.record_issue(issue)

        # Only advance the reference state on clean data. Learning the price
        # from a bad tick would make the *next* jump check compare against a
        # value we already decided not to trust.
        if quality is DataQuality.OK:
            state.clean_streak += 1
            state.last_seq = tick.seq if tick.seq is not None else state.last_seq
            state.last_exchange_ts = tick.exchange_ts
            state.last_ingest_ts = tick.ingest_ts
            if price is not None:
                state.last_price = price
            if state.quality is not DataQuality.OK and state.clean_streak >= self._recovery_ticks:
                state.quality = DataQuality.OK
        else:
            state.clean_streak = 0
            state.quality = DataQuality.DEGRADED
            # Sequence and timestamp still advance so one gap does not make
            # every subsequent tick look like a gap too.
            if tick.seq is not None:
                state.last_seq = tick.seq
            state.last_exchange_ts = max(state.last_exchange_ts or 0, tick.exchange_ts)
            state.last_ingest_ts = tick.ingest_ts

        return QualityReport(
            symbol=tick.symbol,
            quality=state.quality,
            issues=tuple(issues),
            detail="; ".join(details),
        )

    def check_staleness(self, symbol: str, now_ns: int) -> QualityReport:
        """Mark a symbol STALE when nothing has arrived for too long.

        Called on a timer, not on receipt — the whole point is that no tick is
        arriving, so nothing else would trigger it.
        """
        state = self._states.get(symbol)
        if state is None or state.last_ingest_ts is None:
            return QualityReport(symbol, DataQuality.STALE, (QualityIssue.STALE,), "no data yet")

        idle_seconds = (now_ns - state.last_ingest_ts) / NS_PER_SECOND
        if idle_seconds > self._config.stale_threshold_seconds:
            state.quality = DataQuality.STALE
            state.clean_streak = 0
            state.record_issue(QualityIssue.STALE)
            return QualityReport(
                symbol,
                DataQuality.STALE,
                (QualityIssue.STALE,),
                f"no update for {idle_seconds:.1f}s "
                f"(threshold {self._config.stale_threshold_seconds}s)",
            )
        return QualityReport(symbol, state.quality)

    def sweep_staleness(self, now_ns: int) -> list[QualityReport]:
        """Check every known symbol. Returns only the ones that went stale."""
        return [
            report
            for report in (self.check_staleness(symbol, now_ns) for symbol in list(self._states))
            if report.quality is DataQuality.STALE
        ]

    def force_degrade(self, symbol: str, reason: str) -> QualityReport:
        """Block a symbol from outside the tick path.

        Used when two sources disagree (spec §5.3) — neither feed is
        individually faulty, but we no longer know which price is right, and
        trading in that state is the worst of the options.
        """
        state = self.state(symbol)
        state.quality = DataQuality.DEGRADED
        state.clean_streak = 0
        state.record_issue(reason)
        return QualityReport(symbol, DataQuality.DEGRADED, (reason,), reason)
