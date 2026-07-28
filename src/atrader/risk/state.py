"""The state the pre-trade checks evaluate against.

Spec §7.2 checks read account equity, current positions, recent order rates and
today's P&L. Rather than have each check reach into a live service — which would
make them untestable and let two checks in one evaluation see different states —
the engine takes an immutable :class:`RiskSnapshot` and every check is a pure
function of it.

That matters for more than testability. A limit evaluated against a state that
shifts mid-evaluation is a limit that can be crossed by two orders that each
individually looked fine.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import pairwise

from atrader.core.clock import NS_PER_SECOND
from atrader.core.models import AccountState, Position
from atrader.core.types import BreakerLevel, DataQuality, Side, SystemState

__all__ = ["OrderRateTracker", "RecentOrder", "RiskSnapshot", "RiskState"]


@dataclass(frozen=True, slots=True)
class RecentOrder:
    """Enough of a recent order to detect duplicates and round trips."""

    symbol: str
    side: Side
    quantity: Decimal
    at_ns: int


@dataclass(frozen=True, slots=True)
class RiskSnapshot:
    """Immutable view of everything the pre-trade checks need.

    Built once per evaluation. Two checks in the same evaluation always see the
    same numbers.
    """

    now_ns: int
    system_state: SystemState
    breaker_level: BreakerLevel
    account: AccountState
    positions: dict[str, Position] = field(default_factory=dict)
    universe: frozenset[str] = frozenset()
    sectors: dict[str, str] = field(default_factory=dict)
    data_quality: dict[str, DataQuality] = field(default_factory=dict)
    prices: dict[str, Decimal] = field(default_factory=dict)
    adv: dict[str, Decimal] = field(default_factory=dict)
    """Average daily volume in shares, per symbol."""
    correlations: dict[tuple[str, str], Decimal] = field(default_factory=dict)
    open_orders: tuple[RecentOrder, ...] = ()
    """Working orders — needed for the self-cross check."""
    recent_orders: tuple[RecentOrder, ...] = ()
    orders_last_minute: int = 0
    baseline_orders_per_minute: Decimal = Decimal(0)
    daily_pnl_pct: Decimal = Decimal(0)
    """Today's P&L as a percentage of starting equity. Negative is a loss."""
    drawdown_pct: Decimal = Decimal(0)
    """Current drawdown from peak equity, as a positive percentage."""
    is_market_open: dict[str, bool] = field(default_factory=dict)
    kill_switch_engaged: bool = False
    reconciliation_break: bool = False
    """Spec §FR-EXE-05. Blocks new orders until a human resolves it."""
    margin_reduce_only: bool = False
    """Spec §FR-PF-04. Set by :class:`~atrader.portfolio.margin.MarginMonitor`
    when equity falls below the maintenance-margin reduce-only threshold.
    Unlike ``reconciliation_break``, this clears itself once margin recovers —
    see the module docstring in ``portfolio.margin`` for why."""

    def position_of(self, symbol: str) -> Position:
        return self.positions.get(symbol) or Position(symbol=symbol)

    def price_of(self, symbol: str) -> Decimal | None:
        return self.prices.get(symbol)

    def quality_of(self, symbol: str) -> DataQuality:
        # Unknown means we have no data, which is not the same as good data.
        return self.data_quality.get(symbol, DataQuality.STALE)

    def sector_of(self, symbol: str) -> str:
        return self.sectors.get(symbol, "UNKNOWN")

    def correlation(self, a: str, b: str) -> Decimal:
        if a == b:
            return Decimal(1)
        return self.correlations.get((a, b)) or self.correlations.get((b, a)) or Decimal(0)

    def gross_exposure(self) -> Decimal:
        return sum((p.exposure for p in self.positions.values()), Decimal(0))

    def sector_exposure(self, sector: str) -> Decimal:
        return sum(
            (p.exposure for s, p in self.positions.items() if self.sector_of(s) == sector),
            Decimal(0),
        )


class OrderRateTracker:
    """Sliding window of recent orders, for the rate and duplicate checks."""

    __slots__ = ("_orders", "_window_ns")

    def __init__(self, window_seconds: int = 60) -> None:
        self._window_ns = window_seconds * NS_PER_SECOND
        self._orders: deque[RecentOrder] = deque()

    def record(self, order: RecentOrder) -> None:
        self._orders.append(order)
        self._evict(order.at_ns)

    def _evict(self, now_ns: int) -> None:
        cutoff = now_ns - self._window_ns
        while self._orders and self._orders[0].at_ns < cutoff:
            self._orders.popleft()

    def count(self, now_ns: int) -> int:
        self._evict(now_ns)
        return len(self._orders)

    def recent(self, now_ns: int) -> tuple[RecentOrder, ...]:
        self._evict(now_ns)
        return tuple(self._orders)

    def roundtrips(self, symbol: str, now_ns: int) -> int:
        """Count direction changes for a symbol in the window.

        Repeated flip-flopping in one name is a strong signature of a strategy
        stuck in a loop (spec §7.5) — the account bleeds commission while the
        position goes nowhere.
        """
        self._evict(now_ns)
        sides = [order.side for order in self._orders if order.symbol == symbol]
        return sum(1 for a, b in pairwise(sides) if a is not b)


@dataclass(slots=True)
class RiskState:
    """Mutable running state the engine owns between evaluations."""

    system_state: SystemState = SystemState.STARTING
    breaker_level: BreakerLevel = BreakerLevel.NONE
    kill_switch_engaged: bool = False
    starting_equity: Decimal = Decimal(0)
    peak_equity: Decimal = Decimal(0)
    consecutive_losses: int = 0
    last_loss_ns: int | None = None
    breaker_tripped_ns: int | None = None
    reconciliation_break: bool = False
    """Set on a break; blocks new orders until a human clears it (§FR-EXE-05)."""

    def daily_pnl_pct(self, equity: Decimal) -> Decimal:
        if self.starting_equity <= 0:
            return Decimal(0)
        return (equity - self.starting_equity) / self.starting_equity * Decimal(100)

    def drawdown_pct(self, equity: Decimal) -> Decimal:
        if self.peak_equity <= 0:
            return Decimal(0)
        drop = self.peak_equity - equity
        return max(Decimal(0), drop / self.peak_equity * Decimal(100))

    def observe_equity(self, equity: Decimal) -> None:
        if self.starting_equity <= 0:
            self.starting_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    def start_new_day(self, equity: Decimal) -> None:
        """Reset the daily counters. Peak equity deliberately carries over —
        drawdown is measured from the all-time high, not from this morning."""
        self.starting_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        self.consecutive_losses = 0
        self.last_loss_ns = None
