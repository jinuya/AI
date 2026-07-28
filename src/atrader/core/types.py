"""Core enumerations shared across every component.

This module is dependency-free by contract (see the ``core is dependency-free``
import-linter contract in ``pyproject.toml``): everything may import it, it
imports nothing from the rest of the package.
"""

from __future__ import annotations

from enum import StrEnum


class Side(StrEnum):
    """Direction of an order or intent."""

    BUY = "BUY"
    SELL = "SELL"

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def sign(self) -> int:
        """+1 for BUY, -1 for SELL. Used for signed position arithmetic."""
        return 1 if self is Side.BUY else -1


class OrderType(StrEnum):
    """Spec §FR-EXE-02. MARKET is disabled by default in configuration."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class TimeInForce(StrEnum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class OrderStatus(StrEnum):
    """Spec §FR-EXE-01. Transitions are enforced in ``execution.statemachine``."""

    PENDING_NEW = "PENDING_NEW"
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    PENDING_CANCEL = "PENDING_CANCEL"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_STATUSES

    @property
    def is_open(self) -> bool:
        """True when the order may still receive fills at the broker."""
        return self in _OPEN_STATUSES


_TERMINAL_STATUSES: frozenset[OrderStatus] = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
    }
)

_OPEN_STATUSES: frozenset[OrderStatus] = frozenset(
    {
        OrderStatus.PENDING_NEW,
        OrderStatus.NEW,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.PENDING_CANCEL,
    }
)


class Urgency(StrEnum):
    """How aggressively the execution engine should work an intent."""

    PASSIVE = "PASSIVE"
    NORMAL = "NORMAL"
    AGGRESSIVE = "AGGRESSIVE"


class TargetType(StrEnum):
    """How ``TradingIntent.target_value`` should be interpreted."""

    SHARES = "SHARES"
    NOTIONAL = "NOTIONAL"
    TARGET_WEIGHT = "TARGET_WEIGHT"


class DataQuality(StrEnum):
    """Spec §FR-MD-03. Anything other than OK blocks new orders for the symbol."""

    OK = "OK"
    DEGRADED = "DEGRADED"
    STALE = "STALE"


class SystemState(StrEnum):
    """Global run state. Only RUNNING permits exposure-increasing orders."""

    STARTING = "STARTING"
    RUNNING = "RUNNING"
    THROTTLED = "THROTTLED"
    """Circuit breaker L1: no new entries, existing positions still managed."""
    BLOCKED = "BLOCKED"
    """Circuit breaker L2: no new orders at all; liquidation still allowed."""
    LIQUIDATING = "LIQUIDATING"
    """Circuit breaker L3: flattening everything."""
    HALTED = "HALTED"
    """Kill switch engaged or unrecoverable state. Nothing goes out."""


class RiskAction(StrEnum):
    """What the risk engine decided to do with an intent."""

    ALLOW = "ALLOW"
    REDUCE = "REDUCE"
    THROTTLE = "THROTTLE"
    QUEUE = "QUEUE"
    REJECT = "REJECT"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"


class AlertLevel(StrEnum):
    """Spec §FR-MON-02."""

    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


class BreakerLevel(StrEnum):
    """Spec §7.5."""

    NONE = "NONE"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


class AssetClass(StrEnum):
    EQUITY = "EQUITY"
    ETF = "ETF"
    FUTURE = "FUTURE"
    CRYPTO = "CRYPTO"


class CostBasisMethod(StrEnum):
    """Spec §FR-PF-01. Must match tax reporting."""

    FIFO = "FIFO"
    AVERAGE = "AVERAGE"


class SizingMethod(StrEnum):
    FIXED = "fixed"
    VOLATILITY_TARGET = "volatility_target"
    KELLY = "kelly"
