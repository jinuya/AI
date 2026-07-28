"""Pre-trade check primitives.

Every check in :mod:`atrader.risk.checks.pretrade` is a **pure function** of a
:class:`CheckContext`. No I/O, no clock reads, no service calls. Three things
follow from that, and all three matter:

* A check can be tested exhaustively without starting the system.
* Two checks in one evaluation cannot see different states — otherwise two
  orders that each looked fine could together cross a limit.
* Replay is deterministic, because the checks add no ambient inputs.

**The liquidation exemption.** Spec §7.4:

    스톱 주문 자체도 리스크 체크를 통과해야 하지만, 청산 방향 주문은 항상 허용된다.
    손실 한도에 걸려서 손절 주문이 거부되는 상황은 절대 만들면 안 된다.
    리스크 체크는 "익스포저를 늘리는 주문"에만 적용한다.

This is not a nicety. Loss limits exist to stop losses growing; a loss limit
that blocks the stop-loss order does the exact opposite of its purpose, and does
it precisely when it hurts most. Each check therefore declares — via
:attr:`RiskCheck.applies_to_reduce_only` — whether it is about *growing* risk
(skipped when reducing) or about *correctness* (always enforced: a fat-fingered
stop is still a fat finger, and a self-cross is still a regulatory problem
whichever way it points).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal

from atrader.config.schema import InstrumentSpec, RiskConfig
from atrader.core.models import TradingIntent
from atrader.core.money import ZERO
from atrader.core.types import AlertLevel, RiskAction
from atrader.risk.state import RiskSnapshot

__all__ = ["CheckContext", "RiskCheck", "RiskCheckResult", "allow", "reduce_to", "reject"]


@dataclass(frozen=True, slots=True)
class CheckContext:
    """Everything a check may look at."""

    intent: TradingIntent
    snapshot: RiskSnapshot
    config: RiskConfig
    quantity: Decimal
    """Resolved share count. Sizing has already run."""
    price: Decimal
    """Reference price used for notional and concentration maths."""
    instrument: InstrumentSpec | None = None
    reduce_only: bool = False
    """True when the order can only shrink exposure. Drives the §7.4 exemption."""

    @property
    def symbol(self) -> str:
        return self.intent.symbol

    @property
    def notional(self) -> Decimal:
        return abs(self.quantity * self.price)

    @property
    def equity(self) -> Decimal:
        return self.snapshot.account.equity

    def notional_pct_of_equity(self, notional: Decimal | None = None) -> Decimal:
        if self.equity <= ZERO:
            # No equity means every order is infinitely large relative to the
            # account. Returning 100% makes the size checks reject rather than
            # divide by zero.
            return Decimal(100)
        return (notional if notional is not None else self.notional) / self.equity * Decimal(100)

    def with_quantity(self, quantity: Decimal) -> CheckContext:
        """A copy at a reduced size, for re-running checks after a REDUCE."""
        return CheckContext(
            intent=self.intent,
            snapshot=self.snapshot,
            config=self.config,
            quantity=quantity,
            price=self.price,
            instrument=self.instrument,
            reduce_only=self.reduce_only,
        )


@dataclass(frozen=True, slots=True)
class RiskCheckResult:
    """One check's verdict.

    ``reason`` is written for whoever reads it during an incident, so it states
    the observed value *and* the limit rather than just "rejected".
    """

    name: str
    passed: bool
    action: RiskAction = RiskAction.ALLOW
    reason: str = ""
    adjusted_quantity: Decimal | None = None
    """Set with :data:`RiskAction.REDUCE` — the largest size that would pass."""
    alert_level: AlertLevel | None = None
    detail: dict[str, str] = field(default_factory=dict)

    @property
    def is_blocking(self) -> bool:
        return self.action in (RiskAction.REJECT, RiskAction.THROTTLE, RiskAction.QUEUE)


def allow(name: str) -> RiskCheckResult:
    return RiskCheckResult(name=name, passed=True, action=RiskAction.ALLOW)


def reject(
    name: str,
    reason: str,
    *,
    alert_level: AlertLevel | None = None,
    action: RiskAction = RiskAction.REJECT,
    detail: dict[str, str] | None = None,
) -> RiskCheckResult:
    return RiskCheckResult(
        name=name,
        passed=False,
        action=action,
        reason=reason,
        alert_level=alert_level,
        detail=detail or {},
    )


def reduce_to(name: str, quantity: Decimal, reason: str) -> RiskCheckResult:
    """Shrink the order to the largest size that would pass.

    Preferred over rejection where the intent is still valid at a smaller size —
    a strategy that wanted 10% of the account and is allowed 8% is better served
    by 8% than by nothing.
    """
    if quantity <= ZERO:
        return reject(name, f"{reason} (no size remains)")
    return RiskCheckResult(
        name=name,
        passed=False,
        action=RiskAction.REDUCE,
        reason=reason,
        adjusted_quantity=quantity,
    )


@dataclass(frozen=True, slots=True)
class RiskCheck:
    """A named check plus its liquidation policy."""

    name: str
    func: Callable[[CheckContext], RiskCheckResult]
    applies_to_reduce_only: bool
    """False for checks that limit *growth* in risk. Spec §7.4: those must not
    block an order that reduces exposure."""
    description: str = ""

    def __call__(self, ctx: CheckContext) -> RiskCheckResult:
        if ctx.reduce_only and not self.applies_to_reduce_only:
            return allow(self.name)
        return self.func(ctx)
