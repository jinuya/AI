"""Margin monitoring — spec §FR-PF-04.

    유지증거금 비율이 120% 아래로 떨어지면 경고, 110% 아래로 떨어지면 신규
    주문을 축소 전용으로 전환한다.

Unlike the circuit breaker (§7.5), recovery here is automatic rather than
latched behind a human clearing it. A margin ratio is a mechanical fact about
current equity versus current requirement: once equity recovers above the
threshold, the account is not under-margined anymore, and there is nothing
left for a person to investigate that the ratio itself does not already show.
The circuit breaker exists because a *loss* is a signal that a strategy might
be misbehaving; a margin ratio recovering is just arithmetic resolving itself.

What is preserved is the same §7.4 exemption used everywhere else in this
system: reduce-only mode blocks orders that grow exposure, never ones that
shrink it. An account fighting a margin call must still be able to close
positions — that is wired through :class:`~atrader.risk.state.RiskSnapshot`'s
``margin_reduce_only`` flag, which ``check_system_state`` (spec check #1)
enforces with the identical exemption as the kill switch and reconciliation
breaks it sits next to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from atrader.audit.logger import AuditEvent, AuditLogger
from atrader.config.schema import AccountLimits
from atrader.core.models import AccountState
from atrader.core.types import AlertLevel

__all__ = ["MarginMonitor", "MarginStatus", "evaluate_margin"]


class MarginStatus:
    OK = "OK"
    WARN = "WARN"
    REDUCE_ONLY = "REDUCE_ONLY"


_ALERT_LEVEL = {
    MarginStatus.OK: AlertLevel.INFO,
    MarginStatus.WARN: AlertLevel.WARN,
    MarginStatus.REDUCE_ONLY: AlertLevel.CRITICAL,
}


def evaluate_margin(account: AccountState, limits: AccountLimits) -> str:
    """Classify the account's current margin health.

    A ``None`` margin ratio (no maintenance margin required — pure cash
    trading) is always :data:`MarginStatus.OK`: there is nothing to be
    under-margined on.
    """
    ratio = account.margin_ratio
    if ratio is None:
        return MarginStatus.OK
    if ratio < limits.margin_reduce_only_ratio:
        return MarginStatus.REDUCE_ONLY
    if ratio < limits.margin_warn_ratio:
        return MarginStatus.WARN
    return MarginStatus.OK


@dataclass
class MarginMonitor:
    """Tracks margin status over time and alerts only on a transition.

    Re-evaluating every tick would otherwise log the same WARN on every poll;
    what matters operationally is *when the state changed*, not that it is
    still, say, WARN five seconds after the last time it was WARN.
    """

    limits: AccountLimits
    audit: AuditLogger | None = None
    _status: str = field(default=MarginStatus.OK)

    @property
    def status(self) -> str:
        return self._status

    @property
    def reduce_only(self) -> bool:
        """True when new orders must be exposure-reducing only.

        Feed this into :attr:`~atrader.risk.state.RiskSnapshot.margin_reduce_only`
        on every snapshot build — the risk engine is the only place that
        actually blocks an order, this module only decides the flag.
        """
        return self._status == MarginStatus.REDUCE_ONLY

    def evaluate(self, account: AccountState) -> str:
        new_status = evaluate_margin(account, self.limits)
        if new_status != self._status:
            self._log_transition(account, self._status, new_status)
            self._status = new_status
        return self._status

    def _log_transition(self, account: AccountState, previous: str, current: str) -> None:
        if self.audit is None:
            return
        self.audit.append(
            AuditEvent.MARGIN_STATUS_CHANGED,
            payload={
                "previous": previous,
                "current": current,
                "margin_ratio": (
                    str(account.margin_ratio) if account.margin_ratio is not None else None
                ),
                "equity": str(account.equity),
                "maintenance_margin": str(account.maintenance_margin),
                "alert_level": _ALERT_LEVEL[current].value,
            },
        )
