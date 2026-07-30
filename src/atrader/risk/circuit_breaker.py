"""Circuit breakers — spec §7.5.

Three levels, each automatic:

======  ==========================================  ==========================
L1      daily loss -1%, or 3 consecutive losses     throttle; auto-recovers
L2      daily loss -2%, or drawdown -7%             block new orders; **manual** reset
L3      daily loss -3%, drawdown -10%, or a break   liquidate everything; post-mortem
======  ==========================================  ==========================

L2 does not auto-recover, and that is deliberate. Reaching L2 means something
went wrong; a system that quietly resumes after thirty minutes will reach L2
again, and the second time it will have lost more. A human has to look.

There are also anomaly triggers that have nothing to do with P&L — an order rate
far above baseline, or a symbol being round-tripped repeatedly. Spec §7.5 is
direct about why:

    이런 건 대개 버그의 신호고, 버그는 빠르게 돈을 태운다.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from atrader.config.schema import CircuitBreakerConfig
from atrader.core.clock import NS_PER_SECOND
from atrader.core.types import AlertLevel, BreakerLevel, SystemState
from atrader.risk.state import RiskSnapshot, RiskState

__all__ = ["BreakerDecision", "BreakerTrigger", "CircuitBreaker"]


class BreakerTrigger:
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    ORDER_RATE_ANOMALY = "order_rate_anomaly"
    ROUNDTRIP_ANOMALY = "roundtrip_anomaly"
    RECONCILIATION_BREAK = "reconciliation_break"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class BreakerDecision:
    """The breaker's verdict for one evaluation."""

    level: BreakerLevel
    system_state: SystemState
    triggers: tuple[str, ...] = ()
    reason: str = ""
    alert_level: AlertLevel | None = None
    requires_manual_reset: bool = False
    liquidate: bool = False

    @property
    def tripped(self) -> bool:
        return self.level is not BreakerLevel.NONE


_LEVEL_ORDER = {
    BreakerLevel.NONE: 0,
    BreakerLevel.L1: 1,
    BreakerLevel.L2: 2,
    BreakerLevel.L3: 3,
}


class CircuitBreaker:
    """Evaluates breaker conditions and owns escalation/recovery."""

    __slots__ = ("_config",)

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self._config = config or CircuitBreakerConfig()

    @property
    def config(self) -> CircuitBreakerConfig:
        return self._config

    def evaluate(self, snapshot: RiskSnapshot, state: RiskState) -> BreakerDecision:
        """Decide the breaker level for the current state.

        Levels only ever escalate within a session — an L2 that briefly looks
        like an L1 again (because a mark-to-market bounce lifted P&L) must not
        silently re-enable trading. Recovery is handled separately by
        :meth:`try_recover`, which has stricter conditions.
        """
        triggers: list[str] = []
        level = BreakerLevel.NONE
        reasons: list[str] = []

        loss_pct = -snapshot.daily_pnl_pct  # positive number when losing
        drawdown = snapshot.drawdown_pct

        # --- L3 ------------------------------------------------------------
        if snapshot.reconciliation_break:
            level = BreakerLevel.L3
            triggers.append(BreakerTrigger.RECONCILIATION_BREAK)
            reasons.append("reconciliation break: local state disagrees with the broker")
        if loss_pct >= self._config.l3_daily_loss_pct:
            level = BreakerLevel.L3
            triggers.append(BreakerTrigger.DAILY_LOSS)
            reasons.append(f"daily loss {loss_pct:.2f}% >= L3 {self._config.l3_daily_loss_pct}%")
        if drawdown >= self._config.l3_max_drawdown_pct:
            level = BreakerLevel.L3
            triggers.append(BreakerTrigger.DRAWDOWN)
            reasons.append(f"drawdown {drawdown:.2f}% >= L3 {self._config.l3_max_drawdown_pct}%")

        # --- L2 ------------------------------------------------------------
        if level is BreakerLevel.NONE:
            if loss_pct >= self._config.l2_daily_loss_pct:
                level = BreakerLevel.L2
                triggers.append(BreakerTrigger.DAILY_LOSS)
                reasons.append(
                    f"daily loss {loss_pct:.2f}% >= L2 {self._config.l2_daily_loss_pct}%"
                )
            elif drawdown >= self._config.l2_max_drawdown_pct:
                level = BreakerLevel.L2
                triggers.append(BreakerTrigger.DRAWDOWN)
                reasons.append(
                    f"drawdown {drawdown:.2f}% >= L2 {self._config.l2_max_drawdown_pct}%"
                )

        # --- L1 ------------------------------------------------------------
        if level is BreakerLevel.NONE:
            if loss_pct >= self._config.l1_daily_loss_pct:
                level = BreakerLevel.L1
                triggers.append(BreakerTrigger.DAILY_LOSS)
                reasons.append(
                    f"daily loss {loss_pct:.2f}% >= L1 {self._config.l1_daily_loss_pct}%"
                )
            elif state.consecutive_losses >= self._config.l1_consecutive_losses:
                window_ns = self._config.l1_consecutive_window_minutes * 60 * NS_PER_SECOND
                if (
                    state.last_loss_ns is not None
                    and snapshot.now_ns - state.last_loss_ns <= window_ns
                ):
                    level = BreakerLevel.L1
                    triggers.append(BreakerTrigger.CONSECUTIVE_LOSSES)
                    reasons.append(
                        f"{state.consecutive_losses} consecutive losses within "
                        f"{self._config.l1_consecutive_window_minutes} minutes"
                    )

        # --- anomaly triggers ----------------------------------------------
        # Not about P&L: these are bug signatures, and a bug burns money fast.
        anomaly = self._check_anomalies(snapshot)
        if anomaly is not None:
            trigger, detail = anomaly
            triggers.append(trigger)
            reasons.append(detail)
            if _LEVEL_ORDER[level] < _LEVEL_ORDER[BreakerLevel.L2]:
                level = BreakerLevel.L2

        # Never de-escalate here — with one exception, applied before the
        # hold: an L1 whose cooldown has elapsed and whose condition has
        # cleared recovers automatically (that is L1's definition). L2/L3
        # hold until manual_reset(); a mark-to-market bounce lifting P&L
        # must not silently re-enable trading.
        held = state.breaker_level
        if held is BreakerLevel.L1 and self._l1_recovery_due(snapshot, state):
            held = BreakerLevel.NONE
            reasons.append(
                f"L1 cooldown of {self._config.l1_cooldown_minutes} minutes elapsed "
                "and the triggering condition has cleared"
            )
        if _LEVEL_ORDER[level] < _LEVEL_ORDER[held]:
            level = held
            if not reasons:
                reasons.append(f"holding at {level.value} until it is cleared")

        # Persist the level and trip time onto the caller's RiskState. This
        # is what makes the no-de-escalation hold real: the runtime hands the
        # same RiskState back every bar, and holding against a field nobody
        # writes would clear an L2 on the first bounced bar — precisely the
        # silent re-enable the docstring above forbids.
        if _LEVEL_ORDER[level] > _LEVEL_ORDER[state.breaker_level]:
            state.breaker_tripped_ns = snapshot.now_ns
        elif level is BreakerLevel.NONE:
            state.breaker_tripped_ns = None
        state.breaker_level = level

        return self._decision(level, tuple(triggers), "; ".join(reasons))

    def _l1_recovery_due(self, snapshot: RiskSnapshot, state: RiskState) -> bool:
        """Whether an L1 may auto-recover *right now* (cooldown + condition)."""
        if not self._config.l1_auto_recover or state.breaker_tripped_ns is None:
            return False
        cooldown_ns = self._config.l1_cooldown_minutes * 60 * NS_PER_SECOND
        if snapshot.now_ns - state.breaker_tripped_ns < cooldown_ns:
            return False
        if -snapshot.daily_pnl_pct >= self._config.l1_daily_loss_pct:
            return False
        return state.consecutive_losses < self._config.l1_consecutive_losses

    def _check_anomalies(self, snapshot: RiskSnapshot) -> tuple[str, str] | None:
        baseline = snapshot.baseline_orders_per_minute
        if baseline > 0:
            multiple = Decimal(snapshot.orders_last_minute) / baseline
            if multiple >= self._config.anomaly_order_rate_multiple:
                return (
                    BreakerTrigger.ORDER_RATE_ANOMALY,
                    f"order rate {snapshot.orders_last_minute}/min is {multiple:.1f}x the "
                    f"baseline {baseline}/min — usually a loop bug",
                )
        return None

    def check_roundtrips(self, symbol: str, count: int) -> BreakerDecision | None:
        """Flag repeated round trips in one symbol (spec §7.5)."""
        if count < self._config.anomaly_roundtrip_count:
            return None
        return self._decision(
            BreakerLevel.L2,
            (BreakerTrigger.ROUNDTRIP_ANOMALY,),
            f"{symbol} has been round-tripped {count} times in the window — the account "
            "is paying commission to stand still, which usually means a strategy loop",
        )

    def _decision(
        self, level: BreakerLevel, triggers: tuple[str, ...], reason: str
    ) -> BreakerDecision:
        if level is BreakerLevel.L3:
            return BreakerDecision(
                level=level,
                system_state=SystemState.LIQUIDATING,
                triggers=triggers,
                reason=reason,
                alert_level=AlertLevel.CRITICAL,
                requires_manual_reset=True,
                liquidate=self._config.l3_liquidate_all,
            )
        if level is BreakerLevel.L2:
            return BreakerDecision(
                level=level,
                system_state=SystemState.BLOCKED,
                triggers=triggers,
                reason=reason,
                alert_level=AlertLevel.CRITICAL,
                requires_manual_reset=not self._config.l2_auto_recover,
            )
        if level is BreakerLevel.L1:
            return BreakerDecision(
                level=level,
                system_state=SystemState.THROTTLED,
                triggers=triggers,
                reason=reason,
                alert_level=AlertLevel.WARN,
                requires_manual_reset=not self._config.l1_auto_recover,
            )
        return BreakerDecision(level=BreakerLevel.NONE, system_state=SystemState.RUNNING)

    def try_recover(self, snapshot: RiskSnapshot, state: RiskState) -> BreakerDecision | None:
        """Attempt automatic recovery. Only L1 is eligible.

        Requires both that the cooldown has elapsed *and* that the triggering
        condition has actually cleared. Waiting out the clock while still losing
        money would just restart the countdown to L2.

        :meth:`evaluate` performs this same recovery internally on every call,
        so callers that evaluate each bar need not call this; it remains for
        out-of-cycle checks.
        """
        if state.breaker_level is not BreakerLevel.L1:
            return None
        if not self._l1_recovery_due(snapshot, state):
            return None
        state.breaker_level = BreakerLevel.NONE
        state.breaker_tripped_ns = None

        return BreakerDecision(
            level=BreakerLevel.NONE,
            system_state=SystemState.RUNNING,
            reason=(
                f"L1 cooldown of {self._config.l1_cooldown_minutes} minutes elapsed and the "
                "triggering condition has cleared"
            ),
            alert_level=AlertLevel.INFO,
        )

    def manual_reset(
        self,
        to_level: BreakerLevel = BreakerLevel.NONE,
        *,
        state: RiskState | None = None,
    ) -> BreakerDecision:
        """Operator override. Spec §7.6 requires human approval to reach here.

        Pass the live ``RiskState`` so the reset actually sticks — without it
        the next ``evaluate`` re-holds the old level.
        """
        if state is not None:
            state.breaker_level = to_level
            state.breaker_tripped_ns = None
        return BreakerDecision(
            level=to_level,
            system_state=SystemState.RUNNING
            if to_level is BreakerLevel.NONE
            else SystemState.BLOCKED,
            triggers=(BreakerTrigger.MANUAL,),
            reason=f"manually reset to {to_level.value}",
            alert_level=AlertLevel.WARN,
        )
