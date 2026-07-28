"""Dead-man switch — spec §FR-MON-04.

    하트비트를 60초 이상 받지 못하면 미체결 주문을 전량 취소한다.

The premise is that a trading process which has stopped thinking is more
dangerous than one which has stopped entirely. A wedged event loop, a deadlocked
database call, a strategy stuck in a retry storm — in every case the resting
orders stay live at the venue and keep filling into a market the system is no
longer watching. Nobody is left to cancel them, take the other side, or notice
the position growing.

So the process publishes proof-of-life on a timer, and a watchdog that outlives
it cancels everything when the proof stops arriving.

**The watchdog must not be driven by the thing it watches.** That is the whole
design constraint, and it is easy to get wrong: a check called from the main
loop cannot fire when the main loop is the thing that has stopped. Two
consequences follow:

* :meth:`DeadManSwitch.run` is meant to be its own task, and its poll loop does
  nothing but read a timestamp — no I/O, no locks, nothing that can block on the
  same resource the trading path is stuck on.
* :meth:`DeadManSwitch.check` is pure enough to call from a test with a
  :class:`~atrader.core.clock.SimulatedClock`, so the expiry logic is verifiable
  without waiting sixty real seconds.

Firing is **latched**. Once tripped, a fresh heartbeat does not silently rearm
it: a process that recovers on its own has still demonstrated it can stall, and
resuming without a human deciding to is how the same stall trades through twice.
:meth:`DeadManSwitch.rearm` is the explicit path back.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from atrader.audit.logger import AuditEvent, AuditLogger
from atrader.core.clock import NS_PER_SECOND, Clock
from atrader.core.types import AlertLevel

__all__ = ["DEFAULT_TIMEOUT_SECONDS", "DeadManSwitch", "HeartbeatSource", "TriggerReport"]

DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_POLL_SECONDS = 1.0

#: Anything that can be asked "when did you last make progress?".
HeartbeatSource = Callable[[], int]


@dataclass(frozen=True, slots=True)
class TriggerReport:
    """What happened when the switch fired."""

    at_ns: int
    silence_seconds: float
    canceled_order_ids: tuple[str, ...] = ()
    cancel_errors: tuple[str, ...] = ()

    @property
    def alert_level(self) -> AlertLevel:
        return AlertLevel.CRITICAL

    def summary(self) -> str:
        base = (
            f"dead-man switch fired after {self.silence_seconds:.1f}s of silence; "
            f"canceled {len(self.canceled_order_ids)} order(s)"
        )
        if self.cancel_errors:
            return f"{base}; {len(self.cancel_errors)} cancel(s) failed: " + "; ".join(
                self.cancel_errors
            )
        return base


@dataclass
class DeadManSwitch:
    """Cancels every working order when heartbeats stop arriving."""

    cancel_all: Callable[[], Awaitable[list[object]]]
    """Usually :meth:`~atrader.execution.oms.OrderManager.cancel_all`."""
    clock: Clock
    audit: AuditLogger | None = None
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    poll_seconds: float = DEFAULT_POLL_SECONDS
    on_trigger: Callable[[TriggerReport], None] | None = None
    """Called after cancellation — where the alert router and the kill switch
    get hooked up. Kept as a callback so this module does not depend on either."""

    _last_beat_ns: int | None = None
    _fired: bool = field(default=False)
    _last_report: TriggerReport | None = field(default=None)
    _stopping: bool = field(default=False)

    # ------------------------------------------------------------------
    # Proof of life
    # ------------------------------------------------------------------

    def beat(self, at_ns: int | None = None) -> None:
        """Record proof of life.

        Called from wherever real progress happens — after a market data tick is
        processed, not from a bare timer. A timer proves only that the timer is
        running, which is exactly the thing that keeps working while everything
        else is stuck.
        """
        self._last_beat_ns = at_ns if at_ns is not None else self.clock.now_ns()

    @property
    def has_fired(self) -> bool:
        return self._fired

    @property
    def last_report(self) -> TriggerReport | None:
        return self._last_report

    @property
    def last_beat_ns(self) -> int | None:
        return self._last_beat_ns

    def silence_ns(self, now_ns: int | None = None) -> int:
        """Nanoseconds since the last heartbeat. Zero before the first one."""
        if self._last_beat_ns is None:
            return 0
        now = now_ns if now_ns is not None else self.clock.now_ns()
        return max(0, now - self._last_beat_ns)

    def is_expired(self, now_ns: int | None = None) -> bool:
        """True once the silence has exceeded the timeout.

        Returns ``False`` before the first heartbeat: a process that has not
        started yet has no orders to cancel, and firing during startup would
        make the switch a boot-time hazard rather than a safety net.
        """
        if self._last_beat_ns is None:
            return False
        return self.silence_ns(now_ns) >= self.timeout_seconds * NS_PER_SECOND

    def rearm(self, *, by: str, note: str = "") -> None:
        """Re-enable the switch after a human has looked at why it fired."""
        self._fired = False
        self._last_beat_ns = self.clock.now_ns()
        if self.audit is not None:
            self.audit.append(
                AuditEvent.DEADMAN_REARMED,
                actor=by,
                payload={"note": note},
            )

    # ------------------------------------------------------------------
    # Firing
    # ------------------------------------------------------------------

    async def check(self) -> TriggerReport | None:
        """Fire if the heartbeat has gone quiet. Returns the report, or ``None``.

        Safe to call repeatedly: the latch means a second call after firing does
        nothing, so a poll loop cannot cancel-storm the broker.
        """
        if self._fired or not self.is_expired():
            return None
        return await self.trigger(reason="heartbeat timeout")

    async def trigger(self, *, reason: str) -> TriggerReport:
        """Cancel everything working. Also the manual panic path.

        Cancellation failures are collected rather than raised. One order that
        refuses to cancel must not stop the other forty from being cancelled —
        the failures go into the report and the alert, where a human sees them.
        """
        now = self.clock.now_ns()
        self._fired = True

        canceled: list[str] = []
        errors: list[str] = []
        try:
            for order in await self.cancel_all():
                canceled.append(str(getattr(order, "order_id", order)))
        except Exception as exc:  # the report must survive any single cancel failure
            errors.append(f"{type(exc).__name__}: {exc}")

        report = TriggerReport(
            at_ns=now,
            silence_seconds=self.silence_ns(now) / NS_PER_SECOND,
            canceled_order_ids=tuple(canceled),
            cancel_errors=tuple(errors),
        )
        self._last_report = report

        if self.audit is not None:
            self.audit.append(
                AuditEvent.DEADMAN_TRIGGERED,
                payload={
                    "reason": reason,
                    "silence_seconds": f"{report.silence_seconds:.3f}",
                    "canceled_order_ids": list(report.canceled_order_ids),
                    "cancel_errors": list(report.cancel_errors),
                },
            )
        if self.on_trigger is not None:
            self.on_trigger(report)
        return report

    # ------------------------------------------------------------------
    # Watchdog task
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Poll until stopped. Run this as its own task, never inline.

        The body is deliberately trivial. Anything that could block — a database
        read, a lock, an awaited broker call other than the cancel itself —
        would let the watchdog inherit the stall it exists to detect.
        """
        self._stopping = False
        while not self._stopping:
            await self.check()
            await asyncio.sleep(self.poll_seconds)

    def stop(self) -> None:
        """Ask :meth:`run` to return at the next poll."""
        self._stopping = True
