"""Kill switch — spec §FR-MON-03.

    UI 버튼, CLI 명령, HTTP 엔드포인트 세 경로 모두로 접근 가능해야 한다. 발동 시
    (1) 신규 주문 중단, (2) 전체 미체결 주문 취소, (3) 설정에 따라 전 포지션 청산.
    **이 동작은 다른 모든 로직에 우선한다.**

Three access paths because the one you need is always the one that is broken:
the UI is down, or you are on a phone, or the process is wedged and only the
HTTP handler still answers.

The "takes precedence over all other logic" clause is implemented literally:
:meth:`KillSwitch.engage` sets a flag that is checked *first* in the risk
engine, before configuration is read, before a snapshot is built, before any
check runs. Nothing downstream can override it and nothing can fail in a way
that leaves it un-checked.

Acceptance criterion #5 requires it to act within 5 seconds through all three
paths, which is why engaging is a pure state change with no I/O — the cancels
that follow are dispatched by the execution layer, and a slow broker cannot
delay the block on *new* orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from atrader.core.clock import Clock

__all__ = ["KillSwitch", "KillSwitchEvent", "KillSwitchSource"]


class KillSwitchSource(StrEnum):
    """Spec §FR-MON-03 requires all three to work."""

    UI = "ui"
    CLI = "cli"
    HTTP = "http"
    AUTOMATIC = "automatic"
    """Engaged by an L3 circuit breaker rather than a person."""


@dataclass(frozen=True, slots=True)
class KillSwitchEvent:
    """A record of the switch being thrown, for the audit log."""

    engaged: bool
    source: KillSwitchSource
    reason: str
    actor: str
    at_ns: int
    liquidate: bool = False

    def as_payload(self) -> dict[str, str]:
        return {
            "engaged": str(self.engaged),
            "source": self.source.value,
            "reason": self.reason,
            "actor": self.actor,
            "liquidate": str(self.liquidate),
        }


@dataclass(slots=True)
class KillSwitch:
    """Global stop.

    Deliberately trivial. The value of a kill switch is that it cannot fail, so
    there is nothing here to go wrong: no network call, no lock, no lazy
    initialisation. Engaging is a boolean assignment.
    """

    clock: Clock
    _engaged: bool = False
    _history: list[KillSwitchEvent] = field(default_factory=list)

    @property
    def is_engaged(self) -> bool:
        return self._engaged

    @property
    def history(self) -> list[KillSwitchEvent]:
        return list(self._history)

    @property
    def last_event(self) -> KillSwitchEvent | None:
        return self._history[-1] if self._history else None

    def engage(
        self,
        *,
        reason: str,
        source: KillSwitchSource = KillSwitchSource.CLI,
        actor: str = "operator",
        liquidate: bool = False,
    ) -> KillSwitchEvent:
        """Stop all new orders immediately.

        Idempotent: engaging an already-engaged switch records the second
        attempt (useful in a post-mortem — it shows who else reached for it)
        but changes nothing.
        """
        event = KillSwitchEvent(
            engaged=True,
            source=source,
            reason=reason,
            actor=actor,
            at_ns=self.clock.now_ns(),
            liquidate=liquidate,
        )
        self._engaged = True
        self._history.append(event)
        return event

    def release(
        self,
        *,
        reason: str,
        source: KillSwitchSource = KillSwitchSource.CLI,
        actor: str = "operator",
    ) -> KillSwitchEvent:
        """Resume trading.

        Release is always manual. There is no timeout and no automatic recovery:
        whatever made someone hit the kill switch has to be understood before
        the system starts sending orders again.
        """
        event = KillSwitchEvent(
            engaged=False,
            source=source,
            reason=reason,
            actor=actor,
            at_ns=self.clock.now_ns(),
        )
        self._engaged = False
        self._history.append(event)
        return event
