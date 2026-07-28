"""Alert routing — spec §9.2.

    INFO/WARN/CRITICAL 세 단계로 분류하고, 채널(로그·Slack·SMS·전화)에 따라
    라우팅한다.

Real Slack/SMS/phone integrations are explicitly out of scope for this
vertical slice (see the plan's "이번 범위 밖" list) — :class:`AlertChannel` is
the interface a real one plugs into, and :class:`LoggingAlertChannel` is the
only implementation shipped, so every alert is at minimum visible in
structured logs even with no channel configured.
"""

from __future__ import annotations

import contextlib
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from atrader.core.clock import Clock
from atrader.core.types import AlertLevel
from atrader.monitoring.logging import get_logger

__all__ = ["Alert", "AlertChannel", "AlertRouter", "LoggingAlertChannel"]

#: How many alerts :class:`AlertRouter` keeps in memory for inspection
#: (``/health``, tests). Not the record of truth — the audit log and each
#: channel's own history are — just a bounded in-process buffer so a
#: long-running process cannot leak memory here.
_HISTORY_LIMIT = 1000


@dataclass(frozen=True, slots=True)
class Alert:
    level: AlertLevel
    component: str
    message: str
    created_at_ns: int
    context: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class AlertChannel(Protocol):
    def send(self, alert: Alert) -> None: ...


class LoggingAlertChannel:
    """Always-on channel: every alert becomes a structured log line."""

    def send(self, alert: Alert) -> None:
        logger = get_logger(alert.component)
        log_method = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARN: logger.warning,
            AlertLevel.CRITICAL: logger.critical,
        }[alert.level]
        log_method(alert.message, **alert.context)


class AlertRouter:
    """Fans an alert out to every configured channel.

    Never lets a channel's own failure propagate: a broken Slack webhook must
    not be able to take down the code path that triggered the alert in the
    first place — a rate limiter's own throttle warning breaking the rate
    limiter would be exactly backwards.
    """

    def __init__(self, clock: Clock, channels: tuple[AlertChannel, ...] | None = None) -> None:
        self._clock = clock
        self._channels = channels or (LoggingAlertChannel(),)
        self._history: deque[Alert] = deque(maxlen=_HISTORY_LIMIT)

    @property
    def history(self) -> tuple[Alert, ...]:
        return tuple(self._history)

    def emit(self, level: AlertLevel, *, component: str, message: str, **context: object) -> Alert:
        alert = Alert(
            level=level,
            component=component,
            message=message,
            created_at_ns=self._clock.now_ns(),
            context=context,
        )
        self._history.append(alert)
        for channel in self._channels:
            # A broken channel must not break the caller — a rate limiter's
            # own throttle warning taking down the rate limiter would be
            # exactly backwards.
            with contextlib.suppress(Exception):
                channel.send(alert)
        return alert

    def info(self, *, component: str, message: str, **context: object) -> Alert:
        return self.emit(AlertLevel.INFO, component=component, message=message, **context)

    def warn(self, *, component: str, message: str, **context: object) -> Alert:
        return self.emit(AlertLevel.WARN, component=component, message=message, **context)

    def critical(self, *, component: str, message: str, **context: object) -> Alert:
        return self.emit(AlertLevel.CRITICAL, component=component, message=message, **context)
