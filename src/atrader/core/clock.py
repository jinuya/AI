"""Time access.

Every component reads time through a :class:`Clock` rather than calling
``datetime.now()`` directly. Spec §2.2 requires that replaying the same input
event sequence produces the same orders; wall-clock reads scattered through the
code make that impossible, so ``tests/unit/test_determinism_lint.py`` fails the
build if anything outside ``atrader.core`` calls the stdlib time functions.

Timestamps are **UTC nanoseconds since the epoch, as ints** (spec §FR-MD-02).
Nanoseconds because exchange feeds provide them and float seconds lose
precision; ints because they compare and serialise without surprises.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Final, Protocol, runtime_checkable

__all__ = [
    "NS_PER_MILLI",
    "NS_PER_SECOND",
    "Clock",
    "SimulatedClock",
    "SystemClock",
    "datetime_to_ns",
    "ns_to_datetime",
]

NS_PER_SECOND: Final[int] = 1_000_000_000
NS_PER_MILLI: Final[int] = 1_000_000


@runtime_checkable
class Clock(Protocol):
    """Source of the current time."""

    def now_ns(self) -> int:
        """Current UTC time as integer nanoseconds since the epoch."""
        ...

    def now(self) -> datetime:
        """Current UTC time as a timezone-aware ``datetime``."""
        ...

    def monotonic_ns(self) -> int:
        """Monotonic nanoseconds, for measuring durations.

        Separate from :meth:`now_ns` because wall-clock time can step backwards
        (NTP correction) and latency measurements must not go negative.
        """
        ...


class SystemClock:
    """Real time. The only place in the package that reads the system clock."""

    __slots__ = ()

    def now_ns(self) -> int:
        return time.time_ns()

    def now(self) -> datetime:
        return datetime.now(tz=UTC)

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()


class SimulatedClock:
    """Manually advanced clock for backtests, replay and tests.

    Not thread-safe by design: a simulation has a single driver, and locking
    here would only hide a bug where two components advance time concurrently.
    """

    __slots__ = ("_monotonic_ns", "_now_ns")

    def __init__(self, start_ns: int = 0) -> None:
        if start_ns < 0:
            raise ValueError(f"start_ns must be non-negative, got {start_ns}")
        self._now_ns = start_ns
        self._monotonic_ns = 0

    def now_ns(self) -> int:
        return self._now_ns

    def now(self) -> datetime:
        return ns_to_datetime(self._now_ns)

    def monotonic_ns(self) -> int:
        return self._monotonic_ns

    def advance(self, delta_ns: int) -> None:
        """Move time forward. Rejects negative deltas — time never runs backwards."""
        if delta_ns < 0:
            raise ValueError(f"cannot advance by a negative delta: {delta_ns}")
        self._now_ns += delta_ns
        self._monotonic_ns += delta_ns

    def advance_seconds(self, seconds: float) -> None:
        self.advance(int(seconds * NS_PER_SECOND))

    def set_to(self, ns: int) -> None:
        """Jump to an absolute timestamp. Used when replaying recorded events."""
        if ns < self._now_ns:
            raise ValueError(
                f"cannot set clock backwards: {ns} < {self._now_ns}. "
                "Replay events must be in non-decreasing timestamp order."
            )
        self._monotonic_ns += ns - self._now_ns
        self._now_ns = ns


def ns_to_datetime(ns: int) -> datetime:
    """Convert UTC nanoseconds to a timezone-aware ``datetime``.

    Microsecond resolution — ``datetime`` cannot hold nanoseconds. Keep the int
    when you need full precision; this is for display and DB columns.
    """
    return datetime.fromtimestamp(ns / NS_PER_SECOND, tz=UTC)


def datetime_to_ns(value: datetime) -> int:
    """Convert a ``datetime`` to UTC nanoseconds.

    Naive datetimes are rejected rather than assumed to be UTC — an implicit
    timezone is how a feed ends up eight hours out.
    """
    if value.tzinfo is None:
        raise ValueError(f"naive datetime rejected: {value!r} (attach a timezone)")
    return int(value.timestamp() * NS_PER_SECOND)
