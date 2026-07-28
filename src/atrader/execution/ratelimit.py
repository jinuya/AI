"""Broker rate limiting — spec §6.3.

    토큰 버킷으로 브로커 API 한도를 관리하고, 한도의 일부(`reserve_pct`)는
    취소·킬스위치 전용으로 남겨둔다.

A plain token bucket is not enough, and the reason is the whole point of this
module. Consider a strategy that malfunctions and submits four hundred orders a
minute against a two-hundred-per-minute limit. The bucket does its job: the
excess is throttled. Then the operator hits the kill switch — and the cancel
request queues behind three hundred pending submissions from the very
malfunction it is trying to stop.

The bucket has correctly enforced the limit and completely failed at its
purpose. So capacity is partitioned rather than shared:

* **Ordinary traffic** — new orders — may only draw the bucket down to the
  reserve line. It is throttled while capacity remains.
* **Emergency traffic** — cancels, liquidations, the kill switch — may draw the
  bucket to empty, and overtakes anything ordinary that is already waiting.

The reserve is unusable by the traffic that would exhaust it, which is what
makes it a reserve rather than a suggestion.

Everything is measured through an injected :class:`~atrader.core.clock.Clock`,
so a test can prove a sixty-second refill without waiting sixty seconds, and a
replay reproduces the same throttle decisions.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from enum import IntEnum

from atrader.audit.logger import AuditEvent, AuditLogger
from atrader.core.clock import NS_PER_SECOND, Clock

__all__ = ["Priority", "RateLimitError", "RateLimiter", "TokenBucket"]

MIN_POLL_SECONDS = 0.001
"""Floor on the re-check interval while blocked behind a higher priority. Its
only job is to stop the wait loop spinning; tokens are not involved."""


class RateLimitError(RuntimeError):
    """Raised when a caller asked to wait and the wait ran out."""


class Priority(IntEnum):
    """Lower sorts first. Ordering is the point, so the values are explicit."""

    EMERGENCY = 0
    """Kill switch and cancels. May consume the reserve; overtakes everything."""
    REDUCING = 1
    """Orders that shrink exposure. Ahead of new risk, but not into the reserve."""
    NORMAL = 2
    """New orders. Throttled first, always."""

    @property
    def may_use_reserve(self) -> bool:
        return self is Priority.EMERGENCY


@dataclass
class TokenBucket:
    """Classic token bucket, driven by an injected clock.

    Tokens are ``Decimal`` rather than ``float`` because a partial refill is
    computed on every read; accumulated binary rounding would eventually make
    two identical replays disagree about whether an order was throttled.
    """

    capacity: Decimal
    refill_per_second: Decimal
    clock: Clock
    _tokens: Decimal = field(init=False)
    _last_ns: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")
        if self.refill_per_second <= 0:
            raise ValueError(f"refill_per_second must be positive, got {self.refill_per_second}")
        self._tokens = self.capacity  # start full: a fresh process may burst

    def _refill(self) -> None:
        now = self.clock.now_ns()
        if self._last_ns is None:
            self._last_ns = now
            return
        elapsed_ns = now - self._last_ns
        if elapsed_ns <= 0:
            return
        gained = self.refill_per_second * Decimal(elapsed_ns) / Decimal(NS_PER_SECOND)
        self._tokens = min(self.capacity, self._tokens + gained)
        self._last_ns = now

    @property
    def tokens(self) -> Decimal:
        self._refill()
        return self._tokens

    def take(self, cost: Decimal, *, floor: Decimal = Decimal(0)) -> bool:
        """Consume *cost* tokens if that leaves at least *floor* behind."""
        self._refill()
        if self._tokens - cost < floor:
            return False
        self._tokens -= cost
        return True

    def seconds_until(self, cost: Decimal, *, floor: Decimal = Decimal(0)) -> float:
        """How long until :meth:`take` would succeed. ``inf`` if never."""
        self._refill()
        if cost + floor > self.capacity:
            return float("inf")  # will not fit even in a full bucket
        needed = cost + floor - self._tokens
        if needed <= 0:
            return 0.0
        return float(needed / self.refill_per_second)


@dataclass
class RateLimiter:
    """Token bucket plus a reserve that ordinary traffic cannot touch."""

    clock: Clock
    per_minute: int = 200
    burst: int = 20
    reserve_pct: Decimal = Decimal(20)
    audit: AuditLogger | None = None
    sleep: Callable[[float], Awaitable[None]] | None = None
    """Injectable for tests: a simulated clock plus a sleep that advances it."""

    _bucket: TokenBucket = field(init=False)
    _pending: dict[Priority, int] = field(default_factory=dict, init=False)
    _throttled: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not 0 <= self.reserve_pct <= 100:
            raise ValueError(f"reserve_pct must be within 0..100, got {self.reserve_pct}")
        self._bucket = TokenBucket(
            capacity=Decimal(self.burst),
            refill_per_second=Decimal(self.per_minute) / Decimal(60),
            clock=self.clock,
        )

    @property
    def bucket(self) -> TokenBucket:
        return self._bucket

    @property
    def reserve_tokens(self) -> Decimal:
        """Capacity that only :attr:`Priority.EMERGENCY` may spend."""
        return Decimal(self.burst) * self.reserve_pct / Decimal(100)

    @property
    def throttled_count(self) -> int:
        """Requests that have had to wait. Exposed as a metric (spec §9.2)."""
        return self._throttled

    def floor_for(self, priority: Priority) -> Decimal:
        return Decimal(0) if priority.may_use_reserve else self.reserve_tokens

    def available(self, priority: Priority = Priority.NORMAL) -> Decimal:
        """Tokens this priority may actually spend right now."""
        return max(Decimal(0), self._bucket.tokens - self.floor_for(priority))

    def try_acquire(self, priority: Priority = Priority.NORMAL, *, cost: int = 1) -> bool:
        """Take capacity without waiting. ``False`` means throttled.

        Ordinary traffic is refused while something more urgent is queued, even
        when tokens are available — otherwise a steady stream of new orders keeps
        consuming the tokens a waiting cancel is about to need.
        """
        if not self._may_go(priority):
            return False
        return self._bucket.take(Decimal(cost), floor=self.floor_for(priority))

    async def acquire(
        self,
        priority: Priority = Priority.NORMAL,
        *,
        cost: int = 1,
        timeout_s: float | None = None,
    ) -> None:
        """Wait for capacity, most urgent first.

        Raises :class:`RateLimitError` if *timeout_s* elapses. Waiting forever
        would be worse: an order held for two minutes and then sent is an order
        priced against a market that no longer exists.
        """
        sleeper = self.sleep or asyncio.sleep
        floor = self.floor_for(priority)
        if Decimal(cost) + floor > Decimal(self.burst):
            raise RateLimitError(
                f"a cost of {cost} at {priority.name} priority can never be satisfied by a "
                f"bucket of capacity {self.burst} with a reserve of {self.reserve_tokens}"
            )

        deadline_ns = (
            None if timeout_s is None else self.clock.now_ns() + int(timeout_s * NS_PER_SECOND)
        )
        self._pending[priority] = self._pending.get(priority, 0) + 1
        waited = False
        try:
            while True:
                if self.try_acquire(priority, cost=cost):
                    if waited:
                        self._log_throttle(priority, cost)
                    return

                if not waited:
                    waited = True
                    self._throttled += 1

                delay = self._bucket.seconds_until(Decimal(cost), floor=floor)
                if not self._may_go(priority):
                    # Blocked by urgency rather than by tokens. That clears when
                    # the other caller finishes, so re-check soon instead of
                    # sleeping out a refill we may not need.
                    delay = min(delay, MIN_POLL_SECONDS) if delay > 0 else MIN_POLL_SECONDS

                if deadline_ns is not None:
                    remaining_s = (deadline_ns - self.clock.now_ns()) / NS_PER_SECOND
                    if remaining_s <= 0:
                        raise RateLimitError(
                            f"waited {timeout_s}s for rate-limit capacity at "
                            f"{priority.name} priority and gave up"
                        )
                    delay = min(delay, remaining_s)
                await sleeper(max(delay, 0.0))
        finally:
            remaining = self._pending.get(priority, 1) - 1
            if remaining <= 0:
                self._pending.pop(priority, None)
            else:
                self._pending[priority] = remaining

    # ------------------------------------------------------------------

    def _may_go(self, priority: Priority) -> bool:
        """True when nothing more urgent is queued ahead of this caller."""
        return not any(count > 0 for p, count in self._pending.items() if p < priority)

    def _log_throttle(self, priority: Priority, cost: int) -> None:
        if self.audit is None:
            return
        self.audit.append(
            AuditEvent.RATE_LIMIT_THROTTLED,
            payload={
                "priority": priority.name,
                "cost": str(cost),
                "tokens_remaining": str(self._bucket.tokens),
                "reserve_tokens": str(self.reserve_tokens),
            },
        )
