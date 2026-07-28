"""Broker rate limiting — spec §6.3.

토큰 버킷 + `reserve_pct` 20% + 취소/킬스위치 전용 우선순위 큐.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from atrader.core.clock import SimulatedClock
from atrader.execution.ratelimit import Priority, RateLimiter, RateLimitError, TokenBucket

BASE_NS = 1_700_000_000_000_000_000


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(start_ns=BASE_NS)


class TestTokenBucket:
    def test_starts_full(self, clock: SimulatedClock) -> None:
        bucket = TokenBucket(capacity=Decimal(10), refill_per_second=Decimal(1), clock=clock)
        assert bucket.tokens == Decimal(10)

    def test_take_reduces_tokens(self, clock: SimulatedClock) -> None:
        bucket = TokenBucket(capacity=Decimal(10), refill_per_second=Decimal(1), clock=clock)
        assert bucket.take(Decimal(3))
        assert bucket.tokens == Decimal(7)

    def test_take_below_floor_fails_and_does_not_consume(self, clock: SimulatedClock) -> None:
        bucket = TokenBucket(capacity=Decimal(10), refill_per_second=Decimal(1), clock=clock)
        assert not bucket.take(Decimal(9), floor=Decimal(5))
        assert bucket.tokens == Decimal(10)  # untouched

    def test_refills_over_time(self, clock: SimulatedClock) -> None:
        bucket = TokenBucket(capacity=Decimal(10), refill_per_second=Decimal(2), clock=clock)
        bucket.take(Decimal(10))
        assert bucket.tokens == Decimal(0)
        clock.advance_seconds(3)
        assert bucket.tokens == Decimal(6)

    def test_refill_caps_at_capacity(self, clock: SimulatedClock) -> None:
        bucket = TokenBucket(capacity=Decimal(10), refill_per_second=Decimal(2), clock=clock)
        bucket.take(Decimal(1))
        clock.advance_seconds(100)
        assert bucket.tokens == Decimal(10)

    def test_no_time_passing_means_no_refill(self, clock: SimulatedClock) -> None:
        bucket = TokenBucket(capacity=Decimal(10), refill_per_second=Decimal(2), clock=clock)
        bucket.take(Decimal(5))
        assert bucket.tokens == Decimal(5)  # a second read at the same instant

    def test_rejects_non_positive_capacity(self, clock: SimulatedClock) -> None:
        with pytest.raises(ValueError, match="capacity"):
            TokenBucket(capacity=Decimal(0), refill_per_second=Decimal(1), clock=clock)

    def test_rejects_non_positive_refill_rate(self, clock: SimulatedClock) -> None:
        with pytest.raises(ValueError, match="refill_per_second"):
            TokenBucket(capacity=Decimal(10), refill_per_second=Decimal(0), clock=clock)

    def test_seconds_until_available_now_is_zero(self, clock: SimulatedClock) -> None:
        bucket = TokenBucket(capacity=Decimal(10), refill_per_second=Decimal(1), clock=clock)
        assert bucket.seconds_until(Decimal(5)) == 0.0

    def test_seconds_until_computes_the_wait(self, clock: SimulatedClock) -> None:
        bucket = TokenBucket(capacity=Decimal(10), refill_per_second=Decimal(2), clock=clock)
        bucket.take(Decimal(10))
        assert bucket.seconds_until(Decimal(4)) == pytest.approx(2.0)

    def test_seconds_until_is_infinite_when_it_can_never_fit(self, clock: SimulatedClock) -> None:
        bucket = TokenBucket(capacity=Decimal(10), refill_per_second=Decimal(1), clock=clock)
        assert bucket.seconds_until(Decimal(5), floor=Decimal(8)) == float("inf")


class TestPriority:
    def test_only_emergency_may_use_the_reserve(self) -> None:
        assert Priority.EMERGENCY.may_use_reserve
        assert not Priority.REDUCING.may_use_reserve
        assert not Priority.NORMAL.may_use_reserve

    def test_ordering_puts_emergency_first(self) -> None:
        assert Priority.EMERGENCY < Priority.REDUCING < Priority.NORMAL


class TestRateLimiterReserve:
    def test_reserve_tokens_computed_from_burst_and_pct(self, clock: SimulatedClock) -> None:
        limiter = RateLimiter(clock=clock, burst=20, reserve_pct=Decimal(20))
        assert limiter.reserve_tokens == Decimal(4)

    def test_normal_traffic_is_throttled_at_the_reserve_line(self, clock: SimulatedClock) -> None:
        limiter = RateLimiter(clock=clock, per_minute=600, burst=10, reserve_pct=Decimal(20))
        for _ in range(8):
            assert limiter.try_acquire(Priority.NORMAL)
        # 2 tokens left, which is exactly the reserve (20% of 10) — normal
        # traffic must not be able to dip into it.
        assert not limiter.try_acquire(Priority.NORMAL)

    def test_emergency_traffic_can_draw_the_bucket_to_empty(self, clock: SimulatedClock) -> None:
        limiter = RateLimiter(clock=clock, per_minute=600, burst=10, reserve_pct=Decimal(20))
        for _ in range(8):
            assert limiter.try_acquire(Priority.NORMAL)
        assert limiter.try_acquire(Priority.EMERGENCY)
        assert limiter.try_acquire(Priority.EMERGENCY)
        assert not limiter.try_acquire(Priority.EMERGENCY)  # now truly empty

    def test_available_reports_what_this_priority_could_still_spend(
        self, clock: SimulatedClock
    ) -> None:
        limiter = RateLimiter(clock=clock, per_minute=600, burst=10, reserve_pct=Decimal(20))
        assert limiter.available(Priority.NORMAL) == Decimal(8)
        assert limiter.available(Priority.EMERGENCY) == Decimal(10)

    def test_zero_reserve_pct_lets_normal_traffic_use_everything(
        self, clock: SimulatedClock
    ) -> None:
        limiter = RateLimiter(clock=clock, per_minute=600, burst=10, reserve_pct=Decimal(0))
        for _ in range(10):
            assert limiter.try_acquire(Priority.NORMAL)
        assert not limiter.try_acquire(Priority.NORMAL)

    def test_rejects_reserve_pct_outside_0_to_100(self, clock: SimulatedClock) -> None:
        with pytest.raises(ValueError, match="reserve_pct"):
            RateLimiter(clock=clock, reserve_pct=Decimal(150))


class TestAcquireWaiting:
    async def test_acquire_returns_immediately_when_capacity_is_available(
        self, clock: SimulatedClock
    ) -> None:
        limiter = RateLimiter(clock=clock, per_minute=600, burst=10)
        await limiter.acquire(Priority.NORMAL)
        assert limiter.throttled_count == 0

    async def test_acquire_waits_for_a_refill_using_the_injected_sleep(
        self, clock: SimulatedClock
    ) -> None:
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            clock.advance_seconds(seconds)

        limiter = RateLimiter(
            clock=clock, per_minute=60, burst=1, reserve_pct=Decimal(0), sleep=fake_sleep
        )
        await limiter.acquire(Priority.NORMAL)  # drains the only token
        await limiter.acquire(Priority.NORMAL)  # must wait for a refill
        assert limiter.throttled_count == 1
        assert sleeps  # the fake sleep was actually invoked

    async def test_acquire_raises_when_the_cost_can_never_fit(self, clock: SimulatedClock) -> None:
        limiter = RateLimiter(clock=clock, burst=5, reserve_pct=Decimal(50))
        with pytest.raises(RateLimitError, match="can never be satisfied"):
            await limiter.acquire(Priority.NORMAL, cost=4)  # floor=2.5, 4+2.5 > 5

    async def test_acquire_raises_on_timeout(self, clock: SimulatedClock) -> None:
        async def fake_sleep(seconds: float) -> None:
            clock.advance_seconds(seconds)

        limiter = RateLimiter(
            clock=clock, per_minute=1, burst=1, reserve_pct=Decimal(0), sleep=fake_sleep
        )
        await limiter.acquire(Priority.NORMAL)  # drains the bucket
        with pytest.raises(RateLimitError, match="gave up"):
            await limiter.acquire(Priority.NORMAL, timeout_s=0.5)

    async def test_emergency_preempts_queued_normal_traffic(self, clock: SimulatedClock) -> None:
        """The scenario the whole module exists for: a kill-switch cancel must
        not queue behind a strategy malfunction's flood of new orders.

        The clock is advanced explicitly by the test, once both callers are
        confirmed pending, rather than by the fake sleep — letting the fake
        sleep also advance time makes the outcome depend on which of two
        equally-ready tasks the event loop happens to resume first, which is
        exactly the race this test must not have.
        """
        order: list[str] = []

        async def fake_sleep(seconds: float) -> None:
            await asyncio.sleep(0)  # yield only; the test drives the clock

        limiter = RateLimiter(
            clock=clock, per_minute=6000, burst=1, reserve_pct=Decimal(0), sleep=fake_sleep
        )
        await limiter.acquire(Priority.NORMAL)  # drains the single token immediately

        async def normal_call() -> None:
            await limiter.acquire(Priority.NORMAL)
            order.append("normal")

        async def emergency_call() -> None:
            await limiter.acquire(Priority.EMERGENCY)
            order.append("emergency")

        normal_task = asyncio.create_task(normal_call())
        for _ in range(5):
            await asyncio.sleep(0)  # let normal register as pending and start polling

        emergency_task = asyncio.create_task(emergency_call())
        for _ in range(5):
            await asyncio.sleep(0)  # let emergency register as pending too

        clock.advance_seconds(10)  # enough to refill exactly one token
        await emergency_task
        assert order == ["emergency"]
        assert not normal_task.done()  # still blocked — the bucket is empty again

        clock.advance_seconds(10)
        await normal_task
        assert order == ["emergency", "normal"]

    async def test_throttle_is_audited_once_resolved(self, clock: SimulatedClock) -> None:
        from atrader.audit.logger import AuditEvent, AuditLogger
        from atrader.storage.memory import InMemoryStorage

        storage = InMemoryStorage()
        audit = AuditLogger(storage.audit, clock)

        async def fake_sleep(seconds: float) -> None:
            clock.advance_seconds(seconds)

        limiter = RateLimiter(
            clock=clock,
            per_minute=60,
            burst=1,
            reserve_pct=Decimal(0),
            sleep=fake_sleep,
            audit=audit,
        )
        await limiter.acquire(Priority.NORMAL)
        await limiter.acquire(Priority.NORMAL)
        events = [r.event_type for r in storage.audit.read_all()]
        assert AuditEvent.RATE_LIMIT_THROTTLED in events
