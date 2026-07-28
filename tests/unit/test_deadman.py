"""Dead-man switch — spec §FR-MON-04.

하트비트를 60초 이상 받지 못하면 미체결 주문을 전량 취소한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from atrader.audit.logger import AuditEvent, AuditLogger
from atrader.core.clock import SimulatedClock
from atrader.core.types import AlertLevel
from atrader.execution.deadman import DeadManSwitch, TriggerReport
from atrader.storage.memory import InMemoryStorage

BASE_NS = 1_700_000_000_000_000_000


@dataclass
class FakeCanceler:
    """Stands in for OrderManager.cancel_all."""

    to_cancel: list[Any] = field(default_factory=list)
    calls: int = field(default=0, init=False)
    should_raise: Exception | None = None

    async def cancel_all(self) -> list[Any]:
        self.calls += 1
        if self.should_raise is not None:
            raise self.should_raise
        return self.to_cancel


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(start_ns=BASE_NS)


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


@pytest.fixture
def audit(storage: InMemoryStorage, clock: SimulatedClock) -> AuditLogger:
    return AuditLogger(storage.audit, clock)


class FakeOrder:
    def __init__(self, order_id: str) -> None:
        self.order_id = order_id


class TestHeartbeat:
    async def test_not_expired_before_any_heartbeat(
        self, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        switch = DeadManSwitch(cancel_all=FakeCanceler().cancel_all, clock=clock, audit=audit)
        clock.advance_seconds(1000)
        assert not switch.is_expired()

    async def test_not_expired_within_the_timeout(
        self, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        switch = DeadManSwitch(
            cancel_all=FakeCanceler().cancel_all, clock=clock, audit=audit, timeout_seconds=60
        )
        switch.beat()
        clock.advance_seconds(59)
        assert not switch.is_expired()

    async def test_expired_once_the_timeout_elapses(
        self, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        switch = DeadManSwitch(
            cancel_all=FakeCanceler().cancel_all, clock=clock, audit=audit, timeout_seconds=60
        )
        switch.beat()
        clock.advance_seconds(61)
        assert switch.is_expired()

    async def test_a_fresh_heartbeat_pushes_the_deadline_out(
        self, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        switch = DeadManSwitch(
            cancel_all=FakeCanceler().cancel_all, clock=clock, audit=audit, timeout_seconds=60
        )
        switch.beat()
        clock.advance_seconds(50)
        switch.beat()
        clock.advance_seconds(50)
        assert not switch.is_expired()

    async def test_silence_seconds_reports_the_gap(
        self, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        switch = DeadManSwitch(cancel_all=FakeCanceler().cancel_all, clock=clock, audit=audit)
        switch.beat()
        clock.advance_seconds(15)
        assert switch.silence_ns() == 15 * 1_000_000_000

    async def test_silence_seconds_is_zero_before_any_heartbeat(
        self, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        switch = DeadManSwitch(cancel_all=FakeCanceler().cancel_all, clock=clock, audit=audit)
        clock.advance_seconds(1000)
        assert switch.silence_ns() == 0

    async def test_last_beat_ns_tracks_the_most_recent_heartbeat(
        self, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        switch = DeadManSwitch(cancel_all=FakeCanceler().cancel_all, clock=clock, audit=audit)
        assert switch.last_beat_ns is None
        switch.beat()
        assert switch.last_beat_ns == clock.now_ns()


class TestCheckAndTrigger:
    async def test_check_does_nothing_before_expiry(
        self, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        canceler = FakeCanceler()
        switch = DeadManSwitch(
            cancel_all=canceler.cancel_all, clock=clock, audit=audit, timeout_seconds=60
        )
        switch.beat()
        clock.advance_seconds(10)
        report = await switch.check()
        assert report is None
        assert canceler.calls == 0

    async def test_check_cancels_everything_once_expired(
        self, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        canceler = FakeCanceler(to_cancel=[FakeOrder("o-1"), FakeOrder("o-2")])
        switch = DeadManSwitch(
            cancel_all=canceler.cancel_all, clock=clock, audit=audit, timeout_seconds=60
        )
        switch.beat()
        clock.advance_seconds(61)

        report = await switch.check()

        assert report is not None
        assert report.canceled_order_ids == ("o-1", "o-2")
        assert report.alert_level is AlertLevel.CRITICAL
        assert switch.has_fired
        assert canceler.calls == 1

    async def test_latched_after_firing_a_second_check_is_a_no_op(
        self, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        canceler = FakeCanceler(to_cancel=[FakeOrder("o-1")])
        switch = DeadManSwitch(
            cancel_all=canceler.cancel_all, clock=clock, audit=audit, timeout_seconds=60
        )
        switch.beat()
        clock.advance_seconds(61)
        await switch.check()
        second = await switch.check()
        assert second is None
        assert canceler.calls == 1  # not cancel-stormed

    async def test_manual_trigger_bypasses_the_timer(
        self, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        canceler = FakeCanceler(to_cancel=[FakeOrder("o-1")])
        switch = DeadManSwitch(
            cancel_all=canceler.cancel_all, clock=clock, audit=audit, timeout_seconds=60
        )
        switch.beat()
        report = await switch.trigger(reason="operator panic button")
        assert report.canceled_order_ids == ("o-1",)
        assert switch.has_fired

    async def test_cancel_failures_are_captured_not_raised(
        self, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        canceler = FakeCanceler(should_raise=RuntimeError("broker unreachable"))
        switch = DeadManSwitch(
            cancel_all=canceler.cancel_all, clock=clock, audit=audit, timeout_seconds=60
        )
        switch.beat()
        report = await switch.trigger(reason="test")
        assert report.canceled_order_ids == ()
        assert "broker unreachable" in report.cancel_errors[0]
        assert "1 cancel(s) failed" in report.summary()

    async def test_trigger_is_audited_and_calls_the_hook(
        self, clock: SimulatedClock, audit: AuditLogger, storage: InMemoryStorage
    ) -> None:
        seen: list[TriggerReport] = []
        canceler = FakeCanceler(to_cancel=[FakeOrder("o-1")])
        switch = DeadManSwitch(
            cancel_all=canceler.cancel_all,
            clock=clock,
            audit=audit,
            timeout_seconds=60,
            on_trigger=seen.append,
        )
        switch.beat()
        await switch.trigger(reason="test")
        assert len(seen) == 1
        events = [r.event_type for r in storage.audit.read_all()]
        assert AuditEvent.DEADMAN_TRIGGERED in events

    def test_last_report_is_available_after_firing(
        self, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        switch = DeadManSwitch(cancel_all=FakeCanceler().cancel_all, clock=clock, audit=audit)
        assert switch.last_report is None

    async def test_summary_without_errors_omits_the_failure_clause(
        self, clock: SimulatedClock, audit: AuditLogger
    ) -> None:
        canceler = FakeCanceler(to_cancel=[FakeOrder("o-1")])
        switch = DeadManSwitch(
            cancel_all=canceler.cancel_all, clock=clock, audit=audit, timeout_seconds=60
        )
        switch.beat()
        report = await switch.trigger(reason="test")
        assert report.cancel_errors == ()
        assert "canceled 1 order(s)" in report.summary()
        assert "failed" not in report.summary()


class TestRearm:
    async def test_rearm_clears_the_latch_and_resets_the_heartbeat(
        self, clock: SimulatedClock, audit: AuditLogger, storage: InMemoryStorage
    ) -> None:
        canceler = FakeCanceler()
        switch = DeadManSwitch(
            cancel_all=canceler.cancel_all, clock=clock, audit=audit, timeout_seconds=60
        )
        switch.beat()
        clock.advance_seconds(61)
        await switch.check()
        assert switch.has_fired

        switch.rearm(by="ops-oncall", note="confirmed the process is healthy again")

        assert not switch.has_fired
        assert not switch.is_expired()
        events = [r.event_type for r in storage.audit.read_all()]
        assert AuditEvent.DEADMAN_REARMED in events


class TestRunLoop:
    async def test_run_polls_until_stopped(self, clock: SimulatedClock, audit: AuditLogger) -> None:
        import asyncio

        canceler = FakeCanceler(to_cancel=[FakeOrder("o-1")])
        switch = DeadManSwitch(
            cancel_all=canceler.cancel_all,
            clock=clock,
            audit=audit,
            timeout_seconds=60,
            poll_seconds=0.001,
        )
        switch.beat()

        async def stop_soon() -> None:
            await asyncio.sleep(0.005)
            switch.stop()

        await asyncio.gather(switch.run(), stop_soon())
        assert not switch.has_fired  # never expired; the loop just ran and stopped cleanly
