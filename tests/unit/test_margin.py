"""Margin monitoring — spec §FR-PF-04."""

from __future__ import annotations

from decimal import Decimal

from atrader.audit.logger import AuditEvent, AuditLogger, InMemoryAuditSink
from atrader.config.schema import AccountLimits
from atrader.core.clock import SimulatedClock
from atrader.core.models import AccountState
from atrader.portfolio.margin import MarginMonitor, MarginStatus, evaluate_margin

BASE_NS = 1_700_000_000_000_000_000
LIMITS = AccountLimits(margin_warn_ratio=Decimal("1.2"), margin_reduce_only_ratio=Decimal("1.1"))


def account(*, equity: str, maintenance_margin: str) -> AccountState:
    return AccountState(
        cash=Decimal(equity),
        equity=Decimal(equity),
        buying_power=Decimal(equity),
        maintenance_margin=Decimal(maintenance_margin),
    )


class TestEvaluateMargin:
    def test_no_maintenance_margin_required_is_always_ok(self) -> None:
        assert evaluate_margin(account(equity="1000", maintenance_margin="0"), LIMITS) == (
            MarginStatus.OK
        )

    def test_above_the_warn_threshold_is_ok(self) -> None:
        # ratio = 1000/500 = 2.0
        result = evaluate_margin(account(equity="1000", maintenance_margin="500"), LIMITS)
        assert result == MarginStatus.OK

    def test_below_warn_but_above_reduce_only_warns(self) -> None:
        # ratio = 1150/1000 = 1.15, between 1.1 and 1.2
        result = evaluate_margin(account(equity="1150", maintenance_margin="1000"), LIMITS)
        assert result == MarginStatus.WARN

    def test_below_reduce_only_switches_to_reduce_only(self) -> None:
        # ratio = 1050/1000 = 1.05
        result = evaluate_margin(account(equity="1050", maintenance_margin="1000"), LIMITS)
        assert result == MarginStatus.REDUCE_ONLY

    def test_exactly_at_the_threshold_is_not_yet_breached(self) -> None:
        # ratio = 1100/1000 = 1.10, exactly at the reduce-only line
        result = evaluate_margin(account(equity="1100", maintenance_margin="1000"), LIMITS)
        assert result == MarginStatus.WARN


class TestMarginMonitor:
    def test_starts_ok_and_reports_no_reduce_only(self) -> None:
        monitor = MarginMonitor(limits=LIMITS)
        assert monitor.status == MarginStatus.OK
        assert not monitor.reduce_only

    def test_evaluate_transitions_to_reduce_only_and_flags_it(self) -> None:
        monitor = MarginMonitor(limits=LIMITS)
        status = monitor.evaluate(account(equity="1050", maintenance_margin="1000"))
        assert status == MarginStatus.REDUCE_ONLY
        assert monitor.reduce_only

    def test_recovery_clears_reduce_only_automatically(self) -> None:
        monitor = MarginMonitor(limits=LIMITS)
        monitor.evaluate(account(equity="1050", maintenance_margin="1000"))
        assert monitor.reduce_only
        monitor.evaluate(account(equity="3000", maintenance_margin="1000"))
        assert not monitor.reduce_only
        assert monitor.status == MarginStatus.OK

    def test_a_transition_logs_exactly_one_audit_event(self) -> None:
        clock = SimulatedClock(start_ns=BASE_NS)
        sink = InMemoryAuditSink()
        audit = AuditLogger(sink=sink, clock=clock)
        monitor = MarginMonitor(limits=LIMITS, audit=audit)

        monitor.evaluate(account(equity="1050", maintenance_margin="1000"))
        records = [r for r in sink.read_all() if r.event_type == AuditEvent.MARGIN_STATUS_CHANGED]
        assert len(records) == 1
        assert records[0].payload["current"] == MarginStatus.REDUCE_ONLY

    def test_re_evaluating_the_same_status_does_not_log_again(self) -> None:
        clock = SimulatedClock(start_ns=BASE_NS)
        sink = InMemoryAuditSink()
        audit = AuditLogger(sink=sink, clock=clock)
        monitor = MarginMonitor(limits=LIMITS, audit=audit)

        monitor.evaluate(account(equity="1050", maintenance_margin="1000"))
        monitor.evaluate(account(equity="1049", maintenance_margin="1000"))  # still reduce-only
        records = [r for r in sink.read_all() if r.event_type == AuditEvent.MARGIN_STATUS_CHANGED]
        assert len(records) == 1
