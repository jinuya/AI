"""Circuit breakers and the kill switch — spec §7.5, §FR-MON-03."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from atrader.config.schema import CircuitBreakerConfig
from atrader.core.clock import NS_PER_SECOND, SimulatedClock
from atrader.core.models import AccountState
from atrader.core.types import AlertLevel, BreakerLevel, Side, SystemState
from atrader.risk.circuit_breaker import BreakerTrigger, CircuitBreaker
from atrader.risk.killswitch import KillSwitch, KillSwitchSource
from atrader.risk.state import OrderRateTracker, RecentOrder, RiskSnapshot, RiskState

BASE_NS = 1_700_000_000 * NS_PER_SECOND
EQUITY = Decimal("100000")


def snapshot(**overrides: Any) -> RiskSnapshot:
    defaults: dict[str, Any] = {
        "now_ns": BASE_NS,
        "system_state": SystemState.RUNNING,
        "breaker_level": BreakerLevel.NONE,
        "account": AccountState(cash=EQUITY, equity=EQUITY),
    }
    return RiskSnapshot(**{**defaults, **overrides})


class TestEscalation:
    def test_no_trigger_leaves_the_system_running(self) -> None:
        decision = CircuitBreaker().evaluate(snapshot(), RiskState())
        assert not decision.tripped
        assert decision.system_state is SystemState.RUNNING

    @pytest.mark.parametrize(
        ("loss", "level", "state"),
        [
            (Decimal("-1.5"), BreakerLevel.L1, SystemState.THROTTLED),
            (Decimal("-2.5"), BreakerLevel.L2, SystemState.BLOCKED),
            (Decimal("-3.5"), BreakerLevel.L3, SystemState.LIQUIDATING),
        ],
    )
    def test_daily_loss_escalates_through_the_levels(
        self, loss: Decimal, level: BreakerLevel, state: SystemState
    ) -> None:
        decision = CircuitBreaker().evaluate(snapshot(daily_pnl_pct=loss), RiskState())
        assert decision.level is level
        assert decision.system_state is state
        assert BreakerTrigger.DAILY_LOSS in decision.triggers

    def test_drawdown_trips_l2(self) -> None:
        decision = CircuitBreaker().evaluate(snapshot(drawdown_pct=Decimal("8")), RiskState())
        assert decision.level is BreakerLevel.L2
        assert BreakerTrigger.DRAWDOWN in decision.triggers

    def test_l3_liquidates_and_alerts_critically(self) -> None:
        decision = CircuitBreaker().evaluate(snapshot(daily_pnl_pct=Decimal("-5")), RiskState())
        assert decision.liquidate
        assert decision.alert_level is AlertLevel.CRITICAL
        assert decision.requires_manual_reset

    def test_a_reconciliation_break_goes_straight_to_l3(self) -> None:
        # We no longer know our real position; sizing anything is guesswork.
        decision = CircuitBreaker().evaluate(snapshot(reconciliation_break=True), RiskState())
        assert decision.level is BreakerLevel.L3
        assert BreakerTrigger.RECONCILIATION_BREAK in decision.triggers

    def test_consecutive_losses_trip_l1(self) -> None:
        state = RiskState(consecutive_losses=3, last_loss_ns=BASE_NS - 60 * NS_PER_SECOND)
        decision = CircuitBreaker().evaluate(snapshot(), state)
        assert decision.level is BreakerLevel.L1
        assert BreakerTrigger.CONSECUTIVE_LOSSES in decision.triggers

    def test_consecutive_losses_outside_the_window_do_not_trip(self) -> None:
        state = RiskState(consecutive_losses=3, last_loss_ns=BASE_NS - 3600 * NS_PER_SECOND)
        assert not CircuitBreaker().evaluate(snapshot(), state).tripped

    def test_the_level_never_de_escalates_on_its_own(self) -> None:
        # A mark-to-market bounce that lifts P&L back above the L2 threshold
        # must not silently re-enable trading.
        state = RiskState(breaker_level=BreakerLevel.L2)
        decision = CircuitBreaker().evaluate(snapshot(daily_pnl_pct=Decimal("0")), state)
        assert decision.level is BreakerLevel.L2


class TestAnomalyTriggers:
    def test_an_order_rate_spike_blocks_trading(self) -> None:
        # Nothing to do with P&L: a rate far above baseline is a loop bug, and a
        # loop bug burns money fast (spec §7.5).
        decision = CircuitBreaker().evaluate(
            snapshot(orders_last_minute=50, baseline_orders_per_minute=Decimal("5")),
            RiskState(),
        )
        assert decision.level is BreakerLevel.L2
        assert BreakerTrigger.ORDER_RATE_ANOMALY in decision.triggers

    def test_a_normal_rate_does_not_trip(self) -> None:
        decision = CircuitBreaker().evaluate(
            snapshot(orders_last_minute=8, baseline_orders_per_minute=Decimal("5")),
            RiskState(),
        )
        assert not decision.tripped

    def test_no_baseline_means_no_rate_anomaly(self) -> None:
        # On day one there is nothing to compare against; the hard per-minute
        # cap in check 13 still applies.
        decision = CircuitBreaker().evaluate(
            snapshot(orders_last_minute=500, baseline_orders_per_minute=Decimal("0")),
            RiskState(),
        )
        assert not decision.tripped

    def test_repeated_roundtrips_are_flagged(self) -> None:
        decision = CircuitBreaker().check_roundtrips("AAPL", count=6)
        assert decision is not None
        assert decision.level is BreakerLevel.L2
        assert "commission to stand still" in decision.reason

    def test_a_couple_of_roundtrips_are_not_suspicious(self) -> None:
        assert CircuitBreaker().check_roundtrips("AAPL", count=1) is None


class TestRecovery:
    def test_l1_recovers_after_the_cooldown_once_the_condition_clears(self) -> None:
        breaker = CircuitBreaker(CircuitBreakerConfig(l1_cooldown_minutes=30))
        state = RiskState(breaker_level=BreakerLevel.L1, breaker_tripped_ns=BASE_NS)
        later = snapshot(now_ns=BASE_NS + 31 * 60 * NS_PER_SECOND, daily_pnl_pct=Decimal("0"))

        decision = breaker.try_recover(later, state)
        assert decision is not None
        assert decision.level is BreakerLevel.NONE

    def test_the_cooldown_alone_is_not_enough(self) -> None:
        # Waiting out the clock while still losing money just restarts the
        # countdown to L2.
        breaker = CircuitBreaker(CircuitBreakerConfig(l1_cooldown_minutes=30))
        state = RiskState(breaker_level=BreakerLevel.L1, breaker_tripped_ns=BASE_NS)
        later = snapshot(now_ns=BASE_NS + 31 * 60 * NS_PER_SECOND, daily_pnl_pct=Decimal("-1.5"))
        assert breaker.try_recover(later, state) is None

    def test_recovery_before_the_cooldown_is_refused(self) -> None:
        breaker = CircuitBreaker(CircuitBreakerConfig(l1_cooldown_minutes=30))
        state = RiskState(breaker_level=BreakerLevel.L1, breaker_tripped_ns=BASE_NS)
        early = snapshot(now_ns=BASE_NS + 60 * NS_PER_SECOND)
        assert breaker.try_recover(early, state) is None

    def test_l2_never_recovers_automatically(self) -> None:
        """Spec §7.5 — deliberately manual.

        A system that quietly resumes after thirty minutes will reach L2 again,
        and the second time it will have lost more.
        """
        breaker = CircuitBreaker()
        state = RiskState(breaker_level=BreakerLevel.L2, breaker_tripped_ns=BASE_NS)
        later = snapshot(now_ns=BASE_NS + 86400 * NS_PER_SECOND, daily_pnl_pct=Decimal("0"))
        assert breaker.try_recover(later, state) is None

    def test_l3_never_recovers_automatically(self) -> None:
        breaker = CircuitBreaker()
        state = RiskState(breaker_level=BreakerLevel.L3, breaker_tripped_ns=BASE_NS)
        later = snapshot(now_ns=BASE_NS + 86400 * NS_PER_SECOND)
        assert breaker.try_recover(later, state) is None

    def test_a_manual_reset_clears_the_breaker(self) -> None:
        decision = CircuitBreaker().manual_reset()
        assert decision.level is BreakerLevel.NONE
        assert BreakerTrigger.MANUAL in decision.triggers


class TestRiskStateAccounting:
    def test_drawdown_is_measured_from_the_peak(self) -> None:
        state = RiskState()
        state.observe_equity(Decimal("100000"))
        state.observe_equity(Decimal("120000"))
        state.observe_equity(Decimal("108000"))
        assert state.drawdown_pct(Decimal("108000")) == Decimal("10")

    def test_a_new_day_keeps_the_all_time_peak(self) -> None:
        # Drawdown is measured from the high-water mark, not from this morning;
        # resetting it daily would hide a slow bleed.
        state = RiskState()
        state.observe_equity(Decimal("150000"))
        state.start_new_day(Decimal("100000"))
        assert state.peak_equity == Decimal("150000")
        assert state.drawdown_pct(Decimal("100000")) > Decimal("33")

    def test_daily_pnl_is_relative_to_the_days_start(self) -> None:
        state = RiskState()
        state.start_new_day(Decimal("100000"))
        assert state.daily_pnl_pct(Decimal("98000")) == Decimal("-2")


class TestOrderRateTracker:
    def test_it_counts_only_the_window(self) -> None:
        tracker = OrderRateTracker(window_seconds=60)
        for i in range(10):
            tracker.record(RecentOrder("AAPL", Side.BUY, Decimal("1"), BASE_NS + i * NS_PER_SECOND))
        assert tracker.count(BASE_NS + 10 * NS_PER_SECOND) == 10
        assert tracker.count(BASE_NS + 300 * NS_PER_SECOND) == 0

    def test_roundtrips_count_direction_changes(self) -> None:
        tracker = OrderRateTracker(window_seconds=300)
        for i, side in enumerate([Side.BUY, Side.SELL, Side.BUY, Side.SELL]):
            tracker.record(RecentOrder("AAPL", side, Decimal("1"), BASE_NS + i * NS_PER_SECOND))
        assert tracker.roundtrips("AAPL", BASE_NS + 5 * NS_PER_SECOND) == 3

    def test_same_direction_orders_are_not_roundtrips(self) -> None:
        tracker = OrderRateTracker(window_seconds=300)
        for i in range(5):
            tracker.record(RecentOrder("AAPL", Side.BUY, Decimal("1"), BASE_NS + i * NS_PER_SECOND))
        assert tracker.roundtrips("AAPL", BASE_NS + 5 * NS_PER_SECOND) == 0


class TestKillSwitch:
    def test_it_starts_disengaged(self) -> None:
        assert not KillSwitch(clock=SimulatedClock()).is_engaged

    @pytest.mark.parametrize(
        "source", [KillSwitchSource.UI, KillSwitchSource.CLI, KillSwitchSource.HTTP]
    )
    def test_all_three_access_paths_work(self, source: KillSwitchSource) -> None:
        # Spec §FR-MON-03: the path you need is always the one that is broken.
        switch = KillSwitch(clock=SimulatedClock())
        event = switch.engage(reason="emergency", source=source)
        assert switch.is_engaged
        assert event.source is source

    def test_engaging_twice_is_recorded_but_changes_nothing(self) -> None:
        # The second attempt is useful in a post-mortem: it shows who else
        # reached for the switch.
        switch = KillSwitch(clock=SimulatedClock())
        switch.engage(reason="first", actor="alice")
        switch.engage(reason="second", actor="bob")
        assert switch.is_engaged
        assert len(switch.history) == 2

    def test_release_is_recorded(self) -> None:
        switch = KillSwitch(clock=SimulatedClock())
        switch.engage(reason="incident")
        switch.release(reason="resolved", actor="alice")
        assert not switch.is_engaged
        last = switch.last_event
        assert last is not None
        assert last.engaged is False
        assert last.actor == "alice"

    def test_engaging_is_pure_state_with_no_io(self) -> None:
        """Acceptance criterion #5 requires action within 5 seconds.

        Engaging does nothing but set a flag, so it cannot be delayed by a slow
        broker, a network hop, or a lock. The cancels that follow are dispatched
        by the execution layer; blocking *new* orders is instant regardless.
        """
        clock = SimulatedClock(start_ns=BASE_NS)
        switch = KillSwitch(clock=clock)
        switch.engage(reason="timing")
        # The simulated clock only moves when told to, so any I/O would show up
        # as a real-time delay this assertion cannot see — the point is that the
        # code path contains none.
        assert clock.now_ns() == BASE_NS
        assert switch.is_engaged


class TestRemainingBranches:
    """Small cases that complete the §8.3 coverage bar for this module."""

    def test_the_config_is_exposed_for_inspection(self) -> None:
        breaker = CircuitBreaker(CircuitBreakerConfig(l1_cooldown_minutes=45))
        assert breaker.config.l1_cooldown_minutes == 45

    def test_drawdown_alone_can_reach_l3(self) -> None:
        decision = CircuitBreaker().evaluate(snapshot(drawdown_pct=Decimal("12")), RiskState())
        assert decision.level is BreakerLevel.L3
        assert BreakerTrigger.DRAWDOWN in decision.triggers

    def test_a_manual_reset_to_l2_keeps_the_system_blocked(self) -> None:
        decision = CircuitBreaker().manual_reset(to_level=BreakerLevel.L2)
        assert decision.system_state is SystemState.BLOCKED

    def test_recovery_needs_a_recorded_trip_time(self) -> None:
        breaker = CircuitBreaker()
        state = RiskState(breaker_level=BreakerLevel.L1, breaker_tripped_ns=None)
        assert breaker.try_recover(snapshot(), state) is None

    def test_a_held_level_explains_itself(self) -> None:
        state = RiskState(breaker_level=BreakerLevel.L2)
        decision = CircuitBreaker().evaluate(snapshot(), state)
        assert "until it is cleared" in decision.reason

    def test_consecutive_losses_without_a_timestamp_do_not_trip(self) -> None:
        state = RiskState(consecutive_losses=5, last_loss_ns=None)
        assert not CircuitBreaker().evaluate(snapshot(), state).tripped

    def test_equity_is_observed_before_a_starting_value_exists(self) -> None:
        state = RiskState()
        assert state.daily_pnl_pct(Decimal("100")) == Decimal("0")
        assert state.drawdown_pct(Decimal("100")) == Decimal("0")
        state.observe_equity(Decimal("50000"))
        assert state.starting_equity == Decimal("50000")

    def test_a_kill_switch_with_no_history_reports_none(self) -> None:
        assert KillSwitch(clock=SimulatedClock()).last_event is None


class TestTheHoldSurvivesAcrossBars:
    """The no-de-escalation rule has to hold against state the *caller*
    actually maintains.

    It previously held against ``RiskState.breaker_level``, a field no
    production code ever wrote — the runtime kept the prior level only in its
    own attribute and handed ``evaluate`` a permanently-``NONE`` RiskState.
    So an L2 tripped by a -2.5% day cleared itself on the next bar that
    bounced back above the threshold: no human involved, ``/health`` reporting
    RUNNING, which is precisely the silent re-enable the module docstring
    forbids. ``evaluate`` now persists the level, so these tests drive it the
    way the runtime does — one RiskState, reused every bar.
    """

    def test_an_l2_holds_when_the_next_bar_bounces_back(self) -> None:
        breaker = CircuitBreaker(CircuitBreakerConfig())
        state = RiskState()

        tripped = breaker.evaluate(snapshot(daily_pnl_pct=Decimal("-2.5")), state)
        assert tripped.level is BreakerLevel.L2

        bounced = breaker.evaluate(snapshot(daily_pnl_pct=Decimal("-0.5")), state)
        assert bounced.level is BreakerLevel.L2
        assert "holding at" in bounced.reason

    def test_it_holds_even_when_the_day_turns_profitable(self) -> None:
        breaker = CircuitBreaker(CircuitBreakerConfig())
        state = RiskState()
        breaker.evaluate(snapshot(daily_pnl_pct=Decimal("-2.5")), state)

        assert (
            breaker.evaluate(snapshot(daily_pnl_pct=Decimal("1.5")), state).level is BreakerLevel.L2
        )

    def test_an_l3_never_de_escalates_either(self) -> None:
        breaker = CircuitBreaker(CircuitBreakerConfig())
        state = RiskState()
        breaker.evaluate(snapshot(daily_pnl_pct=Decimal("-6")), state)

        assert (
            breaker.evaluate(snapshot(daily_pnl_pct=Decimal("0")), state).level is BreakerLevel.L3
        )

    def test_only_a_manual_reset_clears_it(self) -> None:
        breaker = CircuitBreaker(CircuitBreakerConfig())
        state = RiskState()
        breaker.evaluate(snapshot(daily_pnl_pct=Decimal("-2.5")), state)

        breaker.manual_reset(state=state)

        assert state.breaker_level is BreakerLevel.NONE
        assert breaker.evaluate(snapshot(daily_pnl_pct=Decimal("0")), state).level is (
            BreakerLevel.NONE
        )

    def test_a_reset_without_the_state_does_not_stick(self) -> None:
        """Documents the trap: manual_reset returns a decision either way, but
        without the live state the next bar re-holds the old level."""
        breaker = CircuitBreaker(CircuitBreakerConfig())
        state = RiskState()
        breaker.evaluate(snapshot(daily_pnl_pct=Decimal("-2.5")), state)

        breaker.manual_reset()

        assert breaker.evaluate(snapshot(daily_pnl_pct=Decimal("0")), state).level is (
            BreakerLevel.L2
        )


class TestL1AutoRecoveryIsTheOneExemption:
    """L1 is defined as auto-recovering, so the hold must not trap it — but
    only once both the cooldown has elapsed and the condition has cleared."""

    def _tripped(self) -> tuple[CircuitBreaker, RiskState]:
        breaker = CircuitBreaker(CircuitBreakerConfig())
        state = RiskState()
        decision = breaker.evaluate(snapshot(daily_pnl_pct=Decimal("-1.2")), state)
        assert decision.level is BreakerLevel.L1
        return breaker, state

    def test_it_holds_before_the_cooldown_elapses(self) -> None:
        breaker, state = self._tripped()
        early = snapshot(daily_pnl_pct=Decimal("-0.1"), now_ns=BASE_NS + 29 * 60 * NS_PER_SECOND)
        assert breaker.evaluate(early, state).level is BreakerLevel.L1

    def test_it_recovers_once_the_cooldown_elapses_and_the_loss_clears(self) -> None:
        breaker, state = self._tripped()
        later = snapshot(daily_pnl_pct=Decimal("-0.1"), now_ns=BASE_NS + 31 * 60 * NS_PER_SECOND)
        assert breaker.evaluate(later, state).level is BreakerLevel.NONE
        assert state.breaker_tripped_ns is None

    def test_waiting_out_the_clock_while_still_losing_does_not_recover(self) -> None:
        breaker, state = self._tripped()
        still_losing = snapshot(
            daily_pnl_pct=Decimal("-1.5"), now_ns=BASE_NS + 31 * 60 * NS_PER_SECOND
        )
        assert breaker.evaluate(still_losing, state).level is BreakerLevel.L1
