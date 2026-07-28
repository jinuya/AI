"""The risk engine and its fifteen pre-trade checks — spec §7.2.

This is the file that matters most. Spec §8.3 sets a 95% coverage bar for this
module and says why: *"이 두 모듈은 예외 케이스가 곧 사고다."*

Two behaviours get particular attention because getting them wrong is expensive
in opposite directions:

* **Nothing over a limit gets through.** Covered exhaustively here and
  property-tested in ``tests/property/test_risk_invariants.py``.
* **Nothing that reduces risk gets blocked.** Spec §7.4 — a daily loss limit
  that rejects the stop-loss does the opposite of its job, at the worst moment.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest

from atrader.audit.logger import AuditLogger, InMemoryAuditSink
from atrader.config.schema import (
    AccountLimits,
    AppConfig,
    InstrumentSpec,
    OrderLimits,
    PositionLimits,
    RiskConfig,
    UniverseConfig,
)
from atrader.core.clock import NS_PER_SECOND, SimulatedClock
from atrader.core.ids import DeterministicIdGenerator
from atrader.core.models import AccountState, Position, TradingIntent
from atrader.core.types import (
    AlertLevel,
    BreakerLevel,
    DataQuality,
    OrderType,
    RiskAction,
    Side,
    SystemState,
    TargetType,
)
from atrader.risk.approval import ApprovalGate, ApprovalKind
from atrader.risk.checks.pretrade import PRETRADE_CHECKS, check_names
from atrader.risk.engine import RiskEngine
from atrader.risk.killswitch import KillSwitch, KillSwitchSource
from atrader.risk.state import RecentOrder, RiskSnapshot

BASE_NS = 1_700_000_000 * NS_PER_SECOND
EQUITY = Decimal("100000")
PRICE = Decimal("187.50")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_config(**risk_overrides: Any) -> AppConfig:
    risk = RiskConfig(**risk_overrides) if risk_overrides else RiskConfig()
    return AppConfig(
        account_equity=EQUITY,
        risk=risk,
        universe=UniverseConfig(
            symbols=("AAPL", "MSFT", "JPM"),
            sectors={"AAPL": "TECHNOLOGY", "MSFT": "TECHNOLOGY", "JPM": "FINANCIALS"},
        ),
        instruments=(
            InstrumentSpec(symbol="AAPL", sector="TECHNOLOGY"),
            InstrumentSpec(symbol="MSFT", sector="TECHNOLOGY"),
            InstrumentSpec(symbol="JPM", sector="FINANCIALS"),
        ),
    )


def make_snapshot(**overrides: Any) -> RiskSnapshot:
    defaults: dict[str, Any] = {
        "now_ns": BASE_NS,
        "system_state": SystemState.RUNNING,
        "breaker_level": BreakerLevel.NONE,
        "account": AccountState(cash=EQUITY, equity=EQUITY, buying_power=EQUITY),
        "positions": {},
        "universe": frozenset({"AAPL", "MSFT", "JPM"}),
        "sectors": {"AAPL": "TECHNOLOGY", "MSFT": "TECHNOLOGY", "JPM": "FINANCIALS"},
        "data_quality": {"AAPL": DataQuality.OK, "MSFT": DataQuality.OK, "JPM": DataQuality.OK},
        "prices": {"AAPL": PRICE, "MSFT": Decimal("410.00"), "JPM": Decimal("200.00")},
        "adv": {
            "AAPL": Decimal("50000000"),
            "MSFT": Decimal("20000000"),
            "JPM": Decimal("10000000"),
        },
        "is_market_open": {"AAPL": True, "MSFT": True, "JPM": True},
    }
    return RiskSnapshot(**{**defaults, **overrides})


def make_intent(**overrides: Any) -> TradingIntent:
    defaults: dict[str, Any] = {
        "intent_id": uuid4(),
        "strategy_id": "test_strategy",
        "symbol": "AAPL",
        "side": Side.BUY,
        "target_type": TargetType.SHARES,
        "target_value": Decimal("10"),
        "created_at_ns": BASE_NS,
    }
    return TradingIntent(**{**defaults, **overrides})


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(start_ns=BASE_NS)


@pytest.fixture
def engine(clock: SimulatedClock) -> RiskEngine:
    return RiskEngine(
        config=make_config(),
        clock=clock,
        ids=DeterministicIdGenerator(clock, seed=1),
        kill_switch=KillSwitch(clock=clock),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestCheckRegistry:
    def test_all_fifteen_checks_from_the_spec_are_present(self) -> None:
        assert len(PRETRADE_CHECKS) == 15

    def test_evaluation_order_matches_the_spec(self) -> None:
        # The order is not arbitrary: cheap categorical checks run before any
        # arithmetic, so a malformed intent cannot influence a size calculation.
        assert check_names() == (
            "system_state",
            "universe_whitelist",
            "data_quality",
            "trading_hours",
            "order_notional",
            "price_deviation",
            "adv_participation",
            "position_concentration",
            "sector_concentration",
            "leverage",
            "daily_loss_limit",
            "max_drawdown",
            "order_rate",
            "duplicate_order",
            "self_cross",
        )

    def test_growth_checks_are_exempt_for_reduce_only(self) -> None:
        # Spec §7.4: risk checks apply to orders that *increase* exposure.
        exempt = {c.name for c in PRETRADE_CHECKS if not c.applies_to_reduce_only}
        assert exempt == {
            "order_notional",
            "adv_participation",
            "position_concentration",
            "sector_concentration",
            "leverage",
            "daily_loss_limit",
            "max_drawdown",
            "order_rate",
        }

    def test_correctness_checks_always_apply(self) -> None:
        # A fat-fingered stop is still a fat finger; a self-cross is still a
        # regulatory problem whichever direction it points.
        always = {c.name for c in PRETRADE_CHECKS if c.applies_to_reduce_only}
        assert {"price_deviation", "self_cross", "duplicate_order"} <= always


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


class TestApproval:
    def test_a_normal_order_is_approved(self, engine: RiskEngine) -> None:
        decision = engine.evaluate(make_intent(), make_snapshot())
        assert decision.approved
        assert decision.action is RiskAction.ALLOW
        assert decision.order is not None
        assert decision.order.quantity == Decimal("10")

    def test_the_order_carries_the_risk_check_id(self, engine: RiskEngine) -> None:
        # Spec §4.5 makes this the audit link from an order back to its decision.
        decision = engine.evaluate(make_intent(), make_snapshot())
        assert decision.order is not None
        assert decision.order.risk_check_id == decision.risk_check_id

    def test_the_order_links_back_to_its_intent(self, engine: RiskEngine) -> None:
        intent = make_intent()
        decision = engine.evaluate(intent, make_snapshot())
        assert decision.order is not None
        assert decision.order.parent_intent_id == intent.intent_id

    def test_every_order_gets_a_unique_client_order_id(self, engine: RiskEngine) -> None:
        # The idempotency key (spec §FR-EXE-04). Collisions would defeat it.
        ids = set()
        for _ in range(20):
            decision = engine.evaluate(make_intent(), make_snapshot())
            assert decision.order is not None
            ids.add(decision.order.client_order_id)
        assert len(ids) == 20

    def test_market_orders_are_off_by_default(self, engine: RiskEngine) -> None:
        # Spec §FR-EXE-02: slippage on a market order in a thin name is unbounded.
        decision = engine.evaluate(make_intent(limit_price=None), make_snapshot())
        assert decision.order is not None
        assert decision.order.order_type is OrderType.LIMIT
        assert decision.order.limit_price is not None

    def test_an_aggressive_limit_crosses_the_spread_for_a_buy(self, engine: RiskEngine) -> None:
        decision = engine.evaluate(make_intent(side=Side.BUY), make_snapshot())
        assert decision.order is not None
        assert decision.order.limit_price is not None
        assert decision.order.limit_price > PRICE

    def test_an_aggressive_limit_crosses_downward_for_a_sell(self, engine: RiskEngine) -> None:
        decision = engine.evaluate(make_intent(side=Side.SELL), make_snapshot())
        assert decision.order is not None
        assert decision.order.limit_price is not None
        assert decision.order.limit_price < PRICE

    def test_market_orders_are_used_when_explicitly_enabled(self, clock: SimulatedClock) -> None:
        engine = RiskEngine(
            config=make_config(order=OrderLimits(allow_market_orders=True)),
            clock=clock,
            ids=DeterministicIdGenerator(clock),
            kill_switch=KillSwitch(clock=clock),
        )
        decision = engine.evaluate(make_intent(limit_price=None), make_snapshot())
        assert decision.order is not None
        assert decision.order.order_type is OrderType.MARKET
        assert decision.order.limit_price is None


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


class TestCheck01SystemState:
    @pytest.mark.parametrize(
        "state",
        [SystemState.STARTING, SystemState.THROTTLED, SystemState.BLOCKED, SystemState.HALTED],
    )
    def test_non_running_states_reject(self, engine: RiskEngine, state: SystemState) -> None:
        decision = engine.evaluate(make_intent(), make_snapshot(system_state=state))
        assert not decision.approved
        assert "not RUNNING" in decision.reason or "HALTED" in decision.reason

    def test_a_reconciliation_break_blocks_new_orders(self, engine: RiskEngine) -> None:
        # Spec §FR-EXE-05: we no longer know our real position.
        decision = engine.evaluate(make_intent(), make_snapshot(reconciliation_break=True))
        assert not decision.approved
        assert "reconciliation break" in decision.reason

    def test_a_margin_call_blocks_new_orders(self, engine: RiskEngine) -> None:
        # Spec §FR-PF-04: below the reduce-only threshold, only exposure
        # reduction is allowed — set by portfolio.margin.MarginMonitor.
        decision = engine.evaluate(make_intent(), make_snapshot(margin_reduce_only=True))
        assert not decision.approved
        assert "margin call" in decision.reason

    def test_a_margin_call_does_not_block_reducing_orders(self, engine: RiskEngine) -> None:
        position = Position(symbol="AAPL", quantity=Decimal(50), avg_price=PRICE)
        decision = engine.evaluate(
            make_intent(side=Side.SELL, target_value=Decimal("10")),
            make_snapshot(margin_reduce_only=True, positions={"AAPL": position}),
            reduce_only=True,
        )
        assert decision.approved


class TestCheck02Universe:
    def test_an_unknown_symbol_is_rejected_critically(self, engine: RiskEngine) -> None:
        # The last line of defence against a hallucinated ticker (spec §7.7).
        decision = engine.evaluate(make_intent(symbol="FAKECO"), make_snapshot())
        assert not decision.approved
        assert decision.alert_level is AlertLevel.CRITICAL
        assert "hallucination" in decision.reason

    def test_a_held_position_can_be_closed_after_delisting_from_the_universe(
        self, engine: RiskEngine
    ) -> None:
        # Otherwise removing a name from the whitelist traps the position we own.
        snapshot = make_snapshot(
            universe=frozenset({"MSFT"}),
            positions={
                "AAPL": Position(
                    symbol="AAPL", quantity=Decimal("100"), avg_price=PRICE, last_price=PRICE
                )
            },
        )
        decision = engine.evaluate(
            make_intent(side=Side.SELL, target_value=Decimal("100")),
            snapshot,
            reduce_only=True,
        )
        assert decision.approved


class TestCheck03DataQuality:
    @pytest.mark.parametrize("quality", [DataQuality.DEGRADED, DataQuality.STALE])
    def test_bad_data_blocks_new_orders(self, engine: RiskEngine, quality: DataQuality) -> None:
        snapshot = make_snapshot(data_quality={"AAPL": quality})
        decision = engine.evaluate(make_intent(), snapshot)
        assert not decision.approved
        assert quality.value in decision.reason

    def test_an_unknown_symbols_data_is_treated_as_stale(self, engine: RiskEngine) -> None:
        decision = engine.evaluate(make_intent(), make_snapshot(data_quality={}))
        assert not decision.approved

    def test_exposure_can_still_be_reduced_on_degraded_data(self, engine: RiskEngine) -> None:
        # Being trapped in a position whose feed died is worse than closing it
        # on imperfect data.
        snapshot = make_snapshot(
            data_quality={"AAPL": DataQuality.DEGRADED},
            positions={
                "AAPL": Position(
                    symbol="AAPL", quantity=Decimal("100"), avg_price=PRICE, last_price=PRICE
                )
            },
        )
        decision = engine.evaluate(
            make_intent(side=Side.SELL, target_value=Decimal("100")), snapshot, reduce_only=True
        )
        assert decision.approved


class TestCheck04TradingHours:
    def test_a_closed_market_queues_rather_than_rejecting(self, engine: RiskEngine) -> None:
        decision = engine.evaluate(make_intent(), make_snapshot(is_market_open={"AAPL": False}))
        assert not decision.approved
        assert decision.action is RiskAction.QUEUE


class TestCheck05OrderNotional:
    def test_an_oversized_order_is_reduced_not_rejected(self, engine: RiskEngine) -> None:
        # 2% of 100k = 2000 -> ~10 shares at 187.50. A strategy that wanted 100
        # is better served by 10 than by nothing.
        decision = engine.evaluate(make_intent(target_value=Decimal("100")), make_snapshot())
        assert decision.approved
        assert decision.was_reduced
        assert decision.order is not None
        assert decision.order.quantity * PRICE <= EQUITY * Decimal("0.02")


class TestCheck06FatFinger:
    def test_a_wildly_off_limit_price_is_rejected_critically(self, engine: RiskEngine) -> None:
        # A misplaced decimal point is the classic case.
        decision = engine.evaluate(make_intent(limit_price=Decimal("1875.00")), make_snapshot())
        assert not decision.approved
        assert decision.alert_level is AlertLevel.CRITICAL
        assert "fat-finger" in decision.reason

    def test_a_price_inside_the_band_passes(self, engine: RiskEngine) -> None:
        decision = engine.evaluate(make_intent(limit_price=Decimal("190.00")), make_snapshot())
        assert decision.approved

    def test_the_fat_finger_guard_applies_to_liquidation_too(self, engine: RiskEngine) -> None:
        snapshot = make_snapshot(
            positions={
                "AAPL": Position(
                    symbol="AAPL", quantity=Decimal("100"), avg_price=PRICE, last_price=PRICE
                )
            }
        )
        decision = engine.evaluate(
            make_intent(side=Side.SELL, target_value=Decimal("100"), limit_price=Decimal("18.75")),
            snapshot,
            reduce_only=True,
        )
        assert not decision.approved


class TestCheck07AdvParticipation:
    def test_an_order_above_the_adv_cap_is_reduced(self, engine: RiskEngine) -> None:
        snapshot = make_snapshot(
            adv={"AAPL": Decimal("100")},
            account=AccountState(equity=Decimal("100000000"), cash=Decimal("100000000")),
        )
        decision = engine.evaluate(make_intent(target_value=Decimal("1000")), snapshot)
        assert decision.order is None or decision.order.quantity <= Decimal("5")

    def test_a_missing_adv_estimate_does_not_block_trading(self, engine: RiskEngine) -> None:
        # Rejecting outright would block every newly listed name; the size and
        # concentration caps still bound the damage.
        decision = engine.evaluate(make_intent(), make_snapshot(adv={}))
        assert decision.approved


class TestCheck08PositionConcentration:
    def test_an_order_that_would_breach_the_cap_is_reduced(self, engine: RiskEngine) -> None:
        # Already at 9.5% of a 10% cap: only a sliver of headroom remains.
        snapshot = make_snapshot(
            positions={
                "AAPL": Position(
                    symbol="AAPL", quantity=Decimal("50"), avg_price=PRICE, last_price=PRICE
                )
            }
        )
        decision = engine.evaluate(make_intent(target_value=Decimal("10")), snapshot)
        assert decision.approved
        assert decision.order is not None
        resulting = (Decimal("50") + decision.order.quantity) * PRICE
        assert resulting <= EQUITY * Decimal("0.10") + PRICE


class TestCheck09SectorConcentration:
    def test_the_sector_cap_counts_every_symbol_in_the_sector(self, clock: SimulatedClock) -> None:
        engine = RiskEngine(
            config=make_config(
                position=PositionLimits(max_position_pct=Decimal(50), max_sector_pct=Decimal(20)),
                order=OrderLimits(max_order_notional_pct=Decimal(50)),
            ),
            clock=clock,
            ids=DeterministicIdGenerator(clock),
            kill_switch=KillSwitch(clock=clock),
        )
        # MSFT already fills 19% of the 20% TECHNOLOGY cap.
        snapshot = make_snapshot(
            positions={
                "MSFT": Position(
                    symbol="MSFT",
                    quantity=Decimal("46"),
                    avg_price=Decimal("410"),
                    last_price=Decimal("410"),
                )
            }
        )
        decision = engine.evaluate(make_intent(target_value=Decimal("100")), snapshot)
        tech_exposure = Decimal("46") * Decimal("410")
        if decision.order is not None:
            tech_exposure += decision.order.quantity * PRICE
        assert tech_exposure <= EQUITY * Decimal("0.20") + PRICE


class TestCheck10Leverage:
    def test_gross_leverage_above_the_cap_is_rejected(self, clock: SimulatedClock) -> None:
        engine = RiskEngine(
            config=make_config(
                account=AccountLimits(max_leverage=Decimal("1.0")),
                position=PositionLimits(max_position_pct=Decimal(100)),
                order=OrderLimits(max_order_notional_pct=Decimal(100)),
            ),
            clock=clock,
            ids=DeterministicIdGenerator(clock),
            kill_switch=KillSwitch(clock=clock),
        )
        # The existing position is in a different sector, so only the leverage
        # cap can be what stops this — 100k of JPM plus 18.75k of AAPL is 1.19x
        # gross on 100k of equity.
        snapshot = make_snapshot(
            positions={
                "JPM": Position(
                    symbol="JPM",
                    quantity=Decimal("500"),
                    avg_price=Decimal("200"),
                    last_price=Decimal("200"),
                )
            }
        )
        decision = engine.evaluate(make_intent(target_value=Decimal("100")), snapshot)
        assert not decision.approved
        assert "leverage" in decision.reason


class TestCheck11And12LossLimits:
    def test_the_daily_loss_limit_blocks_new_exposure(self, engine: RiskEngine) -> None:
        decision = engine.evaluate(make_intent(), make_snapshot(daily_pnl_pct=Decimal("-2.5")))
        assert not decision.approved
        assert decision.alert_level is AlertLevel.CRITICAL

    def test_the_drawdown_limit_blocks_new_exposure(self, engine: RiskEngine) -> None:
        decision = engine.evaluate(make_intent(), make_snapshot(drawdown_pct=Decimal("12")))
        assert not decision.approved

    def test_a_stop_loss_is_never_blocked_by_the_loss_limit(self, engine: RiskEngine) -> None:
        """Spec §7.4 — the single most important exemption in the system.

        A daily loss limit that rejects the stop-loss order does the exact
        opposite of its purpose, and does it precisely when it matters.
        """
        snapshot = make_snapshot(
            daily_pnl_pct=Decimal("-5.0"),
            drawdown_pct=Decimal("15.0"),
            positions={
                "AAPL": Position(
                    symbol="AAPL", quantity=Decimal("100"), avg_price=PRICE, last_price=PRICE
                )
            },
        )
        decision = engine.evaluate(
            make_intent(side=Side.SELL, target_value=Decimal("100")), snapshot, reduce_only=True
        )
        assert decision.approved, decision.reason

    def test_liquidation_works_even_in_the_blocked_state(self, engine: RiskEngine) -> None:
        snapshot = make_snapshot(
            system_state=SystemState.BLOCKED,
            breaker_level=BreakerLevel.L2,
            positions={
                "AAPL": Position(
                    symbol="AAPL", quantity=Decimal("100"), avg_price=PRICE, last_price=PRICE
                )
            },
        )
        decision = engine.evaluate(
            make_intent(side=Side.SELL, target_value=Decimal("100")), snapshot, reduce_only=True
        )
        assert decision.approved


class TestCheck13OrderRate:
    def test_the_rate_limit_throttles(self, engine: RiskEngine) -> None:
        decision = engine.evaluate(make_intent(), make_snapshot(orders_last_minute=30))
        assert not decision.approved
        assert decision.action is RiskAction.THROTTLE


class TestCheck14Duplicate:
    def test_a_repeat_inside_the_window_is_rejected(self, engine: RiskEngine) -> None:
        # Far more often a retry bug or a strategy re-firing than a real trade.
        snapshot = make_snapshot(
            recent_orders=(
                RecentOrder("AAPL", Side.BUY, Decimal("10"), BASE_NS - 2 * NS_PER_SECOND),
            )
        )
        decision = engine.evaluate(make_intent(), snapshot)
        assert not decision.approved
        assert "duplicate window" in decision.reason

    def test_an_order_outside_the_window_passes(self, engine: RiskEngine) -> None:
        snapshot = make_snapshot(
            recent_orders=(
                RecentOrder("AAPL", Side.BUY, Decimal("10"), BASE_NS - 30 * NS_PER_SECOND),
            )
        )
        assert engine.evaluate(make_intent(), snapshot).approved

    def test_the_opposite_side_is_not_a_duplicate(self, engine: RiskEngine) -> None:
        snapshot = make_snapshot(
            recent_orders=(RecentOrder("AAPL", Side.SELL, Decimal("10"), BASE_NS - NS_PER_SECOND),)
        )
        assert engine.evaluate(make_intent(side=Side.BUY), snapshot).approved


class TestCheck15SelfCross:
    def test_an_opposing_working_order_blocks_the_other_side(self, engine: RiskEngine) -> None:
        # Spec §7.2 #15: two of our own orders crossing looks like manipulation.
        snapshot = make_snapshot(
            open_orders=(RecentOrder("AAPL", Side.SELL, Decimal("50"), BASE_NS - 1000),)
        )
        decision = engine.evaluate(make_intent(side=Side.BUY), snapshot)
        assert not decision.approved
        assert "cross" in decision.reason

    def test_a_same_side_working_order_is_fine(self, engine: RiskEngine) -> None:
        snapshot = make_snapshot(
            open_orders=(RecentOrder("AAPL", Side.BUY, Decimal("50"), BASE_NS - 1000),)
        )
        assert engine.evaluate(make_intent(side=Side.BUY), snapshot).approved


# ---------------------------------------------------------------------------
# Engine behaviour
# ---------------------------------------------------------------------------


class TestKillSwitch:
    def test_it_blocks_everything_new(self, engine: RiskEngine) -> None:
        engine.kill_switch.engage(reason="test", source=KillSwitchSource.CLI)
        decision = engine.evaluate(make_intent(), make_snapshot())
        assert not decision.approved
        assert decision.reason == "kill switch engaged"

    def test_it_is_checked_before_anything_else(self, engine: RiskEngine) -> None:
        # Spec §FR-MON-03: it takes precedence over all other logic. Even with a
        # snapshot that would fail several other checks, this is the reason given.
        engine.kill_switch.engage(reason="test")
        decision = engine.evaluate(
            make_intent(symbol="FAKECO"),
            make_snapshot(system_state=SystemState.HALTED, data_quality={}),
        )
        assert decision.reason == "kill switch engaged"

    def test_liquidation_still_works_with_the_switch_engaged(self, engine: RiskEngine) -> None:
        engine.kill_switch.engage(reason="test", liquidate=True)
        snapshot = make_snapshot(
            positions={
                "AAPL": Position(
                    symbol="AAPL", quantity=Decimal("100"), avg_price=PRICE, last_price=PRICE
                )
            }
        )
        decision = engine.evaluate(
            make_intent(side=Side.SELL, target_value=Decimal("100")), snapshot, reduce_only=True
        )
        assert decision.approved

    def test_release_restores_trading(self, engine: RiskEngine) -> None:
        engine.kill_switch.engage(reason="test")
        engine.kill_switch.release(reason="resolved")
        assert engine.evaluate(make_intent(), make_snapshot()).approved


class TestFailClosed:
    def test_an_exception_inside_a_check_becomes_a_rejection(
        self, engine: RiskEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crashing check must never become a bypass.

        "The checker threw, so the order went out" is the worst possible
        failure mode for this system, so the engine catches everything.
        """

        def explode(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("check exploded")

        monkeypatch.setattr(engine, "_resolve_quantity", explode)
        decision = engine.evaluate(make_intent(), make_snapshot())

        assert not decision.approved
        assert decision.action is RiskAction.REJECT
        assert decision.alert_level is AlertLevel.CRITICAL
        assert "fail-closed" in decision.reason

    def test_a_missing_price_rejects_rather_than_guessing(self, engine: RiskEngine) -> None:
        decision = engine.evaluate(make_intent(limit_price=None), make_snapshot(prices={}))
        assert not decision.approved
        assert "reference price" in decision.reason

    def test_an_expired_intent_is_rejected(self, engine: RiskEngine) -> None:
        intent = make_intent(valid_until_ns=BASE_NS - 1)
        assert not engine.evaluate(intent, make_snapshot()).approved


class TestReferencePrice:
    def test_the_market_price_wins_over_the_intents_own_limit(self, engine: RiskEngine) -> None:
        # Otherwise a strategy could set its own reference and walk straight
        # past the fat-finger check.
        decision = engine.evaluate(make_intent(limit_price=Decimal("1000")), make_snapshot())
        assert not decision.approved
        assert "fat-finger" in decision.reason


class TestTargetTypes:
    def test_notional_targets_convert_to_shares(self, engine: RiskEngine) -> None:
        decision = engine.evaluate(
            make_intent(target_type=TargetType.NOTIONAL, target_value=Decimal("1875")),
            make_snapshot(),
        )
        assert decision.approved
        assert decision.order is not None
        assert decision.order.quantity == Decimal("10")

    def test_target_weight_computes_the_delta_from_the_current_position(
        self, engine: RiskEngine
    ) -> None:
        decision = engine.evaluate(
            make_intent(target_type=TargetType.TARGET_WEIGHT, target_value=Decimal("0.02")),
            make_snapshot(),
        )
        assert decision.approved
        assert decision.order is not None
        assert decision.order.quantity == Decimal("10")  # 2% of 100k / 187.50

    def test_the_rebalance_deadband_suppresses_tiny_trades(self, engine: RiskEngine) -> None:
        # Spec §FR-PF-03: without it, weight drift causes constant trading and
        # the commission eats the account.
        snapshot = make_snapshot(
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("10"),
                    avg_price=PRICE,
                    last_price=PRICE,
                )
            }
        )
        decision = engine.evaluate(
            make_intent(target_type=TargetType.TARGET_WEIGHT, target_value=Decimal("0.019")),
            snapshot,
        )
        assert not decision.approved
        assert "zero" in decision.reason


class TestHumanApproval:
    def test_a_large_order_requires_sign_off(self, clock: SimulatedClock) -> None:
        gate = ApprovalGate(clock, DeterministicIdGenerator(clock))
        engine = RiskEngine(
            config=make_config(
                order=OrderLimits(
                    max_order_notional_pct=Decimal(50), human_approval_threshold_pct=Decimal(10)
                ),
                position=PositionLimits(max_position_pct=Decimal(60)),
            ),
            clock=clock,
            ids=DeterministicIdGenerator(clock),
            kill_switch=KillSwitch(clock=clock),
            approvals=gate,
        )
        decision = engine.evaluate(make_intent(target_value=Decimal("100")), make_snapshot())

        assert decision.action is RiskAction.REQUIRE_APPROVAL
        assert decision.approval_request_id is not None
        assert gate.get(decision.approval_request_id) is not None

    def test_a_granted_approval_lets_the_order_through(self, clock: SimulatedClock) -> None:
        gate = ApprovalGate(clock, DeterministicIdGenerator(clock))
        engine = RiskEngine(
            config=make_config(
                order=OrderLimits(
                    max_order_notional_pct=Decimal(50), human_approval_threshold_pct=Decimal(10)
                ),
                position=PositionLimits(max_position_pct=Decimal(60)),
            ),
            clock=clock,
            ids=DeterministicIdGenerator(clock),
            kill_switch=KillSwitch(clock=clock),
            approvals=gate,
        )
        first = engine.evaluate(make_intent(target_value=Decimal("100")), make_snapshot())
        assert first.approval_request_id is not None
        gate.grant(first.approval_request_id, by="operator")

        second = engine.evaluate(
            make_intent(target_value=Decimal("100")),
            make_snapshot(),
            approval_request_id=first.approval_request_id,
        )
        assert second.approved

    def test_liquidation_is_never_gated_behind_a_human(self, clock: SimulatedClock) -> None:
        gate = ApprovalGate(clock, DeterministicIdGenerator(clock))
        engine = RiskEngine(
            config=make_config(
                order=OrderLimits(human_approval_threshold_pct=Decimal("0.1")),
            ),
            clock=clock,
            ids=DeterministicIdGenerator(clock),
            kill_switch=KillSwitch(clock=clock),
            approvals=gate,
        )
        snapshot = make_snapshot(
            positions={
                "AAPL": Position(
                    symbol="AAPL", quantity=Decimal("100"), avg_price=PRICE, last_price=PRICE
                )
            }
        )
        decision = engine.evaluate(
            make_intent(side=Side.SELL, target_value=Decimal("100")), snapshot, reduce_only=True
        )
        assert decision.approved


class TestAuditTrail:
    def test_passes_are_recorded_not_just_rejections(self, clock: SimulatedClock) -> None:
        """Spec §9.3 wants all risk check results.

        When a bad order *does* get through, the question is which check let it
        through — only answerable if the passes were recorded too.
        """
        sink = InMemoryAuditSink()
        engine = RiskEngine(
            config=make_config(),
            clock=clock,
            ids=DeterministicIdGenerator(clock),
            kill_switch=KillSwitch(clock=clock),
            audit=AuditLogger(sink, clock),
        )
        engine.evaluate(make_intent(), make_snapshot())

        record = sink.read_all()[-1]
        checks = record.payload["checks"]
        assert len(checks) == 15
        assert all(check["passed"] == "True" for check in checks)

    def test_rejections_record_the_failing_check(self, clock: SimulatedClock) -> None:
        sink = InMemoryAuditSink()
        engine = RiskEngine(
            config=make_config(),
            clock=clock,
            ids=DeterministicIdGenerator(clock),
            kill_switch=KillSwitch(clock=clock),
            audit=AuditLogger(sink, clock),
        )
        engine.evaluate(make_intent(symbol="FAKECO"), make_snapshot())

        record = sink.read_all()[-1]
        failed = [c for c in record.payload["checks"] if c["passed"] == "False"]
        assert failed[0]["name"] == "universe_whitelist"


class TestLatency:
    def test_evaluation_stays_within_the_five_millisecond_budget(self, engine: RiskEngine) -> None:
        # Spec §8.4: the pre-trade path is inline on the order path.
        latencies = [engine.evaluate(make_intent(), make_snapshot()).latency_ms for _ in range(200)]
        latencies.sort()
        p99 = latencies[int(len(latencies) * 0.99)]
        assert p99 < 5.0, f"p99 pre-trade latency {p99:.3f}ms exceeds the 5ms budget"


class TestApprovalGateTimeout:
    def test_an_unanswered_request_times_out_into_denial(self, clock: SimulatedClock) -> None:
        # Spec §7.6: executing later, unattended, is more dangerous than not
        # executing at all.
        gate = ApprovalGate(clock, DeterministicIdGenerator(clock), timeout_seconds=300)
        request = gate.request(ApprovalKind.LARGE_ORDER, "big trade")
        assert request.is_pending

        clock.advance(301 * NS_PER_SECOND)
        assert not gate.is_granted(request.request_id)
        refreshed = gate.get(request.request_id)
        assert refreshed is not None
        assert refreshed.status.value == "timed_out"

    def test_granting_after_the_timeout_is_refused(self, clock: SimulatedClock) -> None:
        gate = ApprovalGate(clock, DeterministicIdGenerator(clock), timeout_seconds=60)
        request = gate.request(ApprovalKind.LARGE_ORDER, "big trade")
        clock.advance(61 * NS_PER_SECOND)
        with pytest.raises(Exception, match="already"):
            gate.grant(request.request_id, by="operator")
