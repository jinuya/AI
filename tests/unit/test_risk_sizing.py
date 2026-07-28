"""Position sizing and the remaining risk-engine edge cases — spec §7.3.

Sizing is where "how much" is decided, and the spec's argument for volatility
targeting is the one worth keeping in mind while reading these tests:

    고정 금액으로 사면 변동성 높은 종목에서 리스크가 훨씬 커진다.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest

from atrader.config.schema import (
    AccountLimits,
    AppConfig,
    DataQualityConfig,
    InstrumentSpec,
    OrderLimits,
    PositionLimits,
    RiskConfig,
    SizingConfig,
    UniverseConfig,
)
from atrader.core.clock import NS_PER_SECOND, SimulatedClock
from atrader.core.ids import DeterministicIdGenerator
from atrader.core.models import AccountState, Position, TradingIntent
from atrader.core.types import (
    BreakerLevel,
    DataQuality,
    RiskAction,
    Side,
    SizingMethod,
    SystemState,
    TargetType,
)
from atrader.risk.approval import ApprovalGate, ApprovalKind, ApprovalStatus, needs_approval
from atrader.risk.engine import RiskEngine
from atrader.risk.killswitch import KillSwitch
from atrader.risk.sizing import correlated_exposure, size_position
from atrader.risk.state import RiskSnapshot

BASE_NS = 1_700_000_000 * NS_PER_SECOND
EQUITY = Decimal("100000")
PRICE = Decimal("100")


def snapshot(**overrides: Any) -> RiskSnapshot:
    defaults: dict[str, Any] = {
        "now_ns": BASE_NS,
        "system_state": SystemState.RUNNING,
        "breaker_level": BreakerLevel.NONE,
        "account": AccountState(cash=EQUITY, equity=EQUITY, buying_power=EQUITY),
    }
    return RiskSnapshot(**{**defaults, **overrides})


def risk_config(**overrides: Any) -> RiskConfig:
    return RiskConfig(**overrides)


class TestVolatilityTargeting:
    def test_a_more_volatile_name_gets_a_smaller_position(self) -> None:
        """The entire reason for the method.

        Same ticket size in a calm name and a wild one are not the same bet;
        sizing by ATR makes them comparable.
        """
        config = risk_config(
            sizing=SizingConfig(risk_per_trade_pct=Decimal("0.5"), atr_multiple=Decimal("2")),
            order=OrderLimits(max_order_notional_pct=Decimal("100")),
            position=PositionLimits(max_position_pct=Decimal("100")),
        )
        calm = size_position(
            symbol="AAPL", price=PRICE, config=config, snapshot=snapshot(), atr=Decimal("1")
        )
        wild = size_position(
            symbol="AAPL", price=PRICE, config=config, snapshot=snapshot(), atr=Decimal("8")
        )
        assert wild.quantity < calm.quantity

    def test_the_designed_loss_at_the_stop_is_the_risk_budget(self) -> None:
        # 0.5% of 100k = 500. A 2 x ATR(5) = 10 point stop -> 50 shares.
        config = risk_config(
            sizing=SizingConfig(risk_per_trade_pct=Decimal("0.5"), atr_multiple=Decimal("2")),
            order=OrderLimits(max_order_notional_pct=Decimal("100")),
            position=PositionLimits(max_position_pct=Decimal("100")),
        )
        result = size_position(
            symbol="AAPL", price=PRICE, config=config, snapshot=snapshot(), atr=Decimal("5")
        )
        assert result.quantity == Decimal("50")
        assert result.quantity * Decimal("2") * Decimal("5") == Decimal("500")

    def test_a_missing_atr_falls_back_rather_than_guessing(self) -> None:
        # An invented ATR would mis-size every position in a new listing, and
        # mis-size it in the dangerous direction if the guess were low.
        config = risk_config(order=OrderLimits(max_order_notional_pct=Decimal("100")))
        result = size_position(
            symbol="AAPL", price=PRICE, config=config, snapshot=snapshot(), atr=None
        )
        assert "fell back to fixed fraction" in result.reason
        assert result.quantity > 0

    def test_a_zero_atr_also_falls_back(self) -> None:
        config = risk_config(order=OrderLimits(max_order_notional_pct=Decimal("100")))
        result = size_position(
            symbol="AAPL", price=PRICE, config=config, snapshot=snapshot(), atr=Decimal("0")
        )
        assert "fell back" in result.reason


class TestKelly:
    def test_it_is_capped_at_the_configured_fraction(self) -> None:
        """Spec §7.3: full Kelly is optimal only if the edge estimate is exact.

        It never is, and the penalty for overestimating is geometric.
        """
        config = risk_config(
            sizing=SizingConfig(method=SizingMethod.KELLY, kelly_fraction=Decimal("0.25")),
            order=OrderLimits(max_order_notional_pct=Decimal("100")),
            position=PositionLimits(max_position_pct=Decimal("100")),
        )
        result = size_position(
            symbol="AAPL",
            price=PRICE,
            config=config,
            snapshot=snapshot(),
            win_probability=Decimal("0.95"),
            win_loss_ratio=Decimal("10"),
        )
        assert result.notional <= EQUITY * Decimal("0.25")

    def test_a_negative_edge_produces_no_position(self) -> None:
        config = risk_config(sizing=SizingConfig(method=SizingMethod.KELLY))
        result = size_position(
            symbol="AAPL",
            price=PRICE,
            config=config,
            snapshot=snapshot(),
            win_probability=Decimal("0.3"),
            win_loss_ratio=Decimal("1"),
        )
        assert result.is_empty
        assert "non-positive" in result.reason

    def test_missing_statistics_fall_back(self) -> None:
        config = risk_config(
            sizing=SizingConfig(method=SizingMethod.KELLY),
            order=OrderLimits(max_order_notional_pct=Decimal("100")),
        )
        result = size_position(symbol="AAPL", price=PRICE, config=config, snapshot=snapshot())
        assert "insufficient statistics" in result.reason

    def test_the_config_refuses_more_than_quarter_kelly(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SizingConfig(kelly_fraction=Decimal("1.0"))


class TestSizingGuards:
    def test_no_price_means_no_position(self) -> None:
        result = size_position(
            symbol="AAPL", price=Decimal("0"), config=risk_config(), snapshot=snapshot()
        )
        assert result.is_empty

    def test_no_equity_means_no_position(self) -> None:
        result = size_position(
            symbol="AAPL",
            price=PRICE,
            config=risk_config(),
            snapshot=snapshot(account=AccountState(equity=Decimal("0"))),
        )
        assert result.is_empty

    def test_confidence_scales_the_position_down(self) -> None:
        # Spec §7.7: a model's stated confidence must shrink the position, not
        # be logged and ignored.
        config = risk_config(order=OrderLimits(max_order_notional_pct=Decimal("100")))
        full = size_position(
            symbol="AAPL", price=PRICE, config=config, snapshot=snapshot(), atr=Decimal("2")
        )
        half = size_position(
            symbol="AAPL",
            price=PRICE,
            config=config,
            snapshot=snapshot(),
            atr=Decimal("2"),
            confidence=Decimal("0.5"),
        )
        assert half.quantity < full.quantity
        assert "confidence" in half.reason

    def test_sizing_respects_the_single_order_cap(self) -> None:
        # Sizing and the pre-trade check must agree, or every order would arrive
        # needing a REDUCE.
        config = risk_config(order=OrderLimits(max_order_notional_pct=Decimal("1")))
        result = size_position(
            symbol="AAPL", price=PRICE, config=config, snapshot=snapshot(), atr=Decimal("1")
        )
        assert result.notional <= EQUITY * Decimal("0.01")
        assert result.capped_by == "max_order_notional_pct"

    def test_sizing_respects_remaining_position_headroom(self) -> None:
        config = risk_config(
            order=OrderLimits(max_order_notional_pct=Decimal("100")),
            position=PositionLimits(max_position_pct=Decimal("10")),
        )
        held = snapshot(
            positions={
                "AAPL": Position(
                    symbol="AAPL", quantity=Decimal("95"), avg_price=PRICE, last_price=PRICE
                )
            }
        )
        result = size_position(
            symbol="AAPL", price=PRICE, config=config, snapshot=held, atr=Decimal("1")
        )
        assert result.quantity <= Decimal("5")
        assert result.capped_by == "max_position_pct"

    def test_quantities_are_floored_to_the_lot_size(self) -> None:
        config = risk_config(order=OrderLimits(max_order_notional_pct=Decimal("100")))
        result = size_position(
            symbol="AAPL",
            price=PRICE,
            config=config,
            snapshot=snapshot(),
            atr=Decimal("1"),
            lot_size=Decimal("100"),
        )
        assert result.quantity % Decimal("100") == 0


class TestCorrelatedExposure:
    def test_correlated_holdings_are_summed(self) -> None:
        # Spec §7.3: adding a name that moves with what you hold is growing one
        # bet, not diversifying.
        state = snapshot(
            positions={
                "MSFT": Position(
                    symbol="MSFT", quantity=Decimal("100"), avg_price=PRICE, last_price=PRICE
                ),
                "JPM": Position(
                    symbol="JPM", quantity=Decimal("100"), avg_price=PRICE, last_price=PRICE
                ),
            },
            correlations={("AAPL", "MSFT"): Decimal("0.85"), ("AAPL", "JPM"): Decimal("0.2")},
        )
        total, contributors = correlated_exposure("AAPL", state, Decimal("0.7"))
        assert contributors == ["MSFT"]
        assert total == Decimal("10000")

    def test_correlation_lookup_is_symmetric(self) -> None:
        state = snapshot(
            positions={
                "MSFT": Position(
                    symbol="MSFT", quantity=Decimal("10"), avg_price=PRICE, last_price=PRICE
                )
            },
            correlations={("MSFT", "AAPL"): Decimal("0.9")},
        )
        _, contributors = correlated_exposure("AAPL", state, Decimal("0.7"))
        assert contributors == ["MSFT"]

    def test_flat_positions_are_ignored(self) -> None:
        state = snapshot(positions={"MSFT": Position(symbol="MSFT", quantity=Decimal("0"))})
        total, contributors = correlated_exposure("AAPL", state, Decimal("0.7"))
        assert total == Decimal("0")
        assert contributors == []


# ---------------------------------------------------------------------------
# Remaining engine and gate edge cases
# ---------------------------------------------------------------------------


def build_engine(clock: SimulatedClock, **risk_overrides: Any) -> RiskEngine:
    config = AppConfig(
        account_equity=EQUITY,
        risk=RiskConfig(**risk_overrides) if risk_overrides else RiskConfig(),
        universe=UniverseConfig(symbols=("AAPL",), sectors={"AAPL": "TECHNOLOGY"}),
        instruments=(InstrumentSpec(symbol="AAPL", sector="TECHNOLOGY"),),
    )
    return RiskEngine(
        config=config,
        clock=clock,
        ids=DeterministicIdGenerator(clock),
        kill_switch=KillSwitch(clock=clock),
    )


def intent(**overrides: Any) -> TradingIntent:
    defaults: dict[str, Any] = {
        "intent_id": uuid4(),
        "strategy_id": "test",
        "symbol": "AAPL",
        "side": Side.BUY,
        "target_type": TargetType.SHARES,
        "target_value": Decimal("10"),
        "created_at_ns": BASE_NS,
    }
    return TradingIntent(**{**defaults, **overrides})


def live_snapshot(**overrides: Any) -> RiskSnapshot:
    defaults: dict[str, Any] = {
        "universe": frozenset({"AAPL"}),
        "sectors": {"AAPL": "TECHNOLOGY"},
        "data_quality": {"AAPL": DataQuality.OK},
        "prices": {"AAPL": PRICE},
        "is_market_open": {"AAPL": True},
    }
    return snapshot(**{**defaults, **overrides})


class TestEngineEdgeCases:
    def test_a_halt_stops_even_liquidation(self) -> None:
        clock = SimulatedClock(start_ns=BASE_NS)
        engine = build_engine(clock)
        state = live_snapshot(
            system_state=SystemState.HALTED,
            positions={
                "AAPL": Position(
                    symbol="AAPL", quantity=Decimal("100"), avg_price=PRICE, last_price=PRICE
                )
            },
        )
        decision = engine.evaluate(
            intent(side=Side.SELL, target_value=Decimal("100")), state, reduce_only=True
        )
        assert not decision.approved
        assert "HALTED" in decision.reason

    def test_a_limit_price_with_no_market_price_is_rejected(self) -> None:
        # Validating a fat finger needs something to compare against.
        clock = SimulatedClock(start_ns=BASE_NS)
        engine = build_engine(clock)
        decision = engine.evaluate(
            intent(limit_price=Decimal("100")), live_snapshot(prices={"AAPL": Decimal("0")})
        )
        assert not decision.approved

    def test_a_reduction_that_leaves_nothing_rejects(self) -> None:
        clock = SimulatedClock(start_ns=BASE_NS)
        engine = build_engine(clock, order=OrderLimits(max_order_notional_pct=Decimal("0.0001")))
        decision = engine.evaluate(intent(target_value=Decimal("1000")), live_snapshot())
        assert not decision.approved
        assert decision.action is RiskAction.REJECT

    def test_the_reduction_loop_terminates(self) -> None:
        # Two checks shrinking each other must not spin forever; the engine
        # rejects rather than shipping an order no check agreed on.
        clock = SimulatedClock(start_ns=BASE_NS)
        engine = build_engine(
            clock,
            order=OrderLimits(max_order_notional_pct=Decimal("5")),
            position=PositionLimits(max_position_pct=Decimal("3")),
        )
        decision = engine.evaluate(intent(target_value=Decimal("100")), live_snapshot())
        assert decision.action in (RiskAction.ALLOW, RiskAction.REJECT)

    def test_a_zero_equity_account_admits_nothing(self) -> None:
        clock = SimulatedClock(start_ns=BASE_NS)
        engine = build_engine(clock)
        decision = engine.evaluate(
            intent(), live_snapshot(account=AccountState(equity=Decimal("0")))
        )
        assert not decision.approved


class TestApprovalGateEdgeCases:
    def test_needs_approval_flags_a_new_symbol_first(self) -> None:
        assert (
            needs_approval(Decimal("1"), Decimal("10"), is_new_symbol=True)
            is ApprovalKind.NEW_SYMBOL
        )

    def test_needs_approval_flags_a_large_order(self) -> None:
        assert needs_approval(Decimal("15"), Decimal("10")) is ApprovalKind.LARGE_ORDER

    def test_a_normal_order_needs_no_approval(self) -> None:
        assert needs_approval(Decimal("1"), Decimal("10")) is None

    def test_deny_records_the_decision(self) -> None:
        clock = SimulatedClock(start_ns=BASE_NS)
        gate = ApprovalGate(clock, DeterministicIdGenerator(clock))
        request = gate.request(ApprovalKind.LARGE_ORDER, "big")
        denied = gate.deny(request.request_id, by="alice", note="too big")
        assert denied.status is ApprovalStatus.DENIED
        assert denied.decided_by == "alice"

    def test_denying_twice_is_idempotent(self) -> None:
        clock = SimulatedClock(start_ns=BASE_NS)
        gate = ApprovalGate(clock, DeterministicIdGenerator(clock))
        request = gate.request(ApprovalKind.LARGE_ORDER, "big")
        gate.deny(request.request_id, by="alice")
        assert gate.deny(request.request_id, by="bob").decided_by == "alice"

    def test_pending_excludes_decided_requests(self) -> None:
        clock = SimulatedClock(start_ns=BASE_NS)
        gate = ApprovalGate(clock, DeterministicIdGenerator(clock))
        first = gate.request(ApprovalKind.LARGE_ORDER, "one")
        gate.request(ApprovalKind.NEW_SYMBOL, "two")
        gate.grant(first.request_id, by="alice")
        assert len(gate.pending()) == 1

    def test_sweep_reports_what_expired(self) -> None:
        clock = SimulatedClock(start_ns=BASE_NS)
        gate = ApprovalGate(clock, DeterministicIdGenerator(clock), timeout_seconds=60)
        gate.request(ApprovalKind.LARGE_ORDER, "one")
        gate.request(ApprovalKind.NEW_SYMBOL, "two")
        clock.advance(61 * NS_PER_SECOND)
        assert len(gate.sweep()) == 2

    def test_seconds_remaining_counts_down_to_zero(self) -> None:
        clock = SimulatedClock(start_ns=BASE_NS)
        gate = ApprovalGate(clock, DeterministicIdGenerator(clock), timeout_seconds=300)
        request = gate.request(ApprovalKind.LARGE_ORDER, "big")
        assert request.seconds_remaining(BASE_NS) == pytest.approx(300.0)
        assert request.seconds_remaining(BASE_NS + 400 * NS_PER_SECOND) == 0.0

    def test_an_unknown_request_id_raises(self) -> None:
        clock = SimulatedClock(start_ns=BASE_NS)
        gate = ApprovalGate(clock, DeterministicIdGenerator(clock))
        assert gate.get(uuid4()) is None
        with pytest.raises(KeyError):
            gate.grant(uuid4(), by="alice")

    def test_a_non_positive_timeout_is_rejected(self) -> None:
        clock = SimulatedClock(start_ns=BASE_NS)
        with pytest.raises(ValueError, match="must be positive"):
            ApprovalGate(clock, DeterministicIdGenerator(clock), timeout_seconds=0)

    def test_is_granted_is_false_without_a_request_id(self) -> None:
        clock = SimulatedClock(start_ns=BASE_NS)
        gate = ApprovalGate(clock, DeterministicIdGenerator(clock))
        assert not gate.is_granted(uuid4())


class TestDataQualityConfigDefaults:
    def test_the_shipped_defaults_match_the_spec(self) -> None:
        config = DataQualityConfig()
        assert config.max_feed_lag_ms == 1000
        assert config.stale_threshold_seconds == 30
        assert config.price_jump_threshold_pct == Decimal("20.0")
        assert config.dual_source_divergence_pct == Decimal("0.5")


class TestAccountLimitDefaults:
    def test_the_shipped_defaults_match_the_spec(self) -> None:
        limits = AccountLimits()
        assert limits.max_leverage == Decimal("1.0")
        assert limits.daily_loss_limit_pct == Decimal("2.0")
        assert limits.max_drawdown_pct == Decimal("10.0")


class TestDecisionSummaries:
    """Human-facing rendering — read during an incident, so it must be right."""

    def test_an_approval_says_so(self) -> None:
        clock = SimulatedClock(start_ns=BASE_NS)
        engine = build_engine(clock)
        decision = engine.evaluate(intent(), live_snapshot())
        assert decision.summary().startswith("approved")
        assert decision.failed_checks == ()

    def test_a_reduced_order_is_labelled(self) -> None:
        clock = SimulatedClock(start_ns=BASE_NS)
        engine = build_engine(clock, order=OrderLimits(max_order_notional_pct=Decimal("1")))
        decision = engine.evaluate(intent(target_value=Decimal("100")), live_snapshot())
        assert decision.was_reduced
        assert "size reduced" in decision.summary()

    def test_a_rejection_carries_its_reason(self) -> None:
        clock = SimulatedClock(start_ns=BASE_NS)
        engine = build_engine(clock)
        decision = engine.evaluate(intent(), live_snapshot(system_state=SystemState.BLOCKED))
        assert decision.summary().startswith("REJECT")
        assert len(decision.failed_checks) == 1

    def test_correlation_with_itself_is_one(self) -> None:
        assert snapshot().correlation("AAPL", "AAPL") == Decimal("1")

    def test_gross_and_sector_exposure_aggregate(self) -> None:
        state = snapshot(
            positions={
                "AAPL": Position(
                    symbol="AAPL", quantity=Decimal("10"), avg_price=PRICE, last_price=PRICE
                ),
                "MSFT": Position(
                    symbol="MSFT", quantity=Decimal("-5"), avg_price=PRICE, last_price=PRICE
                ),
            },
            sectors={"AAPL": "TECHNOLOGY", "MSFT": "TECHNOLOGY"},
        )
        # Gross counts the short at its absolute size — that is the exposure.
        assert state.gross_exposure() == Decimal("1500")
        assert state.sector_exposure("TECHNOLOGY") == Decimal("1500")
        assert state.sector_exposure("ENERGY") == Decimal("0")

    def test_an_unpriced_position_falls_back_to_cost(self) -> None:
        position = Position(symbol="AAPL", quantity=Decimal("10"), avg_price=PRICE)
        assert position.market_value == Decimal("1000")
        assert position.side_to_close() is Side.SELL
        assert Position(symbol="AAPL").side_to_close() is None
