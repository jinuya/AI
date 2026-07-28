"""Target-weight rebalancing — spec §FR-PF-03."""

from __future__ import annotations

from decimal import Decimal

from atrader.core.clock import SimulatedClock
from atrader.core.ids import DeterministicIdGenerator
from atrader.core.models import Position
from atrader.core.types import Side, TargetType
from atrader.portfolio.rebalance import rebalance_to_weights

BASE_NS = 1_700_000_000_000_000_000
EQUITY = Decimal("100000")


def rebalance(
    targets: dict[str, Decimal],
    *,
    positions: dict[str, Position] | None = None,
    deadband_pct: Decimal = Decimal("0.5"),
) -> tuple[list, list]:
    clock = SimulatedClock(start_ns=BASE_NS)
    ids = DeterministicIdGenerator(clock, seed=1)
    return rebalance_to_weights(
        targets,
        positions=positions or {},
        equity=EQUITY,
        strategy_id="index_tracker",
        ids=ids,
        clock=clock,
        deadband_pct=deadband_pct,
    )


class TestRebalanceToWeights:
    def test_an_empty_book_moving_to_a_target_emits_a_buy(self) -> None:
        intents, targets = rebalance({"AAPL": Decimal("0.20")})
        assert len(intents) == 1
        assert intents[0].side is Side.BUY
        assert intents[0].target_type is TargetType.TARGET_WEIGHT
        assert intents[0].target_value == Decimal("0.20")
        assert targets[0].traded is True
        assert targets[0].drift_pct == Decimal("20")

    def test_a_position_already_on_target_emits_nothing(self) -> None:
        position = Position(symbol="AAPL", quantity=Decimal("200"), avg_price=Decimal("100"))
        # market_value = 200*100 = 20000 = 20% of 100000 equity, matches target exactly
        intents, targets = rebalance({"AAPL": Decimal("0.20")}, positions={"AAPL": position})
        assert intents == []
        assert targets[0].traded is False
        assert targets[0].drift_pct == Decimal("0")

    def test_drift_inside_the_deadband_is_skipped(self) -> None:
        # current weight = 19.8%, target 20% -> 0.2pp drift, inside a 0.5pp deadband
        position = Position(symbol="AAPL", quantity=Decimal("198"), avg_price=Decimal("100"))
        intents, targets = rebalance(
            {"AAPL": Decimal("0.20")}, positions={"AAPL": position}, deadband_pct=Decimal("0.5")
        )
        assert intents == []
        assert targets[0].traded is False

    def test_drift_outside_the_deadband_trades(self) -> None:
        # current weight = 15%, target 20% -> 5pp drift, outside a 0.5pp deadband
        position = Position(symbol="AAPL", quantity=Decimal("150"), avg_price=Decimal("100"))
        intents, targets = rebalance(
            {"AAPL": Decimal("0.20")}, positions={"AAPL": position}, deadband_pct=Decimal("0.5")
        )
        assert len(intents) == 1
        assert intents[0].side is Side.BUY
        assert targets[0].drift_pct == Decimal("5")

    def test_overweight_positions_emit_a_sell(self) -> None:
        position = Position(symbol="AAPL", quantity=Decimal("400"), avg_price=Decimal("100"))
        # current weight = 40%, target 20% -> sell
        intents, _ = rebalance({"AAPL": Decimal("0.20")}, positions={"AAPL": position})
        assert intents[0].side is Side.SELL

    def test_negative_target_weight_is_a_short_target(self) -> None:
        intents, targets = rebalance({"AAPL": Decimal("-0.10")})
        assert intents[0].side is Side.SELL
        assert intents[0].target_value == Decimal("-0.10")
        assert targets[0].drift_pct == Decimal("-10")

    def test_multiple_symbols_are_each_evaluated_independently(self) -> None:
        intents, targets = rebalance({"AAPL": Decimal("0.20"), "MSFT": Decimal("0.10")})
        assert {i.symbol for i in intents} == {"AAPL", "MSFT"}
        assert len(targets) == 2
