"""Deterministic replay harness — acceptance criterion #3.

Two things must both be true for this harness to mean anything: it must pass
on a genuinely deterministic scenario, and it must actually *catch* a
scenario seeded to diverge — a harness that always reports a match is not
verifying anything.
"""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from atrader.backtest.cost_model import CostModel
from atrader.backtest.engine import BacktestEngine
from atrader.backtest.replay import assert_replay_matches, compare_replays, order_sequence_bytes
from atrader.config.schema import AppConfig, InstrumentSpec, RiskConfig, UniverseConfig
from atrader.core.clock import NS_PER_SECOND, SimulatedClock
from atrader.core.ids import DeterministicIdGenerator
from atrader.core.models import Order, TradingIntent
from atrader.core.types import OrderStatus, OrderType, Side, TargetType, TimeInForce
from atrader.features.engine import FeatureEngine
from atrader.features.registry import FeatureRegistry
from atrader.features.store import FeatureStore
from atrader.marketdata.models import Bar
from atrader.risk.engine import RiskEngine
from atrader.risk.killswitch import KillSwitch
from atrader.strategy.base import Strategy, StrategyContext

BASE_NS = 1_700_000_000 * NS_PER_SECOND
ONE_DAY_NS = 86_400 * NS_PER_SECOND


def make_bar(*, index: int, price: int) -> Bar:
    return Bar(
        symbol="AAPL",
        interval="1d",
        open_ts=BASE_NS + index * ONE_DAY_NS,
        close_ts=BASE_NS + (index + 1) * ONE_DAY_NS,
        open=Decimal(price),
        high=Decimal(price + 3),
        low=Decimal(price - 3),
        close=Decimal(price + 1),
        volume=Decimal("1000000"),
        is_final=True,
    )


def bars(n: int) -> list[Bar]:
    return [make_bar(index=i, price=100 + i) for i in range(n)]


class _BuyEveryBarStrategy(Strategy):
    def on_bar(self, bar: Bar, context: StrategyContext) -> list[TradingIntent]:
        return [
            TradingIntent(
                intent_id=context.ids.new_id(),
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,
                target_type=TargetType.SHARES,
                target_value=Decimal("1"),
                created_at_ns=context.now_ns,
            )
        ]


def make_config() -> AppConfig:
    return AppConfig(
        account_equity=Decimal("100000"),
        risk=RiskConfig(),
        universe=UniverseConfig(symbols=("AAPL",), sectors={"AAPL": "TECHNOLOGY"}),
        instruments=(InstrumentSpec(symbol="AAPL", sector="TECHNOLOGY"),),
    )


def deterministic_engine_factory() -> BacktestEngine:
    clock = SimulatedClock(start_ns=BASE_NS)
    ids = DeterministicIdGenerator(clock, seed=7)
    risk_engine = RiskEngine(
        config=make_config(), clock=clock, ids=ids, kill_switch=KillSwitch(clock=clock)
    )
    feature_engine = FeatureEngine(store=FeatureStore(), registry=FeatureRegistry(), specs=())
    return BacktestEngine(
        strategies=(_BuyEveryBarStrategy("buyer"),),
        risk_engine=risk_engine,
        clock=clock,
        ids=ids,
        feature_engine=feature_engine,
        starting_cash=Decimal("100000"),
        cost_model=CostModel(impact_coefficient=Decimal("0")),
    )


class _NondeterministicIdGenerator:
    """Deliberately breaks determinism — the replay harness's negative case.

    Reads real system entropy via ``uuid4()`` (outside ``atrader.core``,
    which is exactly what ``tests/unit/test_determinism_lint.py`` forbids in
    production code — this is a test fixture built specifically to violate
    that rule, so this file's harness can prove it actually catches the
    violation rather than rubber-stamping every run as a match).
    """

    def new_id(self) -> UUID:
        return uuid4()


def nondeterministic_engine_factory() -> BacktestEngine:
    clock = SimulatedClock(start_ns=BASE_NS)
    risk_engine = RiskEngine(
        config=make_config(),
        clock=clock,
        ids=_NondeterministicIdGenerator(),  # type: ignore[arg-type]
        kill_switch=KillSwitch(clock=clock),
    )
    feature_engine = FeatureEngine(store=FeatureStore(), registry=FeatureRegistry(), specs=())
    return BacktestEngine(
        strategies=(_BuyEveryBarStrategy("buyer"),),
        risk_engine=risk_engine,
        clock=clock,
        ids=_NondeterministicIdGenerator(),  # type: ignore[arg-type]
        feature_engine=feature_engine,
        starting_cash=Decimal("100000"),
        cost_model=CostModel(impact_coefficient=Decimal("0")),
    )


class TestOrderSequenceBytes:
    def test_empty_sequence_is_stable(self) -> None:
        assert order_sequence_bytes([]) == order_sequence_bytes([])

    def test_identical_orders_encode_identically(self) -> None:
        order_id = uuid4()
        risk_check_id = uuid4()

        def make() -> Order:
            return Order(
                order_id=order_id,
                client_order_id="abc",
                strategy_id="buyer",
                symbol="AAPL",
                side=Side.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("10"),
                limit_price=Decimal("100.02"),
                time_in_force=TimeInForce.DAY,
                status=OrderStatus.NEW,
                risk_check_id=risk_check_id,
            )

        assert order_sequence_bytes([make()]) == order_sequence_bytes([make()])

    def test_order_matters_in_the_encoding(self) -> None:
        a = Order(
            order_id=uuid4(),
            client_order_id="a",
            strategy_id="buyer",
            symbol="AAPL",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
        )
        b = Order(
            order_id=uuid4(),
            client_order_id="b",
            strategy_id="buyer",
            symbol="AAPL",
            side=Side.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
        )
        assert order_sequence_bytes([a, b]) != order_sequence_bytes([b, a])


class TestCompareReplays:
    async def test_a_deterministic_scenario_matches(self) -> None:
        result = await compare_replays(deterministic_engine_factory, bars(3))
        assert result.matches
        assert result.first_divergence is None
        assert len(result.left_orders) > 0
        assert result.left_orders == result.right_orders

    async def test_assert_replay_matches_does_not_raise_on_a_match(self) -> None:
        result = await assert_replay_matches(deterministic_engine_factory, bars(3))
        assert result.matches

    async def test_a_nondeterministic_scenario_is_actually_caught(self) -> None:
        # The harness must catch divergence, not just report "matches" by
        # default — otherwise it verifies nothing.
        result = await compare_replays(nondeterministic_engine_factory, bars(3))
        assert not result.matches
        assert result.first_divergence == 0

    async def test_assert_replay_matches_raises_on_divergence(self) -> None:
        with pytest.raises(AssertionError, match="replay diverged"):
            await assert_replay_matches(nondeterministic_engine_factory, bars(3))

    async def test_summary_reports_the_order_count_on_a_match(self) -> None:
        result = await compare_replays(deterministic_engine_factory, bars(3))
        assert "byte-identical" in result.summary()

    async def test_summary_reports_the_divergence_point_on_a_mismatch(self) -> None:
        result = await compare_replays(nondeterministic_engine_factory, bars(3))
        assert "diverged at order #0" in result.summary()
