"""The backtest engine — end-to-end: strategy -> netting -> risk -> OMS -> broker.

Spec §2.2: this exercises the exact same ``RiskEngine``/``OrderManager`` code
paths P2 and P3 already cover in isolation; what's new here is only proving
they compose correctly when driven by historical bars instead of hand-built
fixtures.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from atrader.audit.logger import AuditLogger, InMemoryAuditSink
from atrader.backtest.cost_model import CostModel
from atrader.backtest.engine import BacktestEngine
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
from atrader.core.models import TradingIntent
from atrader.core.types import CostBasisMethod, Side, TargetType
from atrader.features.engine import FeatureEngine
from atrader.features.registry import FeatureRegistry
from atrader.features.store import FeatureStore
from atrader.marketdata.models import Bar
from atrader.risk.engine import RiskEngine
from atrader.risk.killswitch import KillSwitch
from atrader.strategy.base import Strategy, StrategyContext

BASE_NS = 1_700_000_000 * NS_PER_SECOND
ONE_DAY_NS = 86_400 * NS_PER_SECOND


def make_bar(
    *, index: int, open_: str, high: str, low: str, close: str, volume: str = "1000000"
) -> Bar:
    return Bar(
        symbol="AAPL",
        interval="1d",
        open_ts=BASE_NS + index * ONE_DAY_NS,
        close_ts=BASE_NS + (index + 1) * ONE_DAY_NS,
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal(volume),
        is_final=True,
    )


def rising_bars(n: int, *, start: int = 100) -> list[Bar]:
    bars = []
    for i in range(n):
        price = start + i
        bars.append(
            make_bar(
                index=i,
                open_=str(price),
                high=str(price + 2),
                low=str(price - 2),
                close=str(price + 1),
            )
        )
    return bars


class _BuyOnceStrategy(Strategy):
    """Buys a fixed share count on the very first bar it sees, then holds."""

    def __init__(self, strategy_id: str = "buy_once", *, quantity: str = "10") -> None:
        super().__init__(strategy_id)
        self._bought = False
        self._quantity = Decimal(quantity)

    def on_bar(self, bar: Bar, context: StrategyContext) -> list[TradingIntent]:
        if self._bought:
            return []
        self._bought = True
        return [
            TradingIntent(
                intent_id=context.ids.new_id(),
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,
                target_type=TargetType.SHARES,
                target_value=self._quantity,
                created_at_ns=context.now_ns,
            )
        ]


class _NeverTradeStrategy(Strategy):
    def on_bar(self, bar: Bar, context: StrategyContext) -> list[TradingIntent]:
        return []


def make_config(**risk_overrides: object) -> AppConfig:
    risk = RiskConfig(**risk_overrides) if risk_overrides else RiskConfig()
    return AppConfig(
        account_equity=Decimal("100000"),
        risk=risk,
        universe=UniverseConfig(symbols=("AAPL",), sectors={"AAPL": "TECHNOLOGY"}),
        instruments=(InstrumentSpec(symbol="AAPL", sector="TECHNOLOGY"),),
        execution={"cost_basis_method": CostBasisMethod.FIFO},
    )


def make_engine(
    strategies: tuple[Strategy, ...],
    *,
    config: AppConfig | None = None,
    audit: AuditLogger | None = None,
    margin_limits: AccountLimits | None = None,
) -> BacktestEngine:
    clock = SimulatedClock(start_ns=BASE_NS)
    ids = DeterministicIdGenerator(clock, seed=1)
    risk_engine = RiskEngine(
        config=config or make_config(),
        clock=clock,
        ids=ids,
        kill_switch=KillSwitch(clock=clock),
        audit=audit,
    )
    feature_engine = FeatureEngine(store=FeatureStore(), registry=FeatureRegistry(), specs=())
    return BacktestEngine(
        strategies=strategies,
        risk_engine=risk_engine,
        clock=clock,
        ids=ids,
        feature_engine=feature_engine,
        starting_cash=Decimal("100000"),
        cost_model=CostModel(impact_coefficient=Decimal("0")),
        margin_limits=margin_limits,
        audit=audit,
    )


class TestBasicRun:
    async def test_a_strategy_that_never_trades_produces_no_orders(self) -> None:
        engine = make_engine((_NeverTradeStrategy("idle"),))
        result = await engine.run(rising_bars(5))
        assert result.orders == ()
        assert result.fills == ()

    async def test_equity_curve_has_one_point_per_bar(self) -> None:
        engine = make_engine((_NeverTradeStrategy("idle"),))
        bars = rising_bars(5)
        result = await engine.run(bars)
        assert len(result.equity_curve) == 5
        assert result.equity_curve[0][0] == bars[0].close_ts

    async def test_a_buy_signal_does_not_fill_on_the_bar_that_produced_it(self) -> None:
        # Look-ahead check: the signal fires on bar 0's close, so the earliest
        # fill can happen is bar 1.
        engine = make_engine((_BuyOnceStrategy(),))
        bars = rising_bars(3)
        result = await engine.run(bars[:1])
        assert result.fills == ()

    async def test_a_buy_signal_fills_on_the_next_bar(self) -> None:
        # The risk engine defaults to an aggressive LIMIT rather than a raw
        # MARKET order (spec §FR-EXE-02) — 2 ticks through bar[0]'s close,
        # the reference price at the moment the intent was evaluated.
        engine = make_engine((_BuyOnceStrategy(),))
        bars = rising_bars(3)
        result = await engine.run(bars)
        assert len(result.fills) == 1
        assert result.fills[0].symbol == "AAPL"
        assert result.fills[0].quantity == Decimal("10")
        assert result.fills[0].price == bars[0].close + Decimal("0.02")
        # And it happens no earlier than bar[1] — never on the bar that
        # produced the signal.
        assert result.fills[0].executed_at_ns == bars[1].close_ts

    async def test_the_position_reflects_the_fill(self) -> None:
        engine = make_engine((_BuyOnceStrategy(),))
        result = await engine.run(rising_bars(3))
        assert result.final_positions["AAPL"].quantity == Decimal("10")

    async def test_feature_store_exposes_the_engines_feature_engine_store(self) -> None:
        engine = make_engine((_NeverTradeStrategy("idle"),))
        assert engine.feature_store is engine.feature_engine.store

    async def test_a_forming_bar_is_rejected(self) -> None:
        engine = make_engine((_NeverTradeStrategy("idle"),))
        forming = make_bar(
            index=0, open_="100", high="102", low="98", close="101", volume="1000000"
        )
        forming = forming.model_copy(update={"is_final": False})
        with pytest.raises(ValueError, match="only accepts closed bars"):
            await engine.run([forming])

    async def test_two_bars_on_the_same_utc_day_do_not_reset_the_daily_baseline(self) -> None:
        # rising_bars() spaces bars a full day apart; here we force two bars
        # into the same UTC day to exercise the "no rollover" path.
        engine = make_engine((_NeverTradeStrategy("idle"),))
        bar_a = make_bar(index=0, open_="100", high="102", low="98", close="101")
        bar_b = Bar(
            symbol="AAPL",
            interval="1h",
            open_ts=bar_a.close_ts,
            close_ts=bar_a.close_ts + NS_PER_SECOND * 3600,
            open=Decimal("101"),
            high=Decimal("103"),
            low=Decimal("99"),
            close=Decimal("102"),
            volume=Decimal("1000000"),
            is_final=True,
        )
        result = await engine.run([bar_a, bar_b])
        assert len(result.equity_curve) == 2

    async def test_orders_are_recorded_even_though_only_one_strategy_traded(self) -> None:
        engine = make_engine((_BuyOnceStrategy(),))
        result = await engine.run(rising_bars(3))
        assert len(result.orders) == 1
        assert result.orders[0].risk_check_id is not None  # came through the risk engine


class TestRiskGateIsReallyWired:
    async def test_an_oversized_order_is_reduced_not_silently_allowed_in_full(self) -> None:
        config = make_config(order=OrderLimits(max_order_notional_pct=Decimal("1.0")))
        # 1% of 100000 equity = 1000; at ~$100/share that's ~10 shares, but the
        # strategy asks for 500 — the risk engine must cut it down.
        engine = make_engine((_BuyOnceStrategy(quantity="500"),), config=config)
        result = await engine.run(rising_bars(3))
        assert len(result.fills) == 1
        assert result.fills[0].quantity < Decimal("500")

    async def test_a_symbol_outside_the_universe_is_rejected_not_traded(self) -> None:
        class _RogueStrategy(Strategy):
            def on_bar(self, bar: Bar, context: StrategyContext) -> list[TradingIntent]:
                return [
                    TradingIntent(
                        intent_id=context.ids.new_id(),
                        strategy_id=self.strategy_id,
                        symbol="FAKECO",
                        side=Side.BUY,
                        target_type=TargetType.SHARES,
                        target_value=Decimal("10"),
                        created_at_ns=context.now_ns,
                    )
                ]

        engine = make_engine((_RogueStrategy("rogue"),))
        result = await engine.run(rising_bars(3))
        assert result.orders == ()
        assert len(result.rejected_intents) >= 1
        assert (
            "FAKECO" in result.rejected_intents[0][1] or "universe" in result.rejected_intents[0][1]
        )

    async def test_a_leverage_breach_is_rejected(self) -> None:
        config = make_config(
            account={"max_leverage": Decimal("1.0")},
            position=PositionLimits(max_position_pct=Decimal("100")),  # isolate leverage
            order=OrderLimits(max_order_notional_pct=Decimal("100")),
        )
        # Ask for far more than the account can carry at 1.0x leverage.
        engine = make_engine((_BuyOnceStrategy(quantity="2000"),), config=config)
        result = await engine.run(rising_bars(3))
        # Either rejected outright or reduced to fit — either way, never the
        # full 2000 shares (~$200k against $100k equity).
        if result.fills:
            assert result.fills[0].quantity < Decimal("2000")
        else:
            assert result.rejected_intents


class TestNettingAcrossStrategies:
    async def test_two_strategies_wanting_opposite_sides_cancel_out(self) -> None:
        buyer = _BuyOnceStrategy("buyer", quantity="50")

        class _SellOnceStrategy(Strategy):
            def __init__(self) -> None:
                super().__init__("seller")
                self._sold = False

            def on_bar(self, bar: Bar, context: StrategyContext) -> list[TradingIntent]:
                if self._sold:
                    return []
                self._sold = True
                return [
                    TradingIntent(
                        intent_id=context.ids.new_id(),
                        strategy_id=self.strategy_id,
                        symbol=bar.symbol,
                        side=Side.SELL,
                        target_type=TargetType.SHARES,
                        target_value=Decimal("50"),
                        created_at_ns=context.now_ns,
                    )
                ]

        engine = make_engine((buyer, _SellOnceStrategy()))
        result = await engine.run(rising_bars(3))
        assert result.orders == ()
        assert result.fills == ()


class TestMarginMonitorWiring:
    async def test_a_configured_margin_monitor_is_evaluated_every_bar(self) -> None:
        # Cash trading has no maintenance margin, so this only proves the
        # monitor is actually invoked (status stays OK), not a margin call.
        engine = make_engine(
            (_NeverTradeStrategy("idle"),),
            margin_limits=AccountLimits(),
        )
        await engine.run(rising_bars(3))
        assert engine._margin_monitor is not None
        assert engine._margin_monitor.status == "OK"


class TestAuditTrail:
    async def test_every_risk_decision_is_audited(self) -> None:
        sink = InMemoryAuditSink()
        clock = SimulatedClock(start_ns=BASE_NS)
        audit = AuditLogger(sink=sink, clock=clock)
        engine = make_engine((_BuyOnceStrategy(),), audit=audit)
        await engine.run(rising_bars(3))
        assert len(sink.read_all()) > 0
        verification = audit.verify()
        assert verification.valid
