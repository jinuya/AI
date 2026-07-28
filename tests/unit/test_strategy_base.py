"""Strategy ABC and its context — spec §FR-STR-01."""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

from atrader.core.models import AccountState, Position, TradingIntent
from atrader.core.types import Side, TargetType
from atrader.features.store import FeatureStore, FeatureValue
from atrader.marketdata.models import Bar
from atrader.strategy.base import Strategy, StrategyContext

BASE_NS = 1_700_000_000_000_000_000


def make_bar() -> Bar:
    return Bar(
        symbol="AAPL",
        interval="1d",
        open_ts=BASE_NS,
        close_ts=BASE_NS + 1,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100"),
        is_final=True,
    )


class _AlwaysBuyStrategy(Strategy):
    def on_bar(self, bar: Bar, context: StrategyContext) -> list[TradingIntent]:
        return [
            TradingIntent(
                intent_id=uuid4(),
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,
                target_type=TargetType.SHARES,
                target_value=Decimal("10"),
                created_at_ns=context.now_ns,
            )
        ]


class TestStrategyContext:
    def test_position_of_an_unknown_symbol_is_flat(self) -> None:
        context = StrategyContext(
            now_ns=BASE_NS, account=AccountState(), positions={}, features=FeatureStore()
        )
        position = context.position_of("AAPL")
        assert position.is_flat

    def test_position_of_a_known_symbol_returns_it(self) -> None:
        held = Position(symbol="AAPL", quantity=Decimal("10"))
        context = StrategyContext(
            now_ns=BASE_NS,
            account=AccountState(),
            positions={"AAPL": held},
            features=FeatureStore(),
        )
        assert context.position_of("AAPL") is held

    def test_feature_returns_none_when_nothing_was_published(self) -> None:
        context = StrategyContext(
            now_ns=BASE_NS, account=AccountState(), positions={}, features=FeatureStore()
        )
        assert context.feature("AAPL", "sma_20") is None

    def test_feature_unwraps_the_value_from_the_store(self) -> None:
        store = FeatureStore()
        store.publish(
            FeatureValue(
                symbol="AAPL", name="sma_20", value=Decimal("150"), valid_from_ns=BASE_NS, version=1
            )
        )
        context = StrategyContext(
            now_ns=BASE_NS, account=AccountState(), positions={}, features=store
        )
        assert context.feature("AAPL", "sma_20") == Decimal("150")


class TestStrategy:
    def test_on_bar_produces_intents(self) -> None:
        strategy = _AlwaysBuyStrategy("always_buy")
        context = StrategyContext(
            now_ns=BASE_NS, account=AccountState(), positions={}, features=FeatureStore()
        )
        intents = strategy.on_bar(make_bar(), context)
        assert len(intents) == 1
        assert intents[0].strategy_id == "always_buy"

    def test_on_start_and_on_fill_default_to_doing_nothing(self) -> None:
        strategy = _AlwaysBuyStrategy("always_buy")
        context = StrategyContext(
            now_ns=BASE_NS, account=AccountState(), positions={}, features=FeatureStore()
        )
        assert strategy.on_start(context) is None
        assert strategy.snapshot() == {}
