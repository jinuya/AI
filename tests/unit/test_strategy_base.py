"""Strategy ABC and its context — spec §FR-STR-01."""

from __future__ import annotations

from decimal import Decimal

from atrader.core.clock import SimulatedClock
from atrader.core.ids import DeterministicIdGenerator
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


def make_context(**overrides: object) -> StrategyContext:
    defaults: dict[str, object] = {
        "now_ns": BASE_NS,
        "account": AccountState(),
        "positions": {},
        "features": FeatureStore(),
        "ids": DeterministicIdGenerator(SimulatedClock(start_ns=BASE_NS)),
    }
    return StrategyContext(**{**defaults, **overrides})  # type: ignore[arg-type]


class _AlwaysBuyStrategy(Strategy):
    def on_bar(self, bar: Bar, context: StrategyContext) -> list[TradingIntent]:
        return [
            TradingIntent(
                intent_id=context.ids.new_id(),
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
        context = make_context()
        position = context.position_of("AAPL")
        assert position.is_flat

    def test_position_of_a_known_symbol_returns_it(self) -> None:
        held = Position(symbol="AAPL", quantity=Decimal("10"))
        context = make_context(positions={"AAPL": held})
        assert context.position_of("AAPL") is held

    def test_feature_returns_none_when_nothing_was_published(self) -> None:
        context = make_context()
        assert context.feature("AAPL", "sma_20") is None

    def test_feature_unwraps_the_value_from_the_store(self) -> None:
        store = FeatureStore()
        store.publish(
            FeatureValue(
                symbol="AAPL",
                name="sma_20",
                value=Decimal("150"),
                valid_from_ns=BASE_NS,
                version=1,
            )
        )
        context = make_context(features=store)
        assert context.feature("AAPL", "sma_20") == Decimal("150")

    def test_ids_mints_deterministic_intent_ids(self) -> None:
        # Same seed, same call sequence -> same ids: the whole point of
        # exposing an IdGenerator on the context rather than letting a
        # strategy reach for uuid4() itself (spec §2.2 determinism).
        clock = SimulatedClock(start_ns=BASE_NS)
        context_a = make_context(ids=DeterministicIdGenerator(clock, seed=3))
        context_b = make_context(ids=DeterministicIdGenerator(clock, seed=3))
        assert context_a.ids.new_id() == context_b.ids.new_id()


class TestStrategy:
    def test_on_bar_produces_intents(self) -> None:
        strategy = _AlwaysBuyStrategy("always_buy")
        context = make_context()
        intents = strategy.on_bar(make_bar(), context)
        assert len(intents) == 1
        assert intents[0].strategy_id == "always_buy"

    def test_on_start_and_on_fill_default_to_doing_nothing(self) -> None:
        strategy = _AlwaysBuyStrategy("always_buy")
        context = make_context()
        assert strategy.on_start(context) is None
        assert strategy.snapshot() == {}
