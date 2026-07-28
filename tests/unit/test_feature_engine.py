"""Feature engine and version registry — spec §8.2."""

from __future__ import annotations

from decimal import Decimal

import pytest

from atrader.core.errors import ConfigError
from atrader.features.engine import FeatureEngine, FeatureSpec
from atrader.features.registry import FeatureDefinition, FeatureRegistry
from atrader.features.store import FeatureStore
from atrader.marketdata.models import Bar

BASE_NS = 1_700_000_000_000_000_000
ONE_MINUTE_NS = 60_000_000_000


def make_bar(*, close: str, index: int, is_final: bool = True) -> Bar:
    return Bar(
        symbol="AAPL",
        interval="1m",
        open_ts=BASE_NS + index * ONE_MINUTE_NS,
        close_ts=BASE_NS + (index + 1) * ONE_MINUTE_NS,
        open=Decimal(close),
        high=Decimal(close),
        low=Decimal(close),
        close=Decimal(close),
        is_final=is_final,
    )


class TestFeatureRegistry:
    def test_registering_the_same_feature_twice_at_the_same_version_is_fine(self) -> None:
        registry = FeatureRegistry()
        registry.register(FeatureDefinition("sma_3", 1))
        registry.register(FeatureDefinition("sma_3", 1))
        assert registry.version_of("sma_3") == 1

    def test_registering_a_conflicting_version_is_rejected(self) -> None:
        registry = FeatureRegistry()
        registry.register(FeatureDefinition("sma_3", 1))
        with pytest.raises(ConfigError, match="registered twice"):
            registry.register(FeatureDefinition("sma_3", 2))

    def test_version_of_an_unknown_feature_raises(self) -> None:
        registry = FeatureRegistry()
        with pytest.raises(ConfigError, match="unknown feature"):
            registry.version_of("sma_3")

    def test_validate_required_passes_when_everything_matches(self) -> None:
        registry = FeatureRegistry([FeatureDefinition("sma_3", 1)])
        registry.validate_required({"sma_3": 1})  # does not raise

    def test_validate_required_rejects_a_missing_feature(self) -> None:
        registry = FeatureRegistry()
        with pytest.raises(ConfigError, match="not registered"):
            registry.validate_required({"sma_3": 1})

    def test_validate_required_rejects_a_version_mismatch(self) -> None:
        registry = FeatureRegistry([FeatureDefinition("sma_3", 2)])
        with pytest.raises(ConfigError, match="expects v1, registry has v2"):
            registry.validate_required({"sma_3": 1})


class TestFeatureEngine:
    def make_engine(self) -> FeatureEngine:
        return FeatureEngine(
            store=FeatureStore(),
            registry=FeatureRegistry(),
            specs=(FeatureSpec(name="sma_3", version=1, kind="sma", period=3),),
        )

    def test_registers_its_specs_with_the_registry_on_construction(self) -> None:
        engine = self.make_engine()
        assert engine.registry.version_of("sma_3") == 1

    def test_rejects_a_forming_bar(self) -> None:
        engine = self.make_engine()
        with pytest.raises(ValueError, match="only accepts closed bars"):
            engine.on_bar(make_bar(close="100", index=0, is_final=False))

    def test_insufficient_history_produces_nothing(self) -> None:
        engine = self.make_engine()
        produced = engine.on_bar(make_bar(close="100", index=0))
        assert produced == {}

    def test_enough_history_publishes_to_the_store(self) -> None:
        engine = self.make_engine()
        for index, close in enumerate(["10", "20", "30"]):
            produced = engine.on_bar(make_bar(close=close, index=index))
        assert produced == {"sma_3": Decimal("20")}

        published = engine.store.asof("AAPL", "sma_3", BASE_NS + 3 * ONE_MINUTE_NS)
        assert published is not None
        assert published.value == Decimal("20")

    def test_published_value_is_valid_from_the_bars_close_not_its_open(self) -> None:
        engine = self.make_engine()
        for index, close in enumerate(["10", "20", "30"]):
            engine.on_bar(make_bar(close=close, index=index))

        third_bar_open_ts = BASE_NS + 2 * ONE_MINUTE_NS
        third_bar_close_ts = BASE_NS + 3 * ONE_MINUTE_NS
        # Not knowable while the bar that produced it was still forming.
        assert engine.store.asof("AAPL", "sma_3", third_bar_open_ts) is None
        assert engine.store.asof("AAPL", "sma_3", third_bar_close_ts) is not None

    def test_history_beyond_max_history_is_trimmed(self) -> None:
        engine = FeatureEngine(
            store=FeatureStore(),
            registry=FeatureRegistry(),
            specs=(FeatureSpec(name="sma_3", version=1, kind="sma", period=3),),
            max_history=5,
        )
        for index in range(10):
            engine.on_bar(make_bar(close=str(index + 1), index=index))
        assert len(engine._history["AAPL"]) == 5  # verifying the trim itself

    def test_computes_ema_atr_and_rsi_specs_too(self) -> None:
        engine = FeatureEngine(
            store=FeatureStore(),
            registry=FeatureRegistry(),
            specs=(
                FeatureSpec(name="ema_3", version=1, kind="ema", period=3),
                FeatureSpec(name="atr_3", version=1, kind="atr", period=3),
                FeatureSpec(name="rsi_3", version=1, kind="rsi", period=3),
            ),
        )
        produced: dict[str, Decimal] = {}
        for index, close in enumerate(["10", "11", "12", "13", "14"]):
            produced = engine.on_bar(make_bar(close=close, index=index))
        assert set(produced) == {"ema_3", "atr_3", "rsi_3"}
