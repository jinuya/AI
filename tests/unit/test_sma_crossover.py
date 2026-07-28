"""SMA crossover reference strategy — spec §13."""

from __future__ import annotations

from decimal import Decimal

import pytest

from atrader.core.clock import SimulatedClock
from atrader.core.ids import DeterministicIdGenerator
from atrader.core.models import AccountState, Position
from atrader.core.types import Side
from atrader.features.store import FeatureStore, FeatureValue
from atrader.marketdata.models import Bar
from atrader.strategy.base import StrategyContext
from atrader.strategy.rules.sma_crossover import (
    FAST_FEATURE,
    SLOW_FEATURE,
    SmaCrossoverStrategy,
    feature_specs_for,
)

BASE_NS = 1_700_000_000_000_000_000
ONE_DAY_NS = 86_400_000_000_000


def make_bar(*, index: int, close: str = "100", volume: str = "1000000") -> Bar:
    return Bar(
        symbol="AAPL",
        interval="1d",
        open_ts=BASE_NS + index * ONE_DAY_NS,
        close_ts=BASE_NS + (index + 1) * ONE_DAY_NS,
        open=Decimal(close),
        high=Decimal(close),
        low=Decimal(close),
        close=Decimal(close),
        volume=Decimal(volume),
        is_final=True,
    )


class Harness:
    """Drives the strategy bar-by-bar with directly-controlled SMA features,
    decoupled from indicator arithmetic (already covered by test_indicators.py)."""

    def __init__(self, *, fast_period: int = 20, slow_period: int = 50, min_bars: int = 60) -> None:
        self.store = FeatureStore()
        self.clock = SimulatedClock(start_ns=BASE_NS)
        self.ids = DeterministicIdGenerator(self.clock, seed=1)
        self.strategy = SmaCrossoverStrategy(
            "sma_crossover",
            ["AAPL"],
            fast_period=fast_period,
            slow_period=slow_period,
            min_bars=min_bars,
            volume_multiple=Decimal("1.8"),
        )
        self.position = Position(symbol="AAPL")

    def publish(self, bar: Bar, *, fast: str | None, slow: str | None) -> list:
        if fast is not None:
            self.store.publish(
                FeatureValue(
                    symbol="AAPL",
                    name=FAST_FEATURE,
                    value=Decimal(fast),
                    valid_from_ns=bar.close_ts,
                    version=1,
                )
            )
        if slow is not None:
            self.store.publish(
                FeatureValue(
                    symbol="AAPL",
                    name=SLOW_FEATURE,
                    value=Decimal(slow),
                    valid_from_ns=bar.close_ts,
                    version=1,
                )
            )
        context = StrategyContext(
            now_ns=bar.close_ts,
            account=AccountState(),
            positions={"AAPL": self.position},
            features=self.store,
            ids=self.ids,
        )
        return self.strategy.on_bar(bar, context)

    def fill(self, quantity: Decimal) -> None:
        """Simulate the intent actually being filled — updates the tracked position."""
        self.position = Position(symbol="AAPL", quantity=self.position.quantity + quantity)


class TestConstruction:
    def test_rejects_a_fast_period_not_below_slow(self) -> None:
        with pytest.raises(ValueError, match="must be below"):
            SmaCrossoverStrategy("sma", ["AAPL"], fast_period=50, slow_period=20)


class TestFeatureSpecs:
    def test_feature_specs_for_names_match_the_strategys_constants(self) -> None:
        specs = feature_specs_for(fast_period=20, slow_period=50)
        names = {spec.name for spec in specs}
        assert names == {FAST_FEATURE, SLOW_FEATURE}
        periods = {spec.name: spec.period for spec in specs}
        assert periods[FAST_FEATURE] == 20
        assert periods[SLOW_FEATURE] == 50


class TestWarmup:
    def test_no_intents_before_min_bars(self) -> None:
        h = Harness(min_bars=5)
        for i in range(4):
            intents = h.publish(make_bar(index=i), fast="10", slow="5")
            assert intents == []

    def test_no_intents_without_both_features_available(self) -> None:
        h = Harness(min_bars=1)
        intents = h.publish(make_bar(index=0), fast="10", slow=None)
        assert intents == []

    def test_the_first_bar_with_both_smas_never_signals_a_crossover(self) -> None:
        # There is nothing to compare against yet on the very first observation.
        h = Harness(min_bars=1)
        intents = h.publish(make_bar(index=0, volume="5000000"), fast="10", slow="5")
        assert intents == []


class TestGoldenCross:
    def test_a_golden_cross_with_volume_confirmation_buys(self) -> None:
        h = Harness(min_bars=1)
        h.publish(make_bar(index=0, volume="1000000"), fast="9", slow="10")  # below
        intents = h.publish(make_bar(index=1, volume="5000000"), fast="11", slow="10")  # crosses
        assert len(intents) == 1
        assert intents[0].side is Side.BUY
        assert "golden cross" in intents[0].rationale

    def test_a_golden_cross_without_volume_confirmation_is_skipped(self) -> None:
        h = Harness(min_bars=1)
        h.publish(make_bar(index=0, volume="1000000"), fast="9", slow="10")
        intents = h.publish(make_bar(index=1, volume="1000001"), fast="11", slow="10")
        assert intents == []

    def test_no_second_entry_while_already_holding(self) -> None:
        h = Harness(min_bars=1)
        h.publish(make_bar(index=0, volume="1000000"), fast="9", slow="10")
        h.publish(make_bar(index=1, volume="5000000"), fast="11", slow="10")
        h.fill(Decimal("10"))
        # Fast stays above slow — no new crossover, so no new entry regardless.
        intents = h.publish(make_bar(index=2, volume="5000000"), fast="12", slow="10")
        assert intents == []

    def test_a_rejected_entry_does_not_get_stuck_believing_it_holds(self) -> None:
        # The strategy never fills (context.positions stays flat) — a shadow
        # "in_position" flag would incorrectly block re-entry forever. Reading
        # the real position means it tries again on the next fresh crossover.
        h = Harness(min_bars=1)
        h.publish(make_bar(index=0, volume="1000000"), fast="9", slow="10")
        first = h.publish(make_bar(index=1, volume="5000000"), fast="11", slow="10")
        assert len(first) == 1
        # No h.fill() call: position stays flat, as if the risk engine rejected it.

        h.publish(make_bar(index=2, volume="1000000"), fast="9", slow="10")  # cross back down
        second = h.publish(
            make_bar(index=3, volume="5000000"), fast="11", slow="10"
        )  # cross up again
        assert len(second) == 1
        assert second[0].side is Side.BUY


class TestDeathCross:
    def test_a_death_cross_while_holding_sells_the_full_position(self) -> None:
        h = Harness(min_bars=1)
        h.publish(make_bar(index=0, volume="1000000"), fast="9", slow="10")
        h.publish(make_bar(index=1, volume="5000000"), fast="11", slow="10")
        h.fill(Decimal("10"))

        intents = h.publish(make_bar(index=2), fast="9", slow="10")
        assert len(intents) == 1
        assert intents[0].side is Side.SELL
        assert intents[0].target_value == Decimal("10")
        assert "death cross" in intents[0].rationale

    def test_a_death_cross_while_flat_does_nothing(self) -> None:
        h = Harness(min_bars=1)
        h.publish(make_bar(index=0, volume="1000000"), fast="11", slow="10")
        intents = h.publish(make_bar(index=1), fast="9", slow="10")
        assert intents == []


class TestMultiSymbolIsolation:
    def test_a_bar_for_an_untracked_symbol_is_ignored(self) -> None:
        h = Harness(min_bars=1)
        other = make_bar(index=0).model_copy(update={"symbol": "MSFT"})
        context = StrategyContext(
            now_ns=other.close_ts,
            account=AccountState(),
            positions={},
            features=h.store,
            ids=h.ids,
        )
        assert h.strategy.on_bar(other, context) == []


class TestSnapshot:
    def test_snapshot_reports_bar_counts_and_crossover_state(self) -> None:
        h = Harness(min_bars=1)
        h.publish(make_bar(index=0), fast="9", slow="10")
        snapshot = h.strategy.snapshot()
        assert snapshot["bar_count"]["AAPL"] == 1
        assert snapshot["fast_above_slow"]["AAPL"] is False
