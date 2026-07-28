"""Kill switch timing — acceptance criterion #5.

    5초 내 (1)신규중단 (2)전체 미체결 주문 취소.

Measured against real wall-clock time (``time.monotonic()``), not a
:class:`~atrader.core.clock.SimulatedClock` — a simulated clock can make an
arbitrarily slow implementation look instantaneous, which is exactly the
thing this acceptance criterion is worried about. This file is exempt from
the project's usual "no wall-clock reads outside ``atrader.core``" rule
(see ``tests/unit/test_determinism_lint.py``) because that rule only scans
``src/atrader`` — a test asserting real elapsed time is precisely what a
timing acceptance criterion requires.
"""

from __future__ import annotations

import time
from decimal import Decimal

from atrader.config.schema import (
    AppConfig,
    InstrumentSpec,
    MarketDataConfig,
    UniverseConfig,
)
from atrader.core.clock import NS_PER_SECOND, SimulatedClock
from atrader.core.ids import DeterministicIdGenerator
from atrader.core.models import TradingIntent
from atrader.core.types import Side, TargetType
from atrader.features.engine import FeatureEngine
from atrader.features.registry import FeatureRegistry
from atrader.features.store import FeatureStore
from atrader.marketdata.feeds.replay import ReplayFeed
from atrader.marketdata.models import Tick
from atrader.risk.killswitch import KillSwitchSource
from atrader.strategy.base import Strategy

from atrader.app.runtime import Runtime  # isort: skip

BASE_NS = 1_700_000_000 * NS_PER_SECOND
FIVE_SECONDS = 5.0


class _BuyOnceStrategy(Strategy):
    def __init__(self, strategy_id: str, symbol: str) -> None:
        super().__init__(strategy_id)
        self.symbol = symbol
        self._tried = False

    def on_bar(self, bar, context):  # type: ignore[no-untyped-def]
        if bar.symbol != self.symbol or self._tried:
            return []
        self._tried = True
        return [
            TradingIntent(
                intent_id=context.ids.new_id(),
                strategy_id=self.strategy_id,
                symbol=self.symbol,
                side=Side.BUY,
                target_type=TargetType.SHARES,
                target_value=Decimal("10"),
                created_at_ns=context.now_ns,
            )
        ]


def make_tick(*, offset_seconds: int, price: str = "100", seq: int = 1) -> Tick:
    ts = BASE_NS + offset_seconds * NS_PER_SECOND
    return Tick(
        symbol="AAPL",
        exchange_ts=ts,
        ingest_ts=ts + 1_000_000,
        last=Decimal(price),
        last_size=Decimal("1000"),
        seq=seq,
    )


def make_runtime_with_a_resting_order() -> Runtime:
    clock = SimulatedClock(start_ns=BASE_NS)
    ids = DeterministicIdGenerator(clock, seed=1)
    feature_engine = FeatureEngine(store=FeatureStore(), registry=FeatureRegistry(), specs=())
    config = AppConfig(
        account_equity=Decimal("100000"),
        universe=UniverseConfig(symbols=("AAPL",), sectors={"AAPL": "TECHNOLOGY"}),
        instruments=(
            InstrumentSpec(
                symbol="AAPL",
                sector="TECHNOLOGY",
                market_open_utc="00:00",
                market_close_utc="23:59",
            ),
        ),
        market_data=MarketDataConfig(bar_intervals=("1s",)),
    )
    # Only two ticks: the order gets submitted (bar 0 closes) but the price
    # never trades again, so it is still resting — exactly the state a kill
    # switch needs to prove it can act on.
    ticks = [make_tick(offset_seconds=0, seq=1), make_tick(offset_seconds=1, seq=2)]
    return Runtime(
        config,
        [_BuyOnceStrategy("buyer", "AAPL")],
        feature_engine,
        clock=clock,
        ids=ids,
        feed=ReplayFeed(ticks),
    )


class TestKillSwitchActsWithinFiveSeconds:
    async def test_engage_and_cancel_all_complete_well_under_five_seconds(self) -> None:
        runtime = make_runtime_with_a_resting_order()
        await runtime.run_forever()
        assert runtime.storage.orders.open_orders(), "fixture must leave an order resting"

        started = time.monotonic()
        await runtime.kill(reason="acceptance criterion #5", source=KillSwitchSource.HTTP)
        elapsed = time.monotonic() - started

        assert elapsed < FIVE_SECONDS
        assert runtime.status().kill_switch_engaged is True
        assert runtime.storage.orders.open_orders() == []

    async def test_new_orders_are_blocked_the_instant_engage_returns(self) -> None:
        # KillSwitch.engage() is a synchronous boolean flip with no I/O
        # (see its module docstring) — new-order blocking must be visible
        # immediately, not just "eventually" once cancellation finishes.
        runtime = make_runtime_with_a_resting_order()
        await runtime.start()

        started = time.monotonic()
        runtime.kill_switch.engage(reason="acceptance criterion #5", source=KillSwitchSource.CLI)
        elapsed = time.monotonic() - started

        assert elapsed < 0.1
        assert runtime.kill_switch.is_engaged is True
