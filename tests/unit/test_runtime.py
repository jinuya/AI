"""Runtime — spec §10. Composition-root wiring, end to end."""

from __future__ import annotations

from decimal import Decimal

from atrader.audit.logger import AuditEvent
from atrader.config.schema import (
    AppConfig,
    InstrumentSpec,
    MarketDataConfig,
    MonitoringConfig,
    OrderLimits,
    RiskConfig,
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
from atrader.strategy.base import Strategy, StrategyContext

from atrader.app.runtime import Runtime  # isort: skip

BASE_NS = 1_700_000_000 * NS_PER_SECOND


class _BuyOnceStrategy(Strategy):
    """Buys once, the first time it sees a flat position, then never again."""

    def __init__(self, strategy_id: str, symbol: str, *, quantity: Decimal = Decimal("10")) -> None:
        super().__init__(strategy_id)
        self.symbol = symbol
        self.quantity = quantity
        self._tried = False

    def on_bar(self, bar, context: StrategyContext) -> list[TradingIntent]:  # type: ignore[no-untyped-def]
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
                target_value=self.quantity,
                created_at_ns=context.now_ns,
            )
        ]


def make_config(**risk_overrides: object) -> AppConfig:
    risk = RiskConfig(**risk_overrides) if risk_overrides else RiskConfig()
    return AppConfig(
        account_equity=Decimal("100000"),
        risk=risk,
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
        monitoring=MonitoringConfig(approval_timeout_seconds=5),
    )


def make_tick(*, offset_seconds: int, price: str, size: str = "1000", seq: int = 1) -> Tick:
    ts = BASE_NS + offset_seconds * NS_PER_SECOND
    return Tick(
        symbol="AAPL",
        exchange_ts=ts,
        ingest_ts=ts + 1_000_000,
        bid=Decimal(price) - Decimal("0.01"),
        ask=Decimal(price) + Decimal("0.01"),
        last=Decimal(price),
        last_size=Decimal(size),
        seq=seq,
    )


def make_runtime(strategies: list[Strategy], ticks: list[Tick], **config_kwargs: object) -> Runtime:
    clock = SimulatedClock(start_ns=BASE_NS)
    ids = DeterministicIdGenerator(clock, seed=1)
    feature_engine = FeatureEngine(store=FeatureStore(), registry=FeatureRegistry(), specs=())
    config = make_config(**config_kwargs)
    return Runtime(
        config,
        strategies,
        feature_engine,
        clock=clock,
        ids=ids,
        feed=ReplayFeed(ticks),
    )


def buy_signal_ticks(*, fill_price: str = "100") -> list[Tick]:
    """Two bars' worth of ticks: bar 0 closes on the second tick (triggering
    the strategy's buy). The risk engine's aggressive BUY limit prices above
    the reference (spec §FR-EXE-02, ``aggressive_limit_ticks``), so a BUY
    limit is marketable once price trades *at or below* it (standard limit
    semantics) — later ticks stay at the same reference price to fill it."""
    return [
        make_tick(offset_seconds=0, price="100", seq=1),
        make_tick(offset_seconds=1, price="100", seq=2),  # closes bar 0 -> on_bar fires
        make_tick(offset_seconds=2, price=fill_price, seq=3),
        make_tick(offset_seconds=3, price=fill_price, seq=4),
        make_tick(offset_seconds=4, price=fill_price, seq=5),
    ]


class TestBasicPipeline:
    async def test_a_buy_signal_produces_a_submitted_order(self) -> None:
        runtime = make_runtime([_BuyOnceStrategy("buyer", "AAPL")], buy_signal_ticks())
        await runtime.run_forever()
        orders = runtime.storage.orders.all_orders()
        assert len(orders) == 1
        assert orders[0].symbol == "AAPL"
        assert orders[0].side is Side.BUY

    async def test_the_order_eventually_fills_and_updates_the_position(self) -> None:
        runtime = make_runtime([_BuyOnceStrategy("buyer", "AAPL")], buy_signal_ticks())
        await runtime.run_forever()
        positions = runtime.positions_snapshot()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].quantity > 0

    async def test_status_reflects_ticks_and_bars_processed(self) -> None:
        ticks = buy_signal_ticks()
        runtime = make_runtime([_BuyOnceStrategy("buyer", "AAPL")], ticks)
        await runtime.run_forever()
        status = runtime.status()
        assert status.ticks_processed == len(ticks)
        assert status.bars_processed >= 2  # each new-bucket tick closes the prior bar
        assert status.orders_submitted == 1
        assert status.system_state == "RUNNING"

    async def test_startup_runs_reconciliation_and_records_it(self) -> None:
        runtime = make_runtime([_BuyOnceStrategy("buyer", "AAPL")], buy_signal_ticks())
        await runtime.start()
        events = [
            r
            for r in runtime.storage.audit.read_all()
            if r.event_type == AuditEvent.STATE_RECOVERED
        ]
        assert len(events) == 1
        assert events[0].payload["clean"] is True

    async def test_a_strategy_that_never_signals_submits_nothing(self) -> None:
        class _NeverBuy(Strategy):
            def on_bar(self, bar, context):  # type: ignore[no-untyped-def]
                return []

        runtime = make_runtime([_NeverBuy("idle")], buy_signal_ticks())
        await runtime.run_forever()
        assert runtime.storage.orders.all_orders() == []


class TestKillSwitch:
    async def test_engaging_blocks_a_subsequent_intent(self) -> None:
        runtime = make_runtime([_BuyOnceStrategy("buyer", "AAPL")], buy_signal_ticks())
        await runtime.start()
        await runtime.kill(reason="test", source=KillSwitchSource.HTTP, actor="tester")

        account = await runtime.broker.get_account()
        positions = {p.symbol: p for p in await runtime.broker.get_positions()}
        intent = TradingIntent(
            intent_id=runtime.ids.new_id(),
            strategy_id="buyer",
            symbol="AAPL",
            side=Side.BUY,
            target_type=TargetType.SHARES,
            target_value=Decimal("10"),
            created_at_ns=runtime.clock.now_ns(),
        )
        decision = await runtime._evaluate_and_submit(
            intent, runtime.clock.now_ns(), account, positions
        )
        assert not decision.approved
        assert "kill switch" in decision.reason

    async def test_engaging_cancels_open_orders(self) -> None:
        runtime = make_runtime([_BuyOnceStrategy("buyer", "AAPL")], buy_signal_ticks()[:2])
        await runtime.run_forever()
        assert runtime.storage.orders.open_orders(), "the order must still be open before kill"

        await runtime.kill(reason="test panic", source=KillSwitchSource.CLI, actor="tester")
        assert runtime.storage.orders.open_orders() == []
        assert runtime.status().kill_switch_engaged is True

    async def test_release_reopens_the_gate(self) -> None:
        runtime = make_runtime([_BuyOnceStrategy("buyer", "AAPL")], buy_signal_ticks())
        await runtime.start()
        await runtime.kill(reason="test", source=KillSwitchSource.CLI, actor="tester")
        await runtime.release(reason="resolved", source=KillSwitchSource.CLI, actor="tester")
        assert runtime.status().kill_switch_engaged is False


class TestApprovalFlow:
    async def test_a_large_order_requires_approval_and_is_held_not_submitted(self) -> None:
        clock = SimulatedClock(start_ns=BASE_NS)
        ids = DeterministicIdGenerator(clock, seed=1)
        feature_engine = FeatureEngine(store=FeatureStore(), registry=FeatureRegistry(), specs=())
        # A tiny threshold so this test's ordinary-sized order needs sign-off.
        config = make_config(order=OrderLimits(human_approval_threshold_pct=Decimal("0.001")))
        runtime = Runtime(
            config,
            [_BuyOnceStrategy("buyer", "AAPL")],
            feature_engine,
            clock=clock,
            ids=ids,
            feed=ReplayFeed(buy_signal_ticks()),
        )

        await runtime.run_forever()

        assert runtime.storage.orders.all_orders() == []
        assert len(runtime._pending_approvals) == 1

    async def test_granting_the_request_submits_the_held_intent(self) -> None:
        clock = SimulatedClock(start_ns=BASE_NS)
        ids = DeterministicIdGenerator(clock, seed=1)
        feature_engine = FeatureEngine(store=FeatureStore(), registry=FeatureRegistry(), specs=())
        config = make_config(order=OrderLimits(human_approval_threshold_pct=Decimal("0.001")))
        ticks = buy_signal_ticks()
        runtime = Runtime(
            config,
            [_BuyOnceStrategy("buyer", "AAPL")],
            feature_engine,
            clock=clock,
            ids=ids,
            feed=ReplayFeed(ticks),
        )
        await runtime.run_forever()
        assert len(runtime._pending_approvals) == 1
        request_id = next(iter(runtime._pending_approvals))

        runtime.approvals.grant(request_id, by="a_human")
        # A later bar cycle re-checks pending approvals and submits if granted.
        more_ticks = [
            make_tick(offset_seconds=5, price="101", seq=6),
            make_tick(offset_seconds=6, price="101", seq=7),
        ]
        runtime.feed = ReplayFeed(more_ticks)
        await runtime.run_forever()

        assert len(runtime.storage.orders.all_orders()) == 1
        assert runtime._pending_approvals == {}


class TestPnlAndPositions:
    async def test_pnl_snapshot_has_the_expected_keys(self) -> None:
        runtime = make_runtime([_BuyOnceStrategy("buyer", "AAPL")], buy_signal_ticks())
        await runtime.run_forever()
        pnl = runtime.pnl_snapshot()
        assert set(pnl) == {"realized", "unrealized", "equity"}


class TestStartupReconciliation:
    async def test_a_break_found_at_boot_keeps_the_system_out_of_running(self) -> None:
        from atrader.core.models import Position

        runtime = make_runtime([_BuyOnceStrategy("buyer", "AAPL")], buy_signal_ticks())
        # A local position with nothing at the broker to match it — a break.
        runtime.storage.positions.upsert(Position(symbol="AAPL", quantity=Decimal("999")))

        await runtime.start()

        assert runtime.status().system_state == "STARTING"
        events = [
            r
            for r in runtime.storage.audit.read_all()
            if r.event_type == AuditEvent.STATE_RECOVERED
        ]
        assert events[0].payload["clean"] is False

    async def test_a_clean_boot_reaches_running(self) -> None:
        runtime = make_runtime([_BuyOnceStrategy("buyer", "AAPL")], buy_signal_ticks())
        await runtime.start()
        assert runtime.status().system_state == "RUNNING"


class TestRunFor:
    async def test_stops_after_the_duration_even_with_an_unbounded_feed(self) -> None:
        # SimulatedFeed never exhausts on its own — the only way this loop
        # ends is the timeout, so this is a deterministic test of run_for's
        # wiring rather than a race against how fast a finite feed drains.
        from atrader.marketdata.feeds.simulated import SimulatedFeed

        clock = SimulatedClock(start_ns=BASE_NS)
        ids = DeterministicIdGenerator(clock, seed=1)
        feature_engine = FeatureEngine(store=FeatureStore(), registry=FeatureRegistry(), specs=())
        runtime = Runtime(
            make_config(),
            [_BuyOnceStrategy("buyer", "AAPL")],
            feature_engine,
            clock=clock,
            ids=ids,
            feed=SimulatedFeed.from_symbols(["AAPL"], clock=clock, interval_ns=NS_PER_SECOND),
        )

        await runtime.run_for(0.05)

        assert runtime.status().running is False
        events = [
            r for r in runtime.storage.audit.read_all() if r.event_type == AuditEvent.SYSTEM_STOPPED
        ]
        assert len(events) == 1
        assert "elapsed" in events[0].payload["reason"]
        assert "elapsed" in events[0].payload["reason"]


class TestLiquidation:
    async def test_kill_with_liquidate_flattens_open_positions(self) -> None:
        runtime = make_runtime([_BuyOnceStrategy("buyer", "AAPL")], buy_signal_ticks())
        await runtime.run_forever()
        assert runtime.positions_snapshot(), "fixture must have an open position before liquidating"

        await runtime.kill(reason="panic", actor="tester", liquidate=True)

        # The liquidation order was itself submitted through the normal risk
        # gate (reduce_only=True), so it appears as a second order.
        orders = runtime.storage.orders.all_orders()
        assert len(orders) == 2
        assert orders[1].strategy_id == "killswitch.liquidate"
        assert orders[1].side is Side.SELL


class TestMarketHours:
    async def test_a_symbol_outside_the_universe_is_never_open(self) -> None:
        runtime = make_runtime([_BuyOnceStrategy("buyer", "AAPL")], buy_signal_ticks())
        assert runtime._is_market_open("NOPE", runtime.clock.now_ns()) is False

    async def test_a_time_outside_the_configured_hours_is_closed(self) -> None:
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
                    market_open_utc="14:30",
                    market_close_utc="21:00",
                ),
            ),
            market_data=MarketDataConfig(bar_intervals=("1s",)),
        )
        runtime = Runtime(
            config,
            [_BuyOnceStrategy("buyer", "AAPL")],
            feature_engine,
            clock=clock,
            ids=ids,
            feed=ReplayFeed([]),
        )
        # BASE_NS is a fixed instant; whatever its UTC time-of-day is, either
        # this check or its complement below observes the closed side of the
        # 14:30-21:00 window depending on that instant.
        midnight_ns = (BASE_NS // (86_400 * NS_PER_SECOND)) * 86_400 * NS_PER_SECOND
        assert runtime._is_market_open("AAPL", midnight_ns) is False


class TestDeadmanIntegration:
    async def test_a_deadman_trigger_cancels_open_orders_and_records_a_metric(self) -> None:
        runtime = make_runtime([_BuyOnceStrategy("buyer", "AAPL")], buy_signal_ticks()[:2])
        await runtime.run_forever()
        assert runtime.storage.orders.open_orders(), "fixture must leave an order resting"

        report = await runtime.deadman.trigger(reason="test-forced")

        assert runtime.storage.orders.open_orders() == []
        assert b"atrader_deadman_triggers_total 1.0" in runtime.metrics.render()
        assert report.canceled_order_ids


class TestDailyEquityCurve:
    """The live half of a divergence report (acceptance criterion #7).

    What matters is that the curve stays *daily* — one closing mark per
    calendar day regardless of bar interval — because a per-bar curve on a
    month-long run at second bars would be millions of points, and because
    the criterion and ``performance_report``'s 252-periods-per-year default
    are both stated in days.
    """

    async def test_many_bars_in_one_day_produce_one_point(self) -> None:
        runtime = make_runtime([_BuyOnceStrategy("buyer", "AAPL")], buy_signal_ticks())
        await runtime.run_forever()

        curve = runtime.daily_equity_curve()
        assert runtime.status().bars_processed > 1, "fixture must span several bars"
        assert len(curve) == 1, f"all ticks are the same UTC day, got {len(curve)} points"

    async def test_the_single_point_holds_the_days_latest_equity(self) -> None:
        runtime = make_runtime([_BuyOnceStrategy("buyer", "AAPL")], buy_signal_ticks())
        await runtime.run_forever()

        ((_, equity),) = runtime.daily_equity_curve()
        assert equity == runtime.status().equity

    async def test_a_run_that_processed_no_bars_has_an_empty_curve(self) -> None:
        runtime = make_runtime([_BuyOnceStrategy("buyer", "AAPL")], [])
        await runtime.run_forever()
        assert runtime.daily_equity_curve() == ()

    async def test_a_second_day_appends_rather_than_overwriting(self) -> None:
        one_day_seconds = 86_400
        ticks = [
            make_tick(offset_seconds=0, price="100", seq=1),
            make_tick(offset_seconds=1, price="100", seq=2),
            make_tick(offset_seconds=one_day_seconds, price="101", seq=3),
            make_tick(offset_seconds=one_day_seconds + 1, price="101", seq=4),
        ]
        runtime = make_runtime([_BuyOnceStrategy("buyer", "AAPL")], ticks)
        await runtime.run_forever()

        curve = runtime.daily_equity_curve()
        assert len(curve) == 2
        assert curve[0][0] < curve[1][0], "points must be in chronological order"

    async def test_the_curve_feeds_a_divergence_report_without_conversion(self) -> None:
        """The whole point of the shared ``(at_ns, equity)`` shape: a live
        session's curve is directly comparable to a backtest's."""
        from atrader.backtest.divergence import divergence_report

        runtime = make_runtime([_BuyOnceStrategy("buyer", "AAPL")], buy_signal_ticks())
        await runtime.run_forever()

        report = divergence_report(runtime.daily_equity_curve(), runtime.daily_equity_curve())
        assert report.live.num_periods == len(runtime.daily_equity_curve())
