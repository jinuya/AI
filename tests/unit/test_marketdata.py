"""Market data: models, quality control, aggregation, replay and dual sourcing."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from atrader.config.schema import DataQualityConfig
from atrader.core.clock import NS_PER_SECOND, SimulatedClock
from atrader.core.types import DataQuality
from atrader.marketdata.aggregator import BarAggregator, interval_to_ns
from atrader.marketdata.feeds.dual_source import DualSourceMonitor
from atrader.marketdata.feeds.replay import ReplayFeed, read_ticks, write_ticks
from atrader.marketdata.feeds.simulated import SimulatedFeed
from atrader.marketdata.models import Bar, Tick
from atrader.marketdata.quality import QualityIssue, QualityMonitor

BASE_NS = 1_700_000_000 * NS_PER_SECOND


def tick(**overrides: object) -> Tick:
    defaults: dict[str, object] = {
        "symbol": "AAPL",
        "exchange_ts": BASE_NS,
        "ingest_ts": BASE_NS + 1_000_000,
        "bid": Decimal("187.49"),
        "ask": Decimal("187.51"),
        "bid_size": Decimal("100"),
        "ask_size": Decimal("100"),
        "last": Decimal("187.50"),
        "last_size": Decimal("50"),
        "seq": 1,
        "source": "test",
    }
    return Tick(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestTick:
    def test_mid_and_spread(self) -> None:
        t = tick()
        assert t.mid == Decimal("187.50")
        assert t.spread == Decimal("0.02")

    def test_crossed_detection(self) -> None:
        assert tick(bid=Decimal("187.55"), ask=Decimal("187.50")).is_crossed
        assert tick(bid=Decimal("187.50"), ask=Decimal("187.50")).is_crossed
        assert not tick().is_crossed

    def test_lag_is_the_difference_between_both_timestamps(self) -> None:
        # Keeping only one timestamp would make this unrecoverable.
        assert tick(exchange_ts=BASE_NS, ingest_ts=BASE_NS + 5_000_000).lag_ns == 5_000_000

    def test_reference_price_falls_back_sensibly(self) -> None:
        assert tick().reference_price == Decimal("187.50")
        assert tick(bid=None, ask=None).reference_price == Decimal("187.50")
        assert tick(bid=None, ask=None, last=None).reference_price is None

    def test_negative_prices_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            tick(bid=Decimal("-1"))


class TestBar:
    def test_incoherent_ohlc_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="outside"):
            Bar(
                symbol="AAPL",
                interval="1m",
                open_ts=0,
                close_ts=59,
                open=Decimal("200"),
                high=Decimal("190"),
                low=Decimal("180"),
                close=Decimal("185"),
            )

    def test_backwards_timestamps_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="precedes"):
            Bar(
                symbol="AAPL",
                interval="1m",
                open_ts=100,
                close_ts=50,
                open=Decimal("100"),
                high=Decimal("100"),
                low=Decimal("100"),
                close=Decimal("100"),
            )


class TestQualityMonitor:
    def test_a_clean_tick_passes(self) -> None:
        monitor = QualityMonitor()
        assert monitor.check(tick()).is_ok

    def test_unknown_symbols_are_stale_not_ok(self) -> None:
        # We have never seen data, so we certainly should not trade on it.
        assert QualityMonitor().quality_of("NEVER_SEEN") is DataQuality.STALE

    def test_sequence_gap_degrades(self) -> None:
        monitor = QualityMonitor()
        monitor.check(tick(seq=1))
        report = monitor.check(tick(seq=5, exchange_ts=BASE_NS + 1000))
        assert QualityIssue.SEQUENCE_GAP in report.issues
        assert not monitor.is_tradable("AAPL")

    def test_backwards_timestamp_degrades(self) -> None:
        monitor = QualityMonitor()
        monitor.check(tick(exchange_ts=BASE_NS + 1000, seq=1))
        report = monitor.check(tick(exchange_ts=BASE_NS, seq=2))
        assert QualityIssue.TIMESTAMP_REGRESSION in report.issues

    def test_crossed_market_degrades(self) -> None:
        report = QualityMonitor().check(tick(bid=Decimal("188"), ask=Decimal("187")))
        assert QualityIssue.CROSSED_MARKET in report.issues

    def test_price_jump_beyond_threshold_degrades(self) -> None:
        monitor = QualityMonitor()
        monitor.check(tick(seq=1))
        jumped = tick(
            seq=2,
            exchange_ts=BASE_NS + 1000,
            bid=Decimal("300.00"),
            ask=Decimal("300.02"),
            last=Decimal("300.01"),
        )
        assert QualityIssue.PRICE_JUMP in monitor.check(jumped).issues

    def test_a_move_inside_the_threshold_is_fine(self) -> None:
        monitor = QualityMonitor()
        monitor.check(tick(seq=1))
        moved = tick(
            seq=2,
            exchange_ts=BASE_NS + 1000,
            bid=Decimal("195.00"),
            ask=Decimal("195.02"),
            last=Decimal("195.01"),
        )
        assert monitor.check(moved).is_ok

    def test_bad_ticks_do_not_move_the_reference_price(self) -> None:
        # Otherwise the next jump check compares against a value we already
        # decided not to trust, and the bad price becomes the new normal.
        monitor = QualityMonitor()
        monitor.check(tick(seq=1))
        monitor.check(
            tick(
                seq=2,
                exchange_ts=BASE_NS + 1,
                bid=Decimal("300"),
                ask=Decimal("300.02"),
                last=Decimal("300.01"),
            )
        )
        assert monitor.state("AAPL").last_price == Decimal("187.50")

    def test_excessive_feed_lag_degrades(self) -> None:
        laggy = tick(exchange_ts=BASE_NS, ingest_ts=BASE_NS + 5 * NS_PER_SECOND)
        assert QualityIssue.EXCESSIVE_LAG in QualityMonitor().check(laggy).issues

    def test_recovery_requires_a_streak_of_clean_ticks(self) -> None:
        # A feed that alternates good and bad would otherwise flap between
        # tradable and blocked, which is worse than staying blocked.
        monitor = QualityMonitor(recovery_ticks=3)
        monitor.check(tick(seq=1))
        monitor.check(tick(seq=9, exchange_ts=BASE_NS + 1))
        assert not monitor.is_tradable("AAPL")

        for i in range(2):
            monitor.check(tick(seq=10 + i, exchange_ts=BASE_NS + 10 + i))
            assert not monitor.is_tradable("AAPL")

        monitor.check(tick(seq=12, exchange_ts=BASE_NS + 20))
        assert monitor.is_tradable("AAPL")

    def test_staleness_is_detected_on_a_timer(self) -> None:
        # Nothing is arriving, so nothing else would notice.
        monitor = QualityMonitor(DataQualityConfig(stale_threshold_seconds=30))
        monitor.check(tick())
        assert monitor.is_tradable("AAPL")

        report = monitor.check_staleness("AAPL", BASE_NS + 60 * NS_PER_SECOND)
        assert report.quality is DataQuality.STALE
        assert not monitor.is_tradable("AAPL")

    def test_force_degrade_blocks_from_outside_the_tick_path(self) -> None:
        monitor = QualityMonitor()
        monitor.check(tick())
        monitor.force_degrade("AAPL", "dual_source_divergence")
        assert not monitor.is_tradable("AAPL")


class TestBarAggregator:
    def _ticks(self, count: int, *, interval_ns: int) -> list[Tick]:
        return [
            tick(
                exchange_ts=BASE_NS + i * interval_ns,
                ingest_ts=BASE_NS + i * interval_ns,
                seq=i + 1,
                last=Decimal("187.50") + Decimal(i),
                last_size=Decimal("10"),
            )
            for i in range(count)
        ]

    def test_bucket_boundaries_align(self) -> None:
        minute = interval_to_ns("1m")
        assert BarAggregator.bucket_start(BASE_NS + 90 * NS_PER_SECOND, minute) % minute == 0

    def test_a_bar_closes_when_the_interval_rolls(self) -> None:
        agg = BarAggregator(("1m",))
        completed: list[Bar] = []
        for t in self._ticks(3, interval_ns=61 * NS_PER_SECOND):
            completed.extend(agg.add(t))
        assert len(completed) == 2
        assert all(bar.is_final for bar in completed)

    def test_ohlc_is_computed_correctly(self) -> None:
        agg = BarAggregator(("1m",))
        prices = [Decimal("100"), Decimal("105"), Decimal("95"), Decimal("102")]
        for i, price in enumerate(prices):
            agg.add(
                tick(
                    exchange_ts=BASE_NS + i * NS_PER_SECOND,
                    seq=i + 1,
                    bid=price - Decimal("0.01"),
                    ask=price + Decimal("0.01"),
                    last=price,
                    last_size=Decimal("10"),
                )
            )
        bar = agg.flush()[0]
        assert (bar.open, bar.high, bar.low, bar.close) == (
            Decimal("100"),
            Decimal("105"),
            Decimal("95"),
            Decimal("102"),
        )
        assert bar.volume == Decimal("40")

    def test_forming_bars_are_never_marked_final(self) -> None:
        # Spec §5.2: trading on a forming bar makes live disagree with backtest.
        agg = BarAggregator(("1m",))
        agg.add(tick())
        forming = agg.forming("AAPL", "1m")
        assert forming is not None
        assert forming.is_final is False

    def test_close_expired_finishes_bars_for_thin_symbols(self) -> None:
        # With no further ticks, nothing else would close the bar.
        agg = BarAggregator(("1m",))
        agg.add(tick())
        assert agg.close_expired(BASE_NS + 120 * NS_PER_SECOND)[0].is_final

    def test_a_late_tick_does_not_reopen_a_published_bar(self) -> None:
        agg = BarAggregator(("1m",))
        agg.add(tick(exchange_ts=BASE_NS, seq=1))
        agg.add(tick(exchange_ts=BASE_NS + 120 * NS_PER_SECOND, seq=2))
        before = agg.forming("AAPL", "1m")
        agg.add(tick(exchange_ts=BASE_NS + NS_PER_SECOND, seq=3))
        assert agg.forming("AAPL", "1m") == before

    def test_multiple_intervals_are_tracked_independently(self) -> None:
        agg = BarAggregator(("1m", "5m"))
        completed: list[Bar] = []
        for i in range(20):  # 10 minutes of data at 30s spacing
            completed.extend(agg.add(tick(exchange_ts=BASE_NS + i * 30 * NS_PER_SECOND, seq=i + 1)))
        completed.extend(agg.flush())

        one_minute = [bar for bar in completed if bar.interval == "1m"]
        five_minute = [bar for bar in completed if bar.interval == "5m"]
        assert len(one_minute) > len(five_minute) > 0

        # Each bar spans exactly its own interval, whichever bucket it landed in.
        for bar in completed:
            assert bar.close_ts - bar.open_ts == interval_to_ns(bar.interval) - 1


class TestReplayFeed:
    def test_round_trip_preserves_exact_decimals(self, tmp_path: Path) -> None:
        # A replay that shifts prices by a ULP is not a replay.
        path = tmp_path / "ticks.jsonl"
        original = [tick(seq=i, last=Decimal("187.12345678")) for i in range(1, 4)]
        assert write_ticks(path, original) == 3
        assert read_ticks(path) == original

    def test_replay_is_repeatable(self, tmp_path: Path) -> None:
        path = tmp_path / "ticks.jsonl"
        write_ticks(path, [tick(seq=i) for i in range(1, 11)])
        feed = ReplayFeed(path)

        first = [t for _ in range(10) if (t := feed.next_tick()) is not None]
        feed.reset()
        second = [t for _ in range(10) if (t := feed.next_tick()) is not None]
        assert first == second

    async def test_symbol_filter(self, tmp_path: Path) -> None:
        path = tmp_path / "ticks.jsonl"
        write_ticks(path, [tick(symbol="AAPL", seq=1), tick(symbol="MSFT", seq=2)])
        feed = ReplayFeed(path)
        await feed.connect(["AAPL"])
        collected = []
        while (t := feed.next_tick()) is not None:
            collected.append(t)
        assert {t.symbol for t in collected} == {"AAPL"}


class TestSimulatedFeed:
    def test_the_same_seed_gives_the_same_session(self) -> None:
        def run() -> list[Decimal | None]:
            feed = SimulatedFeed.from_symbols(
                ["AAPL"], seed=42, clock=SimulatedClock(start_ns=BASE_NS)
            )
            return [t.last for t in feed.generate(20)]

        assert run() == run()

    def test_different_seeds_diverge(self) -> None:
        def run(seed: int) -> list[Decimal | None]:
            feed = SimulatedFeed.from_symbols(
                ["AAPL"], seed=seed, clock=SimulatedClock(start_ns=BASE_NS)
            )
            return [t.last for t in feed.generate(20)]

        assert run(1) != run(2)

    def test_generated_ticks_pass_quality_control(self) -> None:
        # A simulator that produces data its own validator rejects is useless
        # for exercising the pipeline.
        feed = SimulatedFeed.from_symbols(
            ["AAPL", "MSFT"], seed=7, clock=SimulatedClock(start_ns=BASE_NS)
        )
        monitor = QualityMonitor()
        for t in feed.generate(200):
            assert monitor.check(t).is_ok, monitor.check(t).detail

    def test_prices_stay_positive_over_a_long_run(self) -> None:
        feed = SimulatedFeed.from_symbols(["AAPL"], seed=3, clock=SimulatedClock(start_ns=BASE_NS))
        assert all(t.last is not None and t.last > 0 for t in feed.generate(2000))


class TestDualSourceDivergence:
    def _monitors(self) -> tuple[QualityMonitor, DualSourceMonitor]:
        quality = QualityMonitor()
        return quality, DualSourceMonitor(
            quality, DataQualityConfig(dual_source_divergence_pct=Decimal("0.5"))
        )

    def test_agreeing_sources_stay_tradable(self) -> None:
        quality, dual = self._monitors()
        quality.check(tick())
        assert dual.observe_primary(tick(last=Decimal("187.50"))) is None
        assert dual.observe_backup(tick(last=Decimal("187.55"))) is None
        assert quality.is_tradable("AAPL")

    def test_divergence_blocks_new_orders(self) -> None:
        # Spec §5.3: trading when you do not know which price is right is the
        # most dangerous state, so we stop rather than pick a side.
        quality, dual = self._monitors()
        quality.check(tick())
        dual.observe_primary(tick(bid=Decimal("187.49"), ask=Decimal("187.51")))
        event = dual.observe_backup(tick(bid=Decimal("200.00"), ask=Decimal("200.02")))

        assert event is not None
        assert event.divergence_pct > Decimal("0.5")
        assert not quality.is_tradable("AAPL")
        assert "no longer know which price is right" in event.message

    def test_a_stale_counterpart_is_not_compared(self) -> None:
        # Comparing against an old quote produces false divergences during a
        # normal fast move, which would block trading for no reason.
        _, dual = self._monitors()
        dual.observe_primary(tick(ingest_ts=BASE_NS))
        late = tick(
            ingest_ts=BASE_NS + 60 * NS_PER_SECOND, bid=Decimal("300"), ask=Decimal("300.02")
        )
        assert dual.observe_backup(late) is None
