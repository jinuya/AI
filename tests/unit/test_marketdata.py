"""Market data: models, quality control, aggregation, replay and dual sourcing."""

from __future__ import annotations

import json
import time
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from atrader.config.schema import DataQualityConfig
from atrader.core.clock import NS_PER_SECOND, SimulatedClock, SystemClock
from atrader.core.errors import DataQualityError
from atrader.core.types import DataQuality
from atrader.marketdata.aggregator import BarAggregator, interval_to_ns
from atrader.marketdata.feeds.dual_source import DualSourceMonitor, choose_active_feed
from atrader.marketdata.feeds.protocol import FeedStatus
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

    def test_a_tick_with_no_usable_price_is_ignored(self) -> None:
        """A quote with nothing to compare cannot diverge from anything, and
        must not evict the good price already cached for that side."""
        _, dual = self._monitors()
        dual.observe_primary(tick(last=Decimal("187.50")))
        assert dual.observe_primary(tick(bid=None, ask=None, last=None)) is None
        assert dual.observe_backup(tick(bid=Decimal("200.00"), ask=Decimal("200.02"))) is not None

    def test_divergences_are_recorded_for_the_post_mortem(self) -> None:
        _, dual = self._monitors()
        dual.observe_primary(tick())
        dual.observe_backup(tick(bid=Decimal("200.00"), ask=Decimal("200.02")))
        assert len(dual.divergences) == 1

    def test_the_divergence_log_is_a_copy(self) -> None:
        _, dual = self._monitors()
        dual.observe_primary(tick())
        dual.observe_backup(tick(bid=Decimal("200.00"), ask=Decimal("200.02")))
        dual.divergences.clear()
        assert len(dual.divergences) == 1

    def test_clearing_a_symbol_drops_both_sides(self) -> None:
        """Used when a source reconnects: the cached price predates the gap,
        so comparing against it would measure the outage, not a disagreement."""
        _, dual = self._monitors()
        dual.observe_primary(tick(last=Decimal("187.50")))
        dual.clear("AAPL")
        assert dual.observe_backup(tick(bid=Decimal("200.00"), ask=Decimal("200.02"))) is None

    def test_clearing_an_unknown_symbol_is_harmless(self) -> None:
        _, dual = self._monitors()
        dual.clear("NEVER_SEEN")


class _StubFeed:
    """Minimal MarketDataFeed for the failover decision, which only reads
    ``status`` — connecting a real feed would test the feed, not the choice."""

    def __init__(self, name: str, status: str = FeedStatus.CONNECTED) -> None:
        self._name = name
        self.status = status

    @property
    def name(self) -> str:
        return self._name


class TestChooseActiveFeed:
    """Spec §5.3 failover. Distinct from divergence: a primary that has
    *stopped* is unambiguous, so the backup is simply used. A primary that is
    producing *wrong* data is the divergence case, where switching would just
    be guessing — which is why nothing here inspects prices.
    """

    def _clock(self, at_ns: int = BASE_NS) -> SimulatedClock:
        return SimulatedClock(start_ns=at_ns)

    def test_with_no_backup_the_primary_is_used_however_bad_it_looks(self) -> None:
        primary = _StubFeed("primary", FeedStatus.DISCONNECTED)
        chosen, reason = choose_active_feed(
            primary, None, clock=self._clock(), last_primary_tick_ns=None
        )
        assert chosen is primary
        assert reason == "no backup configured"

    @pytest.mark.parametrize("status", [FeedStatus.DISCONNECTED, FeedStatus.EXHAUSTED])
    def test_a_dead_primary_fails_over(self, status: str) -> None:
        primary, backup = _StubFeed("primary", status), _StubFeed("backup")
        chosen, reason = choose_active_feed(
            primary, backup, clock=self._clock(), last_primary_tick_ns=BASE_NS
        )
        assert chosen is backup
        assert status in reason

    def test_a_degraded_primary_is_kept(self) -> None:
        """DEGRADED means connected but disagreeing — the divergence path
        already blocked new orders, and swapping feeds would be picking a
        side. Only a feed producing *nothing* justifies failover."""
        primary, backup = _StubFeed("primary", FeedStatus.DEGRADED), _StubFeed("backup")
        chosen, _ = choose_active_feed(
            primary, backup, clock=self._clock(), last_primary_tick_ns=BASE_NS
        )
        assert chosen is primary

    def test_a_primary_that_has_not_started_yet_is_given_the_benefit(self) -> None:
        """At boot there is no last tick. Treating that as silence would fail
        over to the backup on every startup."""
        primary, backup = _StubFeed("primary"), _StubFeed("backup")
        chosen, reason = choose_active_feed(
            primary, backup, clock=self._clock(), last_primary_tick_ns=None
        )
        assert chosen is primary
        assert reason == "primary has not produced yet"

    def test_a_recently_active_primary_is_healthy(self) -> None:
        primary, backup = _StubFeed("primary"), _StubFeed("backup")
        chosen, reason = choose_active_feed(
            primary,
            backup,
            clock=self._clock(BASE_NS + 3 * NS_PER_SECOND),
            last_primary_tick_ns=BASE_NS,
            stale_after_seconds=10,
        )
        assert chosen is primary
        assert reason == "primary healthy"

    def test_a_silent_primary_fails_over_and_says_how_long(self) -> None:
        primary, backup = _StubFeed("primary"), _StubFeed("backup")
        chosen, reason = choose_active_feed(
            primary,
            backup,
            clock=self._clock(BASE_NS + 30 * NS_PER_SECOND),
            last_primary_tick_ns=BASE_NS,
            stale_after_seconds=10,
        )
        assert chosen is backup
        assert "30.0s" in reason

    def test_the_threshold_is_exclusive(self) -> None:
        """Exactly at the limit is not yet stale — a feed on a 10s heartbeat
        with a 10s threshold would otherwise flap every single beat."""
        primary, backup = _StubFeed("primary"), _StubFeed("backup")
        chosen, _ = choose_active_feed(
            primary,
            backup,
            clock=self._clock(BASE_NS + 10 * NS_PER_SECOND),
            last_primary_tick_ns=BASE_NS,
            stale_after_seconds=10,
        )
        assert chosen is primary


class TestReadTicksErrorContract:
    """``read_ticks`` promises ``DataQualityError`` with the file and line
    number — the module docstring's whole argument for JSONL is that a replay
    that disagrees with the original stays findable. Two common corruption
    shapes used to escape that contract: ``decimal.InvalidOperation``
    subclasses ``ArithmeticError`` rather than ``ValueError``, and a line that
    decodes to something other than an object reached ``.get`` on a non-dict.
    Callers that fail closed on ``DataQualityError`` saw neither.
    """

    def _write(self, path: Path, bad_line: str) -> Path:
        good = json.dumps(
            {
                "symbol": "AAPL",
                "exchange_ts": 1,
                "ingest_ts": 2,
                "last": "100",
                "last_size": "1",
                "seq": 1,
            }
        )
        path.write_text(f"{good}\n{bad_line}\n", encoding="utf-8")
        return path

    @pytest.mark.parametrize(
        ("label", "bad_line"),
        [
            (
                "a price that is not a number",
                json.dumps(
                    {
                        "symbol": "AAPL",
                        "exchange_ts": 1,
                        "ingest_ts": 2,
                        "last": "abc",
                        "last_size": "1",
                        "seq": 1,
                    }
                ),
            ),
            ("a JSON array", "[1, 2, 3]"),
            ("a bare number", "42"),
            ("a JSON null", "null"),
            ("a bare string", '"hello"'),
            ("malformed JSON", "{not json"),
        ],
    )
    def test_every_corruption_shape_raises_data_quality_error(
        self, tmp_path: Path, label: str, bad_line: str
    ) -> None:
        path = self._write(tmp_path / "recording.jsonl", bad_line)
        with pytest.raises(DataQualityError):
            read_ticks(path)

    @pytest.mark.parametrize(
        ("label", "bad_line"),
        [
            ("a price that is not a number", '{"symbol": "A", "last": "abc"}'),
            ("a JSON array", "[]"),
        ],
    )
    def test_the_offending_line_number_is_reported(
        self, tmp_path: Path, label: str, bad_line: str
    ) -> None:
        """Without this the operator has a corrupt recording and no idea
        where — which is the diagnostic the format was chosen for."""
        path = self._write(tmp_path / "recording.jsonl", bad_line)
        with pytest.raises(DataQualityError, match="line 2"):
            read_ticks(path)

    def test_a_valid_recording_still_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "recording.jsonl"
        original = [tick(seq=1), tick(seq=2, last=Decimal("187.60"))]
        write_ticks(path, original)
        assert read_ticks(path) == original


class TestSimulatedFeedCadence:
    """Against a real clock the emission rate has to be real too.

    ``_step`` applies one ``interval_ns`` of price diffusion per tick. With a
    ``SimulatedClock`` the loop advances that clock, so the two agree. With a
    real clock there was nothing pacing the loop: it spun as fast as the event
    loop allowed — measured at ~62,000 ticks per second for a one-second
    interval — while every one of those ticks moved the price as though a full
    second had elapsed. A minute bar built from that has an intrabar range of
    tens of percent, tripping the fat-finger and price-jump checks on pure
    artifact.
    """

    async def test_a_real_clock_paces_the_loop(self) -> None:
        feed = SimulatedFeed.from_symbols(
            ["AAPL"], clock=SystemClock(), interval_ns=NS_PER_SECOND // 20
        )
        await feed.connect(["AAPL"])

        started = time.monotonic()
        emitted = 0
        async for _ in feed:
            emitted += 1
            if time.monotonic() - started > 0.5:
                break
        await feed.close()

        # ~10 expected at 50ms cadence over 0.5s. The bound that matters is
        # the upper one: before the fix this was in the tens of thousands.
        assert emitted < 100, f"feed is not pacing itself: {emitted} ticks in 0.5s"
        assert emitted >= 2, "feed produced almost nothing; it should still tick"

    async def test_a_simulated_clock_is_not_slowed_down(self) -> None:
        """Backtests must stay instant — the pacing applies only to real time."""
        clock = SimulatedClock(start_ns=BASE_NS)
        feed = SimulatedFeed.from_symbols(["AAPL"], clock=clock, interval_ns=NS_PER_SECOND)
        await feed.connect(["AAPL"])

        started = time.monotonic()
        emitted = 0
        async for _ in feed:
            emitted += 1
            if emitted >= 500:
                break
        await feed.close()

        assert time.monotonic() - started < 2.0, "a simulated clock must not sleep"
        # emitted - 1, not emitted: the generator is suspended at the yield of
        # the last tick, so that round's clock advance has not run yet.
        assert clock.now_ns() == BASE_NS + (emitted - 1) * NS_PER_SECOND


class TestABarIsPublishedFinalExactlyOnce:
    """The late-tick guard only ever saw an *open* builder. Once the timer
    closed and removed one, a tick belonging to that bucket opened a fresh
    builder at the same ``open_ts`` and the identical bar went out a second
    time as ``is_final=True``, with rewritten OHLCV. Nothing downstream
    de-duplicates bars, so both reached the strategies — breaking the
    exactly-one-final-bar contract and deterministic replay alike.

    The race is routine, not exotic: ``close_expired`` compares wall-clock
    ``now_ns`` against an ``open_ts`` derived from ``exchange_ts``, so any feed
    lag at a bucket boundary delivers in-bucket ticks after the timer fired.
    """

    def _tick(self, second: int, price: str, seq: int) -> Tick:
        ts = second * NS_PER_SECOND
        return Tick(
            symbol="AAPL",
            exchange_ts=ts,
            ingest_ts=ts + 1,
            last=Decimal(price),
            last_size=Decimal("1"),
            seq=seq,
        )

    def test_a_tick_arriving_after_the_timer_closed_its_bar_is_dropped(self) -> None:
        aggregator = BarAggregator(intervals=("1m",))
        aggregator.add(self._tick(10, "100", 1))
        aggregator.add(self._tick(20, "101", 2))

        first = aggregator.close_expired(now_ns=61 * NS_PER_SECOND)
        assert [b.open_ts for b in first] == [0]

        aggregator.add(self._tick(59, "250", 3))
        assert aggregator.close_expired(now_ns=122 * NS_PER_SECOND) == []

    def test_the_published_bar_is_not_rewritten(self) -> None:
        aggregator = BarAggregator(intervals=("1m",))
        aggregator.add(self._tick(10, "100", 1))
        (published,) = aggregator.close_expired(now_ns=61 * NS_PER_SECOND)

        aggregator.add(self._tick(59, "250", 2))

        assert published.close == Decimal("100")
        assert aggregator.forming("AAPL", "1m") is None

    def test_flush_also_closes_the_bucket_to_later_ticks(self) -> None:
        aggregator = BarAggregator(intervals=("1m",))
        aggregator.add(self._tick(10, "100", 1))
        assert len(aggregator.flush()) == 1

        aggregator.add(self._tick(30, "250", 2))
        assert aggregator.flush() == []

    def test_a_tick_for_the_next_bucket_still_opens_a_bar(self) -> None:
        """The watermark must block only what was already published."""
        aggregator = BarAggregator(intervals=("1m",))
        aggregator.add(self._tick(10, "100", 1))
        aggregator.close_expired(now_ns=61 * NS_PER_SECOND)

        aggregator.add(self._tick(70, "105", 2))

        forming = aggregator.forming("AAPL", "1m")
        assert forming is not None
        assert forming.open_ts == 60 * NS_PER_SECOND
