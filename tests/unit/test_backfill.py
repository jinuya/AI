"""Gap detection and backfill merge — spec §FR-MD-04.

``atrader.marketdata.backfill`` had no test at all until this file: 59
statements, zero executed. Pure functions with no infrastructure requirement,
so there was never a reason for that beyond nobody having written it — and
code that has never run is code whose behaviour is a guess.

Two properties carry the weight here. **A feed that died an hour ago must not
look complete** — the last bar it delivered is still the last bar present, so
gap detection that only compares neighbours sees nothing wrong. And **live
data wins on overlap**: a REST backfill may serve a revised bar, and letting
it replace one a strategy already traded on rewrites history.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from atrader.marketdata.aggregator import BarAggregator
from atrader.marketdata.backfill import BackfillPlan, Gap, find_gaps, merge_bars
from atrader.marketdata.models import Bar

MINUTE_NS = 60 * 1_000_000_000

#: Bar buckets are floored to the *epoch* grid, not to whenever the feed
#: happened to start (``BarAggregator.bucket_start``). A base timestamp that is
#: not itself on a bucket boundary therefore produces bars that could never
#: exist, and every arithmetic assertion below would be measuring the offset
#: rather than the logic. 1_700_000_000 is 20 seconds past a minute boundary,
#: so it is floored here — see ``TestBucketAlignmentIsShared``.
BASE_NS = BarAggregator.bucket_start(1_700_000_000 * 1_000_000_000, MINUTE_NS)


def bar(
    *,
    minute: int,
    symbol: str = "AAPL",
    interval: str = "1m",
    close: str = "100",
    source: str = "live",
) -> Bar:
    open_ts = BASE_NS + minute * MINUTE_NS
    price = Decimal(close)
    return Bar(
        symbol=symbol,
        interval=interval,
        open_ts=open_ts,
        close_ts=open_ts + MINUTE_NS - 1,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=Decimal("1000"),
        is_final=True,
        source=source,
    )


class TestFindGapsBetweenBars:
    def test_a_contiguous_series_has_no_gaps(self) -> None:
        plan = find_gaps([bar(minute=m) for m in range(5)])
        assert plan.is_empty
        assert plan.total_bars == 0
        assert plan.describe() == "no gaps"

    def test_no_bars_at_all_is_an_empty_plan(self) -> None:
        assert find_gaps([]) == BackfillPlan(())

    def test_a_single_missing_bar_is_found(self) -> None:
        plan = find_gaps([bar(minute=0), bar(minute=2)])
        assert len(plan.gaps) == 1
        gap = plan.gaps[0]
        assert gap.start_ns == BASE_NS + 1 * MINUTE_NS
        assert gap.end_ns == BASE_NS + 1 * MINUTE_NS
        assert gap.bar_count == 1
        assert plan.total_bars == 1

    def test_a_multi_bar_hole_reports_its_full_extent(self) -> None:
        plan = find_gaps([bar(minute=0), bar(minute=10)])
        (gap,) = plan.gaps
        assert gap.start_ns == BASE_NS + 1 * MINUTE_NS
        assert gap.end_ns == BASE_NS + 9 * MINUTE_NS
        assert gap.bar_count == 9

    def test_several_holes_are_each_reported(self) -> None:
        plan = find_gaps([bar(minute=m) for m in (0, 2, 3, 7)])
        assert [g.bar_count for g in plan.gaps] == [1, 3]
        assert plan.total_bars == 4

    def test_input_need_not_be_sorted(self) -> None:
        shuffled = [bar(minute=m) for m in (7, 0, 3, 2)]
        assert find_gaps(shuffled).gaps == find_gaps([bar(minute=m) for m in (0, 2, 3, 7)]).gaps

    def test_describe_names_the_symbol_and_size(self) -> None:
        plan = find_gaps([bar(minute=0), bar(minute=4)])
        described = plan.describe()
        assert "AAPL" in described
        assert "1m" in described
        assert "3 bar(s)" in described

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            (bar(minute=0, symbol="AAPL"), bar(minute=1, symbol="MSFT")),
            (bar(minute=0, interval="1m"), bar(minute=1, interval="5m")),
        ],
    )
    def test_mixing_symbols_or_intervals_is_refused(self, first: Bar, second: Bar) -> None:
        """Two symbols interleaved would make every other bar look like a gap.
        Refusing is the only answer that cannot silently produce nonsense."""
        with pytest.raises(ValueError, match="one symbol and interval"):
            find_gaps([first, second])


class TestTheStaleFeedCase:
    """The reason ``expected_end_ns`` exists. Without it a feed that stopped
    an hour ago looks perfectly healthy: its last bar is contiguous with the
    one before it, and there is no later bar to compare against."""

    def test_a_series_that_stops_early_looks_complete_without_expected_end(self) -> None:
        bars = [bar(minute=m) for m in range(3)]
        assert find_gaps(bars).is_empty

    def test_and_is_caught_once_the_expected_end_is_supplied(self) -> None:
        bars = [bar(minute=m) for m in range(3)]
        now = BASE_NS + 10 * MINUTE_NS + 30 * 1_000_000_000  # mid-bucket
        plan = find_gaps(bars, expected_end_ns=now)
        (gap,) = plan.gaps
        assert gap.start_ns == BASE_NS + 3 * MINUTE_NS
        # Minute 10 is still forming, so the last bar legitimately expected is 9.
        assert gap.end_ns == BASE_NS + 9 * MINUTE_NS
        assert gap.bar_count == 7

    def test_a_current_series_reports_nothing(self) -> None:
        """Bars through minute 9, now is mid-minute-10: nothing is missing."""
        bars = [bar(minute=m) for m in range(10)]
        now = BASE_NS + 10 * MINUTE_NS + 30 * 1_000_000_000
        assert find_gaps(bars, expected_end_ns=now).is_empty

    def test_the_forming_bucket_is_never_demanded(self) -> None:
        """An exact bucket boundary means that bucket has just opened with
        zero elapsed time — asking for it would demand a bar that cannot
        exist yet."""
        bars = [bar(minute=m) for m in range(10)]
        exactly_on_boundary = BASE_NS + 10 * MINUTE_NS
        assert find_gaps(bars, expected_end_ns=exactly_on_boundary).is_empty

    def test_an_expected_end_in_the_past_asks_for_nothing(self) -> None:
        bars = [bar(minute=m) for m in range(10)]
        stale = BASE_NS + 2 * MINUTE_NS
        assert find_gaps(bars, expected_end_ns=stale).is_empty

    def test_it_combines_with_an_interior_hole(self) -> None:
        bars = [bar(minute=m) for m in (0, 5)]
        now = BASE_NS + 8 * MINUTE_NS + 1
        plan = find_gaps(bars, expected_end_ns=now)
        assert [(g.start_ns, g.bar_count) for g in plan.gaps] == [
            (BASE_NS + 1 * MINUTE_NS, 4),  # minutes 1-4
            (BASE_NS + 6 * MINUTE_NS, 2),  # minutes 6-7 (8 is still forming)
        ]


class TestMergePrecedence:
    def test_live_wins_over_backfill_on_the_same_bucket(self) -> None:
        """Spec §FR-MD-04. Replacing a bar a strategy already acted on would
        make the backtest and the live run disagree about what was known."""
        live = [bar(minute=1, close="100", source="live")]
        backfilled = [bar(minute=1, close="999", source="rest")]
        (merged,) = merge_bars(live, backfilled)
        assert merged.close == Decimal("100")
        assert merged.source == "live"

    def test_precedence_does_not_depend_on_argument_order_within_a_list(self) -> None:
        """Two REST revisions of the same bucket must still both lose to live."""
        live = [bar(minute=1, close="100", source="live")]
        backfilled = [
            bar(minute=1, close="888", source="rest-a"),
            bar(minute=1, close="999", source="rest-b"),
        ]
        (merged,) = merge_bars(live, backfilled)
        assert merged.source == "live"

    def test_backfill_fills_buckets_live_never_had(self) -> None:
        live = [bar(minute=0), bar(minute=3)]
        backfilled = [bar(minute=1, source="rest"), bar(minute=2, source="rest")]
        merged = merge_bars(live, backfilled)
        assert [b.open_ts for b in merged] == [BASE_NS + m * MINUTE_NS for m in range(4)]
        assert [b.source for b in merged] == ["live", "rest", "rest", "live"]

    def test_the_result_is_sorted_by_symbol_interval_and_time(self) -> None:
        merged = merge_bars(
            [bar(minute=2, symbol="MSFT"), bar(minute=1, symbol="AAPL")],
            [bar(minute=0, symbol="MSFT"), bar(minute=0, symbol="AAPL")],
        )
        assert [(b.symbol, b.open_ts) for b in merged] == [
            ("AAPL", BASE_NS),
            ("AAPL", BASE_NS + MINUTE_NS),
            ("MSFT", BASE_NS),
            ("MSFT", BASE_NS + 2 * MINUTE_NS),
        ]

    def test_different_intervals_of_one_symbol_do_not_collide(self) -> None:
        """The merge key includes the interval — a 1m and a 5m bar opening at
        the same instant are different bars, not a conflict."""
        merged = merge_bars(
            [bar(minute=0, interval="1m")],
            [bar(minute=0, interval="5m", source="rest")],
        )
        assert len(merged) == 2

    def test_merging_nothing_into_nothing_is_empty(self) -> None:
        assert merge_bars([], []) == []

    def test_a_backfill_only_merge_keeps_everything(self) -> None:
        backfilled = [bar(minute=m, source="rest") for m in range(3)]
        assert len(merge_bars([], backfilled)) == 3


class TestBucketAlignmentIsShared:
    """``find_gaps`` floors ``expected_end_ns`` to a bucket boundary to decide
    which bar is still forming. If it used a different grid from the one
    ``BarAggregator`` uses to *open* bars, backfill would request bars that
    never existed and skip ones that did — silently, since both sides would
    look internally consistent. This pins the two together.
    """

    @pytest.mark.parametrize("interval_ns", [MINUTE_NS, 5 * MINUTE_NS, 3600 * 1_000_000_000])
    def test_a_bucket_start_is_a_fixed_point(self, interval_ns: int) -> None:
        start = BarAggregator.bucket_start(BASE_NS + 12_345_678_901, interval_ns)
        assert BarAggregator.bucket_start(start, interval_ns) == start

    def test_find_gaps_stops_at_the_bar_the_aggregator_is_still_filling(self) -> None:
        bars = [bar(minute=m) for m in range(10)]
        now = BASE_NS + 10 * MINUTE_NS + 30 * 1_000_000_000
        forming_bucket = BarAggregator.bucket_start(now, MINUTE_NS)

        plan = find_gaps(bars, expected_end_ns=now)

        assert plan.is_empty, "minute 9 is the last closed bar and it is present"
        # ...and the bucket the aggregator is still filling is minute 10, the
        # one find_gaps declined to ask for.
        assert forming_bucket == BASE_NS + 10 * MINUTE_NS

    def test_a_gap_never_extends_into_the_forming_bucket(self) -> None:
        bars = [bar(minute=0)]
        now = BASE_NS + 7 * MINUTE_NS + 1
        (gap,) = find_gaps(bars, expected_end_ns=now).gaps
        assert gap.end_ns < BarAggregator.bucket_start(now, MINUTE_NS)


class TestGapArithmetic:
    def test_bar_count_of_a_single_bucket_gap_is_one(self) -> None:
        gap = Gap("AAPL", "1m", BASE_NS, BASE_NS)
        assert gap.bar_count == 1

    def test_bar_count_is_inclusive_of_both_ends(self) -> None:
        gap = Gap("AAPL", "1m", BASE_NS, BASE_NS + 4 * MINUTE_NS)
        assert gap.bar_count == 5

    def test_an_unsupported_interval_is_refused_rather_than_guessed(self) -> None:
        gap = Gap("AAPL", "3s", BASE_NS, BASE_NS)
        with pytest.raises(ValueError, match="unsupported bar interval"):
            _ = gap.bar_count
