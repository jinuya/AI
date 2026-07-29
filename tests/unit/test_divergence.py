"""Backtest-vs-live divergence — the measuring tool for acceptance criterion #7.

The interesting tests here are not "does it subtract two numbers correctly".
They are the three places :mod:`atrader.backtest.divergence` makes a judgment
call that a naive implementation would get wrong in a way nobody would notice:
a zero baseline, two runs of very different length, and which metrics are
allowed to decide the verdict. Each one is a way a report could claim to have
answered a question it did not answer.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from atrader.backtest.divergence import (
    DEFAULT_GATED_METRICS,
    DEFAULT_TOLERANCE_PCT,
    MetricDivergence,
    divergence_report,
    dump_equity_curve,
    load_equity_curve,
)
from atrader.core.errors import ATraderError

DAY_NS = 86_400 * 1_000_000_000


def curve(*equities: str, start_ns: int = 0) -> tuple[tuple[int, Decimal], ...]:
    return tuple((start_ns + i * DAY_NS, Decimal(e)) for i, e in enumerate(equities))


def flat_curve(points: int, *, value: str = "100000") -> tuple[tuple[int, Decimal], ...]:
    return curve(*([value] * points))


def rising_curve(points: int, *, start: str = "100000", step: str = "100"):  # type: ignore[no-untyped-def]
    base = Decimal(start)
    increment = Decimal(step)
    return tuple((i * DAY_NS, base + increment * i) for i in range(points))


class TestRelativeDivergenceAgainstAZeroBaseline:
    def test_a_zero_backtest_figure_yields_none_not_infinity(self) -> None:
        metric = MetricDivergence(name="total_return_pct", backtest=Decimal(0), live=Decimal("0.5"))
        assert metric.relative_pct is None
        assert metric.absolute == Decimal("0.5")

    def test_an_unmeasurable_metric_never_counts_as_a_breach(self) -> None:
        """`exceeds` must not treat "no baseline" as "over the limit" — that
        would fail a run for a reason the data cannot support."""
        metric = MetricDivergence(name="sharpe_ratio", backtest=Decimal(0), live=Decimal("99"))
        assert metric.exceeds(DEFAULT_TOLERANCE_PCT) is False

    def test_but_an_unmeasurable_gated_metric_blocks_a_pass(self) -> None:
        """...and it must not be silently counted as a pass either. A flat
        backtest earned nothing, so a live run that earned something has no
        baseline to be a percentage of — the correct verdict is "cannot say",
        which `within_tolerance` reports as False."""
        report = divergence_report(flat_curve(10), rising_curve(10))
        assert {m.name for m in report.incomparable} == {"total_return_pct", "sharpe_ratio"}
        assert report.breaches == ()
        assert report.within_tolerance is False

    def test_zero_against_zero_is_agreement_not_incomparability(self) -> None:
        """The one case a zero baseline is still measurable. Both curves here
        rise monotonically, so both have exactly zero max drawdown — they
        agree perfectly, and calling that "cannot compare" would fail a
        matching run on a technicality."""
        report = divergence_report(rising_curve(30), rising_curve(30))
        drawdown = report.by_name("max_drawdown_pct")
        assert (drawdown.backtest, drawdown.live) == (Decimal(0), Decimal(0))
        assert drawdown.relative_pct == Decimal(0)
        assert drawdown not in report.incomparable


class TestPeriodAlignment:
    def test_runs_of_similar_length_are_comparable(self) -> None:
        report = divergence_report(rising_curve(30), rising_curve(30))
        assert report.periods_aligned is True

    def test_a_year_of_backtest_against_a_month_of_live_is_not(self) -> None:
        report = divergence_report(rising_curve(252), rising_curve(21))
        assert report.periods_aligned is False

    def test_misaligned_runs_cannot_pass_however_close_the_numbers_look(self) -> None:
        """Identical per-day behaviour over different spans produces very
        different totals. Passing that would be the tool endorsing a
        comparison it should have refused."""
        report = divergence_report(rising_curve(252), rising_curve(21))
        assert report.within_tolerance is False
        assert "not comparable" in report.summary()

    def test_an_empty_curve_is_never_aligned(self) -> None:
        report = divergence_report(rising_curve(30), ())
        assert report.periods_aligned is False
        assert report.within_tolerance is False


class TestTheVerdict:
    def test_identical_curves_are_within_tolerance(self) -> None:
        report = divergence_report(rising_curve(30), rising_curve(30))
        assert report.breaches == ()
        assert report.incomparable == ()
        assert report.within_tolerance is True
        assert "WITHIN TOLERANCE" in report.summary()

    def test_a_live_run_that_earned_half_as_much_breaches(self) -> None:
        report = divergence_report(rising_curve(30, step="100"), rising_curve(30, step="50"))
        assert report.within_tolerance is False
        assert "total_return_pct" in {m.name for m in report.breaches}

    def test_a_small_shortfall_stays_inside_the_limit(self) -> None:
        """10% below the backtest is well inside the 30% the criterion allows."""
        report = divergence_report(rising_curve(30, step="100"), rising_curve(30, step="90"))
        assert report.by_name("total_return_pct").relative_pct < DEFAULT_TOLERANCE_PCT
        assert report.within_tolerance is True

    def test_a_sign_flip_is_a_breach_not_a_small_difference(self) -> None:
        """Backtest up, live down. The absolute gap may be modest but the
        prediction was qualitatively wrong, and a percentage of the baseline
        says so (>100%)."""
        up = curve("100000", "101000", "102000", "103000")
        down = curve("100000", "99000", "98000", "97000")
        report = divergence_report(up, down)
        assert report.by_name("total_return_pct").relative_pct > Decimal(100)
        assert report.within_tolerance is False

    def test_tolerance_is_configurable(self) -> None:
        """Half the backtest's return is a 50% divergence: a breach at the
        criterion's 30%, acceptable at 60%. Gated on return alone so the knob
        is the only thing under test."""
        backtest, live = rising_curve(30, step="100"), rising_curve(30, step="50")
        gated = ("total_return_pct",)
        strict = divergence_report(backtest, live, gated_metrics=gated)
        lenient = divergence_report(backtest, live, tolerance_pct=Decimal(60), gated_metrics=gated)
        assert strict.within_tolerance is False
        assert lenient.within_tolerance is True


class TestOnlyGatedMetricsDecide:
    def test_the_default_gated_set_is_the_three_documented_metrics(self) -> None:
        """Return says whether the backtest predicted the outcome, Sharpe the
        ride, drawdown the worst moment. Widening this set changes what the
        criterion means, so it should not happen by accident."""
        assert DEFAULT_GATED_METRICS == (
            "total_return_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
        )
        report = divergence_report(rising_curve(5), rising_curve(5))
        assert {m.name for m in report.gated} == set(DEFAULT_GATED_METRICS)

    def test_trade_count_divergence_alone_does_not_fail_a_run(self) -> None:
        """num_trades is reported for the post-mortem but is not part of the
        verdict — a run that reached the same result with fewer trades has not
        failed the criterion."""
        equity = rising_curve(30)
        report = divergence_report(
            equity,
            equity,
            backtest_trade_pnls=[Decimal(10)] * 100,
            live_trade_pnls=[Decimal(10)] * 3,
        )
        trades = report.by_name("num_trades")
        assert trades.relative_pct > DEFAULT_TOLERANCE_PCT
        assert "num_trades" not in {m.name for m in report.breaches}
        assert report.within_tolerance is True

    def test_the_gated_set_can_be_narrowed(self) -> None:
        report = divergence_report(
            rising_curve(30, step="100"),
            rising_curve(30, step="50"),
            gated_metrics=("max_drawdown_pct",),
        )
        assert {m.name for m in report.gated} == {"max_drawdown_pct"}

    def test_an_empty_gated_set_cannot_pass(self) -> None:
        """Gating on nothing is not the same as passing everything."""
        report = divergence_report(rising_curve(30), rising_curve(30), gated_metrics=())
        assert report.within_tolerance is False

    def test_by_name_rejects_an_unknown_metric(self) -> None:
        report = divergence_report(rising_curve(5), rising_curve(5))
        with pytest.raises(KeyError):
            report.by_name("no_such_metric")


class TestEquityCurveRoundTrip:
    def test_a_curve_survives_a_write_and_read_exactly(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        original = curve("100000.12345678", "100001.87654321", "99999.00000001")
        path = tmp_path / "equity.jsonl"
        dump_equity_curve(original, path)
        assert load_equity_curve(path) == original

    def test_equity_is_stored_as_a_string_so_decimals_stay_exact(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A JSON *number* would go through float and lose the last digits —
        the one thing this codebase refuses to do with money."""
        path = tmp_path / "equity.jsonl"
        dump_equity_curve(curve("100000.12345678"), path)
        assert '"equity": "100000.12345678"' in path.read_text(encoding="utf-8")

    def test_missing_parent_directories_are_created(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "nested" / "deeper" / "equity.jsonl"
        dump_equity_curve(curve("100000"), path)
        assert path.is_file()

    def test_an_empty_curve_round_trips_as_empty(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "equity.jsonl"
        dump_equity_curve((), path)
        assert load_equity_curve(path) == ()

    @pytest.mark.parametrize(
        "bad_line",
        [
            "not json at all",
            '{"at_ns": 1}',  # missing equity
            '{"equity": "100"}',  # missing at_ns
            '{"at_ns": 1, "equity": "not-a-number"}',
            '{"at_ns": "not-an-int", "equity": "100"}',
        ],
    )
    def test_a_malformed_point_raises_rather_than_being_skipped(
        self,
        tmp_path,  # type: ignore[no-untyped-def]
        bad_line: str,
    ) -> None:
        """Skipping a bad line would produce a shorter curve and therefore a
        wrong report, with nothing on screen to say so."""
        path = tmp_path / "equity.jsonl"
        path.write_text(f'{{"at_ns": 0, "equity": "100"}}\n{bad_line}\n', encoding="utf-8")
        with pytest.raises(ATraderError, match="equity curve point"):
            load_equity_curve(path)

    def test_blank_lines_are_tolerated(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "equity.jsonl"
        path.write_text('{"at_ns": 0, "equity": "100"}\n\n\n', encoding="utf-8")
        assert load_equity_curve(path) == ((0, Decimal("100")),)
