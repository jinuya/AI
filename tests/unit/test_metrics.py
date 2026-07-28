"""Backtest performance metrics — spec §8.4."""

from __future__ import annotations

from decimal import Decimal

from atrader.backtest.metrics import (
    deflated_sharpe_ratio,
    max_drawdown_pct,
    performance_report,
    period_returns,
    profit_factor,
    sharpe_ratio,
    total_return_pct,
    win_rate_pct,
)

BASE_NS = 1_700_000_000_000_000_000
DAY_NS = 86_400_000_000_000


def curve(*equities: str) -> list[tuple[int, Decimal]]:
    return [(BASE_NS + i * DAY_NS, Decimal(e)) for i, e in enumerate(equities)]


class TestPeriodReturns:
    def test_a_single_point_curve_has_no_returns(self) -> None:
        assert period_returns(curve("100000")) == []

    def test_returns_are_computed_period_over_period(self) -> None:
        returns = period_returns(curve("100", "110", "99"))
        assert returns[0] == Decimal("0.1")
        assert returns[1] == Decimal("99") / Decimal("110") - Decimal(1)

    def test_a_zero_equity_period_is_skipped_rather_than_dividing_by_zero(self) -> None:
        returns = period_returns(curve("100", "0", "50"))
        assert len(returns) == 1  # only 100->0; 0->50 is unrepresentable as a % return


class TestTotalReturn:
    def test_flat_equity_is_zero_return(self) -> None:
        assert total_return_pct(curve("100000", "100000")) == Decimal("0")

    def test_growth_is_a_positive_percentage(self) -> None:
        assert total_return_pct(curve("100000", "110000")) == Decimal("10")

    def test_a_single_point_curve_has_no_return(self) -> None:
        assert total_return_pct(curve("100000")) == Decimal("0")

    def test_a_zero_starting_equity_is_zero_not_a_division_error(self) -> None:
        assert total_return_pct(curve("0", "100")) == Decimal("0")


class TestMaxDrawdown:
    def test_a_monotonically_rising_curve_has_zero_drawdown(self) -> None:
        assert max_drawdown_pct(curve("100", "110", "120")) == Decimal("0")

    def test_drawdown_is_measured_from_the_running_peak(self) -> None:
        # peak 120 -> trough 90 is a 25% drawdown, even after a partial recovery.
        result = max_drawdown_pct(curve("100", "120", "90", "100"))
        assert result == Decimal("25")

    def test_multiple_drawdowns_report_the_worst_one(self) -> None:
        result = max_drawdown_pct(curve("100", "80", "100", "50"))
        assert result == Decimal("50")


class TestSharpeRatio:
    def test_fewer_than_two_returns_is_zero(self) -> None:
        assert sharpe_ratio([Decimal("0.01")], periods_per_year=252) == Decimal("0")

    def test_zero_volatility_is_zero_not_a_division_error(self) -> None:
        returns = [Decimal("0.01")] * 5
        assert sharpe_ratio(returns, periods_per_year=252) == Decimal("0")

    def test_consistently_positive_returns_score_a_positive_sharpe(self) -> None:
        returns = [Decimal("0.01"), Decimal("0.02"), Decimal("0.005"), Decimal("0.015")]
        assert sharpe_ratio(returns, periods_per_year=252) > Decimal("0")

    def test_consistently_negative_returns_score_a_negative_sharpe(self) -> None:
        returns = [Decimal("-0.01"), Decimal("-0.02"), Decimal("-0.005"), Decimal("-0.015")]
        assert sharpe_ratio(returns, periods_per_year=252) < Decimal("0")


class TestWinRateAndProfitFactor:
    def test_no_trades_is_none_not_zero(self) -> None:
        assert win_rate_pct([]) is None
        assert profit_factor([]) is None

    def test_win_rate_is_the_share_of_positive_trades(self) -> None:
        pnls = [Decimal("10"), Decimal("-5"), Decimal("3"), Decimal("-1")]
        assert win_rate_pct(pnls) == Decimal("50")

    def test_profit_factor_is_gross_profit_over_gross_loss(self) -> None:
        pnls = [Decimal("30"), Decimal("-10")]
        assert profit_factor(pnls) == Decimal("3")

    def test_no_losing_trades_makes_profit_factor_undefined(self) -> None:
        assert profit_factor([Decimal("10"), Decimal("5")]) is None


class TestPerformanceReport:
    def test_assembles_every_metric_consistently(self) -> None:
        equity_curve = curve("100000", "101000", "99000", "102000")
        report = performance_report(
            equity_curve,
            periods_per_year=252,
            trade_pnls=[Decimal("500"), Decimal("-200")],
        )
        assert report.num_periods == 4
        assert report.num_trades == 2
        assert report.total_return_pct == total_return_pct(equity_curve)
        assert report.max_drawdown_pct == max_drawdown_pct(equity_curve)
        assert report.win_rate_pct == Decimal("50")

    def test_a_flat_single_point_curve_produces_zeros_not_errors(self) -> None:
        report = performance_report(curve("100000"), periods_per_year=252)
        assert report.total_return_pct == Decimal("0")
        assert report.annualized_return_pct == Decimal("0")
        assert report.sharpe_ratio == Decimal("0")


class TestDeflatedSharpeRatio:
    def test_fewer_than_two_returns_is_zero(self) -> None:
        assert deflated_sharpe_ratio([Decimal("0.01")], num_trials=1) == Decimal("0")

    def test_zero_volatility_is_zero(self) -> None:
        assert deflated_sharpe_ratio([Decimal("0.01")] * 5, num_trials=1) == Decimal("0")

    def test_result_is_always_a_valid_probability(self) -> None:
        returns = [
            Decimal("0.01"),
            Decimal("-0.02"),
            Decimal("0.03"),
            Decimal("0.005"),
            Decimal("-0.01"),
        ]
        result = deflated_sharpe_ratio(returns, num_trials=10)
        assert Decimal("0") <= result <= Decimal("1")

    def test_more_trials_makes_the_same_track_record_less_convincing(self) -> None:
        returns = [
            Decimal("0.02"),
            Decimal("0.01"),
            Decimal("0.03"),
            Decimal("0.015"),
            Decimal("0.025"),
            Decimal("0.01"),
        ]
        few_trials = deflated_sharpe_ratio(returns, num_trials=2)
        many_trials = deflated_sharpe_ratio(returns, num_trials=200)
        assert many_trials < few_trials

    def test_a_single_trial_compares_directly_against_the_benchmark(self) -> None:
        returns = [Decimal("0.02"), Decimal("0.01"), Decimal("0.03"), Decimal("-0.01")]
        high_bar = deflated_sharpe_ratio(returns, num_trials=1, benchmark_sharpe=5.0)
        low_bar = deflated_sharpe_ratio(returns, num_trials=1, benchmark_sharpe=-5.0)
        assert low_bar > high_bar
