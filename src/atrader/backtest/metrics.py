"""Backtest performance metrics — spec §8.4.

Everything here reads an equity curve (and, for trade-level stats, a sequence
of realized per-trade P&L) rather than a live account, so the exact same
functions score a backtest and a paper-trading session — another instance of
spec §2.2's "same code, different data source" principle.

**Deflated Sharpe Ratio** is the one metric here worth a paragraph. A Sharpe
ratio computed after trying many strategy variants and reporting the best one
is inflated by selection — the more variants tried, the more likely *some*
one of them clears a given bar by chance alone. DSR (Bailey & López de Prado,
2014) answers a sharper question than "what is the Sharpe": *what is the
probability this Sharpe is genuinely positive, given that ``num_trials``
variants were tried to find it?* It is why :func:`deflated_sharpe_ratio` takes
a trial count as a required argument rather than defaulting it to one — a
default would make it silent about the very bias it exists to correct for.

Everything monetary here is ``Decimal``. The deflated Sharpe calculation is
the one exception — it needs the normal distribution's CDF and inverse CDF,
which only exist as float-precision functions in the standard library, and a
probability is not money the way an equity curve is.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise
from statistics import NormalDist

from atrader.core.money import ZERO

__all__ = ["PerformanceReport", "deflated_sharpe_ratio", "performance_report"]

_EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    total_return_pct: Decimal
    annualized_return_pct: Decimal
    annualized_volatility_pct: Decimal
    sharpe_ratio: Decimal
    max_drawdown_pct: Decimal
    win_rate_pct: Decimal | None
    profit_factor: Decimal | None
    num_periods: int
    num_trades: int


def period_returns(equity_curve: Sequence[tuple[int, Decimal]]) -> list[Decimal]:
    """Simple period-over-period returns. Empty or single-point curves have none."""
    returns: list[Decimal] = []
    for (_, previous), (_, current) in pairwise(equity_curve):
        if previous == ZERO:
            continue  # a wiped-out account has no meaningful percentage return
        returns.append((current - previous) / previous)
    return returns


def total_return_pct(equity_curve: Sequence[tuple[int, Decimal]]) -> Decimal:
    if len(equity_curve) < 2:
        return ZERO
    start = equity_curve[0][1]
    end = equity_curve[-1][1]
    if start == ZERO:
        return ZERO
    return (end - start) / start * Decimal(100)


def max_drawdown_pct(equity_curve: Sequence[tuple[int, Decimal]]) -> Decimal:
    peak = ZERO
    worst = ZERO
    for _, equity in equity_curve:
        peak = max(peak, equity)
        if peak > ZERO:
            drawdown = (peak - equity) / peak * Decimal(100)
            worst = max(worst, drawdown)
    return worst


def sharpe_ratio(
    returns: Sequence[Decimal], *, periods_per_year: int, risk_free_pct: Decimal = ZERO
) -> Decimal:
    """Annualized Sharpe ratio from a series of period returns.

    ``periods_per_year`` converts the period's own frequency (daily, hourly,
    ...) to an annual figure — 252 for daily equity bars, for instance. Zero
    volatility (a dead-flat return series, including a single observation)
    returns zero rather than dividing by zero.
    """
    n = len(returns)
    if n < 2:
        return ZERO
    period_risk_free = risk_free_pct / Decimal(100) / periods_per_year
    excess = [r - period_risk_free for r in returns]
    mean = sum(excess, ZERO) / n
    variance = sum(((r - mean) ** 2 for r in excess), ZERO) / (n - 1)
    if variance <= ZERO:
        return ZERO
    std = variance.sqrt()
    return (mean / std) * Decimal(periods_per_year).sqrt()


def win_rate_pct(trade_pnls: Sequence[Decimal]) -> Decimal | None:
    if not trade_pnls:
        return None
    wins = sum(1 for pnl in trade_pnls if pnl > ZERO)
    return Decimal(wins) / Decimal(len(trade_pnls)) * Decimal(100)


def profit_factor(trade_pnls: Sequence[Decimal]) -> Decimal | None:
    """Gross profit over gross loss. ``None`` when there is nothing to divide
    by — no trades, or no losing trades to form a ratio against."""
    if not trade_pnls:
        return None
    gross_profit = sum((pnl for pnl in trade_pnls if pnl > ZERO), ZERO)
    gross_loss = sum((-pnl for pnl in trade_pnls if pnl < ZERO), ZERO)
    if gross_loss == ZERO:
        return None
    return gross_profit / gross_loss


def performance_report(
    equity_curve: Sequence[tuple[int, Decimal]],
    *,
    periods_per_year: int = 252,
    risk_free_pct: Decimal = ZERO,
    trade_pnls: Sequence[Decimal] = (),
) -> PerformanceReport:
    returns = period_returns(equity_curve)
    total_periods = len(equity_curve)
    years = Decimal(total_periods) / Decimal(periods_per_year) if periods_per_year else ZERO
    total = total_return_pct(equity_curve)
    annualized_return = (
        (((Decimal(1) + total / Decimal(100)) ** (Decimal(1) / years) - Decimal(1)) * Decimal(100))
        if years > ZERO
        else ZERO
    )
    annualized_vol = ZERO
    if len(returns) >= 2:
        n = len(returns)
        mean = sum(returns, ZERO) / n
        variance = sum(((r - mean) ** 2 for r in returns), ZERO) / (n - 1)
        annualized_vol = variance.sqrt() * Decimal(periods_per_year).sqrt() * Decimal(100)

    return PerformanceReport(
        total_return_pct=total,
        annualized_return_pct=annualized_return,
        annualized_volatility_pct=annualized_vol,
        sharpe_ratio=sharpe_ratio(
            returns, periods_per_year=periods_per_year, risk_free_pct=risk_free_pct
        ),
        max_drawdown_pct=max_drawdown_pct(equity_curve),
        win_rate_pct=win_rate_pct(trade_pnls),
        profit_factor=profit_factor(trade_pnls),
        num_periods=total_periods,
        num_trades=len(trade_pnls),
    )


def _skewness(values: list[float]) -> float:
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = variance**0.5
    if std == 0:
        return 0.0
    return float((sum((v - mean) ** 3 for v in values) / n) / std**3)


def _kurtosis(values: list[float]) -> float:
    """Non-excess kurtosis — a normal distribution scores 3.0, not 0.0."""
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = variance**0.5
    if std == 0:
        return 3.0
    return float((sum((v - mean) ** 4 for v in values) / n) / std**4)


def deflated_sharpe_ratio(
    returns: Sequence[Decimal],
    *,
    num_trials: int,
    benchmark_sharpe: float = 0.0,
) -> Decimal:
    """P(true Sharpe > 0), adjusted for having tried ``num_trials`` variants.

    Bailey & López de Prado (2014), "The Deflated Sharpe Ratio". ``num_trials``
    of 1 skips the multiple-testing correction and tests the observed Sharpe
    directly against ``benchmark_sharpe`` — use that only when this really is
    the one and only strategy variant ever evaluated.
    """
    values = [float(r) for r in returns]
    n = len(values)
    if n < 2:
        return ZERO

    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    if variance <= 0:
        return ZERO
    std = variance**0.5
    sr_hat = mean / std
    skew = _skewness(values)
    kurt = _kurtosis(values)

    denom = 1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat**2
    if denom <= 0:
        return ZERO

    dist = NormalDist()
    if num_trials <= 1:
        sr0 = benchmark_sharpe
    else:
        var_sr = denom / (n - 1)
        sr0 = var_sr**0.5 * (
            (1 - _EULER_MASCHERONI) * dist.inv_cdf(1 - 1 / num_trials)
            + _EULER_MASCHERONI * dist.inv_cdf(1 - 1 / (num_trials * math.e))
        )

    z = (sr_hat - sr0) * (n - 1) ** 0.5 / denom**0.5
    return Decimal(str(dist.cdf(z)))
