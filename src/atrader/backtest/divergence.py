"""Backtest-versus-live divergence — the measuring tool for acceptance criterion #7.

    30일 페이퍼 트레이딩 결과가 백테스트 대비 괴리 30% 미만.

The criterion itself cannot be met inside a development session: it needs
thirty days of wall-clock time. What *can* be built is the instrument that
decides it, so that when operations does run the thirty days, the answer is a
computation rather than an argument. That is this module.

Both sides are scored by the same :func:`~atrader.backtest.metrics.performance_report`
that scores a backtest, because the two curves are the same kind of object —
spec §2.2's "same code, different data source" applied to measurement rather
than to strategy logic.

Three judgment calls are worth stating outright, because each one is a place
where a report could quietly lie:

**A ratio against zero is not a large number, it is no number.** If the
backtest earned exactly nothing and live earned 0.5%, the relative divergence
is not "infinite" or "50000%" — it is undefined. :attr:`MetricDivergence.relative_pct`
returns ``None`` there, and :attr:`DivergenceReport.incomparable` collects
those so they are visible rather than silently counted as passes. A metric
that cannot be judged must not be reported as having been judged.

**Different-length runs are not comparable on totals.** A one-year backtest
and a thirty-day paper run have wildly different ``total_return_pct`` even
when the strategy behaves identically. :attr:`DivergenceReport.periods_aligned`
says whether the two runs are close enough in length for the totals to mean
anything, and it is false by default for anything outside a 2x ratio. Read it
before reading the numbers.

**Not every metric should gate.** Trade counts and volatility are diagnostic —
they explain *why* returns diverged — but a strategy that traded 8 times
instead of 10 has not failed the criterion. :data:`DEFAULT_GATED_METRICS` names
the three that answer the actual question ("does the backtest predict live
behaviour"); the rest are reported for the post-mortem and excluded from the
verdict.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from atrader.backtest.metrics import PerformanceReport, performance_report
from atrader.core.errors import ATraderError
from atrader.core.money import ZERO

__all__ = [
    "DEFAULT_GATED_METRICS",
    "DEFAULT_TOLERANCE_PCT",
    "DivergenceReport",
    "MetricDivergence",
    "divergence_report",
    "dump_equity_curve",
    "load_equity_curve",
]

#: Spec §12-7. Expressed as a percentage of the backtest figure.
DEFAULT_TOLERANCE_PCT = Decimal(30)

#: The metrics whose divergence decides the criterion. Return says whether the
#: backtest predicted the outcome, Sharpe says whether it predicted the ride,
#: and max drawdown says whether it predicted the worst moment — the three
#: things a backtest is actually consulted for.
DEFAULT_GATED_METRICS: tuple[str, ...] = (
    "total_return_pct",
    "sharpe_ratio",
    "max_drawdown_pct",
)

#: Beyond this ratio between the two runs' period counts, totals stop being
#: comparable. Two is deliberately generous: it is a warning threshold, not a
#: precision claim.
_MAX_PERIOD_RATIO = Decimal(2)


@dataclass(frozen=True, slots=True)
class MetricDivergence:
    """One metric, measured on both sides."""

    name: str
    backtest: Decimal
    live: Decimal

    @property
    def absolute(self) -> Decimal:
        """Live minus backtest. Signed — the direction matters."""
        return self.live - self.backtest

    @property
    def relative_pct(self) -> Decimal | None:
        """Divergence as a percentage of the backtest figure, or ``None``.

        ``None`` means the backtest figure was zero *and live's was not*, so
        there is no baseline to be a percentage of. Callers must treat that as
        "unknown", never as "zero divergence" — see this module's docstring.

        Zero against zero is the one case that is not undefined: a backtest
        that never drew down and a live run that never drew down have not
        diverged, they agree exactly. Reporting that pair as incomparable
        would fail a perfectly matching run on a technicality.
        """
        if self.backtest == ZERO:
            return ZERO if self.live == ZERO else None
        return abs(self.absolute) / abs(self.backtest) * Decimal(100)

    def exceeds(self, tolerance_pct: Decimal) -> bool:
        """True only when a divergence is both measurable and over the limit."""
        relative = self.relative_pct
        return relative is not None and relative > tolerance_pct


@dataclass(frozen=True, slots=True)
class DivergenceReport:
    backtest: PerformanceReport
    live: PerformanceReport
    metrics: tuple[MetricDivergence, ...]
    tolerance_pct: Decimal
    gated_metrics: tuple[str, ...]

    def by_name(self, name: str) -> MetricDivergence:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError(f"no metric named {name!r} in this report")

    @property
    def gated(self) -> tuple[MetricDivergence, ...]:
        return tuple(m for m in self.metrics if m.name in self.gated_metrics)

    @property
    def breaches(self) -> tuple[MetricDivergence, ...]:
        """Gated metrics that diverged past the tolerance."""
        return tuple(m for m in self.gated if m.exceeds(self.tolerance_pct))

    @property
    def incomparable(self) -> tuple[MetricDivergence, ...]:
        """Gated metrics with no measurable divergence (zero backtest baseline).

        These are neither passes nor failures. A report with any of these has
        not fully answered the question, and :attr:`verdict` says so.
        """
        return tuple(m for m in self.gated if m.relative_pct is None)

    @property
    def periods_aligned(self) -> bool:
        """Whether the two runs are close enough in length to compare totals.

        A one-year backtest against a thirty-day paper run will differ on
        ``total_return_pct`` for reasons that have nothing to do with strategy
        fidelity. When this is false, the numbers below are describing two
        different questions.
        """
        shorter, longer = sorted((self.backtest.num_periods, self.live.num_periods))
        if shorter == 0:
            return False
        return Decimal(longer) / Decimal(shorter) <= _MAX_PERIOD_RATIO

    @property
    def within_tolerance(self) -> bool:
        """The criterion's verdict: every gated metric measurably inside the limit.

        Deliberately strict about the unmeasurable case — an incomparable
        metric makes this false, because a criterion that cannot be evaluated
        has not been passed. Misaligned periods do the same: a comparison
        between runs of very different length is not evidence either way.
        """
        return (
            self.periods_aligned
            and not self.incomparable
            and not self.breaches
            and bool(self.gated)
        )

    def summary(self) -> str:
        """One-screen human summary — what the runbook prints after 30 days."""
        lines = [
            f"backtest {self.backtest.num_periods} periods vs "
            f"live {self.live.num_periods} periods, tolerance {self.tolerance_pct}%",
        ]
        if not self.periods_aligned:
            lines.append(
                "  WARNING: run lengths differ by more than "
                f"{_MAX_PERIOD_RATIO}x — totals are not comparable"
            )
        for metric in self.metrics:
            relative = metric.relative_pct
            shown = "n/a (backtest baseline is zero)" if relative is None else f"{relative:.1f}%"
            mark = ""
            if metric.name in self.gated_metrics:
                mark = " [FAIL]" if metric.exceeds(self.tolerance_pct) else " [gated]"
            lines.append(
                f"  {metric.name}: backtest {metric.backtest:.4f} -> "
                f"live {metric.live:.4f} (divergence {shown}){mark}"
            )
        lines.append(
            f"verdict: {'WITHIN TOLERANCE' if self.within_tolerance else 'OUT OF TOLERANCE'}"
        )
        return "\n".join(lines)


def divergence_report(
    backtest_equity: Sequence[tuple[int, Decimal]],
    live_equity: Sequence[tuple[int, Decimal]],
    *,
    tolerance_pct: Decimal = DEFAULT_TOLERANCE_PCT,
    periods_per_year: int = 252,
    backtest_trade_pnls: Sequence[Decimal] = (),
    live_trade_pnls: Sequence[Decimal] = (),
    gated_metrics: Sequence[str] = DEFAULT_GATED_METRICS,
) -> DivergenceReport:
    """Score both equity curves and measure how far live drifted from backtest.

    Both curves are ``(at_ns, equity)`` — the shape
    :attr:`~atrader.backtest.engine.BacktestResult.equity_curve` already has,
    and the shape :meth:`~atrader.app.runtime.Runtime.daily_equity_curve`
    produces for a live session, so neither side needs converting.
    """
    backtest = performance_report(
        backtest_equity, periods_per_year=periods_per_year, trade_pnls=backtest_trade_pnls
    )
    live = performance_report(
        live_equity, periods_per_year=periods_per_year, trade_pnls=live_trade_pnls
    )

    metrics = (
        *(
            MetricDivergence(name=name, backtest=getattr(backtest, name), live=getattr(live, name))
            for name in (
                "total_return_pct",
                "annualized_return_pct",
                "annualized_volatility_pct",
                "sharpe_ratio",
                "max_drawdown_pct",
            )
        ),
        # Counts are not PerformanceReport Decimals, so they are built
        # separately rather than pulled by attribute name like the rest.
        MetricDivergence(
            name="num_trades",
            backtest=Decimal(backtest.num_trades),
            live=Decimal(live.num_trades),
        ),
    )

    return DivergenceReport(
        backtest=backtest,
        live=live,
        metrics=metrics,
        tolerance_pct=tolerance_pct,
        gated_metrics=tuple(gated_metrics),
    )


# ---------------------------------------------------------------------------
# On-disk format
# ---------------------------------------------------------------------------
#
# JSONL, one ``{"at_ns": int, "equity": str}`` per line. Two deliberate
# choices: JSONL because ``.gitignore`` excludes ``*.csv``/``*.parquet`` but
# not ``*.jsonl`` (so a recorded curve can be committed as a fixture), and
# equity as a *string* because JSON numbers are floats — writing a Decimal
# equity as a JSON number would silently discard the exactness the whole
# codebase is built to preserve.


def dump_equity_curve(curve: Sequence[tuple[int, Decimal]], path: Path) -> None:
    """Write an equity curve where a later session can read it back exactly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for at_ns, equity in curve:
            handle.write(json.dumps({"at_ns": at_ns, "equity": str(equity)}) + "\n")


def load_equity_curve(path: Path) -> tuple[tuple[int, Decimal], ...]:
    """Read a curve written by :func:`dump_equity_curve`.

    Malformed input raises rather than being skipped: a divergence report
    computed from a partially-read curve would be wrong in a way nobody would
    notice, which is worse than not producing one.
    """
    points: list[tuple[int, Decimal]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                points.append((int(record["at_ns"]), Decimal(record["equity"])))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, InvalidOperation) as exc:
                raise ATraderError(
                    f"{path}:{lineno} is not a valid equity curve point: {exc}"
                ) from exc
    return tuple(points)
