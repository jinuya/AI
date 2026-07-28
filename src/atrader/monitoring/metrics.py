"""Prometheus metrics — spec §9.2.

One :class:`Metrics` instance per process by default, but every instance owns
its own :class:`~prometheus_client.CollectorRegistry` rather than reaching for
the global default one — two ``Metrics()`` instances (e.g. two tests running
in the same process) never collide on a metric name.

This module only *defines* the metrics and exposes ``render()`` for a
``/metrics`` endpoint. It does not know when to update them — that is
:mod:`atrader.app.runtime`'s job, at each point in the pipeline the value
actually changes, exactly like the audit log records events at their source
rather than being reconstructed after the fact.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

__all__ = ["Metrics"]


class Metrics:
    """Every counter/gauge/histogram this system exposes, grouped by component."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()

        # -- Market data (spec §FR-MD-01/03) --------------------------------
        self.feed_lag_seconds = Histogram(
            "atrader_feed_lag_seconds",
            "Age of a tick when it was processed (ingest_ts - exchange_ts)",
            registry=self.registry,
        )
        self.data_quality_ok = Gauge(
            "atrader_data_quality_ok",
            "1 if a symbol's data quality is OK, 0 otherwise",
            ["symbol"],
            registry=self.registry,
        )

        # -- Risk engine (spec §7.2, §7.5) ----------------------------------
        self.risk_checks_total = Counter(
            "atrader_risk_checks_total",
            "Pre-trade check outcomes",
            ["check", "action"],
            registry=self.registry,
        )
        self.risk_decisions_total = Counter(
            "atrader_risk_decisions_total",
            "Risk engine verdicts",
            ["action"],
            registry=self.registry,
        )
        self.circuit_breaker_level = Gauge(
            "atrader_circuit_breaker_level",
            "0=NONE, 1=L1, 2=L2, 3=L3",
            registry=self.registry,
        )
        self.kill_switch_engaged = Gauge(
            "atrader_kill_switch_engaged",
            "1 if the kill switch is engaged",
            registry=self.registry,
        )

        # -- Execution (spec §FR-EXE-*, §6.3) -------------------------------
        self.orders_total = Counter(
            "atrader_orders_total",
            "Orders reaching a terminal or working state",
            ["status"],
            registry=self.registry,
        )
        self.reconciliation_breaks_total = Counter(
            "atrader_reconciliation_breaks_total",
            "Breaks found between local and broker state",
            ["kind"],
            registry=self.registry,
        )
        self.reconciliation_last_clean = Gauge(
            "atrader_reconciliation_last_clean",
            "1 if the most recent reconciliation pass found no breaks",
            registry=self.registry,
        )
        self.deadman_triggers_total = Counter(
            "atrader_deadman_triggers_total",
            "Dead-man switch activations",
            registry=self.registry,
        )
        self.rate_limit_throttled_total = Counter(
            "atrader_rate_limit_throttled_total",
            "Requests the rate limiter made wait",
            registry=self.registry,
        )

        # -- Portfolio (spec §FR-PF-*) ---------------------------------------
        self.equity = Gauge("atrader_equity", "Account equity", registry=self.registry)
        self.daily_pnl_pct = Gauge(
            "atrader_daily_pnl_pct", "Today's P&L as % of starting equity", registry=self.registry
        )
        self.drawdown_pct = Gauge(
            "atrader_drawdown_pct", "Current drawdown from peak equity, %", registry=self.registry
        )
        self.margin_status = Gauge(
            "atrader_margin_status", "0=OK, 1=WARN, 2=REDUCE_ONLY", registry=self.registry
        )

        # -- LLM strategy (spec §7.7, §9.2) ----------------------------------
        self.llm_requests_total = Counter(
            "atrader_llm_requests_total",
            "LLM calls by outcome",
            ["outcome"],
            registry=self.registry,
        )
        self.llm_latency_seconds = Histogram(
            "atrader_llm_latency_seconds", "LLM call latency", registry=self.registry
        )
        self.llm_tokens_total = Counter(
            "atrader_llm_tokens_total",
            "Tokens consumed by kind",
            ["kind"],
            registry=self.registry,
        )
        self.llm_rejections_total = Counter(
            "atrader_llm_rejections_total",
            "Guard rejections on LLM output",
            ["critical"],
            registry=self.registry,
        )

    def render(self) -> bytes:
        """Text-format exposition for a ``/metrics`` endpoint."""
        return generate_latest(self.registry)
