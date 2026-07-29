"""The CLI — spec §10.

    atrader run|paper|backtest|replay|kill|reconcile|verify-audit

``run``/``paper`` start a trading process (composition root:
:class:`~atrader.app.runtime.Runtime` plus its HTTP control surface).
``kill``/``reconcile``/``verify-audit`` (when not given ``--file``) are
*clients* of that process's API — this is the CLI access path spec
§FR-MON-03 requires alongside the UI and HTTP paths, not a separate
mechanism, since a kill switch reachable by two different code paths would be
two kill switches. ``backtest``/``replay`` are self-contained; they need no
running process.

This file is deliberately thin: everything it does is construct objects
already tested in isolation and call one method on them. No trading logic
lives here.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import typer

from atrader.audit.hashchain import (
    load_records_jsonl,
    records_from_json_bytes,
    verify_chain,
)
from atrader.backtest.divergence import (
    DEFAULT_TOLERANCE_PCT,
    divergence_report,
    dump_equity_curve,
    load_equity_curve,
)
from atrader.backtest.engine import BacktestEngine
from atrader.backtest.metrics import performance_report
from atrader.backtest.replay import assert_replay_matches
from atrader.config.loader import load_app_config
from atrader.config.schema import AppConfig, StrategyConfig
from atrader.config.secrets import ChainSecretProvider, EnvSecretProvider
from atrader.core.clock import NS_PER_SECOND, SimulatedClock, SystemClock
from atrader.core.errors import ATraderError
from atrader.core.ids import DeterministicIdGenerator
from atrader.features.engine import FeatureEngine, FeatureSpec
from atrader.features.registry import FeatureRegistry
from atrader.features.store import FeatureStore
from atrader.marketdata.aggregator import BarAggregator
from atrader.marketdata.feeds.simulated import SimulatedFeed
from atrader.marketdata.models import Bar
from atrader.monitoring.logging import configure_logging, get_logger
from atrader.risk.killswitch import KillSwitch
from atrader.strategy.base import Strategy
from atrader.strategy.rules.sma_crossover import SmaCrossoverStrategy, feature_specs_for

app = typer.Typer(help="AI trading agent (spec §1-§13). Paper broker only — see docs/runbook.md.")

_DEFAULT_CONFIG_DIR = Path("config")
_DEFAULT_API_URL = "http://127.0.0.1:8000"
_HTTP_TIMEOUT_SECONDS = 10.0


# ---------------------------------------------------------------------------
# Shared wiring
# ---------------------------------------------------------------------------


def _load_config(config_dir: Path) -> AppConfig:
    try:
        return load_app_config(config_dir)
    except ATraderError as exc:
        typer.echo(f"refusing to boot: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _build_strategy(sc: StrategyConfig) -> tuple[Strategy, tuple[FeatureSpec, ...]]:
    """A rule-based strategy plus the ``FeatureSpec``s it needs.

    Deliberately does not handle ``llm_agent`` — that one needs a live API
    key or a recorded-response file (:mod:`atrader.strategy.llm.client`),
    neither of which this function has access to. ``_build_strategies``
    (the ``run``/``paper`` path) routes it to :func:`_build_llm_strategy`
    instead; ``backtest``/``replay`` do not support it at all yet.
    """
    if sc.strategy_id == "sma_crossover" or sc.strategy_id.startswith("sma_crossover"):
        fast = int(sc.params.get("fast_period", 20))
        slow = int(sc.params.get("slow_period", 50))
        strategy = SmaCrossoverStrategy(
            sc.strategy_id,
            sc.universe,
            fast_period=fast,
            slow_period=slow,
            volume_multiple=Decimal(str(sc.params.get("volume_multiple", "1.8"))),
            min_bars=int(sc.params.get("min_bars", 60)),
        )
        return strategy, feature_specs_for(fast_period=fast, slow_period=slow)

    if sc.strategy_id == "llm_agent" or sc.strategy_id.startswith("llm_agent"):
        raise ATraderError(
            "llm_agent is not supported by backtest/replay in this CLI — it needs a live "
            "API key or a RecordedLLMClient wired up by hand (see atrader.strategy.llm.client)"
        )

    raise ATraderError(
        f"unknown strategy_id {sc.strategy_id!r} in strategies.yaml — no builder registered for it"
    )


def _build_strategies(
    config: AppConfig, *, audit: object, clock: object
) -> tuple[list[Strategy], FeatureEngine]:
    logger = get_logger("cli")
    strategies: list[Strategy] = []
    specs: list[FeatureSpec] = []

    for sc in config.strategies:
        if not sc.enabled:
            continue
        if sc.strategy_id == "llm_agent" or sc.strategy_id.startswith("llm_agent"):
            llm_strategy = _build_llm_strategy(sc, config, audit=audit, clock=clock, logger=logger)
            if llm_strategy is not None:
                strategies.append(llm_strategy)
            continue
        try:
            strategy, strategy_specs = _build_strategy(sc)
        except ATraderError as exc:
            # One misconfigured strategy must not take the whole book down —
            # skip it and keep whatever else is enabled running.
            logger.warning("skipping strategy", strategy_id=sc.strategy_id, reason=str(exc))
            continue
        strategies.append(strategy)
        specs.extend(strategy_specs)

    feature_engine = FeatureEngine(
        store=FeatureStore(), registry=FeatureRegistry(), specs=tuple(specs)
    )
    return strategies, feature_engine


def _build_llm_strategy(
    sc: StrategyConfig, config: AppConfig, *, audit: object, clock: object, logger: object
) -> Strategy | None:
    """The LLM agent needs a live API key; without one it is skipped with a
    WARN rather than crashing the whole runtime on its first decision cycle."""
    secret = ChainSecretProvider(EnvSecretProvider()).try_get("ANTHROPIC_API_KEY")
    if secret is None:
        logger.warning(  # type: ignore[attr-defined]
            "skipping llm_agent: ANTHROPIC_API_KEY is not set", strategy_id=sc.strategy_id
        )
        return None

    from atrader.audit.logger import AuditLogger
    from atrader.core.clock import Clock
    from atrader.strategy.llm.agent import LLMAgentStrategy
    from atrader.strategy.llm.client import AnthropicLLMClient

    assert isinstance(clock, Clock)
    llm_audit = audit if isinstance(audit, AuditLogger) else None
    client = AnthropicLLMClient(
        api_key=secret.reveal(),
        clock=clock,
        enable_refusal_fallback=config.risk.llm.enable_refusal_fallback,
    )
    return LLMAgentStrategy(
        sc.strategy_id, sc.universe, client=client, config=config.risk.llm, audit=llm_audit
    )


# ---------------------------------------------------------------------------
# run / paper
# ---------------------------------------------------------------------------


@app.command()
def run(
    config_dir: Annotated[Path, typer.Option("--config", "-c")] = _DEFAULT_CONFIG_DIR,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    duration: Annotated[
        float | None,
        typer.Option("--duration", help="Bound the run in seconds. Omit to run until killed."),
    ] = None,
    equity_out: Annotated[Path | None, typer.Option("--equity-out")] = None,
) -> None:
    """Start the trading runtime and its HTTP control surface.

    This system's only broker is the paper simulator (a confirmed scope
    decision — see docs/runbook.md), so ``run`` and ``paper`` do the same
    thing; ``paper`` just defaults to a bounded smoke-test duration.

    ``--equity-out`` records one closing equity mark per day, which is the
    live half of a ``divergence-report`` (acceptance criterion #7).
    """
    asyncio.run(_serve(config_dir, host, port, duration, equity_out))


@app.command()
def paper(
    config_dir: Annotated[Path, typer.Option("--config", "-c")] = _DEFAULT_CONFIG_DIR,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    duration: Annotated[float, typer.Option("--duration")] = 60.0,
    equity_out: Annotated[Path | None, typer.Option("--equity-out")] = None,
) -> None:
    """Alias for ``run``, bounded by default (e.g. ``--duration 60s`` style smoke tests)."""
    asyncio.run(_serve(config_dir, host, port, duration, equity_out))


async def _serve(
    config_dir: Path,
    host: str,
    port: int,
    duration: float | None,
    equity_out: Path | None = None,
) -> None:
    import uvicorn

    from atrader.app.api import create_app
    from atrader.app.runtime import Runtime
    from atrader.audit.logger import AuditLogger
    from atrader.storage.memory import InMemoryStorage

    config = _load_config(config_dir)
    configure_logging(level=config.monitoring.log_level)
    logger = get_logger("cli")

    clock = SystemClock()
    storage = InMemoryStorage()
    audit = AuditLogger(storage.audit, clock)
    strategies, feature_engine = _build_strategies(config, audit=audit, clock=clock)
    if not strategies:
        typer.echo("no enabled strategies configured; refusing to run with an idle book", err=True)
        raise typer.Exit(code=1)

    runtime = Runtime(config, strategies, feature_engine, clock=clock, storage=storage)
    api_app = create_app(runtime)
    server = uvicorn.Server(uvicorn.Config(api_app, host=host, port=port, log_level="warning"))

    logger.info("starting", host=host, port=port, strategies=[s.strategy_id for s in strategies])
    server_task = asyncio.create_task(server.serve())
    try:
        if duration is not None:
            await runtime.run_for(duration)
        else:
            await runtime.run_forever()
    finally:
        server.should_exit = True
        await server_task
        # In the `finally` so a session ended by Ctrl-C or the kill switch
        # still leaves its curve behind — a month-long paper run that loses
        # its equity history because it was stopped the usual way would make
        # this option useless for the thing it exists to measure.
        if equity_out is not None:
            curve = runtime.daily_equity_curve()
            dump_equity_curve(curve, equity_out)
            logger.info("equity curve written", path=str(equity_out), points=len(curve))


# ---------------------------------------------------------------------------
# backtest / replay
# ---------------------------------------------------------------------------


def _synthetic_bars(
    symbols: list[str], *, start: date, end: date, interval: str = "1d", seed: int = 0
) -> list[Bar]:
    """Generate a synthetic daily bar series spanning [start, end].

    There is no real historical data source in this vertical slice (see the
    plan's explicit scope decisions) — this is a clearly-labelled stand-in so
    ``backtest``/``replay`` are runnable without external data. Pass
    ``--bars`` with a real recording (spec §8.3) for anything that needs to
    reflect an actual market.
    """
    num_days = max(1, (end - start).days)
    interval_ns = {"1d": 86_400 * NS_PER_SECOND, "1h": 3_600 * NS_PER_SECOND}.get(
        interval, 86_400 * NS_PER_SECOND
    )
    start_ns = (
        int(datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp()) * NS_PER_SECOND
    )
    clock = SimulatedClock(start_ns=start_ns)
    feed = SimulatedFeed.from_symbols(symbols, clock=clock, interval_ns=interval_ns)
    aggregator = BarAggregator(intervals=(interval,))
    bars: list[Bar] = []
    for tick in feed.generate(num_days + 1):
        bars.extend(aggregator.add(tick))
    return bars


def _load_bars(path: Path) -> list[Bar]:
    bars: list[Bar] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                bars.append(Bar.model_validate(json.loads(line)))
    return bars


def _resolve_bars(
    config: AppConfig, symbols: list[str], bars_file: Path | None, from_date: str, to_date: str
) -> list[Bar]:
    if bars_file is not None:
        return _load_bars(bars_file)
    typer.echo(
        f"no --bars file given; generating synthetic data for {from_date}..{to_date} "
        "(not real market history — see docs/runbook.md)"
    )
    return _synthetic_bars(
        symbols, start=date.fromisoformat(from_date), end=date.fromisoformat(to_date)
    )


def _make_engine_factory(config: AppConfig, strategy_id: str):  # type: ignore[no-untyped-def]
    from atrader.risk.engine import RiskEngine

    matches = [sc for sc in config.strategies if sc.strategy_id == strategy_id]
    if not matches:
        raise ATraderError(f"no strategy {strategy_id!r} configured in strategies.yaml")
    sc = matches[0]
    # Validate eagerly (not lazily inside the closure) so an unsupported
    # strategy is a clean CLI error here rather than an exception raised the
    # first time `factory()` runs, deep inside `engine.run(...)`.
    _build_strategy(sc)

    def factory() -> BacktestEngine:
        clock = SimulatedClock(start_ns=0)
        ids = DeterministicIdGenerator(clock, seed=1)
        strategy, specs = _build_strategy(sc)
        risk_engine = RiskEngine(
            config=config, clock=clock, ids=ids, kill_switch=KillSwitch(clock=clock)
        )
        feature_engine = FeatureEngine(
            store=FeatureStore(), registry=FeatureRegistry(), specs=specs
        )
        return BacktestEngine(
            strategies=(strategy,),
            risk_engine=risk_engine,
            clock=clock,
            ids=ids,
            feature_engine=feature_engine,
            starting_cash=config.account_equity,
        )

    return factory


@app.command()
def backtest(
    config_dir: Annotated[Path, typer.Option("--config", "-c")] = _DEFAULT_CONFIG_DIR,
    strategy_id: Annotated[str, typer.Option("--strategy")] = "sma_crossover",
    bars_file: Annotated[Path | None, typer.Option("--bars")] = None,
    from_date: Annotated[str, typer.Option("--from")] = "2024-01-01",
    to_date: Annotated[str, typer.Option("--to")] = "2024-12-31",
    equity_out: Annotated[Path | None, typer.Option("--equity-out")] = None,
) -> None:
    """Run a strategy over historical (or, absent ``--bars``, synthetic) bars.

    ``--equity-out`` records the equity curve for a later
    ``divergence-report`` against a live session (acceptance criterion #7).
    """
    asyncio.run(_run_backtest(config_dir, strategy_id, bars_file, from_date, to_date, equity_out))


async def _run_backtest(
    config_dir: Path,
    strategy_id: str,
    bars_file: Path | None,
    from_date: str,
    to_date: str,
    equity_out: Path | None = None,
) -> None:
    config = _load_config(config_dir)
    try:
        factory = _make_engine_factory(config, strategy_id)
    except ATraderError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    bars = _resolve_bars(config, list(config.universe.symbols), bars_file, from_date, to_date)
    engine = factory()
    result = await engine.run(bars)
    report = performance_report(list(result.equity_curve))

    typer.echo(f"bars processed: {len(bars)}")
    typer.echo(f"orders: {len(result.orders)}  fills: {len(result.fills)}")
    typer.echo(f"rejected intents: {len(result.rejected_intents)}")
    for field_name, value in asdict(report).items():
        typer.echo(f"{field_name}: {value}")

    if equity_out is not None:
        dump_equity_curve(result.equity_curve, equity_out)
        typer.echo(f"equity curve written to {equity_out}")


@app.command()
def replay(
    config_dir: Annotated[Path, typer.Option("--config", "-c")] = _DEFAULT_CONFIG_DIR,
    strategy_id: Annotated[str, typer.Option("--strategy")] = "sma_crossover",
    bars_file: Annotated[Path | None, typer.Option("--bars")] = None,
    from_date: Annotated[str, typer.Option("--from")] = "2024-01-01",
    to_date: Annotated[str, typer.Option("--to")] = "2024-12-31",
) -> None:
    """Run a strategy twice from a fresh engine and byte-compare the order
    sequence — acceptance criterion #3."""
    asyncio.run(_run_replay(config_dir, strategy_id, bars_file, from_date, to_date))


async def _run_replay(
    config_dir: Path, strategy_id: str, bars_file: Path | None, from_date: str, to_date: str
) -> None:
    config = _load_config(config_dir)
    try:
        factory = _make_engine_factory(config, strategy_id)
    except ATraderError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    bars = _resolve_bars(config, list(config.universe.symbols), bars_file, from_date, to_date)
    try:
        result = await assert_replay_matches(factory, bars)
    except AssertionError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(result.summary())


# ---------------------------------------------------------------------------
# kill / reconcile — HTTP clients of a running `run`/`paper` process
# ---------------------------------------------------------------------------


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
            result: dict[str, object] = json.loads(response.read())
            return result
    except urllib.error.URLError as exc:
        typer.echo(f"could not reach {url}: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def kill(
    reason: Annotated[str, typer.Option("--reason")],
    liquidate: Annotated[bool, typer.Option("--liquidate")] = False,
    api_url: Annotated[str, typer.Option("--api-url")] = _DEFAULT_API_URL,
) -> None:
    """Engage the kill switch on a running process — the CLI access path
    spec §FR-MON-03 requires (acceptance criterion #5: within 5 seconds)."""
    result = _post_json(
        f"{api_url}/kill", {"reason": reason, "liquidate": liquidate, "actor": "cli"}
    )
    typer.echo(json.dumps(result, indent=2))


@app.command()
def reconcile(
    inject_break: Annotated[bool, typer.Option("--inject-break")] = False,
    api_url: Annotated[str, typer.Option("--api-url")] = _DEFAULT_API_URL,
) -> None:
    """Trigger an out-of-cycle reconciliation on a running process
    (acceptance criterion #6 with ``--inject-break``)."""
    result = _post_json(f"{api_url}/reconcile", {"inject_break": inject_break})
    typer.echo(json.dumps(result, indent=2))
    raise typer.Exit(code=0 if result.get("clean") else 1)


# ---------------------------------------------------------------------------
# verify-audit
# ---------------------------------------------------------------------------


@app.command(name="verify-audit")
def verify_audit(
    file: Annotated[
        Path | None,
        typer.Option("--file", help="Offline JSONL export. Omit to fetch a running process."),
    ] = None,
    api_url: Annotated[str, typer.Option("--api-url")] = _DEFAULT_API_URL,
) -> None:
    """Verify the audit log's hash chain — acceptance criterion #9."""
    if file is not None:
        records = load_records_jsonl(file)
    else:
        request = urllib.request.Request(f"{api_url}/audit")
        try:
            with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
                records = records_from_json_bytes(response.read())
        except urllib.error.URLError as exc:
            typer.echo(f"could not reach {api_url}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    result = verify_chain(records)
    typer.echo(f"{result.records_checked} record(s) checked; valid={result.valid}")
    if not result.valid:
        typer.echo(f"first bad seq: {result.first_bad_seq}; {result.reason}", err=True)
    raise typer.Exit(code=0 if result.valid else 1)


# ---------------------------------------------------------------------------
# divergence-report
# ---------------------------------------------------------------------------


@app.command(name="divergence-report")
def divergence_report_command(
    backtest_curve: Annotated[
        Path, typer.Option("--backtest", help="Equity curve from `backtest --equity-out`.")
    ],
    live_curve: Annotated[
        Path, typer.Option("--live", help="Equity curve from `run/paper --equity-out`.")
    ],
    tolerance_pct: Annotated[float, typer.Option("--tolerance")] = float(DEFAULT_TOLERANCE_PCT),
) -> None:
    """Measure how far a live session drifted from its backtest.

    This is the instrument for acceptance criterion #7 (30일 페이퍼 트레이딩
    괴리 30% 미만), not the criterion itself: producing the verdict needs a
    real thirty-day paper run to point it at. Exits non-zero when the runs
    diverged past ``--tolerance``, so it can gate a promotion step.
    """
    try:
        backtest_points = load_equity_curve(backtest_curve)
        live_points = load_equity_curve(live_curve)
    except (OSError, ATraderError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    report = divergence_report(
        backtest_points, live_points, tolerance_pct=Decimal(str(tolerance_pct))
    )
    typer.echo(report.summary())
    raise typer.Exit(code=0 if report.within_tolerance else 1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
