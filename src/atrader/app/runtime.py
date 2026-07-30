"""The runtime — spec §10. Composition root for live/paper trading.

Every safety property this module leans on was built and tested in
isolation in earlier phases: the risk engine is the only path from an intent
to an order (spec §1, §7.1), fills are idempotent (§4.3), reconciliation
never auto-corrects (§FR-EXE-05), the kill switch takes precedence over
everything (§FR-MON-03). This module's job is wiring, plus the handful of
spots where the *order* of wiring is itself a safety property:

* **Startup state recovery (§10.4).** Query the broker, compare to local
  state, and only start accepting new orders if they agree. A break found at
  boot behaves exactly like a break found mid-session — it blocks new orders
  until a human clears it.
* **Bar processing order.** Exactly like
  :class:`~atrader.backtest.engine.BacktestEngine` (spec §2.2: strategy code
  and this ordering are identical in both): resting orders are matched
  against a bar *before* strategies react to it, so a newly-submitted order
  can never fill against the very bar that produced its signal.
* **The circuit breaker is evaluated once per bar**, before any intent for
  that bar is risk-checked, and its verdict is what every risk check in that
  bar sees as ``system_state``/``breaker_level`` — never recomputed
  mid-cycle, for the same reason :class:`~atrader.risk.state.RiskSnapshot` is
  built once per evaluation.

Because this system's only broker is the paper simulator (a confirmed scope
decision — see the plan in the repository root's commit history), "live" and
"paper" are the same runtime; see ``docs/runbook.md``.

**Known simplifications**, each because building it fully is orthogonal to
what P7 (the operational surface) is actually about, not because it is
unimportant — each is a reasonable place to extend this vertical slice:

* Average daily volume is a rolling window over observed bar volume, not a
  true trailing-20-session ADV (no historical warm-up data source exists in
  this slice).
* Cross-symbol correlation is not tracked (:attr:`~atrader.risk.state.RiskSnapshot.correlations`
  stays empty), so the correlated-exposure check never fires. Wiring a real
  correlation estimator is future work.
* Market-hours checking uses the instrument's UTC open/close time-of-day only
  — no holiday calendar.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from dataclasses import dataclass
from datetime import date, time
from decimal import Decimal
from typing import cast
from uuid import UUID

from atrader.audit.logger import AuditEvent, AuditLogger
from atrader.brokers.models import BrokerEvent, BrokerEventType
from atrader.brokers.paper import PaperBroker, PaperBrokerConfig
from atrader.config.schema import AppConfig
from atrader.core.clock import Clock, SystemClock, ns_to_datetime
from atrader.core.ids import IdGenerator, SystemIdGenerator
from atrader.core.models import AccountState, Fill, Position, TradingIntent
from atrader.core.money import ZERO
from atrader.core.rng import Rng, SeededRng
from atrader.core.types import (
    AlertLevel,
    BreakerLevel,
    RiskAction,
    Side,
    SystemState,
    TargetType,
)
from atrader.execution.deadman import DeadManSwitch, TriggerReport
from atrader.execution.oms import OrderManager
from atrader.execution.ratelimit import Priority, RateLimiter
from atrader.execution.reconciliation import Reconciler
from atrader.features.engine import FeatureEngine
from atrader.marketdata.aggregator import BarAggregator
from atrader.marketdata.feeds.protocol import MarketDataFeed
from atrader.marketdata.feeds.simulated import SimulatedFeed
from atrader.marketdata.models import Bar, Tick
from atrader.marketdata.quality import QualityMonitor
from atrader.monitoring.alerts import AlertRouter
from atrader.monitoring.metrics import Metrics
from atrader.portfolio.margin import MarginMonitor, MarginStatus
from atrader.portfolio.netting import net_intents
from atrader.portfolio.positions import PositionBook
from atrader.risk.approval import ApprovalGate
from atrader.risk.circuit_breaker import CircuitBreaker
from atrader.risk.engine import RiskDecision, RiskEngine
from atrader.risk.killswitch import KillSwitch, KillSwitchEvent, KillSwitchSource
from atrader.risk.state import OrderRateTracker, RecentOrder, RiskSnapshot, RiskState
from atrader.storage.memory import InMemoryStorage
from atrader.strategy.base import Strategy, StrategyContext

__all__ = ["Runtime", "RuntimeStatus"]

#: Rolling window (in bars) for the average-daily-volume approximation.
_ADV_WINDOW = 20

_BREAKER_LEVEL_NUM = {
    BreakerLevel.NONE: 0,
    BreakerLevel.L1: 1,
    BreakerLevel.L2: 2,
    BreakerLevel.L3: 3,
}
_MARGIN_STATUS_NUM = {MarginStatus.OK: 0, MarginStatus.WARN: 1, MarginStatus.REDUCE_ONLY: 2}


def _parse_hhmm(value: str) -> time:
    hours, minutes = value.split(":")
    return time(hour=int(hours), minute=int(minutes))


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    """Snapshot for ``/health`` and the CLI."""

    running: bool
    environment: str
    started_at_ns: int | None
    ticks_processed: int
    bars_processed: int
    orders_submitted: int
    kill_switch_engaged: bool
    system_state: str
    breaker_level: str
    reconciliation_clean: bool
    equity: Decimal
    margin_status: str


class Runtime:
    """Composition root: wires every component and drives the tick loop."""

    def __init__(
        self,
        config: AppConfig,
        strategies: list[Strategy],
        feature_engine: FeatureEngine,
        *,
        clock: Clock | None = None,
        ids: IdGenerator | None = None,
        rng: Rng | None = None,
        feed: MarketDataFeed | None = None,
        storage: InMemoryStorage | None = None,
        alerts: AlertRouter | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        self.config = config
        self.clock = clock or SystemClock()
        self.ids = ids or SystemIdGenerator(self.clock)
        self.rng = rng or SeededRng(seed=0)
        self.storage = storage or InMemoryStorage()
        self.audit = AuditLogger(self.storage.audit, self.clock)
        self.metrics = metrics or Metrics()
        self.alerts = alerts or AlertRouter(self.clock)

        self.kill_switch = KillSwitch(clock=self.clock)
        self.approvals = ApprovalGate(
            self.clock, self.ids, timeout_seconds=config.monitoring.approval_timeout_seconds
        )
        self.circuit_breaker = CircuitBreaker(config.risk.circuit_breaker)
        self.risk_engine = RiskEngine(
            config=config,
            clock=self.clock,
            ids=self.ids,
            kill_switch=self.kill_switch,
            approvals=self.approvals,
            audit=self.audit,
        )

        self.broker = PaperBroker(
            self.clock,
            config=PaperBrokerConfig(starting_cash=config.account_equity),
            rng=self.rng,
        )
        self.oms = OrderManager(
            broker=self.broker,
            orders=self.storage.orders,
            fills=self.storage.fills,
            clock=self.clock,
            ids=self.ids,
            audit=self.audit,
        )
        self.reconciler = Reconciler(
            broker=self.broker,
            orders=self.storage.orders,
            positions=self.storage.positions,
            clock=self.clock,
            audit=self.audit,
            interval_seconds=config.execution.reconciliation_interval_seconds,
        )
        self.rate_limiter = RateLimiter(
            clock=self.clock,
            per_minute=config.execution.rate_limit_per_minute,
            burst=config.execution.rate_limit_burst,
            reserve_pct=config.execution.rate_limit_reserve_pct,
            audit=self.audit,
        )
        self.deadman = DeadManSwitch(
            cancel_all=self._cancel_all_for_deadman,
            clock=self.clock,
            audit=self.audit,
            timeout_seconds=config.execution.deadman_timeout_seconds,
            on_trigger=self._on_deadman_trigger,
        )

        self.position_book = PositionBook(
            store=self.storage.positions,
            clock=self.clock,
            method=config.execution.cost_basis_method,
        )
        self.margin_monitor = MarginMonitor(limits=config.risk.account, audit=self.audit)
        self.quality_monitor = QualityMonitor(config.risk.data_quality)

        bar_interval = (
            config.market_data.bar_intervals[0] if config.market_data.bar_intervals else "1m"
        )
        self.aggregator = BarAggregator(intervals=(bar_interval,))
        self.feature_engine = feature_engine
        self.strategies = tuple(strategies)
        self.feed = feed or SimulatedFeed.from_symbols(
            list(config.universe.symbols), clock=self.clock
        )

        self._order_rate = OrderRateTracker()
        self._risk_state = RiskState()
        self._breaker_level = BreakerLevel.NONE
        self._system_state = SystemState.STARTING
        self._prices: dict[str, Decimal] = {}
        self._volume_history: dict[str, deque[Decimal]] = {}
        self._pending_approvals: dict[UUID, TradingIntent] = {}
        self._last_date: date | None = None
        self._last_equity: Decimal = config.account_equity
        #: One closing mark per calendar day, for the backtest-vs-live
        #: comparison acceptance criterion #7 needs (see
        #: :mod:`atrader.backtest.divergence`). Daily rather than per-bar so a
        #: month-long session stays bounded at ~30 entries whatever the bar
        #: interval — and because daily is the granularity the criterion and
        #: ``performance_report``'s 252-periods-per-year default both assume.
        self._daily_equity: list[tuple[int, Decimal]] = []
        self._running = False
        self._started_at_ns: int | None = None
        self._ticks_processed = 0
        self._bars_processed = 0
        self._orders_submitted = 0
        self._deadman_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Boot sequence: on_start hooks, then state recovery (spec §10.4)."""
        if self._started_at_ns is not None:
            raise RuntimeError("Runtime.start() called twice")
        self._started_at_ns = self.clock.now_ns()
        self.audit.append(
            AuditEvent.SYSTEM_STARTED, payload={"environment": self.config.environment}
        )

        account = await self.broker.get_account()
        context = self._context(self.clock.now_ns(), account, {})
        for strategy in self.strategies:
            strategy.on_start(context)

        # §10.4: broker query -> compare to local -> block on break -> only
        # resume if clean. A fresh paper session has nothing to disagree
        # with, but the *procedure* is what matters — this is the same code
        # path a restart against persisted state runs.
        report = await self.reconciler.reconcile()
        self.metrics.reconciliation_last_clean.set(1.0 if report.is_clean else 0.0)
        if report.is_clean:
            self._system_state = SystemState.RUNNING
            self.audit.append(AuditEvent.STATE_RECOVERED, payload={"clean": True})
        else:
            self.alerts.critical(
                component="runtime",
                message="startup reconciliation found a break; refusing to resume trading",
                breaks=[b.message for b in report.breaks],
            )
            self.audit.append(
                AuditEvent.STATE_RECOVERED,
                payload={"clean": False, "breaks": [b.message for b in report.breaks]},
            )
            # system_state stays STARTING: RiskEngine only allows exposure-
            # increasing orders when RUNNING (see module docstring).

        self.deadman.beat(self.clock.now_ns())
        self._running = True

    async def run_forever(self) -> None:
        """Consume the feed until :meth:`stop` is called or it is exhausted."""
        if self._started_at_ns is None:
            await self.start()

        self._deadman_task = asyncio.create_task(self.deadman.run())
        try:
            await self.feed.connect(list(self.config.universe.symbols))
            async for tick in self.feed:
                await self._on_tick(tick)
                # A feed driven by a SimulatedClock never truly suspends (no
                # real I/O, no sleep), so without an explicit yield here this
                # loop could starve the event loop forever — including the
                # timer behind run_for()'s timeout. asyncio.sleep(0) is a
                # real, if minimal, suspension point that guarantees the loop
                # gets a turn every tick.
                await asyncio.sleep(0)
                if not self._running:
                    break
        finally:
            self.deadman.stop()
            task, self._deadman_task = self._deadman_task, None
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def run_for(self, duration_seconds: float) -> None:
        """Run the feed for a bounded duration — the CLI's smoke-test path."""
        try:
            await asyncio.wait_for(self.run_forever(), timeout=duration_seconds)
        except TimeoutError:
            await self.stop(reason=f"run_for({duration_seconds}s) elapsed")

    async def stop(self, *, reason: str) -> None:
        self._running = False
        self.audit.append(AuditEvent.SYSTEM_STOPPED, payload={"reason": reason})

    # ------------------------------------------------------------------
    # Kill switch (spec §FR-MON-03) — takes precedence over everything else.
    # Engaging is synchronous (a boolean flip on KillSwitch itself), so a new
    # order is blocked the instant this call returns; cancellation of what is
    # already working follows in the same coroutine.
    # ------------------------------------------------------------------

    async def kill(
        self,
        *,
        reason: str,
        source: KillSwitchSource = KillSwitchSource.CLI,
        actor: str = "operator",
        liquidate: bool = False,
    ) -> KillSwitchEvent:
        event = self.kill_switch.engage(
            reason=reason, source=source, actor=actor, liquidate=liquidate
        )
        self.metrics.kill_switch_engaged.set(1.0)
        self.audit.append(AuditEvent.KILL_SWITCH_ENGAGED, actor=actor, payload=event.as_payload())
        self.alerts.critical(
            component="killswitch", message=f"kill switch engaged: {reason}", source=source.value
        )
        await self.oms.cancel_all()
        if liquidate:
            await self._liquidate_all()
        return event

    async def release(
        self,
        *,
        reason: str,
        source: KillSwitchSource = KillSwitchSource.CLI,
        actor: str = "operator",
    ) -> KillSwitchEvent:
        event = self.kill_switch.release(reason=reason, source=source, actor=actor)
        self.metrics.kill_switch_engaged.set(0.0)
        self.audit.append(AuditEvent.KILL_SWITCH_RELEASED, actor=actor, payload=event.as_payload())
        self.alerts.info(component="killswitch", message=f"kill switch released: {reason}")
        return event

    def reset_breaker(self, *, actor: str, reason: str) -> None:
        """Operator path for clearing an L2/L3 circuit breaker (spec: 수동해제).

        L2 and L3 hold across bars by design — nothing in the evaluation loop
        may clear them — so without this method the only "manual reset" would
        be restarting the process, which throws away every other piece of
        state along with the breaker.
        """
        decision = self.circuit_breaker.manual_reset(state=self._risk_state)
        self._breaker_level = decision.level
        self._system_state = decision.system_state
        self.metrics.circuit_breaker_level.set(_BREAKER_LEVEL_NUM[self._breaker_level])
        self.audit.append(
            AuditEvent.CIRCUIT_BREAKER_RESET,
            actor=actor,
            payload={"reason": reason, "to_level": decision.level.value},
        )
        self.alerts.warn(
            component="circuit_breaker", message=f"manually reset by {actor}: {reason}"
        )

    async def _liquidate_all(self) -> None:
        positions = {p.symbol: p for p in await self.broker.get_positions()}
        account = await self.broker.get_account()
        at_ns = self.clock.now_ns()
        for symbol, position in positions.items():
            if position.is_flat:
                continue
            intent = TradingIntent(
                intent_id=self.ids.new_id(),
                strategy_id="killswitch.liquidate",
                symbol=symbol,
                side=Side.SELL if position.is_long else Side.BUY,
                target_type=TargetType.SHARES,
                target_value=abs(position.quantity),
                created_at_ns=at_ns,
                rationale="kill switch liquidation",
            )
            await self._evaluate_and_submit(intent, at_ns, account, positions, reduce_only=True)

    # ------------------------------------------------------------------
    # Status / accessors — used by app.api and the CLI.
    # ------------------------------------------------------------------

    def status(self) -> RuntimeStatus:
        return RuntimeStatus(
            running=self._running,
            environment=self.config.environment,
            started_at_ns=self._started_at_ns,
            ticks_processed=self._ticks_processed,
            bars_processed=self._bars_processed,
            orders_submitted=self._orders_submitted,
            kill_switch_engaged=self.kill_switch.is_engaged,
            system_state=self._system_state.value,
            breaker_level=self._breaker_level.value,
            reconciliation_clean=not self.reconciler.has_active_break,
            equity=self._last_equity,
            margin_status=self.margin_monitor.status,
        )

    def positions_snapshot(self) -> list[Position]:
        return [p for p in self.storage.positions.all_positions() if not p.is_flat]

    def pnl_snapshot(self) -> dict[str, Decimal]:
        positions = self.storage.positions.all_positions()
        return {
            "realized": sum((p.realized_pnl for p in positions), ZERO),
            "unrealized": sum((p.unrealized_pnl for p in positions), ZERO),
            "equity": self._last_equity,
        }

    def daily_equity_curve(self) -> tuple[tuple[int, Decimal], ...]:
        """Closing equity per calendar day, oldest first.

        Same ``(at_ns, equity)`` shape as
        :attr:`~atrader.backtest.engine.BacktestResult.equity_curve`, so a live
        session and a backtest can be handed straight to
        :func:`~atrader.backtest.divergence.divergence_report` without either
        side being converted first.
        """
        return tuple(self._daily_equity)

    # ------------------------------------------------------------------
    # Tick / bar processing
    # ------------------------------------------------------------------

    async def _on_tick(self, tick: Tick) -> None:
        self._ticks_processed += 1
        lag_seconds = max(0.0, (tick.ingest_ts - tick.exchange_ts) / 1_000_000_000)
        self.metrics.feed_lag_seconds.observe(lag_seconds)

        report = self.quality_monitor.check(tick)
        self.metrics.data_quality_ok.labels(symbol=tick.symbol).set(1.0 if report.is_ok else 0.0)

        price, size = tick.last, tick.last_size
        if price is not None:
            self._prices[tick.symbol] = price
            self.broker.set_price(tick.symbol, price)
            if size is not None and size > ZERO:
                for event in self.broker.advance_market(tick.symbol, price, size):
                    await self._apply_broker_event(event)

        for bar in self.aggregator.add(tick):
            await self._on_bar(bar)

    async def _apply_broker_event(self, event: BrokerEvent) -> None:
        order = self.oms.apply_event(event)
        if order is not None:
            self.metrics.orders_total.labels(status=order.status.value).inc()
        if event.event_type is not BrokerEventType.FILL or order is None:
            return
        fill = self._find_fill(order.order_id, event.broker_fill_id)
        if fill is None:
            return
        self.position_book.apply_fill(fill)
        account = await self.broker.get_account()
        positions = {p.symbol: p for p in await self.broker.get_positions()}
        context = self._context(self.clock.now_ns(), account, positions)
        for strategy in self.strategies:
            strategy.on_fill(fill, context)

    async def _on_bar(self, bar: Bar) -> None:
        self._bars_processed += 1
        self._record_volume(bar)
        self.deadman.beat(self.clock.now_ns())

        if self.reconciler.is_due(self.clock.now_ns()):
            recon_report = await self.reconciler.reconcile()
            self.metrics.reconciliation_last_clean.set(1.0 if recon_report.is_clean else 0.0)
            if not recon_report.is_clean:
                for one_break in recon_report.breaks:
                    self.metrics.reconciliation_breaks_total.labels(kind=one_break.kind).inc()
                self.alerts.critical(component="reconciliation", message=recon_report.summary())

        self.feature_engine.on_bar(bar)

        account = await self.broker.get_account()
        positions = {p.symbol: p for p in await self.broker.get_positions()}
        self._last_equity = account.equity
        self._roll_day(bar.close_ts, account.equity)
        self.margin_monitor.evaluate(account)
        self.metrics.equity.set(float(account.equity))
        self.metrics.margin_status.set(_MARGIN_STATUS_NUM[self.margin_monitor.status])
        self.metrics.daily_pnl_pct.set(float(self._risk_state.daily_pnl_pct(account.equity)))
        self.metrics.drawdown_pct.set(float(self._risk_state.drawdown_pct(account.equity)))

        self._update_breaker_state(bar.close_ts, account, positions)
        if self._breaker_level is BreakerLevel.L3 and not self.kill_switch.is_engaged:
            await self.kill(
                reason="circuit breaker L3",
                source=KillSwitchSource.AUTOMATIC,
                actor="circuit_breaker",
                liquidate=True,
            )

        await self._process_ready_approvals(bar.close_ts, account, positions)

        context = self._context(bar.close_ts, account, positions)
        intents = [
            intent for strategy in self.strategies for intent in strategy.on_bar(bar, context)
        ]

        # Netting, then the risk gate — mirrors BacktestEngine exactly (spec
        # §2.2: this ordering is part of "strategy code is identical in
        # backtest and live").
        netted, conflicts = net_intents(
            intents,
            positions=positions,
            prices=self._prices,
            equity=account.equity,
            ids=self.ids,
            clock=self.clock,
        )
        if conflicts:
            self.audit.append(
                AuditEvent.INTENTS_NETTED,
                payload={
                    "symbols": [c.symbol for c in conflicts],
                    "contributing_strategies": [
                        list(c.contributing_strategy_ids) for c in conflicts
                    ],
                },
            )
        for intent in netted:
            await self._evaluate_and_submit(intent, bar.close_ts, account, positions)

    async def _evaluate_and_submit(
        self,
        intent: TradingIntent,
        at_ns: int,
        account: AccountState,
        positions: dict[str, Position],
        *,
        reduce_only: bool | None = None,
        approval_request_id: UUID | None = None,
    ) -> RiskDecision:
        snapshot = self._snapshot(at_ns, account, positions)
        effective_reduce_only = (
            self.margin_monitor.reduce_only if reduce_only is None else reduce_only
        )
        decision = self.risk_engine.evaluate(
            intent,
            snapshot,
            reduce_only=effective_reduce_only,
            approval_request_id=approval_request_id,
        )
        self.metrics.risk_decisions_total.labels(action=decision.action.value).inc()
        for result in decision.results:
            self.metrics.risk_checks_total.labels(
                check=result.name, action=result.action.value
            ).inc()

        if (
            decision.action is RiskAction.REQUIRE_APPROVAL
            and decision.approval_request_id is not None
        ):
            self._pending_approvals[decision.approval_request_id] = intent
            return decision
        if not decision.approved or decision.order is None:
            return decision

        # REDUCING outranks new risk but, unlike EMERGENCY, still respects the
        # reserve — EMERGENCY is reserved for cancels (spec §6.3), which this
        # is not: it is a *new* order that happens to shrink exposure.
        priority = Priority.REDUCING if effective_reduce_only else Priority.NORMAL
        await self.rate_limiter.acquire(priority)
        report = await self.oms.submit(decision.order)
        self._orders_submitted += 1
        self.metrics.orders_total.labels(status=report.order.status.value).inc()
        if report.is_live:
            self._order_rate.record(
                RecentOrder(
                    symbol=decision.order.symbol,
                    side=decision.order.side,
                    quantity=decision.order.quantity,
                    at_ns=at_ns,
                )
            )
        return decision

    async def _process_ready_approvals(
        self, at_ns: int, account: AccountState, positions: dict[str, Position]
    ) -> None:
        if not self._pending_approvals:
            return
        self.approvals.sweep()
        for request_id in list(self._pending_approvals):
            request = self.approvals.get(request_id)
            if request is None or request.is_pending:
                continue
            intent = self._pending_approvals.pop(request_id)
            if request.is_granted:
                await self._evaluate_and_submit(
                    intent, at_ns, account, positions, approval_request_id=request_id
                )

    # ------------------------------------------------------------------
    # Snapshot / breaker
    # ------------------------------------------------------------------

    def _update_breaker_state(
        self, at_ns: int, account: AccountState, positions: dict[str, Position]
    ) -> None:
        """Evaluate the circuit breaker exactly once per bar (see module docstring)."""
        prelim = self._snapshot(at_ns, account, positions)
        # evaluate() persists the level and trip time into self._risk_state,
        # which is what makes L2/L3 hold across bars (they clear only via
        # reset_breaker) and lets L1 auto-recover after its cooldown.
        decision = self.circuit_breaker.evaluate(prelim, self._risk_state)

        self._breaker_level = decision.level
        self._system_state = decision.system_state
        self.metrics.circuit_breaker_level.set(_BREAKER_LEVEL_NUM[self._breaker_level])
        if decision.tripped:
            self.alerts.emit(
                decision.alert_level or AlertLevel.WARN,
                component="circuit_breaker",
                message=decision.reason,
                triggers=list(decision.triggers),
            )
            self.audit.append(
                AuditEvent.CIRCUIT_BREAKER_TRIPPED,
                payload={"level": self._breaker_level.value, "reason": decision.reason},
            )

    def _snapshot(
        self, at_ns: int, account: AccountState, positions: dict[str, Position]
    ) -> RiskSnapshot:
        universe = frozenset(self.config.universe.symbols)
        open_orders = tuple(
            RecentOrder(o.symbol, o.side, o.remaining_quantity, o.updated_at_ns)
            for o in self.storage.orders.open_orders()
        )
        adv = {
            symbol: volume
            for symbol in universe
            if (volume := self._average_volume(symbol)) is not None
        }
        return RiskSnapshot(
            now_ns=at_ns,
            system_state=self._system_state,
            breaker_level=self._breaker_level,
            account=account,
            positions=positions,
            universe=universe,
            sectors=dict(self.config.universe.sectors),
            data_quality={s: self.quality_monitor.quality_of(s) for s in universe},
            prices=dict(self._prices),
            adv=adv,
            open_orders=open_orders,
            recent_orders=self._order_rate.recent(at_ns),
            orders_last_minute=self._order_rate.count(at_ns),
            daily_pnl_pct=self._risk_state.daily_pnl_pct(account.equity),
            drawdown_pct=self._risk_state.drawdown_pct(account.equity),
            is_market_open={s: self._is_market_open(s, at_ns) for s in universe},
            kill_switch_engaged=self.kill_switch.is_engaged,
            reconciliation_break=self.reconciler.has_active_break,
            margin_reduce_only=self.margin_monitor.reduce_only,
        )

    def _is_market_open(self, symbol: str, at_ns: int) -> bool:
        instrument = self.config.instrument(symbol)
        if instrument is None:
            return False
        current = ns_to_datetime(at_ns).time()
        return (
            _parse_hhmm(instrument.market_open_utc)
            <= current
            <= _parse_hhmm(instrument.market_close_utc)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _context(
        self, now_ns: int, account: AccountState, positions: dict[str, Position]
    ) -> StrategyContext:
        return StrategyContext(
            now_ns=now_ns,
            account=account,
            positions=positions,
            features=self.feature_engine.store,
            ids=self.ids,
        )

    def _record_volume(self, bar: Bar) -> None:
        history = self._volume_history.setdefault(bar.symbol, deque(maxlen=_ADV_WINDOW))
        history.append(bar.volume)

    def _average_volume(self, symbol: str) -> Decimal | None:
        history = self._volume_history.get(symbol)
        if not history:
            return None
        return sum(history, ZERO) / len(history)

    def _find_fill(self, order_id: UUID, broker_fill_id: str | None) -> Fill | None:
        for fill in self.storage.fills.for_order(order_id):
            if fill.broker_fill_id == broker_fill_id:
                return fill
        return None

    def _roll_day(self, at_ns: int, equity: Decimal) -> None:
        today = ns_to_datetime(at_ns).date()
        if self._last_date is None or today != self._last_date:
            self._last_date = today
            self._risk_state.start_new_day(equity)
            self._daily_equity.append((at_ns, equity))
        else:
            self._risk_state.observe_equity(equity)
            # Overwrite rather than append: the day's entry should be its
            # latest mark, so when the day ends the curve holds its close.
            self._daily_equity[-1] = (at_ns, equity)

    async def _cancel_all_for_deadman(self) -> list[object]:
        canceled = await self.oms.cancel_all()
        return cast("list[object]", canceled)

    def _on_deadman_trigger(self, report: TriggerReport) -> None:
        self.metrics.deadman_triggers_total.inc()
        self.alerts.critical(component="deadman", message=report.summary())
