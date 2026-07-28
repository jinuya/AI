"""The backtest engine — spec §2.2, §8.

    전략 코드는 백테스트와 실거래에서 문자 그대로 동일하다. 차이는 데이터
    소스와 실행 레이어뿐이다.

This is the one place that assembles the whole pipeline the way a live
runtime eventually will: strategies emit intents,
:func:`~atrader.portfolio.netting.net_intents` collapses same-symbol intents
from different strategies into one, the
:class:`~atrader.risk.engine.RiskEngine` is the only thing that turns an
intent into an order, and :class:`~atrader.execution.oms.OrderManager` is the
only thing that sends it anywhere. The backtest supplies two things the live
system supplies differently — historical bars instead of a live feed, and
:class:`~atrader.backtest.broker.BacktestBroker` instead of a venue — and
nothing else changes.

**Bar ordering within one cycle is the whole look-ahead guarantee.** For each
bar: resting orders are matched against it *first* (they were placed after
some earlier bar, so this is the earliest they could fill), *then* features
and strategies react to this bar's close, *then* whatever they decide is
submitted. A submitted order is never matched against the bar that produced
the signal — only :meth:`~atrader.backtest.broker.BacktestBroker.advance_bar`
on a *later* bar can fill it. Getting this ordering backwards is the single
most common way a backtest cheats.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from uuid import UUID

from atrader.audit.logger import AuditLogger
from atrader.backtest.broker import BacktestBroker
from atrader.backtest.cost_model import CostModel
from atrader.backtest.fill_model import ConservativeFillModel
from atrader.brokers.models import BrokerEventType
from atrader.config.schema import AccountLimits, InstrumentSpec
from atrader.core.clock import SimulatedClock, ns_to_datetime
from atrader.core.ids import IdGenerator
from atrader.core.models import AccountState, Fill, Order, Position, TradingIntent
from atrader.core.money import ZERO
from atrader.core.types import BreakerLevel, DataQuality, SystemState
from atrader.execution.oms import OrderManager
from atrader.features.engine import FeatureEngine
from atrader.features.store import FeatureStore
from atrader.marketdata.models import Bar
from atrader.portfolio.margin import MarginMonitor
from atrader.portfolio.netting import net_intents
from atrader.portfolio.positions import PositionBook
from atrader.risk.engine import RiskEngine
from atrader.risk.state import OrderRateTracker, RecentOrder, RiskSnapshot, RiskState
from atrader.storage.memory import (
    InMemoryFillStore,
    InMemoryOrderStore,
    InMemoryPositionStore,
)
from atrader.strategy.base import Strategy, StrategyContext

__all__ = ["BacktestEngine", "BacktestResult"]


@dataclass(frozen=True, slots=True)
class BacktestResult:
    orders: tuple[Order, ...]
    fills: tuple[Fill, ...]
    equity_curve: tuple[tuple[int, Decimal], ...]
    """``(at_ns, equity)`` after every bar."""
    final_positions: dict[str, Position]
    rejected_intents: tuple[tuple[TradingIntent, str], ...]
    """Every intent the risk engine refused, and why — the first place to
    look when a strategy trades far less in the backtest than expected."""


def _no_instrument(symbol: str) -> InstrumentSpec | None:
    return None


@dataclass
class BacktestEngine:
    """Drives strategies over historical bars through the live risk/execution
    pipeline, unmodified."""

    strategies: tuple[Strategy, ...]
    risk_engine: RiskEngine
    clock: SimulatedClock
    ids: IdGenerator
    feature_engine: FeatureEngine
    starting_cash: Decimal = Decimal("100000")
    fill_model: ConservativeFillModel = field(default_factory=ConservativeFillModel)
    cost_model: CostModel = field(default_factory=CostModel)
    instrument_of: Callable[[str], InstrumentSpec | None] = field(default=_no_instrument)
    adv_window: int = 20
    deadband_pct: Decimal = ZERO
    margin_limits: AccountLimits | None = None
    audit: AuditLogger | None = None

    orders_store: InMemoryOrderStore = field(default_factory=InMemoryOrderStore)
    fills_store: InMemoryFillStore = field(default_factory=InMemoryFillStore)
    position_book_store: InMemoryPositionStore = field(default_factory=InMemoryPositionStore)

    _broker: BacktestBroker = field(init=False)
    _oms: OrderManager = field(init=False)
    _position_book: PositionBook = field(init=False)
    _order_rate: OrderRateTracker = field(init=False)
    _risk_state: RiskState = field(init=False)
    _margin_monitor: MarginMonitor | None = field(init=False)
    _volume_history: dict[str, deque[Decimal]] = field(default_factory=dict, init=False)
    _prices: dict[str, Decimal] = field(default_factory=dict, init=False)
    _equity_curve: list[tuple[int, Decimal]] = field(default_factory=list, init=False)
    _rejections: list[tuple[TradingIntent, str]] = field(default_factory=list, init=False)
    _last_date: date | None = field(default=None, init=False)
    _started: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._broker = BacktestBroker(
            clock=self.clock,
            fill_model=self.fill_model,
            cost_model=self.cost_model,
            instrument_of=self.instrument_of,
            adv_of=self._average_volume,
            starting_cash=self.starting_cash,
        )
        self._oms = OrderManager(
            broker=self._broker,
            orders=self.orders_store,
            fills=self.fills_store,
            clock=self.clock,
            ids=self.ids,
            audit=self.audit,
        )
        self._position_book = PositionBook(
            store=self.position_book_store,
            clock=self.clock,
            method=self.risk_engine.config.execution.cost_basis_method,
        )
        self._order_rate = OrderRateTracker()
        self._risk_state = RiskState(
            starting_equity=self.starting_cash, peak_equity=self.starting_cash
        )
        self._margin_monitor = (
            MarginMonitor(limits=self.margin_limits, audit=self.audit)
            if self.margin_limits is not None
            else None
        )

    @property
    def feature_store(self) -> FeatureStore:
        return self.feature_engine.store

    # ------------------------------------------------------------------

    async def run(self, bars: Iterable[Bar]) -> BacktestResult:
        if not self._started:
            self._started = True
            account = await self._broker.get_account()
            context = self._context(self.clock.now_ns(), account, {})
            for strategy in self.strategies:
                strategy.on_start(context)

        for bar in bars:
            await self._process_bar(bar)

        positions = {p.symbol: p for p in self.position_book_store.all_positions()}
        return BacktestResult(
            orders=tuple(self.orders_store.all_orders()),
            fills=tuple(self.fills_store.all_fills()),
            equity_curve=tuple(self._equity_curve),
            final_positions=positions,
            rejected_intents=tuple(self._rejections),
        )

    async def _process_bar(self, bar: Bar) -> None:
        if not bar.is_final:
            raise ValueError(
                f"BacktestEngine only accepts closed bars, got a forming bar for {bar.symbol}"
            )
        self.clock.set_to(bar.close_ts)
        self._prices[bar.symbol] = bar.close
        self._record_volume(bar)

        # 1. Resting orders from earlier bars are matched against THIS bar
        # first — the earliest point they could plausibly have filled.
        for event in self._broker.advance_bar(bar):
            order = self._oms.apply_event(event)
            if event.event_type is BrokerEventType.FILL and order is not None:
                fill = self._find_fill(order.order_id, event.broker_fill_id)
                if fill is not None:
                    self._position_book.apply_fill(fill)
                    account = await self._broker.get_account()
                    positions = {p.symbol: p for p in await self._broker.get_positions()}
                    context = self._context(bar.close_ts, account, positions)
                    for strategy in self.strategies:
                        strategy.on_fill(fill, context)

        # 2. Features and strategies react to this bar closing.
        self.feature_engine.on_bar(bar)
        account = await self._broker.get_account()
        positions = {p.symbol: p for p in await self._broker.get_positions()}
        self._roll_day(bar.close_ts, account.equity)
        if self._margin_monitor is not None:
            self._margin_monitor.evaluate(account)

        context = self._context(bar.close_ts, account, positions)
        intents = [
            intent for strategy in self.strategies for intent in strategy.on_bar(bar, context)
        ]

        # 3. Netting, then the risk gate — never more than one intent per
        # symbol reaches it, since this cycle only ever concerns one symbol.
        netted, _conflicts = net_intents(
            intents,
            positions=positions,
            prices=self._prices,
            equity=account.equity,
            ids=self.ids,
            clock=self.clock,
            deadband_pct=self.deadband_pct,
        )
        for intent in netted:
            await self._evaluate_and_submit(intent, bar.close_ts, account, positions)

        self._equity_curve.append((bar.close_ts, account.equity))

    async def _evaluate_and_submit(
        self,
        intent: TradingIntent,
        at_ns: int,
        account: AccountState,
        positions: dict[str, Position],
    ) -> None:
        snapshot = self._build_snapshot(at_ns, account, positions)
        decision = self.risk_engine.evaluate(intent, snapshot)
        if not decision.approved or decision.order is None:
            self._rejections.append((intent, decision.reason))
            return

        report = await self._oms.submit(decision.order)
        if report.is_live:
            self._order_rate.record(
                RecentOrder(
                    symbol=decision.order.symbol,
                    side=decision.order.side,
                    quantity=decision.order.quantity,
                    at_ns=at_ns,
                )
            )

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

    def _build_snapshot(
        self, at_ns: int, account: AccountState, positions: dict[str, Position]
    ) -> RiskSnapshot:
        universe = frozenset(self.risk_engine.config.universe.symbols)
        open_orders = tuple(
            RecentOrder(o.symbol, o.side, o.remaining_quantity, o.updated_at_ns)
            for o in self.orders_store.open_orders()
        )
        adv = {
            symbol: volume
            for symbol in universe
            if (volume := self._average_volume(symbol)) is not None
        }
        return RiskSnapshot(
            now_ns=at_ns,
            system_state=SystemState.RUNNING,
            breaker_level=BreakerLevel.NONE,
            account=account,
            positions=positions,
            universe=universe,
            sectors=dict(self.risk_engine.config.universe.sectors),
            data_quality=dict.fromkeys(universe, DataQuality.OK),
            prices=dict(self._prices),
            adv=adv,
            open_orders=open_orders,
            recent_orders=self._order_rate.recent(at_ns),
            orders_last_minute=self._order_rate.count(at_ns),
            daily_pnl_pct=self._risk_state.daily_pnl_pct(account.equity),
            drawdown_pct=self._risk_state.drawdown_pct(account.equity),
            is_market_open=dict.fromkeys(universe, True),
            kill_switch_engaged=False,
            reconciliation_break=False,
            margin_reduce_only=self._margin_monitor.reduce_only if self._margin_monitor else False,
        )

    def _record_volume(self, bar: Bar) -> None:
        history = self._volume_history.setdefault(bar.symbol, deque(maxlen=self.adv_window))
        history.append(bar.volume)

    def _average_volume(self, symbol: str) -> Decimal | None:
        history = self._volume_history.get(symbol)
        if not history:
            return None
        return sum(history, ZERO) / len(history)

    def _find_fill(self, order_id: UUID, broker_fill_id: str | None) -> Fill | None:
        for fill in self.fills_store.for_order(order_id):
            if fill.broker_fill_id == broker_fill_id:
                return fill
        return None

    def _roll_day(self, at_ns: int, equity: Decimal) -> None:
        """Reset the daily loss/drawdown-baseline counters on a UTC date change.

        Peak equity deliberately is not reset (see
        :meth:`~atrader.risk.state.RiskState.start_new_day`) — drawdown is
        measured from the all-time high, not from this morning.
        """
        today = ns_to_datetime(at_ns).date()
        if self._last_date is None:
            self._last_date = today
            self._risk_state.start_new_day(equity)
            return
        if today != self._last_date:
            self._last_date = today
            self._risk_state.start_new_day(equity)
        else:
            self._risk_state.observe_equity(equity)
