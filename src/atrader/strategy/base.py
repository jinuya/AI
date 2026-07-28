"""The strategy interface — spec §FR-STR-01.

    전략은 시장 데이터를 받아 TradingIntent를 내보내는 것 외에 아무것도 하지
    않는다. 주문을 만들지 않고, 브로커를 호출하지 않고, 리스크 한도를 알지
    못한다.

That restriction is enforced twice: at design time by :class:`Strategy`'s
signature (it can only return :class:`~atrader.core.models.TradingIntent`
objects, never an order), and structurally by the import-linter contracts in
``pyproject.toml`` (``atrader.strategy`` cannot import ``atrader.brokers``,
``atrader.execution``, or ``atrader.risk`` at all — the central "the model
never touches the broker" rule from spec §1 made mechanically unbypassable
rather than merely documented).

:class:`StrategyContext` is the strategy's entire view of the world: current
positions, account state, and point-in-time features. No broker handle, no
risk config, nothing the strategy could use to act rather than merely decide.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

from atrader.core.models import AccountState, Fill, Position, TradingIntent
from atrader.features.store import FeatureValue
from atrader.marketdata.models import Bar

__all__ = ["FeatureView", "Strategy", "StrategyContext"]


class FeatureView(Protocol):
    """The read-only slice of :class:`~atrader.features.store.FeatureStore` a
    strategy is allowed to see — a narrower Protocol so a strategy only ever
    has ``asof`` (point-in-time) available, never ``latest`` (which bypasses
    the point-in-time guarantee)."""

    def asof(self, symbol: str, name: str, at_ns: int) -> FeatureValue | None: ...


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Everything a strategy may look at while deciding what it wants."""

    now_ns: int
    account: AccountState
    positions: dict[str, Position]
    features: FeatureView

    def position_of(self, symbol: str) -> Position:
        return self.positions.get(symbol) or Position(symbol=symbol)

    def feature(self, symbol: str, name: str) -> Decimal | None:
        """Point-in-time lookup: only ever what was knowable by ``now_ns``."""
        value = self.features.asof(symbol, name, self.now_ns)
        return None if value is None else value.value


class Strategy(ABC):
    """Base class for every trading strategy — rule-based or LLM-driven alike.

    Both this system's reference strategy and the LLM agent strategy subclass
    this, and the backtest engine and the live runtime drive both through the
    exact same three methods (spec §2.2: strategy code is identical in
    backtest and live trading). Only ``on_bar`` is required; the other two
    default to doing nothing, since a stateless strategy has no fills to react
    to and no state worth persisting.
    """

    def __init__(self, strategy_id: str) -> None:
        self.strategy_id = strategy_id

    def on_start(self, context: StrategyContext) -> None:  # noqa: B027 — optional hook, not abstract
        """Called once before the first bar. Override to warm up state."""

    @abstractmethod
    def on_bar(self, bar: Bar, context: StrategyContext) -> list[TradingIntent]:
        """React to one closed bar. Return zero or more intents.

        ``bar`` is always final (spec §5.2) — the caller is responsible for
        never invoking this on a still-forming bar, the same way
        :meth:`~atrader.features.engine.FeatureEngine.on_bar` refuses one.
        """
        ...

    def on_fill(self, fill: Fill, context: StrategyContext) -> None:  # noqa: B027
        """React to an execution. Override for strategies that track their
        own fills (e.g. to manage a trailing stop). Default: no-op."""

    def snapshot(self) -> dict[str, Any]:
        """Serializable internal state, for logging and crash recovery.

        Default is empty — a strategy with no state beyond what
        ``StrategyContext`` already provides has nothing to snapshot.
        """
        return {}
