"""Net P&L — spec §FR-PF-02.

    실시간 시가평가와 수수료·세금·차입비용·환율을 반영한 순손익을 제공해야 한다.

:mod:`atrader.portfolio.positions` computes *trading* P&L — the pure price
difference a FIFO or average-cost book produces, which is also what a tax lot
reports. Net P&L is a different, portfolio-level number: what the trading P&L
becomes after every real cost of holding and executing is taken out. Keeping
the two separate means neither one lies — the position book still matches the
tax lots, and the net figure still matches the account statement.

Costs handled here:

* **Commission and tax** — summed directly from the fills that produced them.
* **Borrow cost** — accrues only on short positions, priced off
  :attr:`~atrader.config.schema.InstrumentSpec.borrow_bps_annual`. What
  :meth:`BorrowCostModel.daily_cost` returns is a same-day run rate, not an
  accumulated ledger — accumulating it over time is an operational job (run
  once a day, persist the sum), not something a point-in-time position can
  reconstruct on its own.
* **FX** — a position's raw P&L is in the instrument's own currency;
  converting it to the account's base currency mixes in a currency-movement
  component that trading skill had nothing to do with. ``fx_adjustment``
  isolates that component so it can be reported (and, if it turns out to
  matter, hedged) separately from trading performance.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from decimal import Decimal

from atrader.config.schema import InstrumentSpec
from atrader.core.models import Fill, Position
from atrader.core.money import ZERO
from atrader.storage.protocol import PositionStore

__all__ = ["BorrowCostModel", "PnLReport", "mark_to_market", "net_pnl_report"]

DAYS_PER_YEAR = Decimal(365)
BPS = Decimal(10_000)


@dataclass(frozen=True, slots=True)
class BorrowCostModel:
    """Daily accrual for holding a short position (spec §FR-PF-02)."""

    def daily_cost(self, position: Position, spec: InstrumentSpec | None) -> Decimal:
        if not position.is_short or spec is None:
            return ZERO
        annual_rate = spec.borrow_bps_annual / BPS
        return position.exposure * annual_rate / DAYS_PER_YEAR


@dataclass(frozen=True, slots=True)
class PnLReport:
    as_of_ns: int
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    commission: Decimal
    tax: Decimal
    borrow_cost: Decimal
    fx_adjustment: Decimal
    """Portion of ``realized_pnl + unrealized_pnl`` attributable to currency
    movement rather than trading — positive when the base currency weakened
    against a held instrument's currency."""
    per_symbol_pnl: dict[str, Decimal] = field(default_factory=dict)
    """Realized + unrealized, converted to base currency, per symbol."""

    @property
    def gross_trading_pnl(self) -> Decimal:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def net_pnl(self) -> Decimal:
        """What actually hit the account: trading P&L minus every real cost.

        ``fx_adjustment`` is deliberately *not* added. It reports how much of
        ``gross_trading_pnl`` came from the currency rather than the trade —
        and ``realized_pnl``/``unrealized_pnl`` are already converted at that
        rate, so gross contains it. Adding it again inflated the figure by the
        currency component twice over: 100 EUR realized at 1.08 arrives as 108
        USD in the account, and this reported 116.
        """
        return self.gross_trading_pnl - self.commission - self.tax - self.borrow_cost


def mark_to_market(
    store: PositionStore, prices: dict[str, Decimal], *, now_ns: int
) -> list[Position]:
    """Refresh ``last_price``/``unrealized_pnl`` for every open position.

    A symbol with no fresh price keeps its previous mark rather than falling
    back to cost — a stale mark should read as stale (via ``updated_at_ns``
    lagging), not silently agree with the entry price as if nothing changed.
    """
    updated: list[Position] = []
    for position in store.all_positions():
        if position.is_flat:
            continue
        price = prices.get(position.symbol)
        if price is None:
            continue
        unrealized = (price - position.avg_price) * position.quantity
        new_position = position.model_copy(
            update={"last_price": price, "unrealized_pnl": unrealized, "updated_at_ns": now_ns}
        )
        store.upsert(new_position)
        updated.append(new_position)
    return updated


def net_pnl_report(
    positions: Iterable[Position],
    fills: Iterable[Fill],
    *,
    instrument_of: Callable[[str], InstrumentSpec | None],
    as_of_ns: int,
    base_currency: str = "USD",
    fx_rates: dict[str, Decimal] | None = None,
    borrow_model: BorrowCostModel | None = None,
) -> PnLReport:
    """Roll positions and fills up into one portfolio-level P&L figure.

    ``fx_rates`` maps a currency code to units of base currency per unit of
    that currency, e.g. ``{"EUR": Decimal("1.08")}`` when the base is USD.
    A currency absent from the map — the common case, since every sample
    instrument in this system trades in the base currency — is treated as
    parity, which makes ``fx_adjustment`` correctly come out to zero rather
    than silently misprice an untracked currency.
    """
    fx_rates = fx_rates or {}
    borrow_model = borrow_model or BorrowCostModel()

    realized = ZERO
    unrealized = ZERO
    fx_adjustment = ZERO
    borrow_cost = ZERO
    per_symbol: dict[str, Decimal] = {}

    for position in positions:
        spec = instrument_of(position.symbol)
        currency = spec.currency if spec is not None else base_currency
        rate = fx_rates.get(currency, Decimal(1)) if currency != base_currency else Decimal(1)

        local_pnl = position.realized_pnl + position.unrealized_pnl
        per_symbol[position.symbol] = local_pnl * rate
        fx_adjustment += local_pnl * (rate - Decimal(1))

        realized += position.realized_pnl * rate
        unrealized += position.unrealized_pnl * rate
        borrow_cost += borrow_model.daily_cost(position, spec)

    commission = ZERO
    tax = ZERO
    for fill in fills:
        commission += fill.commission
        tax += fill.tax

    return PnLReport(
        as_of_ns=as_of_ns,
        realized_pnl=realized,
        unrealized_pnl=unrealized,
        commission=commission,
        tax=tax,
        borrow_cost=borrow_cost,
        fx_adjustment=fx_adjustment,
        per_symbol_pnl=per_symbol,
    )
