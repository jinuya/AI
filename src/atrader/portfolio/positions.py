"""Position tracking — spec §FR-PF-01.

    비용 기준(FIFO 또는 평균단가)은 설정 가능해야 하고, 세무 보고와 일치해야 한다.

:class:`PositionBook` turns a stream of fills into the running
:class:`~atrader.core.models.Position` per symbol. Two cost-basis methods are
supported because the spec requires the choice to match tax reporting, not
trading convenience — a book kept on average cost cannot be reconciled against
a 1099 that requires FIFO lots, and vice versa.

**Realized P&L here is pure trading P&L**: exit price minus entry price, times
the quantity closed. Commission, tax and borrow cost are deliberately *not*
folded in at this layer — see :mod:`atrader.portfolio.pnl` for why they are a
separate, portfolio-level rollup instead. A FIFO lot's cost basis is the
execution price; smuggling a fee into that number would make this module's
output disagree with the tax lots it exists to mirror.

FIFO keeps an explicit queue of lots and consumes the oldest first, which is
the only reason it needs any state beyond what :class:`Position` already
stores. Average cost needs none — the running quantity and average price live
on the :class:`Position` record itself, so re-deriving them from the store is
enough to make the next fill's arithmetic correct.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal

from atrader.core.clock import Clock
from atrader.core.models import Fill, Position
from atrader.core.money import ZERO
from atrader.core.types import CostBasisMethod, Side
from atrader.storage.protocol import PositionStore

__all__ = ["FillApplication", "PositionBook"]


@dataclass(frozen=True, slots=True)
class _Lot:
    quantity: Decimal
    """Positive magnitude — direction is tracked separately per symbol."""
    price: Decimal
    opened_at_ns: int


@dataclass(frozen=True, slots=True)
class FillApplication:
    """What one fill did. Returned so callers can log or report without
    re-deriving the delta from a before/after position diff."""

    position: Position
    realized_pnl_delta: Decimal
    closed_quantity: Decimal


@dataclass
class PositionBook:
    """Applies fills to positions using the configured cost-basis method."""

    store: PositionStore
    clock: Clock
    method: CostBasisMethod = CostBasisMethod.FIFO
    _lots: dict[str, deque[_Lot]] = field(default_factory=dict)
    _lot_side: dict[str, Side] = field(default_factory=dict)

    def apply_fill(self, fill: Fill) -> FillApplication:
        position = self.store.get(fill.symbol) or Position(symbol=fill.symbol)

        if self.method is CostBasisMethod.FIFO:
            signed_qty, avg_price, realized, closed = self._apply_fifo(position, fill)
        else:
            signed_qty, avg_price, realized, closed = self._apply_average(position, fill)

        now = fill.executed_at_ns or self.clock.now_ns()
        was_flat = position.is_flat
        now_flat = signed_qty == ZERO
        if was_flat and not now_flat:
            opened_at_ns: int | None = now
        elif now_flat:
            opened_at_ns = None
        else:
            opened_at_ns = position.opened_at_ns

        updated = position.model_copy(
            update={
                "quantity": signed_qty,
                "avg_price": avg_price,
                "realized_pnl": position.realized_pnl + realized,
                "opened_at_ns": opened_at_ns,
                "updated_at_ns": now,
            }
        )
        self.store.upsert(updated)
        return FillApplication(
            position=updated, realized_pnl_delta=realized, closed_quantity=closed
        )

    def rebuild(self, fills: Iterable[Fill]) -> None:
        """Replay fill history from scratch — startup recovery (spec §10.4).

        Fills must be passed in execution order: cost basis is path-dependent,
        so applying them out of order produces a different (wrong) lot
        structure even though the final signed quantity would come out the same.
        """
        self._lots.clear()
        self._lot_side.clear()
        for fill in fills:
            self.apply_fill(fill)

    # ------------------------------------------------------------------

    def _apply_fifo(
        self, position: Position, fill: Fill
    ) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        symbol = fill.symbol
        lots = self._lots.setdefault(symbol, deque())
        current_side = self._lot_side.get(symbol)
        realized = ZERO
        closed = ZERO
        remaining = fill.quantity

        if current_side is not None and fill.side is current_side.opposite:
            while remaining > ZERO and lots:
                lot = lots[0]
                take = min(lot.quantity, remaining)
                realized += current_side.sign * (fill.price - lot.price) * take
                closed += take
                remaining -= take
                if take >= lot.quantity:
                    lots.popleft()
                else:
                    lots[0] = _Lot(lot.quantity - take, lot.price, lot.opened_at_ns)
            if not lots:
                self._lot_side.pop(symbol, None)

        if remaining > ZERO:
            lots.append(_Lot(remaining, fill.price, fill.executed_at_ns))
            self._lot_side[symbol] = fill.side

        if lots:
            total_qty = sum((lot.quantity for lot in lots), ZERO)
            avg_price = sum((lot.quantity * lot.price for lot in lots), ZERO) / total_qty
            signed_qty = total_qty * self._lot_side[symbol].sign
        else:
            self._lots.pop(symbol, None)
            signed_qty, avg_price = ZERO, ZERO

        return signed_qty, avg_price, realized, closed

    def _apply_average(
        self, position: Position, fill: Fill
    ) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        old_qty = position.quantity
        old_avg_price = position.avg_price
        current_side = Side.BUY if old_qty > ZERO else (Side.SELL if old_qty < ZERO else None)
        realized = ZERO
        closed = ZERO
        remaining = fill.quantity

        if current_side is not None and fill.side is current_side.opposite:
            closed = min(abs(old_qty), remaining)
            realized = current_side.sign * (fill.price - old_avg_price) * closed
            remaining -= closed
            old_qty = current_side.sign * (abs(old_qty) - closed)

        if remaining > ZERO:
            if old_qty == ZERO:
                return fill.side.sign * remaining, fill.price, realized, closed
            existing_qty = abs(old_qty)
            new_qty_abs = existing_qty + remaining
            new_avg_price = (existing_qty * old_avg_price + remaining * fill.price) / new_qty_abs
            return old_qty + fill.side.sign * remaining, new_avg_price, realized, closed

        if old_qty == ZERO:
            return ZERO, ZERO, realized, closed
        return old_qty, old_avg_price, realized, closed
