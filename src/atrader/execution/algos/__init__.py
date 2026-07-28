"""Execution algorithms — spec §FR-EXE-03.

    ADV의 1%를 초과하는 주문은 반드시 분할 실행한다.

A single order that is a large share of a name's daily volume moves the price
against itself before it finishes filling — the market sees the order and
reprices ahead of it. Splitting the same quantity into smaller pieces, worked
over time, is strictly about not announcing the whole size at once.

Three ways to slice, in increasing order of how much they react to what the
market is actually doing:

* :mod:`~atrader.execution.algos.twap` — equal pieces, equal time. No market
  data needed, which makes it the fallback when nothing better is available.
* :mod:`~atrader.execution.algos.vwap` — pieces sized to a historical intraday
  volume curve, so participation tracks *when* volume normally trades rather
  than spreading it flat across the clock.
* :mod:`~atrader.execution.algos.pov` — reactive: sized to a fixed share of
  volume as it is actually observed to trade, live. The only one of the three
  that cannot be planned in advance.

TWAP and VWAP produce a :class:`SliceSchedule` up front — a fixed, replayable
plan — which is what makes them checkable in a deterministic-replay test
without a market feed. POV cannot: it is a stream, not a plan, so it is a
stateful executor instead (:class:`~atrader.execution.algos.pov.POVExecutor`).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from atrader.core.money import ZERO

__all__ = ["Slice", "SliceSchedule"]


@dataclass(frozen=True, slots=True)
class Slice:
    """One child order's worth of an algo's schedule."""

    sequence: int
    offset_seconds: float
    """Seconds after the algo starts that this slice should be sent."""
    quantity: Decimal

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError(f"sequence must be non-negative, got {self.sequence}")
        if self.offset_seconds < 0:
            raise ValueError(f"offset_seconds must be non-negative, got {self.offset_seconds}")
        if self.quantity <= ZERO:
            raise ValueError(f"slice quantity must be positive, got {self.quantity}")


@dataclass(frozen=True, slots=True)
class SliceSchedule:
    """An ordered, immutable execution plan.

    The invariant that matters is in :meth:`__post_init__`: the slices must sum
    to exactly the requested total. A schedule that silently drops or invents a
    few shares to rounding is indistinguishable from a bug in the risk-approved
    order size, and nothing downstream would catch it.
    """

    total_quantity: Decimal
    slices: tuple[Slice, ...]

    def __post_init__(self) -> None:
        summed = sum((s.quantity for s in self.slices), start=ZERO)
        if summed != self.total_quantity:
            raise ValueError(
                f"slices sum to {summed}, not the requested total {self.total_quantity} — "
                "an execution schedule must account for every share"
            )
        offsets = [s.offset_seconds for s in self.slices]
        if offsets != sorted(offsets):
            raise ValueError("slices must be in non-decreasing offset order")
        sequences = [s.sequence for s in self.slices]
        if sequences != list(range(len(sequences))):
            raise ValueError("slice sequence numbers must be 0, 1, 2, ... with no gaps")

    @property
    def slice_count(self) -> int:
        return len(self.slices)

    @property
    def is_empty(self) -> bool:
        return not self.slices
