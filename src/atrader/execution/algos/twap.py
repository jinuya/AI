"""TWAP — equal slices, equal spacing (spec §FR-EXE-03).

The fallback algorithm: it needs nothing but a quantity and a duration, so it
is always available even when there is no volume curve or live tape to react
to. What it is not is smart — it does not know that most of a day's volume
trades in the first and last hour, and will happily send the same size into a
dead lunch session as into the open. Use :mod:`~atrader.execution.algos.vwap`
when a volume curve is available; reach for this when one is not.
"""

from __future__ import annotations

from decimal import Decimal

from atrader.core.money import ZERO, floor_to_lot
from atrader.execution.algos import Slice, SliceSchedule

__all__ = ["plan_twap"]


def plan_twap(
    total_quantity: Decimal,
    *,
    duration_seconds: float,
    slice_count: int,
    lot_size: Decimal | None = None,
) -> SliceSchedule:
    """Split *total_quantity* into *slice_count* equal pieces over the window.

    Remainder handling matters more than it looks: dividing 1000 shares into 3
    slices of 333.33 each loses a third of a share to truncation unless the
    leftover is folded back in. It goes onto the *last* slice rather than the
    first — the closing slices of a TWAP already tend to carry more urgency (a
    partial fill running out of clock has nowhere left to go), so this does not
    change the algorithm's risk profile, just where the rounding remainder lands.
    """
    if total_quantity <= ZERO:
        raise ValueError(f"total_quantity must be positive, got {total_quantity}")
    if slice_count <= 0:
        raise ValueError(f"slice_count must be positive, got {slice_count}")
    if duration_seconds < 0:
        raise ValueError(f"duration_seconds must be non-negative, got {duration_seconds}")

    lot = lot_size if lot_size is not None and lot_size > ZERO else Decimal(1)
    base = floor_to_lot(total_quantity / Decimal(slice_count), lot)
    if base <= ZERO:
        # The order is too small to spread over this many slices without a slice
        # rounding to zero. One slice for the whole thing beats a schedule with
        # empty entries.
        return SliceSchedule(total_quantity, (Slice(0, 0.0, total_quantity),))

    spacing = duration_seconds / slice_count if slice_count > 1 else 0.0
    slices = [Slice(i, i * spacing, base) for i in range(slice_count)]

    remainder = total_quantity - base * slice_count
    if remainder > ZERO:
        last = slices[-1]
        slices[-1] = Slice(last.sequence, last.offset_seconds, last.quantity + remainder)

    return SliceSchedule(total_quantity, tuple(slices))
