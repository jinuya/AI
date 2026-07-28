"""VWAP — slices sized to a historical intraday volume curve (spec §FR-EXE-03).

The curve is a shape, not a forecast — a sequence of weights, one per bucket,
describing what fraction of the day's volume that bucket normally carries (the
U-shaped open/lunch/close pattern most liquid names show). Slicing proportional
to it means the order's participation rate stays roughly constant across the
day instead of spiking whenever the market happens to be quiet.

Deliberately *not* fed live volume. A version that renormalised against actual
volume as the day went would be adaptive and would not produce a schedule that
can be computed once, logged, and byte-compared in a replay test — that
reactive job belongs to :mod:`~atrader.execution.algos.pov` instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from atrader.core.money import ZERO, floor_to_lot
from atrader.execution.algos import Slice, SliceSchedule

__all__ = ["plan_vwap"]


def plan_vwap(
    total_quantity: Decimal,
    *,
    volume_curve: Sequence[Decimal],
    bucket_seconds: float,
    lot_size: Decimal | None = None,
) -> SliceSchedule:
    """Split *total_quantity* proportional to *volume_curve*.

    The curve need not sum to 1 — it is normalised here — but every weight must
    be non-negative, and at least one must be positive. A bucket with zero
    weight gets no slice; that is a real VWAP shape (e.g. no participation during
    a lunch lull), not a lot-rounding artefact.
    """
    if total_quantity <= ZERO:
        raise ValueError(f"total_quantity must be positive, got {total_quantity}")
    if not volume_curve:
        raise ValueError("volume_curve must have at least one bucket")
    if any(w < ZERO for w in volume_curve):
        raise ValueError(f"volume_curve weights must be non-negative, got {volume_curve!r}")
    curve_total = sum(volume_curve, start=ZERO)
    if curve_total <= ZERO:
        raise ValueError("volume_curve must have at least one positive weight")
    if bucket_seconds < 0:
        raise ValueError(f"bucket_seconds must be non-negative, got {bucket_seconds}")

    lot = lot_size if lot_size is not None and lot_size > ZERO else Decimal(1)

    raw = [total_quantity * weight / curve_total for weight in volume_curve]
    floored = [floor_to_lot(q, lot) for q in raw]

    # Largest-remainder method: hand the rounding shortfall to the buckets whose
    # true share was furthest above what flooring gave them. Anything simpler —
    # e.g. dumping it all on the last bucket — would systematically overweight
    # the close relative to the curve that was asked for.
    shortfall = total_quantity - sum(floored, start=ZERO)
    if shortfall > ZERO:
        remainders = sorted(
            (i for i in range(len(raw)) if volume_curve[i] > ZERO),
            key=lambda i: raw[i] - floored[i],
            reverse=True,
        )
        remaining = shortfall
        for index in remainders:
            if remaining <= ZERO:
                break
            add = min(lot, remaining)
            floored[index] += add
            remaining -= add
        if remaining > ZERO and floored:  # pragma: no cover
            # Unreachable in practice: each bucket's own rounding remainder is
            # strictly less than one lot, so summed over the N eligible buckets
            # the aggregate shortfall is strictly less than N * lot — which
            # guarantees the loop above always finishes distributing it. Kept as
            # a backstop so the schedule invariant (slices sum to the total)
            # cannot silently break if that reasoning is ever invalidated by a
            # future change to the distribution loop.
            floored[-1] += remaining

    non_zero = [(i, qty) for i, qty in enumerate(floored) if qty > ZERO]
    if not non_zero:  # pragma: no cover
        # Unreachable given the redistribution above: whenever total_quantity is
        # positive, either some bucket already floored to a positive quantity,
        # or the full shortfall was folded into floored[-1] at line 78, so at
        # least one entry is always positive. Kept as a backstop — one slice
        # for the whole order beats an empty schedule if that invariant is
        # ever broken by a future edit to the redistribution logic above.
        return SliceSchedule(total_quantity, (Slice(0, 0.0, total_quantity),))

    # Re-sequence: dropping zero-quantity buckets above must not leave gaps in
    # the sequence numbers the schedule's invariant requires.
    slices = tuple(
        Slice(seq, bucket_index * bucket_seconds, qty)
        for seq, (bucket_index, qty) in enumerate(non_zero)
    )
    return SliceSchedule(total_quantity, slices)
