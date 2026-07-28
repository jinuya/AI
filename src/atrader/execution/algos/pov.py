"""POV — percentage of volume, reactive to the live tape (spec §FR-EXE-03).

TWAP and VWAP commit to a schedule before the first share trades. POV commits
to nothing but a rate: for every unit of volume the market actually prints,
send at most *participation_rate_pct* of it as a child order, until the parent
quantity is exhausted. There is no plan to log and byte-compare in a replay
test, because there is no plan — only a rule applied to whatever the tape
does, which is the point: an order sized against a curve trades too fast on a
quiet day and too slow on a busy one, and POV cannot make either mistake
because it only ever reacts to the volume actually seen.

That reactivity is why this is a class with state rather than a pure function
like :func:`~atrader.execution.algos.twap.plan_twap`. Determinism is preserved
a different way: the same sequence of ``on_market_volume`` calls always
produces the same sequence of slice quantities, so a replay that feeds back
the recorded tape reproduces the same child orders without needing the
schedule itself to be precomputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from atrader.core.money import ZERO, floor_to_lot

__all__ = ["POVExecutor"]


@dataclass
class POVExecutor:
    """Tracks one parent order's worth of participation-of-volume execution."""

    total_quantity: Decimal
    participation_rate_pct: Decimal
    lot_size: Decimal = Decimal(1)
    min_slice_quantity: Decimal = Decimal(0)
    """Below this, an eligible slice is held rather than sent — sending single
    shares on every tick would be indistinguishable from an order-rate anomaly
    to the circuit breaker (spec §7.5), and would just spam the venue."""
    max_slice_quantity: Decimal | None = None
    """Caps a single burst of volume from releasing the whole remaining order
    at once, which defeats the purpose of participating gradually."""

    _sent: Decimal = field(default=ZERO, init=False)
    _entitled_unsent: Decimal = field(default=ZERO, init=False)
    """Participation earned from volume seen but not yet released as a slice —
    carried forward rather than dropped, so quiet ticks are not simply lost."""
    _volume_seen: Decimal = field(default=ZERO, init=False)

    def __post_init__(self) -> None:
        if self.total_quantity <= ZERO:
            raise ValueError(f"total_quantity must be positive, got {self.total_quantity}")
        if not ZERO < self.participation_rate_pct <= 100:
            raise ValueError(
                f"participation_rate_pct must be within (0, 100], got {self.participation_rate_pct}"
            )
        if self.lot_size <= ZERO:
            raise ValueError(f"lot_size must be positive, got {self.lot_size}")
        if self.min_slice_quantity < ZERO:
            raise ValueError(
                f"min_slice_quantity must be non-negative, got {self.min_slice_quantity}"
            )
        if self.max_slice_quantity is not None and self.max_slice_quantity <= ZERO:
            raise ValueError(f"max_slice_quantity must be positive, got {self.max_slice_quantity}")

    @property
    def remaining_quantity(self) -> Decimal:
        return self.total_quantity - self._sent

    @property
    def sent_quantity(self) -> Decimal:
        return self._sent

    @property
    def volume_seen(self) -> Decimal:
        return self._volume_seen

    @property
    def is_complete(self) -> bool:
        return self._sent >= self.total_quantity

    def on_market_volume(self, traded_quantity: Decimal) -> Decimal:
        """Feed one observation of market volume; get back a slice to send.

        Returns ``ZERO`` when nothing crosses the minimum-slice threshold yet —
        that is the normal case between prints, not an error. The unspent
        entitlement is retained, so a string of tiny prints eventually adds up
        to a real slice instead of being discarded tick by tick.
        """
        if traded_quantity < ZERO:
            raise ValueError(f"traded_quantity must be non-negative, got {traded_quantity}")
        if self.is_complete:
            return ZERO

        self._volume_seen += traded_quantity
        self._entitled_unsent += traded_quantity * self.participation_rate_pct / Decimal(100)
        self._entitled_unsent = min(self._entitled_unsent, self.remaining_quantity)

        candidate = floor_to_lot(self._entitled_unsent, self.lot_size)
        candidate = min(candidate, self.remaining_quantity)
        if self.max_slice_quantity is not None:
            candidate = min(candidate, self.max_slice_quantity)

        if candidate < self.min_slice_quantity and candidate < self.remaining_quantity:
            return ZERO

        if candidate <= ZERO:
            return ZERO

        self._entitled_unsent -= candidate
        self._sent += candidate
        return candidate

    def finish_remaining(self) -> Decimal:
        """Release whatever is left, bypassing the participation rate.

        Called when the algo's time budget runs out — spec §FR-EXE-03 requires
        the split, not that the order never completes. An unfinished POV order
        sitting open past its deadline is worse than finishing it directly.
        """
        remainder = self.remaining_quantity
        if remainder <= ZERO:
            return ZERO
        self._sent = self.total_quantity
        self._entitled_unsent = ZERO
        return remainder
