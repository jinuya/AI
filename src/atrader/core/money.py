"""Decimal arithmetic for prices, quantities and money.

Spec §4.4: *"금액은 절대 float로 저장하지 않는다 — ``NUMERIC(20,8)``을 쓴다."*
Floating point error accumulating in an account balance is genuinely painful, so
this module makes the safe path the easy one and the unsafe path explicit.

Every monetary or quantity value in the system is a :class:`decimal.Decimal`
quantised to 8 decimal places, with at most 12 integer digits — exactly what
``NUMERIC(20,8)`` can round-trip.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_HALF_EVEN, ROUND_UP, Decimal, InvalidOperation
from typing import Final

__all__ = [
    "HUNDRED",
    "ONE",
    "QUANTUM",
    "ZERO",
    "D",
    "MoneyError",
    "as_pct",
    "floor_to_lot",
    "from_float",
    "pct_of",
    "quantize",
    "round_to_tick",
    "safe_div",
]

#: Smallest representable increment — matches ``NUMERIC(20,8)``.
QUANTUM: Final[Decimal] = Decimal("0.00000001")
DECIMAL_PLACES: Final[int] = 8
MAX_INTEGER_DIGITS: Final[int] = 12
_MAX_ABS: Final[Decimal] = Decimal(10) ** MAX_INTEGER_DIGITS

ZERO: Final[Decimal] = Decimal(0)
ONE: Final[Decimal] = Decimal(1)
HUNDRED: Final[Decimal] = Decimal(100)


class MoneyError(ValueError):
    """Raised when a value cannot be represented as ``NUMERIC(20,8)``."""


def D(value: Decimal | int | str) -> Decimal:  # noqa: N802 — deliberate short name
    """Build a :class:`Decimal` from an exact source.

    ``float`` is rejected on purpose: ``Decimal(0.1)`` is
    ``0.1000000000000000055511151231257827021181583404541015625``, and silently
    admitting that into a price feed is how rounding drift starts. Use
    :func:`from_float` when the value genuinely came from a float API and you
    have accepted the loss.
    """
    if isinstance(value, float):  # pragma: no cover — guarded by type checker too
        raise MoneyError(
            "float is not an exact source for Decimal; use from_float() if you accept the loss"
        )
    try:
        return Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise MoneyError(f"cannot convert {value!r} to Decimal") from exc


def from_float(value: float) -> Decimal:
    """Convert a float to Decimal via its shortest repr.

    Explicit escape hatch for values arriving from JSON APIs that use floats.
    Goes through ``str`` so ``0.1`` becomes ``Decimal("0.1")``, not the exact
    binary expansion.
    """
    return Decimal(str(value))


def quantize(value: Decimal, *, rounding: str = ROUND_HALF_EVEN) -> Decimal:
    """Snap to 8 decimal places and verify the value fits ``NUMERIC(20,8)``."""
    result = value.quantize(QUANTUM, rounding=rounding)
    if result.copy_abs() >= _MAX_ABS:
        raise MoneyError(
            f"{value} overflows NUMERIC(20,8) (max {MAX_INTEGER_DIGITS} integer digits)"
        )
    return result


def round_to_tick(price: Decimal, tick_size: Decimal, *, side: str | None = None) -> Decimal:
    """Round *price* to a valid multiple of *tick_size*.

    ``side`` biases the rounding so an order never becomes *more* aggressive
    than intended: a BUY limit rounds down, a SELL limit rounds up. Passing
    ``None`` rounds to nearest.
    """
    if tick_size <= ZERO:
        raise MoneyError(f"tick_size must be positive, got {tick_size}")
    if side == "BUY":
        rounding = ROUND_DOWN
    elif side == "SELL":
        rounding = ROUND_UP
    else:
        rounding = ROUND_HALF_EVEN
    ticks = (price / tick_size).quantize(ONE, rounding=rounding)
    return quantize(ticks * tick_size)


def floor_to_lot(quantity: Decimal, lot_size: Decimal) -> Decimal:
    """Round *quantity* **down** to a whole multiple of *lot_size*.

    Always floors: rounding a quantity up could push an order past a risk limit
    that was checked against the pre-rounding value.
    """
    if lot_size <= ZERO:
        raise MoneyError(f"lot_size must be positive, got {lot_size}")
    lots = (quantity / lot_size).to_integral_value(rounding=ROUND_DOWN)
    return quantize(lots * lot_size)


def pct_of(value: Decimal, percent: Decimal) -> Decimal:
    """``percent`` percent of ``value``. ``pct_of(1000, 2)`` -> ``20``."""
    return quantize(value * percent / HUNDRED)


def as_pct(part: Decimal, whole: Decimal) -> Decimal:
    """``part`` expressed as a percentage of ``whole``. Zero whole -> zero."""
    if whole == ZERO:
        return ZERO
    return quantize(part / whole * HUNDRED)


def safe_div(numerator: Decimal, denominator: Decimal, *, default: Decimal = ZERO) -> Decimal:
    """Divide, returning *default* instead of raising when the denominator is zero."""
    if denominator == ZERO:
        return default
    return quantize(numerator / denominator)
