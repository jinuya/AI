"""Identifier generation.

Spec §FR-EXE-04 requires every order to carry a client-generated
``client_order_id`` so a retry after a lost response cannot produce a duplicate
fill. UUIDv7 is used because it is time-ordered, which makes the ``orders``
table index well and makes an ID roughly sortable by creation time during an
incident review.

Like :mod:`atrader.core.clock`, generation goes through an injected object so
replay can be deterministic (spec §8.3).
"""

from __future__ import annotations

import secrets
from typing import Protocol, runtime_checkable
from uuid import UUID

from atrader.core.clock import NS_PER_MILLI, Clock

__all__ = ["DeterministicIdGenerator", "IdGenerator", "SystemIdGenerator", "compose_uuid7"]

_MASK_48 = (1 << 48) - 1
_MASK_12 = (1 << 12) - 1
_MASK_62 = (1 << 62) - 1


def compose_uuid7(unix_ms: int, rand_a: int, rand_b: int) -> UUID:
    """Assemble a UUIDv7 from its three fields (RFC 9562 §5.7).

    Layout: 48-bit big-endian millisecond timestamp, 4-bit version (``0111``),
    12 bits of randomness, 2-bit variant (``10``), 62 bits of randomness.
    """
    value = (unix_ms & _MASK_48) << 80
    value |= 0x7 << 76
    value |= (rand_a & _MASK_12) << 64
    value |= 0b10 << 62
    value |= rand_b & _MASK_62
    return UUID(int=value)


@runtime_checkable
class IdGenerator(Protocol):
    """Source of unique identifiers."""

    def new_id(self) -> UUID:
        """Return a fresh UUIDv7."""
        ...


class SystemIdGenerator:
    """Production generator: real time plus cryptographic randomness."""

    __slots__ = ("_clock",)

    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    def new_id(self) -> UUID:
        unix_ms = self._clock.now_ns() // NS_PER_MILLI
        return compose_uuid7(
            unix_ms,
            secrets.randbits(12),
            secrets.randbits(62),
        )


class DeterministicIdGenerator:
    """Reproducible generator for backtests, replay and tests.

    The random fields are replaced by a monotonic counter mixed with a seed, so
    the same run produces the same IDs. That is what lets the replay harness
    compare order sequences byte-for-byte (spec §8.3, acceptance criterion #3).

    Not suitable for production: the IDs are predictable, and the whole point of
    ``client_order_id`` is that two processes never collide on one.
    """

    __slots__ = ("_clock", "_counter", "_seed")

    def __init__(self, clock: Clock, seed: int = 0) -> None:
        self._clock = clock
        self._seed = seed & _MASK_62
        self._counter = 0

    def new_id(self) -> UUID:
        self._counter += 1
        unix_ms = self._clock.now_ns() // NS_PER_MILLI
        # Mixing the seed into rand_b keeps runs with different seeds distinct
        # while remaining a pure function of (seed, counter).
        return compose_uuid7(
            unix_ms,
            self._counter & _MASK_12,
            (self._seed ^ (self._counter * 0x9E3779B97F4A7C15)) & _MASK_62,
        )

    def reset(self) -> None:
        """Restart the counter. Call between replay runs."""
        self._counter = 0
