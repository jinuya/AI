"""Randomness.

Same pattern as :mod:`atrader.core.clock`: randomness is injected rather than
taken from ambient state, so a backtest or a replay run reproduces exactly
(spec §2.2). ``tests/unit/test_determinism_lint.py`` fails the build if anything
outside this module imports ``random`` directly.

The two places the system legitimately needs randomness are the simulated market
feed and reconnect jitter (spec §FR-MD-01 — without jitter every instance
reconnects in lockstep and the venue throttles all of them). Neither is on the
order path, but both must be reproducible.
"""

from __future__ import annotations

import random
from decimal import Decimal
from typing import Protocol, runtime_checkable

__all__ = ["Rng", "SeededRng"]


@runtime_checkable
class Rng(Protocol):
    """Source of pseudo-random values."""

    def uniform(self, low: float, high: float) -> float: ...

    def gauss(self, mu: float, sigma: float) -> float: ...

    def randint(self, low: int, high: int) -> int:
        """Inclusive on both bounds."""
        ...

    def choice_index(self, length: int) -> int:
        """Index into a sequence of *length* items."""
        ...


class SeededRng:
    """Deterministic generator. The same seed always yields the same stream."""

    __slots__ = ("_random", "_seed")

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._random = random.Random(seed)

    @property
    def seed(self) -> int:
        return self._seed

    def uniform(self, low: float, high: float) -> float:
        return self._random.uniform(low, high)

    def gauss(self, mu: float, sigma: float) -> float:
        return self._random.gauss(mu, sigma)

    def randint(self, low: int, high: int) -> int:
        return self._random.randint(low, high)

    def choice_index(self, length: int) -> int:
        if length <= 0:
            raise ValueError("cannot choose from an empty sequence")
        return self._random.randrange(length)

    def jitter(self, base_seconds: float, jitter_pct: Decimal) -> float:
        """Apply +/- *jitter_pct* to a backoff delay (spec §FR-MD-01)."""
        fraction = float(jitter_pct) / 100.0
        return base_seconds * (1.0 + self._random.uniform(-fraction, fraction))

    def reset(self) -> None:
        """Restart the stream. Call between replay runs."""
        self._random = random.Random(self._seed)
