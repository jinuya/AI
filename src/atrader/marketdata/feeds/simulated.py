"""Synthetic market data.

Drives development, paper trading and the backtest fixtures without any vendor
connection. Prices follow a seeded geometric random walk with a configurable
spread, so the same seed always produces the same session — a strategy bug found
here is reproducible.

This is a *simulator*, not a model of any real market: no volatility clustering,
no intraday seasonality, no news. It is honest about that because using it to
judge whether a strategy is profitable would be self-deception. What it is good
for is exercising the plumbing — order flow, risk gating, reconciliation — under
data that keeps moving.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from decimal import Decimal

from atrader.core.clock import NS_PER_SECOND, Clock, SimulatedClock
from atrader.core.money import from_float, quantize, round_to_tick
from atrader.core.rng import Rng, SeededRng
from atrader.marketdata.feeds.protocol import FeedStatus
from atrader.marketdata.models import Tick

__all__ = ["SimulatedFeed", "SymbolSimulation"]


@dataclass(slots=True)
class SymbolSimulation:
    """Per-symbol simulation parameters and running state."""

    symbol: str
    price: Decimal
    annual_volatility: Decimal = Decimal("0.25")
    spread_bps: Decimal = Decimal("2")
    tick_size: Decimal = Decimal("0.01")
    base_size: Decimal = Decimal("100")
    drift_annual: Decimal = Decimal("0.05")
    seq: int = 0


#: Trading seconds in a year: 252 days x 6.5 hours.
_SECONDS_PER_TRADING_YEAR = Decimal(252 * 6 * 60 * 60 + 252 * 30 * 60)


class SimulatedFeed:
    """Generates ticks for a set of symbols on a fixed cadence."""

    __slots__ = ("_clock", "_interval_ns", "_name", "_rng", "_status", "_symbols", "_ticks_emitted")

    def __init__(
        self,
        symbols: Sequence[SymbolSimulation],
        *,
        clock: Clock | None = None,
        rng: Rng | None = None,
        interval_ns: int = NS_PER_SECOND,
        name: str = "simulated",
    ) -> None:
        if not symbols:
            raise ValueError("at least one symbol is required")
        self._symbols = {sim.symbol: sim for sim in symbols}
        self._clock = clock or SimulatedClock(start_ns=1_700_000_000 * NS_PER_SECOND)
        self._rng = rng or SeededRng(seed=0)
        self._interval_ns = interval_ns
        self._name = name
        self._status = FeedStatus.DISCONNECTED
        self._ticks_emitted = 0

    @classmethod
    def from_symbols(
        cls,
        symbols: Sequence[str],
        *,
        start_price: Decimal = Decimal("100"),
        seed: int = 0,
        clock: Clock | None = None,
        interval_ns: int = NS_PER_SECOND,
    ) -> SimulatedFeed:
        return cls(
            [SymbolSimulation(symbol=s, price=start_price) for s in symbols],
            clock=clock,
            rng=SeededRng(seed),
            interval_ns=interval_ns,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> str:
        return self._status

    @property
    def ticks_emitted(self) -> int:
        return self._ticks_emitted

    async def connect(self, symbols: Sequence[str]) -> None:
        unknown = sorted(set(symbols) - set(self._symbols))
        if unknown:
            raise ValueError(f"no simulation configured for {unknown}")
        self._status = FeedStatus.CONNECTED

    def _step(self, sim: SymbolSimulation) -> Tick:
        """Advance one symbol by a single interval."""
        dt_years = Decimal(self._interval_ns) / Decimal(NS_PER_SECOND) / _SECONDS_PER_TRADING_YEAR
        sigma = float(sim.annual_volatility) * float(dt_years) ** 0.5
        drift = float(sim.drift_annual) * float(dt_years)
        shock = self._rng.gauss(drift, sigma)

        # Geometric walk keeps the price positive no matter how long it runs.
        new_price = sim.price * (Decimal(1) + from_float(shock))
        floor = sim.tick_size * Decimal(2)
        sim.price = max(quantize(new_price), floor)

        half_spread = sim.price * sim.spread_bps / Decimal(20_000)
        bid = round_to_tick(sim.price - half_spread, sim.tick_size, side="BUY")
        ask = round_to_tick(sim.price + half_spread, sim.tick_size, side="SELL")
        if bid >= ask:
            # A spread narrower than one tick would produce a crossed quote,
            # which the quality monitor would (correctly) reject as bad data.
            ask = quantize(bid + sim.tick_size)

        sim.seq += 1
        now_ns = self._clock.now_ns()
        size_multiple = Decimal(self._rng.randint(1, 10))
        return Tick(
            symbol=sim.symbol,
            exchange_ts=now_ns,
            ingest_ts=now_ns + 1_000_000,  # a plausible 1ms of feed latency
            bid=bid,
            ask=ask,
            bid_size=sim.base_size * size_multiple,
            ask_size=sim.base_size * size_multiple,
            last=round_to_tick(sim.price, sim.tick_size),
            last_size=sim.base_size,
            seq=sim.seq,
            source=self._name,
        )

    def generate(self, count: int) -> list[Tick]:
        """Produce *count* ticks per symbol, advancing the clock as it goes.

        Only usable with a :class:`~atrader.core.clock.SimulatedClock`; against
        real time the caller controls the cadence instead.
        """
        if not isinstance(self._clock, SimulatedClock):
            raise TypeError("generate() advances the clock and needs a SimulatedClock")
        ticks: list[Tick] = []
        for _ in range(count):
            for sim in self._symbols.values():
                ticks.append(self._step(sim))
                self._ticks_emitted += 1
            self._clock.advance(self._interval_ns)
        return ticks

    async def _iterate(self) -> AsyncIterator[Tick]:
        while self._status == FeedStatus.CONNECTED:
            for sim in self._symbols.values():
                self._ticks_emitted += 1
                yield self._step(sim)
            if isinstance(self._clock, SimulatedClock):
                self._clock.advance(self._interval_ns)

    def __aiter__(self) -> AsyncIterator[Tick]:
        return self._iterate()

    async def close(self) -> None:
        self._status = FeedStatus.DISCONNECTED
