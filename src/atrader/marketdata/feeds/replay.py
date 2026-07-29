"""Replay a recorded tick file.

This is the feed the deterministic replay harness uses (spec §8.3, acceptance
criterion #3): record a session's ticks, play them back, and the order sequence
must come out byte-identical.

Storage is JSONL — one tick per line — because it appends cheaply during
recording, streams without loading the whole file, and stays diffable when a
replay disagrees with the original and you need to find out where.

The file is *not* rewritten to fix bad data. A recording of a session that
included a crossed market must still contain that crossed market, or the replay
is not testing what actually happened.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from atrader.core.errors import DataQualityError
from atrader.marketdata.feeds.protocol import FeedStatus
from atrader.marketdata.models import Tick

__all__ = ["ReplayFeed", "TickRecorder", "read_ticks", "write_ticks"]

_DECIMAL_FIELDS = ("bid", "ask", "bid_size", "ask_size", "last", "last_size")


def _tick_to_json(tick: Tick) -> dict[str, Any]:
    data = tick.model_dump()
    for field in _DECIMAL_FIELDS:
        value = data.get(field)
        if value is not None:
            # str() keeps the exact digits; a float round-trip would not, and a
            # replay that shifts prices by a ULP is not a replay.
            data[field] = str(value)
    data["quality"] = tick.quality.value
    return data


def _tick_from_json(data: Any) -> Tick:
    # Typed ``Any`` rather than ``dict``: the caller hands over whatever
    # ``json.loads`` produced, and a corrupt recording can legitimately
    # contain a line that is a list, a bare number, or ``null``. Declaring a
    # dict here would only move the failure to an AttributeError below.
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    for field in _DECIMAL_FIELDS:
        value = data.get(field)
        if value is not None:
            try:
                data[field] = Decimal(str(value))
            except InvalidOperation as exc:
                # InvalidOperation subclasses ArithmeticError, not ValueError,
                # so it would otherwise escape read_ticks' except clause and
                # reach the caller without the file and line number that make
                # a corrupt recording findable.
                raise ValueError(f"{field}={value!r} is not a number") from exc
    return Tick.model_validate(data)


def write_ticks(path: Path, ticks: Iterable[Tick]) -> int:
    """Write ticks as JSONL. Returns the number written."""
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for tick in ticks:
            handle.write(json.dumps(_tick_to_json(tick), sort_keys=True) + "\n")
            count += 1
    return count


def read_ticks(path: Path) -> list[Tick]:
    """Read a JSONL tick file."""
    if not path.is_file():
        raise DataQualityError(str(path), "recording not found")
    ticks: list[Tick] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                ticks.append(_tick_from_json(json.loads(stripped)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise DataQualityError(str(path), f"line {line_number}: {exc}") from exc
    return ticks


class TickRecorder:
    """Append ticks to a JSONL file as they arrive.

    Used in paper and live trading so any session can be replayed afterwards —
    which is how a production incident becomes a reproducible test case.
    """

    __slots__ = ("_count", "_handle", "_path")

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def record(self, tick: Tick) -> None:
        self._handle.write(json.dumps(_tick_to_json(tick), sort_keys=True) + "\n")
        self._count += 1

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self._handle.flush()
        self._handle.close()

    def __enter__(self) -> TickRecorder:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class ReplayFeed:
    """Deterministic feed backed by a recorded tick list or file."""

    __slots__ = ("_index", "_name", "_status", "_symbols", "_ticks")

    def __init__(self, ticks: Sequence[Tick] | Path, *, name: str = "replay") -> None:
        self._ticks: list[Tick] = read_ticks(ticks) if isinstance(ticks, Path) else list(ticks)
        self._name = name
        self._index = 0
        self._status = FeedStatus.DISCONNECTED
        self._symbols: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> str:
        return self._status

    @property
    def remaining(self) -> int:
        return max(0, len(self._ticks) - self._index)

    async def connect(self, symbols: Sequence[str]) -> None:
        self._symbols = tuple(symbols)
        self._status = FeedStatus.CONNECTED

    def next_tick(self) -> Tick | None:
        """Pull the next tick synchronously.

        The backtest engine drives the clock itself, so it needs a pull
        interface rather than an async stream it would have to race against.
        """
        while self._index < len(self._ticks):
            tick = self._ticks[self._index]
            self._index += 1
            if not self._symbols or tick.symbol in self._symbols:
                return tick
        self._status = FeedStatus.EXHAUSTED
        return None

    async def _iterate(self) -> AsyncIterator[Tick]:
        while True:
            tick = self.next_tick()
            if tick is None:
                return
            yield tick

    def __aiter__(self) -> AsyncIterator[Tick]:
        return self._iterate()

    def reset(self) -> None:
        """Rewind. Two replay runs must produce identical output."""
        self._index = 0
        self._status = FeedStatus.CONNECTED

    async def close(self) -> None:
        self._status = FeedStatus.DISCONNECTED
