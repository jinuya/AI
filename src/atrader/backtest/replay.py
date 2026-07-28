"""Deterministic replay — spec §2.2, acceptance criterion #3.

    동일한 입력(봉 시퀀스, 설정, 시드)으로 다시 실행했을 때 동일한 주문
    시퀀스가 나와야 한다. 그렇지 않으면 사고 조사 시 재현이 불가능하다.

Determinism is not a nice property here — it is what makes an incident
forensically reconstructible. If the same input can produce two different
order sequences, "run it again and see what happened" is not a valid
investigation technique. Every source of non-determinism the rest of this
system removes — wall-clock reads outside :mod:`atrader.core.clock`, unseeded
randomness outside :mod:`atrader.core.rng`, ``uuid4()`` outside
:mod:`atrader.core.ids` — is removed specifically so this harness can prove
replay produces *identical bytes*, not just "looks about the same".

:func:`compare_replays` runs an engine factory twice — from two *freshly
constructed* engines, since reusing one engine's mutated state would prove
nothing about determinism — and canonically encodes each run's order sequence
to bytes before comparing. Order matters in that encoding: two runs that
produced the same orders in a different sequence are not the same run.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import zip_longest

from atrader.audit.hashchain import canonical_json
from atrader.backtest.engine import BacktestEngine, BacktestResult
from atrader.core.models import Order
from atrader.marketdata.models import Bar

__all__ = [
    "ReplayResult",
    "assert_replay_matches",
    "compare_replays",
    "order_sequence_bytes",
    "run_and_record",
]


def order_sequence_bytes(orders: Iterable[Order]) -> bytes:
    """Canonical byte encoding of an order sequence, in the order produced.

    Not sorted first — sequence order is part of what is under test.
    """
    return canonical_json([order.model_dump() for order in orders])


async def run_and_record(
    engine_factory: Callable[[], BacktestEngine], bars: Sequence[Bar]
) -> tuple[BacktestResult, bytes]:
    """Run one fresh engine over *bars* and canonically encode its orders."""
    engine = engine_factory()
    result = await engine.run(bars)
    return result, order_sequence_bytes(result.orders)


@dataclass(frozen=True, slots=True)
class ReplayResult:
    matches: bool
    left_orders: tuple[Order, ...]
    right_orders: tuple[Order, ...]
    first_divergence: int | None
    """Index of the first order that differs (or is missing on one side).
    ``None`` when the runs match."""

    def summary(self) -> str:
        if self.matches:
            return f"replay matches: {len(self.left_orders)} order(s), byte-identical"
        return (
            f"replay diverged at order #{self.first_divergence}: "
            f"{len(self.left_orders)} order(s) on the first run, "
            f"{len(self.right_orders)} on the second. Non-determinism is almost "
            "always ambient state read outside core.clock/core.ids/core.rng — "
            "see tests/unit/test_determinism_lint.py."
        )


def _first_divergence(left: Sequence[Order], right: Sequence[Order]) -> int:
    for index, (left_order, right_order) in enumerate(zip_longest(left, right)):
        if left_order != right_order:
            return index
    return min(len(left), len(right))  # pragma: no cover — unreachable when bytes differ


async def compare_replays(
    engine_factory: Callable[[], BacktestEngine], bars: Iterable[Bar]
) -> ReplayResult:
    """Run the same scenario twice from scratch and compare byte-for-byte.

    ``bars`` is consumed twice, once per run — pass a re-iterable sequence
    (a list), not a one-shot generator.
    """
    bar_list = list(bars)
    left_result, left_bytes = await run_and_record(engine_factory, bar_list)
    right_result, right_bytes = await run_and_record(engine_factory, bar_list)

    if left_bytes == right_bytes:
        return ReplayResult(
            matches=True,
            left_orders=left_result.orders,
            right_orders=right_result.orders,
            first_divergence=None,
        )
    return ReplayResult(
        matches=False,
        left_orders=left_result.orders,
        right_orders=right_result.orders,
        first_divergence=_first_divergence(left_result.orders, right_result.orders),
    )


async def assert_replay_matches(
    engine_factory: Callable[[], BacktestEngine], bars: Iterable[Bar]
) -> ReplayResult:
    """:func:`compare_replays`, raising ``AssertionError`` on divergence.

    The CI replay gate (acceptance criterion #3) calls this directly; a
    caller that wants the comparison without raising should call
    :func:`compare_replays` instead.
    """
    result = await compare_replays(engine_factory, bars)
    if not result.matches:
        raise AssertionError(result.summary())
    return result
