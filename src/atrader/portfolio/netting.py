"""Multi-strategy intent netting — spec §FR-STR-04.

    여러 전략이 같은 종목에 대해 서로 다른(때로는 반대되는) 의도를 낼 수 있다.
    시장에 각각 내보내는 대신, 하나의 순 목표로 넷팅한 뒤에 리스크 게이트로
    보낸다.

Two strategies each independently deciding to trade AAPL in one cycle is not
a special case to handle after the fact — it is routine the moment more than
one strategy is enabled (spec §FR-STR-04; allocations already sum to at most
100% of capital, enforced in :class:`~atrader.config.schema.AppConfig`). Left
alone, opposing intents would show up as two orders that the self-cross check
(§7.2 #15) then has to catch — or worse, cross each other during whatever gap
separates them and print a wash trade neither strategy intended.

Netting removes that failure mode structurally rather than defensively: two
exactly-opposing intents of equal size net to zero and never become an order
at all. This runs *before* the risk engine, on raw strategy output — the risk
engine only ever sees at most one intent per symbol per cycle.

Every intent type is reduced to a common unit — a signed share delta,
positive for "wants more long exposure" — before summing:

* ``SHARES``/``NOTIONAL`` are already a delta (per :class:`TradingIntent`'s
  own contract: direction lives in ``side``, not in the sign of
  ``target_value``).
* ``TARGET_WEIGHT`` is an absolute target, converted to a delta against the
  *current* position the same way the risk engine does it for a single intent
  (``target_shares - current_shares``). Netting needs its own copy of that
  arithmetic because it runs on several intents before any one of them reaches
  the risk engine's ``RiskSnapshot``.

A synthesized multi-strategy intent deliberately drops ``limit_price``,
``stop_loss`` and ``take_profit``: those are one strategy's risk parameters,
and carrying one contributor's stop onto a blended position that contributor
did not fully originate would silently apply their exit logic to size they
never sized for.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

from atrader.core.clock import Clock
from atrader.core.ids import IdGenerator
from atrader.core.models import Position, TradingIntent
from atrader.core.money import ZERO
from atrader.core.types import Side, TargetType, Urgency

__all__ = ["NETTED_STRATEGY_ID", "NettingConflict", "net_intents"]

NETTED_STRATEGY_ID = "portfolio.netted"

_URGENCY_RANK = {Urgency.PASSIVE: 0, Urgency.NORMAL: 1, Urgency.AGGRESSIVE: 2}


@dataclass(frozen=True, slots=True)
class NettingConflict:
    """Reported whenever more than one strategy targeted the same symbol —
    even when they agreed and the net simply added up."""

    symbol: str
    contributing_strategy_ids: tuple[str, ...]
    gross_shares: Decimal
    """Sum of |each strategy's desired delta| — what would have traded without
    netting, had every strategy gone to market independently."""
    net_shares: Decimal
    """Signed. What is actually sent onward, before the deadband is applied."""

    @property
    def cancelled_shares(self) -> Decimal:
        """How much volume netting kept off the market."""
        return self.gross_shares - abs(self.net_shares)


def net_intents(
    intents: Iterable[TradingIntent],
    *,
    positions: dict[str, Position],
    prices: dict[str, Decimal],
    equity: Decimal,
    ids: IdGenerator,
    clock: Clock,
    deadband_pct: Decimal = ZERO,
) -> tuple[list[TradingIntent], list[NettingConflict]]:
    """Combine same-symbol intents from different strategies into one.

    Returns the intents to hand to the risk engine — at most one per symbol —
    plus a conflict report for every symbol two or more strategies weighed in
    on. A symbol only one strategy targeted passes through unchanged,
    including its ``limit_price``/``stop_loss``/``urgency``: there is nothing
    to blend.
    """
    by_symbol: dict[str, list[TradingIntent]] = {}
    for intent in intents:
        by_symbol.setdefault(intent.symbol, []).append(intent)

    netted: list[TradingIntent] = []
    conflicts: list[NettingConflict] = []

    for symbol, group in by_symbol.items():
        if len(group) == 1:
            netted.append(group[0])
            continue

        price = prices.get(symbol)
        position = positions.get(symbol)
        deltas = [_to_share_delta(intent, position, price, equity) for intent in group]
        gross = sum((abs(d) for d in deltas), ZERO)
        net = sum(deltas, ZERO)

        conflicts.append(
            NettingConflict(
                symbol=symbol,
                contributing_strategy_ids=tuple(i.strategy_id for i in group),
                gross_shares=gross,
                net_shares=net,
            )
        )

        share_deadband = ZERO
        if price is not None and price > ZERO:
            share_deadband = (equity * deadband_pct / Decimal(100)) / price
        if abs(net) <= share_deadband:
            continue  # the strategies cancelled each other out (or nearly did)

        netted.append(_synthesize(symbol, group, deltas, net, ids, clock))

    return netted, conflicts


def _to_share_delta(
    intent: TradingIntent,
    position: Position | None,
    price: Decimal | None,
    equity: Decimal,
) -> Decimal:
    """Reduce one intent to a signed share delta.

    A missing or non-positive price starves the conversion to zero rather than
    raising — a symbol with no price yet cannot be sized, and if this intent
    ends up alone (no conflict), the risk engine rejects it on its own terms
    when it reaches ``check_data_quality``/pricing.
    """
    if intent.target_type is TargetType.SHARES:
        return intent.target_value * intent.side.sign
    if price is None or price <= ZERO:
        return ZERO
    if intent.target_type is TargetType.NOTIONAL:
        return (intent.target_value / price) * intent.side.sign
    # TARGET_WEIGHT
    current_qty = position.quantity if position is not None else ZERO
    target_qty = (equity * intent.target_value) / price
    return target_qty - current_qty


def _synthesize(
    symbol: str,
    group: list[TradingIntent],
    deltas: list[Decimal],
    net: Decimal,
    ids: IdGenerator,
    clock: Clock,
) -> TradingIntent:
    # Guaranteed positive: total_weight >= |net| > 0 by the triangle
    # inequality, and this is only called once `net` has cleared the deadband.
    total_weight = sum((abs(d) for d in deltas), ZERO)
    confidence = (
        sum((i.confidence * abs(d) for i, d in zip(group, deltas, strict=True)), ZERO)
        / total_weight
    )

    deadlines = [i.valid_until_ns for i in group if i.valid_until_ns is not None]
    contributors = ", ".join(sorted({i.strategy_id for i in group}))

    return TradingIntent(
        intent_id=ids.new_id(),
        strategy_id=NETTED_STRATEGY_ID,
        symbol=symbol,
        side=Side.BUY if net > ZERO else Side.SELL,
        target_type=TargetType.SHARES,
        target_value=abs(net),
        urgency=max((i.urgency for i in group), key=lambda u: _URGENCY_RANK[u]),
        time_in_force=group[0].time_in_force,
        valid_until_ns=min(deadlines) if deadlines else None,
        confidence=confidence,
        rationale=f"netted from {len(group)} strategies ({contributors})",
        created_at_ns=clock.now_ns(),
    )
