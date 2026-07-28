"""Target-weight rebalancing — spec §FR-PF-03.

    포트폴리오 전체를 목표 비중으로 재조정할 때, 목표와 현재 비중의 차이가
    데드밴드(기본 ±0.5%p) 이내인 종목은 건너뛴다.

A strategy that thinks in *weights* — "40% AAPL, 35% MSFT, 25% cash" — needs
something that turns a whole target vector into the individual per-symbol
intents the rest of the system understands. That is all this module does:
:func:`rebalance_to_weights` diffs a target-weight map against current
positions and emits one ``TARGET_WEIGHT`` :class:`~atrader.core.models.TradingIntent`
per symbol that actually needs to move.

The deadband is applied twice in this system, deliberately, not redundantly.
The risk engine (``risk/engine.py::_resolve_quantity``) applies it again to
whatever intent reaches it, because that is the last line of defence and it
cannot assume every intent arrived through this module. Filtering here first
is what keeps a portfolio that is already on target from emitting fifty
`TARGET_WEIGHT` intents a cycle that the risk engine would each reduce to a
zero-quantity rejection — noise in the audit log for an outcome that was
never in question.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from atrader.core.clock import Clock
from atrader.core.ids import IdGenerator
from atrader.core.models import Position, TradingIntent
from atrader.core.money import ZERO
from atrader.core.types import Side, TargetType, Urgency

__all__ = ["RebalanceTarget", "rebalance_to_weights"]


@dataclass(frozen=True, slots=True)
class RebalanceTarget:
    """One symbol's contribution to a rebalance — kept for reporting even when
    it falls inside the deadband and produces no intent."""

    symbol: str
    current_weight_pct: Decimal
    target_weight_pct: Decimal
    drift_pct: Decimal
    """``target - current``, in percentage points. Positive means underweight
    (needs buying). The true drift is always reported here, even when it fell
    inside the deadband and produced no intent — see :attr:`traded`."""
    traded: bool
    """Whether this symbol's drift exceeded the deadband and an intent was
    emitted for it."""


def rebalance_to_weights(
    target_weights: dict[str, Decimal],
    *,
    positions: dict[str, Position],
    equity: Decimal,
    strategy_id: str,
    ids: IdGenerator,
    clock: Clock,
    deadband_pct: Decimal,
    urgency: Urgency = Urgency.NORMAL,
) -> tuple[list[TradingIntent], list[RebalanceTarget]]:
    """Diff target weights against current positions.

    ``target_weights`` values are fractions of equity (0.10 means 10%),
    matching :attr:`TradingIntent.target_value` for ``TARGET_WEIGHT`` — signed,
    so a negative entry targets a short. Every symbol in the map is reported in
    the returned list of :class:`RebalanceTarget`; only the ones outside the
    deadband also appear in the intent list.
    """
    intents: list[TradingIntent] = []
    targets: list[RebalanceTarget] = []
    now = clock.now_ns()

    for symbol, target_weight in target_weights.items():
        position = positions.get(symbol)
        current_value = position.market_value if position is not None else ZERO
        current_weight = current_value / equity if equity > ZERO else ZERO
        drift = target_weight - current_weight

        within_deadband = abs(drift) * Decimal(100) <= deadband_pct
        targets.append(
            RebalanceTarget(
                symbol=symbol,
                current_weight_pct=current_weight * Decimal(100),
                target_weight_pct=target_weight * Decimal(100),
                drift_pct=drift * Decimal(100),
                traded=not within_deadband,
            )
        )
        if within_deadband:
            continue

        intents.append(
            TradingIntent(
                intent_id=ids.new_id(),
                strategy_id=strategy_id,
                symbol=symbol,
                side=Side.BUY if drift > ZERO else Side.SELL,
                target_type=TargetType.TARGET_WEIGHT,
                target_value=target_weight,
                urgency=urgency,
                rationale=(
                    f"rebalance: {current_weight * 100:.2f}% -> {target_weight * 100:.2f}% "
                    f"target (drift {drift * 100:+.2f}pp)"
                ),
                created_at_ns=now,
            )
        )

    return intents, targets
