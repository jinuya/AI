"""Position sizing — spec §7.3.

The default is volatility targeting, for the reason the spec gives:

    고정 금액으로 사면 변동성 높은 종목에서 리스크가 훨씬 커진다.

$10,000 of a name that moves 1% a day and $10,000 of one that moves 8% a day are
not the same bet, even though the ticket sizes match. Sizing by ATR makes them
comparable: each position is sized so that an adverse move to its stop costs
roughly ``risk_per_trade_pct`` of the account.

Kelly is available but capped at quarter-Kelly, and the cap is not negotiable:

    풀 켈리는 이론적으로 최적이지만 추정 오차에 극도로 민감해서 실전에서는 파산 확률이 높다.

Full Kelly is optimal only if your edge estimate is exact. It never is, and the
penalty for overestimating is geometric.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from atrader.config.schema import RiskConfig
from atrader.core.money import ZERO, floor_to_lot, quantize, safe_div
from atrader.core.types import SizingMethod
from atrader.risk.state import RiskSnapshot

__all__ = ["SizingResult", "correlated_exposure", "size_position"]


@dataclass(frozen=True, slots=True)
class SizingResult:
    """A sized order, with the reasoning kept for the audit log."""

    quantity: Decimal
    notional: Decimal
    method: SizingMethod
    reason: str
    capped_by: str | None = None
    """Which constraint bound the size, when one did."""

    @property
    def is_empty(self) -> bool:
        return self.quantity <= ZERO


def size_position(
    *,
    symbol: str,
    price: Decimal,
    config: RiskConfig,
    snapshot: RiskSnapshot,
    atr: Decimal | None = None,
    win_probability: Decimal | None = None,
    win_loss_ratio: Decimal | None = None,
    lot_size: Decimal = Decimal(1),
    confidence: Decimal = Decimal(1),
) -> SizingResult:
    """Compute the share count for a new position.

    ``confidence`` scales the result linearly — spec §7.7 requires an LLM's
    stated confidence to shrink the position rather than being ignored.
    """
    if price <= ZERO:
        return SizingResult(ZERO, ZERO, config.sizing.method, "no valid price")

    equity = snapshot.account.equity
    if equity <= ZERO:
        return SizingResult(ZERO, ZERO, config.sizing.method, "no equity to allocate")

    method = config.sizing.method
    if method is SizingMethod.VOLATILITY_TARGET:
        notional, reason = _volatility_target(equity, price, config, atr)
    elif method is SizingMethod.KELLY:
        notional, reason = _kelly(equity, config, win_probability, win_loss_ratio)
    else:
        notional, reason = _fixed(equity, config)

    if confidence < Decimal(1):
        notional = quantize(notional * confidence)
        reason += f", scaled by confidence {confidence}"

    # Never exceed the single-order cap: sizing and the pre-trade check must not
    # disagree, or every order would arrive needing a REDUCE.
    max_notional = equity * config.order.max_order_notional_pct / Decimal(100)
    capped_by: str | None = None
    if notional > max_notional:
        notional = max_notional
        capped_by = "max_order_notional_pct"

    # And never exceed the remaining headroom under the per-position cap.
    position_cap = equity * config.position.max_position_pct / Decimal(100)
    held = abs(snapshot.position_of(symbol).market_value)
    headroom = max(ZERO, position_cap - held)
    if notional > headroom:
        notional = headroom
        capped_by = "max_position_pct"

    quantity = floor_to_lot(notional / price, lot_size)
    return SizingResult(
        quantity=quantity,
        notional=quantize(quantity * price),
        method=method,
        reason=reason,
        capped_by=capped_by,
    )


def _volatility_target(
    equity: Decimal, price: Decimal, config: RiskConfig, atr: Decimal | None
) -> tuple[Decimal, str]:
    """Size so that a move to the stop costs ``risk_per_trade_pct`` of equity.

    Spec §7.3 writes this as::

        position_notional = (account_equity x risk_per_trade) / (ATR_multiple x ATR)

    The name in the spec is a slight misnomer: the units there are
    ``money / (money per share)``, i.e. **shares**. Notional is that share count
    multiplied by the price, which is what this returns.

    Without an ATR estimate it falls back to the fixed fraction rather than
    inventing a volatility — a made-up ATR would silently mis-size every
    position in a newly listed name, and mis-size it in the dangerous direction
    if the guess was low.
    """
    risk_budget = equity * config.sizing.risk_per_trade_pct / Decimal(100)
    if atr is None or atr <= ZERO:
        notional, _ = _fixed(equity, config)
        return notional, "no ATR available; fell back to fixed fraction"

    stop_distance = config.sizing.atr_multiple * atr
    shares = safe_div(risk_budget, stop_distance)
    return (
        quantize(shares * price),
        f"volatility target: {config.sizing.risk_per_trade_pct}% of equity ({risk_budget:.2f}) "
        f"at risk over a {config.sizing.atr_multiple}x ATR ({atr}) stop of {stop_distance:.4f} "
        f"-> {shares:.4f} shares",
    )


def _kelly(
    equity: Decimal,
    config: RiskConfig,
    win_probability: Decimal | None,
    win_loss_ratio: Decimal | None,
) -> tuple[Decimal, str]:
    """Fractional Kelly, hard-capped at :attr:`SizingConfig.kelly_fraction`."""
    if win_probability is None or win_loss_ratio is None or win_loss_ratio <= ZERO:
        notional, _ = _fixed(equity, config)
        return notional, "insufficient statistics for Kelly; fell back to fixed fraction"

    # f* = p - (1 - p) / b
    edge = win_probability - (Decimal(1) - win_probability) / win_loss_ratio
    if edge <= ZERO:
        return ZERO, f"Kelly edge is {edge:.4f} (non-positive); no position"

    fraction = min(edge * config.sizing.kelly_fraction, config.sizing.kelly_fraction)
    return (
        quantize(equity * fraction),
        f"fractional Kelly: edge {edge:.4f} x {config.sizing.kelly_fraction} = {fraction:.4f}",
    )


def _fixed(equity: Decimal, config: RiskConfig) -> tuple[Decimal, str]:
    fraction = config.sizing.risk_per_trade_pct / Decimal(100)
    return (
        quantize(equity * fraction),
        f"fixed fraction: {config.sizing.risk_per_trade_pct}% of equity",
    )


def correlated_exposure(
    symbol: str,
    snapshot: RiskSnapshot,
    threshold: Decimal,
) -> tuple[Decimal, list[str]]:
    """Total exposure to positions correlated with *symbol* above *threshold*.

    Spec §7.3: adding a name that moves with what you already hold is growing
    one bet, not diversifying. The combined exposure is what the limit should
    apply to — five uncorrelated 5% positions and five 0.9-correlated 5%
    positions are very different portfolios.
    """
    total = ZERO
    contributors: list[str] = []
    for held_symbol, position in snapshot.positions.items():
        if position.is_flat:
            continue
        if held_symbol == symbol or snapshot.correlation(symbol, held_symbol) >= threshold:
            total += position.exposure
            contributors.append(held_symbol)
    return quantize(total), sorted(contributors)
