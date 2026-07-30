"""Deterministic guards on LLM output — spec §7.7.

Everything :mod:`atrader.strategy.llm.client` returns is treated as untrusted
input, exactly like a market data feed: well-formed per its schema, but not
yet trusted to be *sane* or *honest*. This module is the one place that
decides whether a decision becomes a :class:`~atrader.core.models.TradingIntent`
at all, applying four checks in order:

1. **Symbol whitelist** — a decision for a symbol outside the strategy's
   configured universe is rejected outright, flagged critical. This is the
   line prompt injection has to cross to do anything: even a model fully
   fooled by adversarial text can only act on symbols this strategy was
   configured to trade.
2. **Numeric sanity** — quantity must parse as a positive whole number of
   shares; confidence must parse as a decimal in ``[0, 1]``; stop-loss/
   take-profit, if given, must parse and land within a sane band around a
   reference price.
3. **Confidence floor** — below ``min_confidence`` the position is scaled
   down proportionally; if that rounds the quantity to zero, the decision is
   dropped rather than sent as a zero-size order.
4. Anything that survives becomes a :class:`~atrader.core.models.TradingIntent`
   with its ``intent_id`` minted from ``context.ids`` (never a fresh
   ``uuid4()`` — see ``strategy/base.py`` on why that would break replay).

None of this is optional or configurable away except the whitelist check
itself (``require_symbol_whitelist``, present because the risk engine's own
universe check — spec §7.2 check #2 — is a second, independent enforcement of
the same rule; this one exists to reject *before* spending a risk-engine
cycle on something that was never going anywhere).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal, InvalidOperation

from atrader.core.models import TradingIntent
from atrader.core.types import Side, TargetType
from atrader.strategy.base import StrategyContext
from atrader.strategy.llm.client import LLMTradingResponse

__all__ = ["GuardRejection", "GuardResult", "apply_guards"]

#: A stop-loss/take-profit further than this multiple from the reference
#: price is treated as a numeric-sanity failure rather than an aggressive but
#: legitimate order — spec §7.7 "현재가 대비 합리 범위".
_MAX_PRICE_BAND = Decimal("2")
_MIN_PRICE_BAND = Decimal("1") / _MAX_PRICE_BAND


@dataclass(frozen=True, slots=True)
class GuardRejection:
    symbol: str
    reason: str
    critical: bool = False


@dataclass(frozen=True, slots=True)
class GuardResult:
    intents: tuple[TradingIntent, ...]
    rejections: tuple[GuardRejection, ...]


def _parse_decimal(raw: str | None) -> Decimal | None:
    """Parse a model-supplied number, or ``None`` if it is not one.

    ``is_finite`` is the part that matters: ``Decimal("NaN")``,
    ``Decimal("sNaN")`` and ``Decimal("Infinity")`` all construct *without*
    raising, and every one of them is reachable from a schema-valid response
    because the wire format carries decimals as strings. A NaN then raises
    ``InvalidOperation`` from the very range checks below (NaN has no
    ordering), and an Infinity slips past them to fail inside pydantic —
    either way an exception escapes ``apply_guards`` instead of a
    ``GuardRejection``, taking the whole cycle with it and discarding the
    legitimate decisions alongside the crafted one. The deterministic guard
    layer must fail closed, never fail loudly.
    """
    if raw is None:
        return None
    try:
        value = Decimal(raw)
    except (InvalidOperation, ValueError, TypeError):
        return None
    return value if value.is_finite() else None


def _within_band(price: Decimal, reference: Decimal) -> bool:
    if reference <= 0:
        return False
    ratio = price / reference
    return _MIN_PRICE_BAND <= ratio <= _MAX_PRICE_BAND


def apply_guards(
    response: LLMTradingResponse,
    *,
    context: StrategyContext,
    strategy_id: str,
    symbol_whitelist: frozenset[str],
    min_confidence: Decimal,
    require_symbol_whitelist: bool = True,
    reference_prices: Mapping[str, Decimal] | None = None,
) -> GuardResult:
    """Turn a raw :class:`LLMTradingResponse` into guarded intents.

    ``reference_prices`` (last known price per symbol) is optional so this
    also works in contexts with no price to sanity-check against — the band
    check is then simply skipped for that decision, not treated as a failure.
    """
    reference_prices = reference_prices or {}
    intents: list[TradingIntent] = []
    rejections: list[GuardRejection] = []

    for decision in response.decisions:
        symbol = decision.symbol

        if decision.action == "hold":
            continue

        if require_symbol_whitelist and symbol not in symbol_whitelist:
            rejections.append(
                GuardRejection(
                    symbol=symbol,
                    reason=f"{symbol!r} is not in this strategy's symbol whitelist",
                    critical=True,
                )
            )
            continue

        confidence = _parse_decimal(decision.confidence)
        if confidence is None or not (Decimal(0) <= confidence <= Decimal(1)):
            rejections.append(
                GuardRejection(
                    symbol=symbol,
                    reason=f"confidence {decision.confidence!r} is not a decimal in [0, 1]",
                )
            )
            continue

        if decision.quantity is None:
            rejections.append(
                GuardRejection(symbol=symbol, reason="quantity is required for a buy/sell decision")
            )
            continue
        raw_quantity = _parse_decimal(decision.quantity)
        if (
            raw_quantity is None
            or raw_quantity <= 0
            or raw_quantity != raw_quantity.to_integral_value()
        ):
            rejections.append(
                GuardRejection(
                    symbol=symbol,
                    reason=f"quantity must be a positive whole number of shares, got {decision.quantity!r}",
                )
            )
            continue

        quantity = raw_quantity
        scaled_down = False
        if confidence < min_confidence:
            scale = confidence / min_confidence
            quantity = (raw_quantity * scale).to_integral_value(rounding=ROUND_FLOOR)
            scaled_down = True
            if quantity <= 0:
                rejections.append(
                    GuardRejection(
                        symbol=symbol,
                        reason=(
                            f"confidence {confidence} is below the floor {min_confidence}; "
                            f"scaling {raw_quantity} shares by it rounds to zero"
                        ),
                    )
                )
                continue

        stop_loss = _parse_decimal(decision.stop_loss)
        if decision.stop_loss is not None and (stop_loss is None or stop_loss <= 0):
            rejections.append(
                GuardRejection(
                    symbol=symbol,
                    reason=f"stop_loss {decision.stop_loss!r} is not a positive decimal",
                )
            )
            continue
        take_profit = _parse_decimal(decision.take_profit)
        if decision.take_profit is not None and (take_profit is None or take_profit <= 0):
            rejections.append(
                GuardRejection(
                    symbol=symbol,
                    reason=f"take_profit {decision.take_profit!r} is not a positive decimal",
                )
            )
            continue

        reference_price = reference_prices.get(symbol)
        if reference_price is not None:
            out_of_band = next(
                (
                    (name, price)
                    for name, price in (("stop_loss", stop_loss), ("take_profit", take_profit))
                    if price is not None and not _within_band(price, reference_price)
                ),
                None,
            )
            if out_of_band is not None:
                name, price = out_of_band
                rejections.append(
                    GuardRejection(
                        symbol=symbol,
                        reason=(
                            f"{name} {price} is not within a sane range of reference price "
                            f"{reference_price}"
                        ),
                    )
                )
                continue

        rationale = decision.rationale
        if scaled_down:
            rationale = f"{rationale} [confidence {confidence} scaled quantity from {raw_quantity}]"

        intents.append(
            TradingIntent(
                intent_id=context.ids.new_id(),
                strategy_id=strategy_id,
                symbol=symbol,
                side=Side.BUY if decision.action == "buy" else Side.SELL,
                target_type=TargetType.SHARES,
                target_value=quantity,
                confidence=confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                rationale=rationale,
                created_at_ns=context.now_ns,
            )
        )

    return GuardResult(intents=tuple(intents), rejections=tuple(rejections))
