"""The fifteen pre-trade checks — spec §7.2.

Evaluated in the order the spec lists them, first failure wins. The order is not
arbitrary: the cheap categorical checks (is the system running, is the symbol
allowed, is the data good) come before anything that does arithmetic, so a
malformed intent is rejected before it can influence a size calculation.

Every check names the observed value and the limit in its reason, because the
audience for that string is someone reading an audit log at 09:32 trying to work
out why an order did not go out.
"""

from __future__ import annotations

from decimal import Decimal

from atrader.core.money import ZERO, as_pct, safe_div
from atrader.core.types import AlertLevel, DataQuality, RiskAction, SystemState
from atrader.risk.checks.base import (
    CheckContext,
    RiskCheck,
    RiskCheckResult,
    allow,
    reduce_to,
    reject,
)

__all__ = ["PRETRADE_CHECKS", "check_names"]


# --------------------------------------------------------------------------
# 1. System state
# --------------------------------------------------------------------------


def check_system_state(ctx: CheckContext) -> RiskCheckResult:
    name = "system_state"
    state = ctx.snapshot.system_state

    if ctx.snapshot.kill_switch_engaged and not ctx.reduce_only:
        return reject(
            name,
            "kill switch is engaged; no new orders",
            alert_level=AlertLevel.CRITICAL,
        )
    if ctx.snapshot.reconciliation_break and not ctx.reduce_only:
        # Spec §FR-EXE-05: local state disagrees with the broker. Until a human
        # resolves it we do not know our real position, so sizing anything new
        # would be guesswork.
        return reject(
            name,
            "unresolved reconciliation break; local state disagrees with the broker",
            alert_level=AlertLevel.CRITICAL,
        )
    if ctx.snapshot.margin_reduce_only and not ctx.reduce_only:
        # Spec §FR-PF-04: below the maintenance-margin reduce-only threshold,
        # only exposure-shrinking orders go out. Unlike the two rejections
        # above, this one clears itself the moment margin recovers.
        return reject(
            name,
            "account is under a margin call; only exposure-reducing orders are permitted",
            alert_level=AlertLevel.CRITICAL,
        )

    if ctx.reduce_only:
        # Liquidation must work in every state except a full halt — that is
        # what BLOCKED and LIQUIDATING are *for*.
        if state is SystemState.HALTED:
            return reject(name, "system is HALTED; even liquidation is suspended")
        return allow(name)

    if state is not SystemState.RUNNING:
        return reject(
            name,
            f"system state is {state.value}, not RUNNING; new orders are suspended",
            alert_level=AlertLevel.WARN,
        )
    return allow(name)


# --------------------------------------------------------------------------
# 2. Universe whitelist
# --------------------------------------------------------------------------


def check_universe(ctx: CheckContext) -> RiskCheckResult:
    """Spec §7.2 #2 — and the last line of defence against a hallucinated ticker.

    An LLM inventing a symbol is a real failure mode (spec §7.7), so a miss here
    is CRITICAL rather than a routine rejection.

    One deliberate relaxation: a *reduce-only* order is allowed on a symbol we
    actually hold even if it has since been removed from the universe. Otherwise
    dropping a name from the whitelist would trap the position we already own.
    """
    name = "universe_whitelist"
    if ctx.symbol in ctx.snapshot.universe:
        return allow(name)

    if ctx.reduce_only and not ctx.snapshot.position_of(ctx.symbol).is_flat:
        return allow(name)

    return reject(
        name,
        f"{ctx.symbol} is not in the approved universe "
        f"({len(ctx.snapshot.universe)} symbols). If a model produced this symbol, "
        "treat it as a hallucination.",
        alert_level=AlertLevel.CRITICAL,
        detail={"symbol": ctx.symbol},
    )


# --------------------------------------------------------------------------
# 3. Data quality
# --------------------------------------------------------------------------


def check_data_quality(ctx: CheckContext) -> RiskCheckResult:
    """Spec §7.2 #3. Degraded or stale data blocks new orders on that symbol.

    Reducing exposure is allowed to proceed with a warning: being stuck in a
    position whose feed has died is worse than closing it on imperfect data.
    The execution layer prices these conservatively rather than crossing blind.
    """
    name = "data_quality"
    quality = ctx.snapshot.quality_of(ctx.symbol)
    if quality is DataQuality.OK:
        return allow(name)

    if ctx.reduce_only:
        return RiskCheckResult(
            name=name,
            passed=True,
            action=RiskAction.ALLOW,
            reason=f"{ctx.symbol} data is {quality.value}; permitting exposure reduction only",
            alert_level=AlertLevel.WARN,
        )

    return reject(
        name,
        f"{ctx.symbol} market data is {quality.value}; no new orders until it recovers",
        alert_level=AlertLevel.WARN,
    )


# --------------------------------------------------------------------------
# 4. Trading hours
# --------------------------------------------------------------------------


def check_trading_hours(ctx: CheckContext) -> RiskCheckResult:
    """Spec §7.2 #4. Outside hours the order is queued rather than rejected —
    the intent is still valid, it just cannot be worked yet."""
    name = "trading_hours"
    is_open = ctx.snapshot.is_market_open.get(ctx.symbol)
    if is_open is None or is_open:
        return allow(name)
    return reject(
        name,
        f"{ctx.symbol} market is closed; queued until the next session",
        action=RiskAction.QUEUE,
    )


# --------------------------------------------------------------------------
# 5. Single-order notional
# --------------------------------------------------------------------------


def check_order_notional(ctx: CheckContext) -> RiskCheckResult:
    name = "order_notional"
    limit_pct = ctx.config.order.max_order_notional_pct
    actual_pct = ctx.notional_pct_of_equity()
    if actual_pct <= limit_pct:
        return allow(name)

    if ctx.price <= ZERO:
        return reject(name, "cannot size an order without a price")
    max_notional = ctx.equity * limit_pct / Decimal(100)
    return reduce_to(
        name,
        max_notional / ctx.price,
        f"order notional {ctx.notional:.2f} is {actual_pct:.2f}% of equity, "
        f"above the {limit_pct}% single-order limit",
    )


# --------------------------------------------------------------------------
# 6. Fat-finger limit price deviation
# --------------------------------------------------------------------------


def check_price_deviation(ctx: CheckContext) -> RiskCheckResult:
    """Spec §7.2 #6. A limit price far from the market is usually a mistake —
    a misplaced decimal point, a stale quote, or a model output nobody checked.

    Always enforced, including on reduce-only orders: a fat-fingered stop is
    still a fat finger, and one priced 90% away either never fills or fills
    catastrophically.
    """
    name = "price_deviation"
    limit_price = ctx.intent.limit_price
    if limit_price is None:
        return allow(name)

    market = ctx.snapshot.price_of(ctx.symbol)
    if market is None or market <= ZERO:
        return reject(
            name,
            f"no market price for {ctx.symbol}; cannot validate the limit price",
            alert_level=AlertLevel.WARN,
        )

    deviation = abs(as_pct(limit_price - market, market))
    threshold = ctx.config.order.limit_price_deviation_pct
    if deviation <= threshold:
        return allow(name)

    return reject(
        name,
        f"limit price {limit_price} is {deviation:.2f}% away from the market price "
        f"{market}, beyond the {threshold}% fat-finger threshold",
        alert_level=AlertLevel.CRITICAL,
        detail={"limit_price": str(limit_price), "market_price": str(market)},
    )


# --------------------------------------------------------------------------
# 7. ADV participation
# --------------------------------------------------------------------------


def check_adv_participation(ctx: CheckContext) -> RiskCheckResult:
    """Spec §7.2 #7. Too large a share of daily volume moves the price against
    you before the order is done."""
    name = "adv_participation"
    adv = ctx.snapshot.adv.get(ctx.symbol)
    if adv is None or adv <= ZERO:
        # No ADV estimate means no basis for the check. Rejecting outright would
        # block every newly listed name; the size and concentration checks still
        # bound the damage.
        return allow(name)

    participation = as_pct(abs(ctx.quantity), adv)
    limit_pct = ctx.config.order.max_adv_participation_pct
    if participation <= limit_pct:
        return allow(name)

    return reduce_to(
        name,
        adv * limit_pct / Decimal(100),
        f"order is {participation:.2f}% of ADV ({adv}), above the {limit_pct}% cap",
    )


# --------------------------------------------------------------------------
# 8. Post-fill position concentration
# --------------------------------------------------------------------------


def _post_fill_exposure(ctx: CheckContext) -> Decimal:
    """Exposure in this symbol assuming the order fills completely."""
    current = ctx.snapshot.position_of(ctx.symbol)
    signed_delta = ctx.quantity * ctx.intent.side.sign
    resulting_quantity = current.quantity + signed_delta
    return abs(resulting_quantity * ctx.price)


def check_position_concentration(ctx: CheckContext) -> RiskCheckResult:
    name = "position_concentration"
    limit_pct = ctx.config.position.max_position_pct
    resulting = _post_fill_exposure(ctx)
    resulting_pct = ctx.notional_pct_of_equity(resulting)
    if resulting_pct <= limit_pct:
        return allow(name)

    if ctx.price <= ZERO:
        return reject(name, "cannot evaluate concentration without a price")

    max_exposure = ctx.equity * limit_pct / Decimal(100)
    current = ctx.snapshot.position_of(ctx.symbol)
    headroom_shares = (max_exposure - abs(current.market_value)) / ctx.price
    return reduce_to(
        name,
        max(ZERO, headroom_shares),
        f"post-fill {ctx.symbol} exposure would be {resulting_pct:.2f}% of equity, "
        f"above the {limit_pct}% per-position cap",
    )


# --------------------------------------------------------------------------
# 9. Post-fill sector concentration
# --------------------------------------------------------------------------


def check_sector_concentration(ctx: CheckContext) -> RiskCheckResult:
    name = "sector_concentration"
    sector = ctx.snapshot.sector_of(ctx.symbol)
    limit_pct = ctx.config.position.max_sector_pct

    current_symbol_exposure = abs(ctx.snapshot.position_of(ctx.symbol).market_value)
    sector_total = ctx.snapshot.sector_exposure(sector)
    resulting_sector = sector_total - current_symbol_exposure + _post_fill_exposure(ctx)
    resulting_pct = ctx.notional_pct_of_equity(resulting_sector)

    if resulting_pct <= limit_pct:
        return allow(name)
    if ctx.price <= ZERO:
        return reject(name, "cannot evaluate sector concentration without a price")

    max_sector = ctx.equity * limit_pct / Decimal(100)
    others = sector_total - current_symbol_exposure
    headroom_shares = (max_sector - others - current_symbol_exposure) / ctx.price
    return reduce_to(
        name,
        max(ZERO, headroom_shares),
        f"post-fill {sector} exposure would be {resulting_pct:.2f}% of equity, "
        f"above the {limit_pct}% sector cap",
    )


# --------------------------------------------------------------------------
# 10. Gross leverage
# --------------------------------------------------------------------------


def check_leverage(ctx: CheckContext) -> RiskCheckResult:
    name = "leverage"
    limit = ctx.config.account.max_leverage
    current_symbol_exposure = abs(ctx.snapshot.position_of(ctx.symbol).market_value)
    gross = ctx.snapshot.gross_exposure() - current_symbol_exposure + _post_fill_exposure(ctx)
    leverage = safe_div(gross, ctx.equity)

    if leverage <= limit:
        return allow(name)
    return reject(
        name,
        f"post-fill gross leverage would be {leverage:.3f}x, above the {limit}x limit "
        f"(gross exposure {gross:.2f} on equity {ctx.equity:.2f})",
        alert_level=AlertLevel.WARN,
    )


# --------------------------------------------------------------------------
# 11. Daily loss limit
# --------------------------------------------------------------------------


def check_daily_loss(ctx: CheckContext) -> RiskCheckResult:
    """Spec §7.2 #11.

    Skipped entirely for reduce-only orders — see the note in
    :mod:`atrader.risk.checks.base`. A daily loss limit that blocks the
    stop-loss is worse than no limit at all.
    """
    name = "daily_loss_limit"
    limit_pct = ctx.config.account.daily_loss_limit_pct
    pnl_pct = ctx.snapshot.daily_pnl_pct
    if pnl_pct > -limit_pct:
        return allow(name)
    return reject(
        name,
        f"daily P&L is {pnl_pct:.2f}%, at or beyond the -{limit_pct}% limit; "
        "only exposure-reducing orders are permitted",
        alert_level=AlertLevel.CRITICAL,
    )


# --------------------------------------------------------------------------
# 12. Maximum drawdown
# --------------------------------------------------------------------------


def check_drawdown(ctx: CheckContext) -> RiskCheckResult:
    name = "max_drawdown"
    limit_pct = ctx.config.account.max_drawdown_pct
    drawdown = ctx.snapshot.drawdown_pct
    if drawdown < limit_pct:
        return allow(name)
    return reject(
        name,
        f"drawdown from peak is {drawdown:.2f}%, at or beyond the {limit_pct}% limit",
        alert_level=AlertLevel.CRITICAL,
    )


# --------------------------------------------------------------------------
# 13. Order rate
# --------------------------------------------------------------------------


def check_order_rate(ctx: CheckContext) -> RiskCheckResult:
    """Spec §7.2 #13. Throttled rather than rejected: the intent is fine, we
    just need to slow down. A runaway loop is also the usual cause, which is why
    §7.5 escalates on a sustained rate spike."""
    name = "order_rate"
    limit = ctx.config.order.max_orders_per_minute
    count = ctx.snapshot.orders_last_minute
    if count < limit:
        return allow(name)
    return reject(
        name,
        f"{count} orders in the last minute, at the {limit}/min throttle",
        action=RiskAction.THROTTLE,
        alert_level=AlertLevel.WARN,
    )


# --------------------------------------------------------------------------
# 14. Duplicate suppression
# --------------------------------------------------------------------------


def check_duplicate(ctx: CheckContext) -> RiskCheckResult:
    """Spec §7.2 #14.

    A near-identical order moments after the last one is far more often a retry
    bug or a strategy re-firing on the same bar than a genuine second trade.
    Applies to reduce-only too — double-closing a position opens a short.
    """
    name = "duplicate_order"
    window_ns = ctx.config.order.duplicate_window_seconds * 1_000_000_000
    if window_ns <= 0:
        return allow(name)

    for recent in ctx.snapshot.recent_orders:
        if recent.symbol != ctx.symbol or recent.side is not ctx.intent.side:
            continue
        age_ns = ctx.snapshot.now_ns - recent.at_ns
        if 0 <= age_ns <= window_ns:
            return reject(
                name,
                f"a {recent.side.value} order for {ctx.symbol} was sent "
                f"{age_ns / 1_000_000_000:.2f}s ago, inside the "
                f"{ctx.config.order.duplicate_window_seconds}s duplicate window",
                alert_level=AlertLevel.WARN,
            )
    return allow(name)


# --------------------------------------------------------------------------
# 15. Self-cross
# --------------------------------------------------------------------------


def check_self_cross(ctx: CheckContext) -> RiskCheckResult:
    """Spec §7.2 #15 — a regulatory matter, not just an efficiency one.

        내 매수 주문과 내 매도 주문이 시장에서 서로 체결되면 시세조종으로 오인될 수 있다.

    Enforced in both directions, including reduce-only: a wash trade is a wash
    trade regardless of which side initiated it. The fix is to cancel the
    opposing order and net internally, which the execution layer does.
    """
    name = "self_cross"
    opposing = [
        order
        for order in ctx.snapshot.open_orders
        if order.symbol == ctx.symbol and order.side is not ctx.intent.side
    ]
    if not opposing:
        return allow(name)

    total = sum((order.quantity for order in opposing), ZERO)
    return reject(
        name,
        f"{len(opposing)} opposing {opposing[0].side.value} order(s) for {ctx.symbol} "
        f"totalling {total} are already working; cancel and net them before sending "
        "the other side, or the two could cross in the market",
        alert_level=AlertLevel.WARN,
    )


# --------------------------------------------------------------------------
# Registry — evaluation order is the spec's order
# --------------------------------------------------------------------------

#: Checks 1-4. These read no quantity and no price, so they run **before**
#: sizing. That ordering is load-bearing rather than cosmetic: a hallucinated
#: symbol has no market price, and if sizing ran first it would be rejected for
#: "no reference price" — a routine-looking WARN — instead of the CRITICAL
#: universe violation that tells an operator a model invented a ticker.
CATEGORICAL_CHECKS: tuple[RiskCheck, ...] = (
    RiskCheck("system_state", check_system_state, True, "System is accepting orders"),
    RiskCheck("universe_whitelist", check_universe, True, "Symbol is approved for trading"),
    RiskCheck("data_quality", check_data_quality, True, "Market data is trustworthy"),
    RiskCheck("trading_hours", check_trading_hours, True, "Market is open"),
)

#: Checks 5-15. These need a resolved quantity and price, and any of them may
#: shrink the order, so they run in the reduction loop.
QUANTITATIVE_CHECKS: tuple[RiskCheck, ...] = (
    RiskCheck("order_notional", check_order_notional, False, "Single-order size cap"),
    RiskCheck("price_deviation", check_price_deviation, True, "Fat-finger guard"),
    RiskCheck("adv_participation", check_adv_participation, False, "Share of daily volume"),
    RiskCheck("position_concentration", check_position_concentration, False, "Per-symbol cap"),
    RiskCheck("sector_concentration", check_sector_concentration, False, "Per-sector cap"),
    RiskCheck("leverage", check_leverage, False, "Gross leverage cap"),
    RiskCheck("daily_loss_limit", check_daily_loss, False, "Daily loss circuit"),
    RiskCheck("max_drawdown", check_drawdown, False, "Drawdown circuit"),
    RiskCheck("order_rate", check_order_rate, False, "Orders per minute"),
    RiskCheck("duplicate_order", check_duplicate, True, "Duplicate suppression"),
    RiskCheck("self_cross", check_self_cross, True, "Self-trade prevention"),
)

#: All fifteen, in the spec's order.
PRETRADE_CHECKS: tuple[RiskCheck, ...] = (*CATEGORICAL_CHECKS, *QUANTITATIVE_CHECKS)


def check_names() -> tuple[str, ...]:
    return tuple(check.name for check in PRETRADE_CHECKS)
