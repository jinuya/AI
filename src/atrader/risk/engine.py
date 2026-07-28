"""The risk engine — spec §7.1.

    리스크 엔진은 **주문 경로상의 필수 통과 지점**이다. 라이브러리로 호출하는 게
    아니라, ``TradingIntent``를 받아 ``Order``를 내보내는 유일한 컴포넌트로 만든다.

This class is the only thing in the system that constructs an :class:`Order`.
Strategies cannot import the broker adapter or the execution layer at all
(enforced statically — see ``tests/unit/test_import_boundaries.py``), so the
only route from an intent to the market runs through :meth:`RiskEngine.evaluate`.

**Fail-closed.** Spec §7.1: if the engine cannot answer, orders are rejected; if
its configuration will not load, the process refuses to boot. Every unexpected
exception inside evaluation is caught and converted into a rejection rather than
propagating — a crash in a risk check must not become a bypass, and "the checker
threw, so the order went out" is the single worst failure this system could have.

**Every decision is audited, including the passes.** Spec §9.3 asks for all risk
check results, not just the rejections. When a bad order does get through, the
question is *which check let it through*, and that is only answerable if the
passes were recorded too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from uuid import UUID

from atrader.audit.logger import AuditEvent, AuditLogger
from atrader.config.schema import AppConfig, InstrumentSpec, RiskConfig
from atrader.core.clock import Clock
from atrader.core.errors import FailClosedError
from atrader.core.ids import IdGenerator
from atrader.core.models import Order, TradingIntent
from atrader.core.money import ZERO, floor_to_lot, round_to_tick
from atrader.core.types import (
    AlertLevel,
    OrderStatus,
    OrderType,
    RiskAction,
    Side,
    TargetType,
)
from atrader.risk.approval import ApprovalGate, ApprovalKind, needs_approval
from atrader.risk.checks.base import CheckContext, RiskCheckResult
from atrader.risk.checks.pretrade import CATEGORICAL_CHECKS, QUANTITATIVE_CHECKS
from atrader.risk.killswitch import KillSwitch
from atrader.risk.state import RiskSnapshot

__all__ = ["RiskDecision", "RiskEngine"]

#: Spec §8.4 budget for the inline pre-trade path.
PRETRADE_LATENCY_BUDGET_MS = 5.0


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """The engine's verdict on one intent."""

    risk_check_id: UUID
    intent_id: UUID
    action: RiskAction
    order: Order | None = None
    results: tuple[RiskCheckResult, ...] = ()
    reason: str = ""
    alert_level: AlertLevel | None = None
    latency_ms: float = 0.0
    approval_request_id: UUID | None = None
    original_quantity: Decimal = ZERO
    final_quantity: Decimal = ZERO

    @property
    def approved(self) -> bool:
        return self.order is not None

    @property
    def was_reduced(self) -> bool:
        return self.approved and self.final_quantity < self.original_quantity

    @property
    def failed_checks(self) -> tuple[RiskCheckResult, ...]:
        return tuple(result for result in self.results if not result.passed)

    def summary(self) -> str:
        if self.approved:
            note = " (size reduced)" if self.was_reduced else ""
            return f"approved{note}: {self.final_quantity} shares"
        return f"{self.action.value}: {self.reason}"


@dataclass
class RiskEngine:
    """The mandatory gate between intents and orders."""

    config: AppConfig
    clock: Clock
    ids: IdGenerator
    kill_switch: KillSwitch
    approvals: ApprovalGate | None = None
    audit: AuditLogger | None = None
    max_reduce_iterations: int = 4
    _known_symbols: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        # Fail-closed at construction: a system that boots without limits would
        # trade without them (spec §7.1).
        if self.config.risk is None:  # pragma: no cover — pydantic guarantees this
            raise FailClosedError("risk configuration is missing; refusing to start")
        self._known_symbols = set(self.config.universe.symbols)

    @property
    def risk_config(self) -> RiskConfig:
        return self.config.risk

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        intent: TradingIntent,
        snapshot: RiskSnapshot,
        *,
        reduce_only: bool = False,
        approval_request_id: UUID | None = None,
    ) -> RiskDecision:
        """Turn an intent into an order, or explain why not.

        Never raises. Any unexpected failure becomes a rejection — see the
        fail-closed note in the module docstring.
        """
        started = self.clock.monotonic_ns()
        risk_check_id = self.ids.new_id()

        try:
            decision = self._evaluate(
                risk_check_id, intent, snapshot, reduce_only, approval_request_id
            )
        except Exception as exc:
            # Catching everything is the entire point: a crashing check must
            # never become a bypass. Fail-closed (spec §7.1).
            decision = RiskDecision(
                risk_check_id=risk_check_id,
                intent_id=intent.intent_id,
                action=RiskAction.REJECT,
                reason=(
                    f"risk evaluation raised {type(exc).__name__}: {exc}. "
                    "Rejecting fail-closed — a crashing check must never become a bypass."
                ),
                alert_level=AlertLevel.CRITICAL,
            )

        latency_ms = (self.clock.monotonic_ns() - started) / 1_000_000
        decision = RiskDecision(
            risk_check_id=decision.risk_check_id,
            intent_id=decision.intent_id,
            action=decision.action,
            order=decision.order,
            results=decision.results,
            reason=decision.reason,
            alert_level=decision.alert_level,
            latency_ms=latency_ms,
            approval_request_id=decision.approval_request_id,
            original_quantity=decision.original_quantity,
            final_quantity=decision.final_quantity,
        )
        self._audit(intent, decision)
        return decision

    def _evaluate(
        self,
        risk_check_id: UUID,
        intent: TradingIntent,
        snapshot: RiskSnapshot,
        reduce_only: bool,
        approval_request_id: UUID | None,
    ) -> RiskDecision:
        # The kill switch is checked before anything else, including config
        # reads and snapshot use. Spec §FR-MON-03: it takes precedence over all
        # other logic, so nothing may run ahead of it.
        if self.kill_switch.is_engaged and not reduce_only:
            return RiskDecision(
                risk_check_id=risk_check_id,
                intent_id=intent.intent_id,
                action=RiskAction.REJECT,
                reason="kill switch engaged",
                alert_level=AlertLevel.CRITICAL,
            )

        if intent.is_expired(snapshot.now_ns):
            return RiskDecision(
                risk_check_id=risk_check_id,
                intent_id=intent.intent_id,
                action=RiskAction.REJECT,
                reason=f"intent expired at {intent.valid_until_ns}, now {snapshot.now_ns}",
            )

        instrument = self.config.instrument(intent.symbol)

        # Checks 1-4 run before sizing. A hallucinated symbol has no market
        # price, so sizing first would reject it as "no reference price" — a
        # routine WARN — and bury the CRITICAL universe violation that tells an
        # operator a model invented a ticker.
        categorical_ctx = CheckContext(
            intent=intent,
            snapshot=snapshot,
            config=self.risk_config,
            quantity=ZERO,
            price=snapshot.price_of(intent.symbol) or ZERO,
            instrument=instrument,
            reduce_only=reduce_only,
        )
        categorical_results: list[RiskCheckResult] = []
        for check in CATEGORICAL_CHECKS:
            result = check(categorical_ctx)
            categorical_results.append(result)
            if result.is_blocking:
                return RiskDecision(
                    risk_check_id=risk_check_id,
                    intent_id=intent.intent_id,
                    action=result.action,
                    results=tuple(categorical_results),
                    reason=result.reason,
                    alert_level=result.alert_level,
                )

        price = self._reference_price(intent, snapshot)
        if price is None or price <= ZERO:
            return RiskDecision(
                risk_check_id=risk_check_id,
                intent_id=intent.intent_id,
                action=RiskAction.REJECT,
                results=tuple(categorical_results),
                reason=f"no usable reference price for {intent.symbol}",
                alert_level=AlertLevel.WARN,
            )

        quantity = self._resolve_quantity(intent, snapshot, price, instrument)
        if quantity <= ZERO:
            return RiskDecision(
                risk_check_id=risk_check_id,
                intent_id=intent.intent_id,
                action=RiskAction.REJECT,
                results=tuple(categorical_results),
                reason="resolved order quantity is zero",
            )

        original_quantity = quantity
        ctx = CheckContext(
            intent=intent,
            snapshot=snapshot,
            config=self.risk_config,
            quantity=quantity,
            price=price,
            instrument=instrument,
            reduce_only=reduce_only,
        )

        results, ctx, blocked = self._run_checks(ctx)
        results = [*categorical_results, *results]
        if blocked is not None:
            return RiskDecision(
                risk_check_id=risk_check_id,
                intent_id=intent.intent_id,
                action=blocked.action,
                results=tuple(results),
                reason=blocked.reason,
                alert_level=blocked.alert_level,
                original_quantity=original_quantity,
            )

        # Human approval (spec §7.6) is evaluated after the checks: there is no
        # point waking someone up for an order that would have been rejected.
        approval = self._approval_needed(ctx, reduce_only)
        if approval is not None and not self._approval_satisfied(approval_request_id):
            request_id = approval_request_id
            if self.approvals is not None and request_id is None:
                request = self.approvals.request(
                    approval,
                    f"{intent.side.value} {ctx.quantity} {intent.symbol} "
                    f"(~{ctx.notional:.2f}, {ctx.notional_pct_of_equity():.1f}% of equity)",
                    context={"intent_id": str(intent.intent_id), "symbol": intent.symbol},
                )
                request_id = request.request_id
            return RiskDecision(
                risk_check_id=risk_check_id,
                intent_id=intent.intent_id,
                action=RiskAction.REQUIRE_APPROVAL,
                results=tuple(results),
                reason=f"{approval.value} requires human approval before execution",
                alert_level=AlertLevel.WARN,
                approval_request_id=request_id,
                original_quantity=original_quantity,
            )

        order = self._build_order(risk_check_id, intent, ctx, instrument, reduce_only)
        return RiskDecision(
            risk_check_id=risk_check_id,
            intent_id=intent.intent_id,
            action=RiskAction.ALLOW,
            order=order,
            results=tuple(results),
            reason="all pre-trade checks passed",
            original_quantity=original_quantity,
            final_quantity=ctx.quantity,
        )

    def _run_checks(
        self, ctx: CheckContext
    ) -> tuple[list[RiskCheckResult], CheckContext, RiskCheckResult | None]:
        """Run every check, applying reductions and re-running from the top.

        A reduction can change the answer to a check that already passed — a
        smaller order might now pass ADV but the *new* size still has to be
        re-tested against concentration. Re-running from the start is the only
        way to be sure the final size satisfies all fifteen simultaneously.
        The iteration cap stops two checks reducing each other forever.
        """
        collected: list[RiskCheckResult] = []

        for _ in range(self.max_reduce_iterations):
            pass_results: list[RiskCheckResult] = []
            reduced: RiskCheckResult | None = None

            for check in QUANTITATIVE_CHECKS:
                result = check(ctx)
                pass_results.append(result)
                if result.action is RiskAction.REDUCE:
                    reduced = result
                    break
                if result.is_blocking:
                    return [*collected, *pass_results], ctx, result

            collected.extend(pass_results)
            if reduced is None:
                return collected, ctx, None

            new_quantity = reduced.adjusted_quantity or ZERO
            if ctx.instrument is not None:
                new_quantity = floor_to_lot(new_quantity, ctx.instrument.lot_size)
            if new_quantity <= ZERO:
                return (
                    collected,
                    ctx,
                    RiskCheckResult(
                        name=reduced.name,
                        passed=False,
                        action=RiskAction.REJECT,
                        reason=f"{reduced.reason}; no tradable size remains after reduction",
                    ),
                )
            if new_quantity >= ctx.quantity:
                # Defensive: a "reduction" that does not reduce would loop.
                return (
                    collected,
                    ctx,
                    RiskCheckResult(
                        name=reduced.name,
                        passed=False,
                        action=RiskAction.REJECT,
                        reason=f"{reduced.reason}; reduction did not shrink the order",
                    ),
                )
            ctx = ctx.with_quantity(new_quantity)

        return (
            collected,
            ctx,
            RiskCheckResult(
                name="reduce_convergence",
                passed=False,
                action=RiskAction.REJECT,
                reason=(
                    f"size did not converge within {self.max_reduce_iterations} reductions; "
                    "rejecting rather than shipping an order no check agreed on"
                ),
                alert_level=AlertLevel.WARN,
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reference_price(self, intent: TradingIntent, snapshot: RiskSnapshot) -> Decimal | None:
        """Market price if we have one, else the intent's own limit.

        Order matters: using the intent's limit price when a market price exists
        would let a strategy set its own reference and walk straight past the
        fat-finger check.
        """
        market = snapshot.price_of(intent.symbol)
        if market is not None and market > ZERO:
            return market
        return intent.limit_price

    def _resolve_quantity(
        self,
        intent: TradingIntent,
        snapshot: RiskSnapshot,
        price: Decimal,
        instrument: InstrumentSpec | None,
    ) -> Decimal:
        """Convert the intent's target into a share count."""
        if intent.target_type is TargetType.SHARES:
            quantity = intent.target_value
        elif intent.target_type is TargetType.NOTIONAL:
            quantity = intent.target_value / price
        else:  # TARGET_WEIGHT
            target_value = snapshot.account.equity * intent.target_value
            current_value = snapshot.position_of(intent.symbol).market_value
            delta = target_value - current_value
            # Deadband: without it, tiny weight drift causes constant trading
            # and the commission eats the account (spec §FR-PF-03).
            deadband = (
                snapshot.account.equity
                * self.config.execution.rebalance_deadband_pct
                / Decimal(100)
            )
            if abs(delta) < deadband:
                return ZERO
            quantity = abs(delta) / price

        lot_size = instrument.lot_size if instrument is not None else Decimal(1)
        return floor_to_lot(abs(quantity), lot_size)

    def _approval_needed(self, ctx: CheckContext, reduce_only: bool) -> ApprovalKind | None:
        if reduce_only:
            # Never gate a liquidation behind a human. Spec §7.4's principle
            # applies here too: the thing that reduces risk must not be blocked.
            return None
        is_new_symbol = ctx.symbol not in self._known_symbols
        return needs_approval(
            ctx.notional_pct_of_equity(),
            self.risk_config.order.human_approval_threshold_pct,
            is_new_symbol=is_new_symbol,
        )

    def _approval_satisfied(self, approval_request_id: UUID | None) -> bool:
        if approval_request_id is None:
            return False
        if self.approvals is None:
            return False
        return self.approvals.is_granted(approval_request_id)

    def _build_order(
        self,
        risk_check_id: UUID,
        intent: TradingIntent,
        ctx: CheckContext,
        instrument: InstrumentSpec | None,
        reduce_only: bool,
    ) -> Order:
        """Construct the order. The only place in the system that does this."""
        order_type = self._order_type(intent)
        limit_price = intent.limit_price

        if order_type is OrderType.LIMIT and limit_price is None:
            limit_price = self._aggressive_limit(intent.side, ctx.price, instrument)
        if limit_price is not None and instrument is not None:
            limit_price = round_to_tick(limit_price, instrument.tick_size, side=intent.side.value)

        now = self.clock.now_ns()
        return Order(
            order_id=self.ids.new_id(),
            client_order_id=str(self.ids.new_id()),
            parent_intent_id=intent.intent_id,
            strategy_id=intent.strategy_id,
            symbol=intent.symbol,
            side=intent.side,
            order_type=order_type,
            quantity=ctx.quantity,
            limit_price=limit_price if order_type is not OrderType.MARKET else None,
            time_in_force=intent.time_in_force,
            status=OrderStatus.PENDING_NEW,
            risk_check_id=risk_check_id,
            reduce_only=reduce_only,
            created_at_ns=now,
            updated_at_ns=now,
        )

    def _order_type(self, intent: TradingIntent) -> OrderType:
        """Spec §FR-EXE-02: market orders are disabled by default.

        In a thin name the slippage on a market order is unbounded, so an
        aggressive limit is the default instead — it crosses the spread but
        stops at a price you chose.
        """
        if intent.limit_price is not None:
            return OrderType.LIMIT
        if self.risk_config.order.allow_market_orders:
            return OrderType.MARKET
        return OrderType.LIMIT

    def _aggressive_limit(
        self, side: Side, price: Decimal, instrument: InstrumentSpec | None
    ) -> Decimal:
        """A limit priced N ticks through the touch (spec §FR-EXE-02)."""
        tick = instrument.tick_size if instrument is not None else Decimal("0.01")
        offset = tick * Decimal(self.risk_config.order.aggressive_limit_ticks)
        return price + offset if side is Side.BUY else max(tick, price - offset)

    def _audit(self, intent: TradingIntent, decision: RiskDecision) -> None:
        """Record the decision — every check, pass or fail (spec §9.3)."""
        if self.audit is None:
            return
        event = AuditEvent.RISK_DECISION if decision.approved else AuditEvent.RISK_CHECK_REJECTED
        self.audit.append(
            event,
            actor=intent.strategy_id,
            payload={
                "risk_check_id": str(decision.risk_check_id),
                "intent_id": str(intent.intent_id),
                "symbol": intent.symbol,
                "side": intent.side.value,
                "action": decision.action.value,
                "reason": decision.reason,
                "latency_ms": f"{decision.latency_ms:.3f}",
                "original_quantity": str(decision.original_quantity),
                "final_quantity": str(decision.final_quantity),
                "order_id": str(decision.order.order_id) if decision.order else None,
                # Passes are recorded too: when a bad order gets through, the
                # question is which check let it through.
                "checks": [
                    {
                        "name": result.name,
                        "passed": str(result.passed),
                        "action": result.action.value,
                        "reason": result.reason,
                    }
                    for result in decision.results
                ],
            },
        )


def measure_latency_budget(decision: RiskDecision) -> bool:
    """Whether a decision met the §8.4 p99 budget of 5ms."""
    return decision.latency_ms <= PRETRADE_LATENCY_BUDGET_MS
