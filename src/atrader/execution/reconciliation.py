"""Reconciliation — spec §FR-EXE-05.

    최소 30초 주기, 그리고 시작 시 무조건 한 번. 브로커의 포지션·미체결주문·현금 잔고를
    조회해서 로컬 상태와 대조한다. 불일치(break)가 발견되면 즉시 신규 주문을 중단하고
    알림을 발송한다. **자동 수정은 금지 — 사람이 확인해야 한다.**

The no-auto-correct rule is the important one, and it is counter-intuitive: the
system knows the broker is the source of truth (spec §2.2), so why not just copy
the broker's numbers over the local ones?

Because a break means a *bug*, and the position mismatch is a symptom rather
than the disease. Silently adopting the broker's state fixes the symptom, hides
the bug, and lets whatever caused it keep running. The next break will be
larger. Stopping and calling a person is how the cause gets found.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal

from atrader.audit.logger import AuditEvent, AuditLogger
from atrader.brokers.models import OrderState
from atrader.brokers.protocol import BrokerAdapter
from atrader.core.clock import NS_PER_SECOND, Clock
from atrader.core.models import AccountState, Order, Position
from atrader.core.money import ZERO
from atrader.core.types import AlertLevel
from atrader.storage.protocol import OrderStore, PositionStore

__all__ = [
    "Break",
    "BreakKind",
    "Reconciler",
    "ReconciliationReport",
    "expected_positions_from_fills",
]

DEFAULT_INTERVAL_SECONDS = 30
DEFAULT_QUANTITY_TOLERANCE = Decimal("0")
DEFAULT_CASH_TOLERANCE = Decimal("0.01")


class BreakKind:
    POSITION_QUANTITY = "position_quantity"
    POSITION_MISSING_LOCALLY = "position_missing_locally"
    POSITION_MISSING_AT_BROKER = "position_missing_at_broker"
    ORDER_MISSING_LOCALLY = "order_missing_locally"
    ORDER_MISSING_AT_BROKER = "order_missing_at_broker"
    ORDER_QUANTITY = "order_filled_quantity"
    CASH = "cash"
    CURRENCY = "currency"


@dataclass(frozen=True, slots=True)
class Break:
    """One disagreement between us and the broker."""

    kind: str
    identifier: str
    local_value: str
    broker_value: str
    detail: str = ""

    @property
    def message(self) -> str:
        return (
            f"{self.kind} on {self.identifier}: local={self.local_value} "
            f"broker={self.broker_value}. {self.detail}".strip()
        )


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    at_ns: int
    breaks: tuple[Break, ...] = ()
    positions_checked: int = 0
    orders_checked: int = 0
    duration_ms: float = 0.0

    @property
    def is_clean(self) -> bool:
        return not self.breaks

    @property
    def alert_level(self) -> AlertLevel:
        return AlertLevel.INFO if self.is_clean else AlertLevel.CRITICAL

    def summary(self) -> str:
        if self.is_clean:
            return (
                f"clean: {self.positions_checked} position(s), "
                f"{self.orders_checked} order(s) agree with the broker"
            )
        return f"{len(self.breaks)} break(s): " + "; ".join(b.message for b in self.breaks)


@dataclass
class Reconciler:
    """Compares local state against the broker on a timer."""

    broker: BrokerAdapter
    orders: OrderStore
    positions: PositionStore
    clock: Clock
    audit: AuditLogger | None = None
    local_account: Callable[[], AccountState | None] | None = None
    """Our own view of the account, if we keep one.

    Left unset in a pure paper configuration where the broker is the only
    bookkeeper — there is nothing to disagree with, and inventing a local figure
    just to compare against itself would produce a check that can never fail.
    """
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS
    quantity_tolerance: Decimal = DEFAULT_QUANTITY_TOLERANCE
    cash_tolerance: Decimal = DEFAULT_CASH_TOLERANCE
    _last_run_ns: int | None = None
    _break_active: bool = field(default=False)

    @property
    def has_active_break(self) -> bool:
        """True while a break is unresolved. Blocks new orders."""
        return self._break_active

    def clear_break(self, *, by: str, note: str = "") -> None:
        """Mark a break resolved. Requires human approval (spec §7.6)."""
        self._break_active = False
        if self.audit is not None:
            self.audit.append(
                AuditEvent.MANUAL_INTERVENTION,
                actor=by,
                payload={"action": "reconciliation_break_cleared", "note": note},
            )

    def is_due(self, now_ns: int | None = None) -> bool:
        now = now_ns if now_ns is not None else self.clock.now_ns()
        if self._last_run_ns is None:
            return True  # spec §FR-EXE-05: always once at startup
        return now - self._last_run_ns >= self.interval_seconds * NS_PER_SECOND

    async def reconcile(self) -> ReconciliationReport:
        """Run one comparison. Never mutates local state."""
        started = self.clock.monotonic_ns()
        now = self.clock.now_ns()
        breaks: list[Break] = []

        broker_positions = {p.symbol: p for p in await self.broker.get_positions()}
        local_positions = {p.symbol: p for p in self.positions.all_positions() if not p.is_flat}
        breaks.extend(self._compare_positions(local_positions, broker_positions))

        broker_orders = {o.client_order_id: o for o in await self.broker.get_open_orders()}
        local_orders = {o.client_order_id: o for o in self.orders.open_orders()}
        breaks.extend(self._compare_orders(local_orders, broker_orders))

        account = await self.broker.get_account()
        breaks.extend(self._compare_cash(account))

        self._last_run_ns = now
        report = ReconciliationReport(
            at_ns=now,
            breaks=tuple(breaks),
            positions_checked=len(broker_positions | local_positions),
            orders_checked=len(broker_orders | local_orders),
            duration_ms=(self.clock.monotonic_ns() - started) / 1_000_000,
        )

        if not report.is_clean:
            # Latched: it stays set until a human clears it. Auto-correcting
            # here would hide the bug that caused the break.
            self._break_active = True

        if self.audit is not None:
            self.audit.append(
                AuditEvent.RECONCILIATION_BREAK
                if not report.is_clean
                else AuditEvent.RECONCILIATION_OK,
                payload={
                    "breaks": [b.message for b in report.breaks],
                    "positions_checked": str(report.positions_checked),
                    "orders_checked": str(report.orders_checked),
                    "duration_ms": f"{report.duration_ms:.3f}",
                },
            )
        return report

    def _compare_positions(
        self, local: dict[str, Position], broker: dict[str, Position]
    ) -> list[Break]:
        breaks: list[Break] = []
        for symbol in sorted(set(local) | set(broker)):
            ours = local.get(symbol)
            theirs = broker.get(symbol)

            if ours is None and theirs is not None:
                breaks.append(
                    Break(
                        BreakKind.POSITION_MISSING_LOCALLY,
                        symbol,
                        "flat",
                        str(theirs.quantity),
                        "the broker holds a position we do not know about",
                    )
                )
            elif ours is not None and theirs is None:
                breaks.append(
                    Break(
                        BreakKind.POSITION_MISSING_AT_BROKER,
                        symbol,
                        str(ours.quantity),
                        "flat",
                        "we believe we hold this and the broker does not",
                    )
                )
            elif ours is not None and theirs is not None:
                difference = abs(ours.quantity - theirs.quantity)
                if difference > self.quantity_tolerance:
                    breaks.append(
                        Break(
                            BreakKind.POSITION_QUANTITY,
                            symbol,
                            str(ours.quantity),
                            str(theirs.quantity),
                            f"differ by {difference}",
                        )
                    )
        return breaks

    def _compare_orders(
        self, local: dict[str, Order], broker: dict[str, OrderState]
    ) -> list[Break]:
        breaks: list[Break] = []
        for coid in sorted(set(local) | set(broker)):
            ours = local.get(coid)
            theirs = broker.get(coid)

            if ours is None and theirs is not None:
                breaks.append(
                    Break(
                        BreakKind.ORDER_MISSING_LOCALLY,
                        coid,
                        "unknown",
                        theirs.status.value,
                        "an order is working at the broker that we have no record of — "
                        "this is what a duplicate submission looks like",
                    )
                )
            elif ours is not None and theirs is None:
                breaks.append(
                    Break(
                        BreakKind.ORDER_MISSING_AT_BROKER,
                        coid,
                        ours.status.value,
                        "absent",
                        "we believe this order is working and the broker does not",
                    )
                )
            elif (
                ours is not None
                and theirs is not None
                and abs(ours.filled_quantity - theirs.filled_quantity) > self.quantity_tolerance
            ):
                breaks.append(
                    Break(
                        BreakKind.ORDER_QUANTITY,
                        coid,
                        str(ours.filled_quantity),
                        str(theirs.filled_quantity),
                        "filled quantities disagree — a fill was missed or double-counted",
                    )
                )
        return breaks

    def _compare_cash(self, account: AccountState) -> list[Break]:
        """Compare the broker's cash balance against our own ledger.

        Only *cash* is compared, not equity. Equity moves with marks, and our
        mark and the broker's are taken at different instants — comparing them
        would produce breaks that mean nothing except "the price moved", and a
        check that cries wolf is worse than no check at all. Cash changes only
        on a fill, a fee or a transfer, so a disagreement there is a real one.
        """
        if self.local_account is None:
            return []
        ours = self.local_account()
        if ours is None:
            return []

        breaks: list[Break] = []
        if ours.currency != account.currency:
            # Comparing numbers across currencies is worse than not comparing.
            breaks.append(
                Break(
                    BreakKind.CURRENCY,
                    "account",
                    ours.currency,
                    account.currency,
                    "account currency disagrees — the cash comparison was skipped",
                )
            )
            return breaks

        difference = abs(ours.cash - account.cash)
        if difference > self.cash_tolerance:
            breaks.append(
                Break(
                    BreakKind.CASH,
                    "account",
                    str(ours.cash),
                    str(account.cash),
                    f"differ by {difference} {account.currency} — a fill or fee is unaccounted for",
                )
            )
        return breaks


def expected_positions_from_fills(fills: list[tuple[str, Decimal]]) -> dict[str, Decimal]:
    """Rebuild expected positions from fills, for startup recovery (spec §10.4)."""
    positions: dict[str, Decimal] = {}
    for symbol, signed_quantity in fills:
        positions[symbol] = positions.get(symbol, ZERO) + signed_quantity
    return {symbol: quantity for symbol, quantity in positions.items() if quantity != ZERO}
