"""Human approval gate — spec §7.6.

Five situations stop automation and ask a person:

* a single order above ``human_approval_threshold_pct`` of the account
* the first ever trade in a symbol new to the universe
* any risk limit change
* clearing an L2 or L3 circuit breaker
* resolving a reconciliation break

Requests expire, and the expiry **denies**:

    승인 요청은 타임아웃(기본 5분)을 두고, 시간이 지나면 자동 거부한다.
    응답이 없는데 나중에 실행되는 게 더 위험하다.

That default direction is the whole design. An approval that sits unanswered
and then fires forty minutes later executes against a market nobody was looking
at, on a decision nobody consciously made. Timing out into *denial* means the
worst case of an unattended system is that it does nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from uuid import UUID

from atrader.core.clock import NS_PER_SECOND, Clock
from atrader.core.errors import ApprovalTimeoutError
from atrader.core.ids import IdGenerator

__all__ = ["ApprovalGate", "ApprovalKind", "ApprovalRequest", "ApprovalStatus"]

DEFAULT_TIMEOUT_SECONDS = 300


class ApprovalKind(StrEnum):
    LARGE_ORDER = "large_order"
    NEW_SYMBOL = "new_symbol"
    LIMIT_CHANGE = "limit_change"
    BREAKER_RESET = "breaker_reset"
    RECONCILIATION_BREAK = "reconciliation_break"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    TIMED_OUT = "timed_out"


@dataclass(slots=True)
class ApprovalRequest:
    """One outstanding request."""

    request_id: UUID
    kind: ApprovalKind
    summary: str
    requested_at_ns: int
    expires_at_ns: int
    status: ApprovalStatus = ApprovalStatus.PENDING
    decided_at_ns: int | None = None
    decided_by: str | None = None
    decision_note: str = ""
    context: dict[str, str] = field(default_factory=dict)

    @property
    def is_pending(self) -> bool:
        return self.status is ApprovalStatus.PENDING

    @property
    def is_granted(self) -> bool:
        return self.status is ApprovalStatus.GRANTED

    def seconds_remaining(self, now_ns: int) -> float:
        return max(0.0, (self.expires_at_ns - now_ns) / NS_PER_SECOND)


class ApprovalGate:
    """Tracks pending approvals and expires them into denial."""

    __slots__ = ("_clock", "_ids", "_requests", "_timeout_ns")

    def __init__(
        self,
        clock: Clock,
        ids: IdGenerator,
        *,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("approval timeout must be positive")
        self._clock = clock
        self._ids = ids
        self._timeout_ns = timeout_seconds * NS_PER_SECOND
        self._requests: dict[UUID, ApprovalRequest] = {}

    def request(
        self,
        kind: ApprovalKind,
        summary: str,
        *,
        context: dict[str, str] | None = None,
    ) -> ApprovalRequest:
        now = self._clock.now_ns()
        request = ApprovalRequest(
            request_id=self._ids.new_id(),
            kind=kind,
            summary=summary,
            requested_at_ns=now,
            expires_at_ns=now + self._timeout_ns,
            context=dict(context or {}),
        )
        self._requests[request.request_id] = request
        return request

    def get(self, request_id: UUID) -> ApprovalRequest | None:
        request = self._requests.get(request_id)
        if request is not None:
            self._expire_if_due(request)
        return request

    def pending(self) -> list[ApprovalRequest]:
        self.sweep()
        return [r for r in self._requests.values() if r.is_pending]

    def grant(self, request_id: UUID, *, by: str, note: str = "") -> ApprovalRequest:
        request = self._require(request_id)
        self._expire_if_due(request)
        if not request.is_pending:
            raise ApprovalTimeoutError(
                f"approval {request_id} is already {request.status.value}; it cannot be granted now"
            )
        request.status = ApprovalStatus.GRANTED
        request.decided_at_ns = self._clock.now_ns()
        request.decided_by = by
        request.decision_note = note
        return request

    def deny(self, request_id: UUID, *, by: str, note: str = "") -> ApprovalRequest:
        request = self._require(request_id)
        self._expire_if_due(request)
        if not request.is_pending:
            return request
        request.status = ApprovalStatus.DENIED
        request.decided_at_ns = self._clock.now_ns()
        request.decided_by = by
        request.decision_note = note
        return request

    def is_granted(self, request_id: UUID) -> bool:
        """Whether the action may proceed *right now*.

        Re-checked at the point of use, not just at decision time — an approval
        granted before the market moved is not consent for the market as it is
        after.
        """
        request = self.get(request_id)
        return request is not None and request.is_granted

    def sweep(self) -> list[ApprovalRequest]:
        """Expire everything past its deadline. Returns what timed out."""
        expired: list[ApprovalRequest] = []
        for request in self._requests.values():
            if self._expire_if_due(request):
                expired.append(request)
        return expired

    def _expire_if_due(self, request: ApprovalRequest) -> bool:
        if not request.is_pending:
            return False
        now = self._clock.now_ns()
        if now < request.expires_at_ns:
            return False
        request.status = ApprovalStatus.TIMED_OUT
        request.decided_at_ns = now
        request.decided_by = "system"
        request.decision_note = "no response within the approval window; auto-denied"
        return True

    def _require(self, request_id: UUID) -> ApprovalRequest:
        request = self._requests.get(request_id)
        if request is None:
            raise KeyError(f"no approval request {request_id}")
        return request


def needs_approval(
    notional_pct_of_equity: Decimal,
    threshold_pct: Decimal,
    *,
    is_new_symbol: bool = False,
) -> ApprovalKind | None:
    """Whether an order requires sign-off (spec §7.6, first two bullets)."""
    if is_new_symbol:
        return ApprovalKind.NEW_SYMBOL
    if notional_pct_of_equity > threshold_pct:
        return ApprovalKind.LARGE_ORDER
    return None
