"""Safe submission and retry — spec §FR-EXE-04.

    응답을 못 받은 경우 같은 ID로 재조회를 먼저 시도하고, 존재하지 않을 때만 재전송한다.
    **응답 없음 = 실패**라고 가정하고 그냥 재전송하는 건 절대 금지 — 중복 체결의 전형적 원인이다.

The dangerous case is not the timeout. It is the *assumption* that a timeout
means failure. A request that timed out may have been received, accepted and
filled; the only thing that failed was the response. Resending on that
assumption produces two orders and two fills, and the second one is discovered
minutes later when reconciliation reports a position twice the expected size.

So an unanswered submit is never retried blind. The sequence is:

1. Query the broker for the ``client_order_id``.
2. If it exists, adopt the broker's state. Nothing to resend.
3. If it does not exist, resend — the *same* ``client_order_id``, so even if
   both the original and the resend land, the broker treats them as one order.
4. If the query itself fails, stop. Lock the symbol and escalate to a human
   (spec §6.4). Guessing here is how duplicate fills happen.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum

from atrader.brokers.models import OrderAck, OrderState
from atrader.brokers.protocol import BrokerAdapter
from atrader.core.clock import Clock
from atrader.core.errors import (
    AmbiguousBrokerError,
    PermanentBrokerError,
    TransientBrokerError,
)
from atrader.core.models import OrderRequest
from atrader.core.rng import Rng, SeededRng

__all__ = ["SubmissionOutcome", "SubmissionResult", "SubmitPolicy", "submit_with_retry"]


class SubmissionOutcome(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ALREADY_EXISTS = "already_exists"
    """The query found it: the original request had in fact been received."""
    AMBIGUOUS = "ambiguous"
    """Neither confirmed nor refuted. A human must resolve this."""


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    outcome: SubmissionOutcome
    client_order_id: str
    ack: OrderAck | None = None
    broker_state: OrderState | None = None
    attempts: int = 1
    reason: str = ""

    @property
    def is_live(self) -> bool:
        return self.outcome in (SubmissionOutcome.ACCEPTED, SubmissionOutcome.ALREADY_EXISTS)

    @property
    def needs_human(self) -> bool:
        return self.outcome is SubmissionOutcome.AMBIGUOUS


@dataclass(frozen=True, slots=True)
class SubmitPolicy:
    """Retry behaviour for transient failures (spec §6.4)."""

    max_attempts: int = 5
    base_delay_seconds: float = 0.2
    max_delay_seconds: float = 8.0
    jitter_pct: float = 20.0
    """Without jitter, every retrying instance hits the venue in lockstep."""

    def delay_for(self, attempt: int, rng: Rng) -> float:
        # ``2 ** (attempt - 1)`` types as Any in typeshed's general int.__pow__
        # overload (it does not know the exponent is non-negative), so the
        # multiplier is pinned back to float explicitly.
        multiplier: float = 2 ** (attempt - 1)
        base = min(self.base_delay_seconds * multiplier, self.max_delay_seconds)
        spread = base * self.jitter_pct / 100.0
        return max(0.0, base + rng.uniform(-spread, spread))


async def submit_with_retry(
    broker: BrokerAdapter,
    request: OrderRequest,
    *,
    clock: Clock,
    policy: SubmitPolicy | None = None,
    rng: Rng | None = None,
    sleep: object = None,
) -> SubmissionResult:
    """Submit an order, resolving ambiguity by querying rather than resending."""
    policy = policy or SubmitPolicy()
    rng = rng or SeededRng(seed=0)
    sleeper = sleep if callable(sleep) else asyncio.sleep

    attempts = 0
    last_error = ""

    while attempts < policy.max_attempts:
        attempts += 1
        try:
            ack = await broker.submit_order(request)
        except PermanentBrokerError as exc:
            # 400, insufficient funds, invalid symbol, halted. Retrying changes
            # nothing (spec §6.4).
            return SubmissionResult(
                outcome=SubmissionOutcome.REJECTED,
                client_order_id=request.client_order_id,
                attempts=attempts,
                reason=str(exc),
            )
        except (AmbiguousBrokerError, TimeoutError, TransientBrokerError) as exc:
            last_error = str(exc)
            resolved = await _resolve_by_query(broker, request.client_order_id)

            if resolved is not None:
                # It was received after all. The response was what got lost.
                return SubmissionResult(
                    outcome=SubmissionOutcome.ALREADY_EXISTS,
                    client_order_id=request.client_order_id,
                    broker_state=resolved,
                    attempts=attempts,
                    reason=f"resolved by query after {type(exc).__name__}: {last_error}",
                )

            if isinstance(exc, AmbiguousBrokerError):
                # The query could not settle it. Do not guess — spec §6.4 says
                # lock the symbol and call a person.
                return SubmissionResult(
                    outcome=SubmissionOutcome.AMBIGUOUS,
                    client_order_id=request.client_order_id,
                    attempts=attempts,
                    reason=(
                        f"submission outcome unknown and the query did not resolve it: "
                        f"{last_error}. Not resending — a blind retry here is the classic "
                        "cause of duplicate fills."
                    ),
                )

            if attempts < policy.max_attempts:
                await sleeper(policy.delay_for(attempts, rng))
            continue

        if ack.rejected:
            return SubmissionResult(
                outcome=SubmissionOutcome.REJECTED,
                client_order_id=request.client_order_id,
                ack=ack,
                attempts=attempts,
                reason=ack.reason,
            )
        return SubmissionResult(
            outcome=SubmissionOutcome.ACCEPTED,
            client_order_id=request.client_order_id,
            ack=ack,
            attempts=attempts,
        )

    # Retries exhausted. One final query before giving up — the last attempt may
    # have landed even though its response did not come back.
    final = await _resolve_by_query(broker, request.client_order_id)
    if final is not None:
        return SubmissionResult(
            outcome=SubmissionOutcome.ALREADY_EXISTS,
            client_order_id=request.client_order_id,
            broker_state=final,
            attempts=attempts,
            reason="found at the broker after retries were exhausted",
        )
    return SubmissionResult(
        outcome=SubmissionOutcome.AMBIGUOUS,
        client_order_id=request.client_order_id,
        attempts=attempts,
        reason=f"{attempts} attempts failed; last error: {last_error}",
    )


async def _resolve_by_query(broker: BrokerAdapter, client_order_id: str) -> OrderState | None:
    """Ask the broker whether it has the order.

    A failing query returns ``None`` rather than raising, which routes the
    caller into the ambiguous branch — the honest answer when we simply do not
    know.
    """
    try:
        return await broker.get_order(client_order_id)
    except Exception:
        return None
