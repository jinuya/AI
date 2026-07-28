"""The append-only audit log.

Spec §9.3 lists what must be recorded, and the list is deliberately exhaustive:
every signal and its inputs, every risk check result *including the passes*,
every order request and raw response, every fill, every configuration change
with before/after values, every manual intervention, every LLM prompt and
response.

Recording only the rejections would be the obvious economy and the wrong one —
when a bad order *does* get through, the question is which check let it through,
and that is only answerable if the passes were recorded too.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from atrader.audit.hashchain import (
    GENESIS_HASH,
    AuditRecord,
    ChainVerification,
    compute_hash,
    verify_chain,
)
from atrader.config.secrets import Secret, is_sensitive_key, redact
from atrader.core.clock import Clock

__all__ = ["AuditEvent", "AuditLogger", "AuditSink", "InMemoryAuditSink"]


class AuditEvent:
    """Canonical ``event_type`` values.

    A fixed vocabulary rather than free strings: filtering an audit log by
    ``event_type`` only works if writers agree on the spelling.
    """

    INTENT_CREATED = "intent.created"
    RISK_CHECK_PASSED = "risk.check_passed"
    RISK_CHECK_REJECTED = "risk.check_rejected"
    RISK_DECISION = "risk.decision"
    ORDER_REQUESTED = "order.requested"
    ORDER_ACK = "order.ack"
    ORDER_REJECTED = "order.rejected"
    ORDER_CANCELED = "order.canceled"
    ORDER_STATE_CHANGED = "order.state_changed"
    FILL_RECEIVED = "fill.received"
    RECONCILIATION_OK = "reconciliation.ok"
    RECONCILIATION_BREAK = "reconciliation.break"
    CIRCUIT_BREAKER_TRIPPED = "circuit_breaker.tripped"
    CIRCUIT_BREAKER_RESET = "circuit_breaker.reset"
    KILL_SWITCH_ENGAGED = "kill_switch.engaged"
    KILL_SWITCH_RELEASED = "kill_switch.released"
    DEADMAN_TRIGGERED = "deadman.triggered"
    DEADMAN_REARMED = "deadman.rearmed"
    RATE_LIMIT_THROTTLED = "rate_limit.throttled"
    MARGIN_STATUS_CHANGED = "margin.status_changed"
    POSITION_UPDATED = "position.updated"
    INTENTS_NETTED = "portfolio.intents_netted"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_DENIED = "approval.denied"
    APPROVAL_TIMED_OUT = "approval.timed_out"
    CONFIG_CHANGED = "config.changed"
    MANUAL_INTERVENTION = "manual.intervention"
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_REJECTED = "llm.rejected"
    SYSTEM_STARTED = "system.started"
    SYSTEM_STOPPED = "system.stopped"
    STATE_RECOVERED = "system.state_recovered"


@runtime_checkable
class AuditSink(Protocol):
    """Where audit records are persisted.

    Defined here rather than imported from ``storage`` so the audit package
    stays self-contained; ``storage`` supplies implementations.
    """

    def append(self, record: AuditRecord) -> None:
        """Persist one record. Must preserve ``seq`` ordering."""
        ...

    def read_all(self) -> list[AuditRecord]:
        """Every record, ordered by ``seq``."""
        ...

    def last(self) -> AuditRecord | None:
        """The most recent record, or ``None`` when the log is empty."""
        ...


class InMemoryAuditSink:
    """In-memory sink for tests and dry runs."""

    __slots__ = ("_records",)

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []

    def append(self, record: AuditRecord) -> None:
        self._records.append(record)

    def read_all(self) -> list[AuditRecord]:
        return list(self._records)

    def last(self) -> AuditRecord | None:
        return self._records[-1] if self._records else None

    def __len__(self) -> int:
        return len(self._records)


def _sanitise(value: object, key: str = "") -> object:
    """Strip credentials from a payload before it is hashed and stored.

    The audit log is retained for seven years (spec §9.3); a key written into it
    is a key you cannot delete without breaking the chain that makes the log
    trustworthy. So masking happens on the way in, not on the way out.
    """
    if isinstance(value, Secret):
        return repr(value)
    if isinstance(value, str):
        return "***REDACTED***" if is_sensitive_key(key) else redact(value)
    if isinstance(value, dict):
        return {str(k): _sanitise(v, str(k)) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitise(item, key) for item in value]
    return value


class AuditLogger:
    """Appends hash-chained records.

    Not thread-safe: the chain is a strict sequence, so writes must be
    serialised by the caller. In the runtime that means a single writer per
    process, which is also what makes ``seq`` meaningful.
    """

    __slots__ = ("_clock", "_next_seq", "_prev_hash", "_sink")

    def __init__(self, sink: AuditSink, clock: Clock) -> None:
        self._sink = sink
        self._clock = clock
        last = sink.last()
        if last is None:
            self._prev_hash = GENESIS_HASH
            self._next_seq = 1
        else:
            # Resume the existing chain rather than starting a new one — a
            # restart must not create a second, unverifiable segment.
            self._prev_hash = last.hash
            self._next_seq = last.seq + 1

    @property
    def next_seq(self) -> int:
        return self._next_seq

    def append(
        self,
        event_type: str,
        *,
        actor: str = "system",
        payload: dict[str, Any] | None = None,
    ) -> AuditRecord:
        """Record an event and return the stored record."""
        sanitised = {str(k): _sanitise(v, str(k)) for k, v in (payload or {}).items()}
        created_at_ns = self._clock.now_ns()
        seq = self._next_seq
        record = AuditRecord(
            seq=seq,
            event_type=event_type,
            actor=actor,
            payload=sanitised,
            created_at_ns=created_at_ns,
            prev_hash=self._prev_hash,
            hash=compute_hash(
                seq=seq,
                event_type=event_type,
                actor=actor,
                payload=sanitised,
                created_at_ns=created_at_ns,
                prev_hash=self._prev_hash,
            ),
        )
        self._sink.append(record)
        self._prev_hash = record.hash
        self._next_seq += 1
        return record

    def verify(self) -> ChainVerification:
        """Verify the whole chain (acceptance criterion #9)."""
        return verify_chain(self._sink.read_all())
