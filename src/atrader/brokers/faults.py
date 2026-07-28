"""Fault injection — spec §8.3 (카오스 테스트).

A broker adapter that wraps another one and misbehaves on purpose: timeouts,
5xx, dropped connections, duplicated and reordered event streams, and the case
that matters most — a request that **actually landed** but whose response never
came back.

That last one is why this module exists rather than a handful of
``unittest.mock`` side effects in the test file. The dangerous failure is not
"the call raised"; it is "the call raised *and the order is now working at the
venue*". A mock that raises before delegating tests the easy half. A retry that
looks correct against it will still produce duplicate fills in production,
because the resend lands on top of an order that was already there.

:class:`FaultInjector` therefore separates the two:

* ``timeout_pct`` — fails **before** the inner broker sees it. Nothing happened.
* ``ambiguous_pct`` — delegates first, **then** raises. Everything happened; only
  the answer was lost.

Only the second one can catch a blind resend, and only
:mod:`atrader.execution.idempotency`'s query-first rule survives it.

Every decision is drawn from an injected :class:`~atrader.core.rng.Rng`, so a
chaos run that finds a bug can be replayed exactly by reusing its seed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from decimal import Decimal

from atrader.brokers.models import (
    BrokerEvent,
    CancelAck,
    OrderAck,
    OrderModification,
    OrderState,
)
from atrader.brokers.protocol import BrokerAdapter, BrokerCapabilities
from atrader.core.errors import (
    AmbiguousBrokerError,
    PermanentBrokerError,
    TransientBrokerError,
)
from atrader.core.models import AccountState, OrderRequest, Position
from atrader.core.rng import Rng, SeededRng

__all__ = ["Fault", "FaultInjector", "FaultProfile", "FaultyBroker", "ScriptedFaults"]


class Fault:
    """Names for the failure modes. Used by :class:`ScriptedFaults`."""

    NONE = "none"
    TIMEOUT = "timeout"
    """Raised before delegating: the request never reached the venue."""
    TRANSIENT = "transient"
    """5xx / 429 / connection reset. Also raised before delegating."""
    PERMANENT = "permanent"
    """400-class refusal. Retrying cannot help."""
    AMBIGUOUS = "ambiguous"
    """Delegated first, then raised. The order exists; the response was lost."""
    DISCONNECT = "disconnect"
    """The event stream ends mid-session."""


@dataclass(frozen=True, slots=True)
class FaultProfile:
    """How often each failure mode fires, as percentages."""

    timeout_pct: Decimal = Decimal(0)
    transient_pct: Decimal = Decimal(0)
    permanent_pct: Decimal = Decimal(0)
    ambiguous_pct: Decimal = Decimal(0)
    duplicate_event_pct: Decimal = Decimal(0)
    """Redelivery. Spec §4.3 says the stream is at-least-once, so a consumer
    that breaks under this is broken in production too, not just under chaos."""
    drop_event_pct: Decimal = Decimal(0)
    reorder_events: bool = False
    disconnect_after_events: int | None = None

    def __post_init__(self) -> None:
        total = self.timeout_pct + self.transient_pct + self.permanent_pct + self.ambiguous_pct
        if total > 100:
            raise ValueError(f"fault probabilities sum to {total}%, which exceeds 100%")
        for name, value in (
            ("duplicate_event_pct", self.duplicate_event_pct),
            ("drop_event_pct", self.drop_event_pct),
        ):
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be within 0..100, got {value}")

    @classmethod
    def flaky(cls) -> FaultProfile:
        """A moderately unreliable venue — the default chaos profile."""
        return cls(
            timeout_pct=Decimal(5),
            transient_pct=Decimal(10),
            permanent_pct=Decimal(2),
            ambiguous_pct=Decimal(3),
            duplicate_event_pct=Decimal(10),
            reorder_events=True,
        )

    @classmethod
    def hostile(cls) -> FaultProfile:
        """Ambiguity-heavy. Aimed squarely at the duplicate-submission path."""
        return cls(
            timeout_pct=Decimal(10),
            transient_pct=Decimal(10),
            ambiguous_pct=Decimal(20),
            duplicate_event_pct=Decimal(25),
            drop_event_pct=Decimal(5),
            reorder_events=True,
        )


@dataclass
class ScriptedFaults:
    """A fixed sequence of faults, consumed one per call.

    Preferred over probabilities in unit tests: "the second submit is ambiguous"
    is a statement a test can assert on, whereas "3% of submits are ambiguous"
    is not.
    """

    sequence: list[str] = field(default_factory=list)
    _index: int = field(default=0, init=False)

    @classmethod
    def of(cls, *faults: str) -> ScriptedFaults:
        return cls(list(faults))

    def next_fault(self) -> str:
        if self._index >= len(self.sequence):
            return Fault.NONE
        fault = self.sequence[self._index]
        self._index += 1
        return fault

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self.sequence)

    def reset(self) -> None:
        self._index = 0


@dataclass
class FaultInjector:
    """Decides which fault, if any, applies to the next call."""

    profile: FaultProfile = field(default_factory=FaultProfile)
    rng: Rng = field(default_factory=lambda: SeededRng(seed=0))
    scripted: ScriptedFaults | None = None
    """When set, takes precedence — deterministic tests beat dice."""

    _calls: int = field(default=0, init=False)
    _faults_fired: dict[str, int] = field(default_factory=dict, init=False)

    @property
    def call_count(self) -> int:
        return self._calls

    @property
    def faults_fired(self) -> dict[str, int]:
        """Tally by fault name — chaos tests assert that a mode actually fired."""
        return dict(self._faults_fired)

    def next_fault(self) -> str:
        self._calls += 1
        fault = self._draw()
        if fault != Fault.NONE:
            self._faults_fired[fault] = self._faults_fired.get(fault, 0) + 1
        return fault

    def _draw(self) -> str:
        if self.scripted is not None:
            return self.scripted.next_fault()
        roll = Decimal(str(self.rng.uniform(0.0, 100.0)))
        for threshold, fault in (
            (self.profile.timeout_pct, Fault.TIMEOUT),
            (self.profile.transient_pct, Fault.TRANSIENT),
            (self.profile.permanent_pct, Fault.PERMANENT),
            (self.profile.ambiguous_pct, Fault.AMBIGUOUS),
        ):
            if roll < threshold:
                return fault
            roll -= threshold
        return Fault.NONE

    def roll_pct(self, probability_pct: Decimal) -> bool:
        """True with the given probability. Used for per-event decisions."""
        if probability_pct <= 0:
            return False
        return Decimal(str(self.rng.uniform(0.0, 100.0))) < probability_pct


@dataclass
class FaultyBroker:
    """A :class:`~atrader.brokers.protocol.BrokerAdapter` that misbehaves.

    Wraps a real adapter — usually :class:`~atrader.brokers.paper.PaperBroker` —
    so the fault path and the happy path exercise the same accounting.
    """

    inner: BrokerAdapter
    injector: FaultInjector = field(default_factory=FaultInjector)

    @property
    def name(self) -> str:
        return f"faulty:{self.inner.name}"

    @property
    def capabilities(self) -> BrokerCapabilities:
        return self.inner.capabilities

    async def submit_order(self, request: OrderRequest) -> OrderAck:
        fault = self.injector.next_fault()

        if fault == Fault.TIMEOUT:
            raise TimeoutError(f"submit timed out for {request.client_order_id}")
        if fault == Fault.TRANSIENT:
            raise TransientBrokerError(
                f"503 from {self.inner.name} for {request.client_order_id}",
                details={"injected": True},
            )
        if fault == Fault.PERMANENT:
            raise PermanentBrokerError(
                f"400 from {self.inner.name} for {request.client_order_id}",
                details={"injected": True},
            )

        if fault == Fault.AMBIGUOUS:
            # The order really is placed, and *then* the response is lost. A
            # caller that resends blindly now has two orders working.
            await self.inner.submit_order(request)
            raise AmbiguousBrokerError(
                f"no response from {self.inner.name}; the order may or may not exist",
                client_order_id=request.client_order_id,
                details={"injected": True, "landed": True},
            )

        return await self.inner.submit_order(request)

    async def cancel_order(self, client_order_id: str) -> CancelAck:
        fault = self.injector.next_fault()
        if fault == Fault.TIMEOUT:
            raise TimeoutError(f"cancel timed out for {client_order_id}")
        if fault == Fault.TRANSIENT:
            raise TransientBrokerError(f"503 cancelling {client_order_id}")
        if fault == Fault.AMBIGUOUS:
            await self.inner.cancel_order(client_order_id)
            raise AmbiguousBrokerError(
                f"no response to the cancel of {client_order_id}",
                client_order_id=client_order_id,
            )
        return await self.inner.cancel_order(client_order_id)

    async def modify_order(self, client_order_id: str, mods: OrderModification) -> OrderAck:
        fault = self.injector.next_fault()
        if fault == Fault.TIMEOUT:
            raise TimeoutError(f"modify timed out for {client_order_id}")
        if fault == Fault.TRANSIENT:
            raise TransientBrokerError(f"503 modifying {client_order_id}")
        return await self.inner.modify_order(client_order_id, mods)

    async def get_order(self, client_order_id: str) -> OrderState | None:
        # Queries fail transiently but are never made ambiguous: an ambiguous
        # *query* is indistinguishable from a failed one to the caller, and
        # idempotency.py already treats a failed query as unresolved.
        if self.injector.next_fault() in (Fault.TIMEOUT, Fault.TRANSIENT):
            raise TransientBrokerError(f"query failed for {client_order_id}")
        return await self.inner.get_order(client_order_id)

    async def get_open_orders(self) -> list[OrderState]:
        if self.injector.next_fault() in (Fault.TIMEOUT, Fault.TRANSIENT):
            raise TransientBrokerError("open-order query failed")
        return await self.inner.get_open_orders()

    async def get_positions(self) -> list[Position]:
        if self.injector.next_fault() in (Fault.TIMEOUT, Fault.TRANSIENT):
            raise TransientBrokerError("position query failed")
        return await self.inner.get_positions()

    async def get_account(self) -> AccountState:
        if self.injector.next_fault() in (Fault.TIMEOUT, Fault.TRANSIENT):
            raise TransientBrokerError("account query failed")
        return await self.inner.get_account()

    def stream_updates(self) -> AsyncIterator[BrokerEvent]:
        return self._corrupted_stream()

    async def _corrupted_stream(self) -> AsyncIterator[BrokerEvent]:
        """Duplicate, drop, reorder and truncate the inner stream.

        Reordering holds one event back and emits it after the next, which is
        what a two-connection venue feed looks like in practice. Doing it this
        way rather than shuffling a buffer keeps the stream lazy — a consumer
        still receives events as they occur, just not in the order they occurred.
        """
        emitted = 0
        held: BrokerEvent | None = None

        async for event in self.inner.stream_updates():
            batch: list[BrokerEvent] = []

            if self.injector.profile.reorder_events and held is None:
                held = event
                continue
            if held is not None:
                batch.append(event)
                batch.append(held)
                held = None
            else:
                batch.append(event)

            for item in batch:
                if self.injector.roll_pct(self.injector.profile.drop_event_pct):
                    continue
                yield item
                emitted += 1
                if self.injector.roll_pct(self.injector.profile.duplicate_event_pct):
                    yield item  # at-least-once, exactly as spec §4.3 warns
                    emitted += 1

                limit = self.injector.profile.disconnect_after_events
                if limit is not None and emitted >= limit:
                    raise TransientBrokerError(
                        f"connection dropped after {emitted} events (injected)"
                    )

        if held is not None:
            yield held

    async def close(self) -> None:
        await self.inner.close()


def replay_events(events: Iterable[BrokerEvent], *, times: int = 2) -> list[BrokerEvent]:
    """Duplicate every event *times* over. For idempotency assertions."""
    out: list[BrokerEvent] = []
    for event in events:
        out.extend([event] * times)
    return out
