"""Message bus interface.

Spec §4.3: Redis Streams by default, NATS JetStream swappable behind a thin
abstraction. Delivery is **at-least-once**, and the spec is blunt about the
consequence:

    그래서 모든 컨슈머는 멱등해야 한다. 같은 체결 이벤트를 두 번 받아도 포지션이
    두 배가 되면 안 된다.

The interface reflects that. Messages are explicitly acknowledged, so a consumer
that crashes mid-processing gets the message again rather than losing it — which
means redelivery is the normal case, not the exception, and every handler has to
be written for it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["Message", "MessageBus", "Subscription"]


@dataclass(frozen=True, slots=True)
class Message:
    """One delivery.

    ``delivery_count`` is above 1 when this is a redelivery after a consumer
    failed to ack — useful for spotting a poison message that keeps killing its
    handler.
    """

    topic: str
    message_id: str
    payload: dict[str, Any]
    published_at_ns: int
    delivery_count: int = 1
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def trace_id(self) -> str | None:
        """Trace id propagated from the publisher (spec §9.2)."""
        return self.headers.get("trace_id")


@runtime_checkable
class Subscription(Protocol):
    """A consumer's view of one topic pattern."""

    def __aiter__(self) -> AsyncIterator[Message]: ...

    async def ack(self, message: Message) -> None:
        """Confirm processing. Unacked messages are redelivered."""
        ...

    async def close(self) -> None: ...


@runtime_checkable
class MessageBus(Protocol):
    async def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> str:
        """Publish and return the assigned message id."""
        ...

    async def subscribe(
        self,
        pattern: str,
        *,
        group: str,
        consumer: str,
    ) -> Subscription:
        """Subscribe to topics matching *pattern*.

        ``pattern`` supports a trailing ``*`` wildcard (``md.tick.*``). Members
        of the same ``group`` share the stream; different groups each get a copy.
        """
        ...

    async def close(self) -> None: ...
