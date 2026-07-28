"""In-process message bus.

Backtests, replay and tests run on this. It reproduces the parts of Redis
Streams semantics that consumers must cope with — consumer groups, explicit
acks, redelivery of unacked messages, and a bounded buffer that drops the
*oldest* entries under backpressure — so code written against it behaves the
same in production.

Delivery order is deterministic: messages are handed out in publication order,
per topic, with no concurrency in the dispatch path. Spec §2.2 requires that
replaying an event sequence produces the same orders, and a bus that reorders
messages under load would make that impossible to guarantee.

Spec §5.4 on backpressure: *"컨슈머가 못 따라가면 오래된 틱을 드롭하되, 드롭
카운터를 메트릭으로 노출한다. 조용히 버리면 안 된다."* — :attr:`InMemoryBus.dropped`
is that counter.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

from atrader.bus.protocol import Message
from atrader.core.clock import Clock, SystemClock

__all__ = ["InMemoryBus", "InMemorySubscription"]

DEFAULT_MAX_QUEUE = 10_000


def _matches(pattern: str, topic: str) -> bool:
    """``md.tick.*`` matches ``md.tick.AAPL``; exact strings match exactly."""
    if pattern == topic:
        return True
    if pattern.endswith("*"):
        return topic.startswith(pattern[:-1])
    return False


class InMemorySubscription:
    """One consumer's queue, with at-least-once redelivery."""

    __slots__ = ("_bus", "_closed", "_consumer", "_group", "_pattern", "_pending", "_queue")

    def __init__(
        self,
        bus: InMemoryBus,
        pattern: str,
        group: str,
        consumer: str,
        max_queue: int,
    ) -> None:
        self._bus = bus
        self._pattern = pattern
        self._group = group
        self._consumer = consumer
        self._queue: deque[Message] = deque(maxlen=max_queue)
        self._pending: dict[str, Message] = {}
        self._closed = False

    @property
    def pattern(self) -> str:
        return self._pattern

    @property
    def group(self) -> str:
        return self._group

    @property
    def pending_count(self) -> int:
        """Delivered but not yet acked. A number that only grows is a stuck consumer."""
        return len(self._pending)

    def _offer(self, message: Message) -> bool:
        """Enqueue, reporting whether an older message had to be dropped."""
        dropped = len(self._queue) == self._queue.maxlen
        self._queue.append(message)
        return dropped

    async def next(self, *, timeout_s: float | None = None) -> Message | None:
        """Return the next message, or ``None`` if the timeout elapses first."""
        waited = 0.0
        poll = 0.001
        while not self._closed:
            if self._queue:
                message = self._queue.popleft()
                self._pending[message.message_id] = message
                return message
            if timeout_s is not None and waited >= timeout_s:
                return None
            waited += poll
            await asyncio.sleep(poll)
        return None

    async def _iterate(self) -> AsyncIterator[Message]:
        while not self._closed:
            message = await self.next(timeout_s=0.05)
            if message is not None:
                yield message

    def __aiter__(self) -> AsyncIterator[Message]:
        # Must be a *sync* method returning an async iterator. Declaring it
        # `async def` with a yield would make it an async generator function,
        # so `__aiter__()` would return a coroutine and `async for` would fail.
        return self._iterate()

    async def ack(self, message: Message) -> None:
        self._pending.pop(message.message_id, None)

    async def nack(self, message: Message) -> None:
        """Return a message to the front of the queue for redelivery.

        The delivery count is incremented so a handler can recognise a message
        that keeps failing rather than retrying it forever.
        """
        self._pending.pop(message.message_id, None)
        redelivered = Message(
            topic=message.topic,
            message_id=message.message_id,
            payload=message.payload,
            published_at_ns=message.published_at_ns,
            delivery_count=message.delivery_count + 1,
            headers=message.headers,
        )
        self._queue.appendleft(redelivered)

    async def redeliver_pending(self) -> int:
        """Requeue everything unacked. This is what a consumer restart does."""
        pending = list(self._pending.values())
        self._pending.clear()
        for message in reversed(pending):
            self._queue.appendleft(
                Message(
                    topic=message.topic,
                    message_id=message.message_id,
                    payload=message.payload,
                    published_at_ns=message.published_at_ns,
                    delivery_count=message.delivery_count + 1,
                    headers=message.headers,
                )
            )
        return len(pending)

    async def close(self) -> None:
        self._closed = True
        self._bus._remove(self)


class InMemoryBus:
    """Deterministic in-process bus."""

    __slots__ = ("_clock", "_counter", "_dropped", "_max_queue", "_published", "_subscriptions")

    def __init__(self, clock: Clock | None = None, *, max_queue: int = DEFAULT_MAX_QUEUE) -> None:
        self._clock = clock or SystemClock()
        self._subscriptions: list[InMemorySubscription] = []
        self._counter = 0
        self._dropped = 0
        self._published = 0
        self._max_queue = max_queue

    @property
    def dropped(self) -> int:
        """Messages discarded under backpressure. Exposed as a metric — spec §5.4
        forbids dropping silently."""
        return self._dropped

    @property
    def published(self) -> int:
        return self._published

    async def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> str:
        self._counter += 1
        # Zero-padded so ids sort lexicographically in the order they were
        # published — replay diffs are much easier to read that way.
        message_id = f"{self._counter:012d}"
        message = Message(
            topic=topic,
            message_id=message_id,
            payload=payload,
            published_at_ns=self._clock.now_ns(),
            headers=dict(headers or {}),
        )
        self._published += 1

        # One delivery per group, matching Redis Streams: members of a group
        # share the stream, separate groups each get a copy.
        delivered_groups: set[str] = set()
        for subscription in self._subscriptions:
            if not _matches(subscription.pattern, topic):
                continue
            if subscription.group in delivered_groups:
                continue
            delivered_groups.add(subscription.group)
            if subscription._offer(message):
                self._dropped += 1
        return message_id

    async def subscribe(
        self,
        pattern: str,
        *,
        group: str,
        consumer: str = "default",
    ) -> InMemorySubscription:
        subscription = InMemorySubscription(self, pattern, group, consumer, self._max_queue)
        self._subscriptions.append(subscription)
        return subscription

    def _remove(self, subscription: InMemorySubscription) -> None:
        if subscription in self._subscriptions:
            self._subscriptions.remove(subscription)

    async def close(self) -> None:
        for subscription in list(self._subscriptions):
            await subscription.close()
        self._subscriptions.clear()
