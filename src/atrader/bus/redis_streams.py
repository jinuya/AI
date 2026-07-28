"""Redis Streams bus — the production backend (spec §4.3).

Chosen over NATS JetStream as the default because the operational burden is
lower; the :class:`~atrader.bus.protocol.MessageBus` Protocol is what makes
swapping it later a contained change.

**Not exercised by the default test suite.** There is no Redis in the standard
development environment, so these paths run only under ``pytest -m integration``
against a real server. The in-memory bus deliberately mirrors the semantics that
matter — consumer groups, explicit acks, redelivery, oldest-first drops under
backpressure — so logic written against it does not discover surprises here.

Streams are capped with ``MAXLEN ~`` rather than left to grow: an unbounded
market-data stream will exhaust Redis memory in a single session, and the
approximate form is dramatically cheaper than the exact one.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from atrader.bus.protocol import Message
from atrader.core.clock import Clock, SystemClock

__all__ = ["RedisStreamsBus", "RedisStreamsSubscription"]

DEFAULT_MAXLEN = 100_000
DEFAULT_BLOCK_MS = 1_000


def _encode(
    payload: dict[str, Any], headers: dict[str, str], published_at_ns: int
) -> dict[str, str]:
    return {
        "payload": json.dumps(payload, sort_keys=True, default=str),
        "headers": json.dumps(headers, sort_keys=True),
        "published_at_ns": str(published_at_ns),
    }


def _decode(topic: str, message_id: str, fields: dict[str, str], delivery_count: int) -> Message:
    return Message(
        topic=topic,
        message_id=message_id,
        payload=json.loads(fields.get("payload", "{}")),
        published_at_ns=int(fields.get("published_at_ns", "0")),
        delivery_count=delivery_count,
        headers=json.loads(fields.get("headers", "{}")),
    )


class RedisStreamsSubscription:
    """Consumer-group reader over one or more streams."""

    __slots__ = ("_block_ms", "_closed", "_consumer", "_group", "_redis", "_streams")

    def __init__(
        self,
        redis: Any,
        streams: list[str],
        group: str,
        consumer: str,
        block_ms: int,
    ) -> None:
        self._redis = redis
        self._streams = streams
        self._group = group
        self._consumer = consumer
        self._block_ms = block_ms
        self._closed = False

    @property
    def group(self) -> str:
        return self._group

    async def _ensure_groups(self) -> None:
        for stream in self._streams:
            try:
                # mkstream so subscribing before the first publish still works.
                await self._redis.xgroup_create(stream, self._group, id="0", mkstream=True)
            except Exception as exc:
                # BUSYGROUP just means the group already exists — expected on restart.
                if "BUSYGROUP" not in str(exc):
                    raise

    async def claim_stale(self, min_idle_ms: int = 60_000) -> list[Message]:
        """Take over messages left pending by a consumer that died.

        Without this, a crash between delivery and ack strands those messages in
        the dead consumer's PEL and the fills they carry are never applied.
        """
        claimed: list[Message] = []
        for stream in self._streams:
            result = await self._redis.xautoclaim(
                stream, self._group, self._consumer, min_idle_time=min_idle_ms, start_id="0-0"
            )
            entries = result[1] if len(result) > 1 else []
            for message_id, fields in entries:
                claimed.append(_decode(stream, message_id, fields, delivery_count=2))
        return claimed

    async def next(self, *, timeout_ms: int | None = None) -> Message | None:
        await self._ensure_groups()
        response = await self._redis.xreadgroup(
            groupname=self._group,
            consumername=self._consumer,
            streams=dict.fromkeys(self._streams, ">"),
            count=1,
            block=self._block_ms if timeout_ms is None else timeout_ms,
        )
        if not response:
            return None
        stream, entries = response[0]
        if not entries:
            return None
        message_id, fields = entries[0]
        topic = stream.decode() if isinstance(stream, bytes) else str(stream)
        return _decode(topic, message_id, fields, delivery_count=1)

    async def _iterate(self) -> AsyncIterator[Message]:
        await self._ensure_groups()
        while not self._closed:
            message = await self.next()
            if message is not None:
                yield message

    def __aiter__(self) -> AsyncIterator[Message]:
        # Sync method returning an async iterator — see the note in bus/memory.py.
        return self._iterate()

    async def ack(self, message: Message) -> None:
        await self._redis.xack(message.topic, self._group, message.message_id)

    async def close(self) -> None:
        self._closed = True


class RedisStreamsBus:
    """Redis Streams implementation of :class:`~atrader.bus.protocol.MessageBus`."""

    __slots__ = ("_block_ms", "_clock", "_maxlen", "_redis")

    def __init__(
        self,
        redis: Any,
        *,
        clock: Clock | None = None,
        maxlen: int = DEFAULT_MAXLEN,
        block_ms: int = DEFAULT_BLOCK_MS,
    ) -> None:
        self._redis = redis
        self._clock = clock or SystemClock()
        self._maxlen = maxlen
        self._block_ms = block_ms

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> RedisStreamsBus:
        from redis.asyncio import Redis as AsyncRedis

        return cls(AsyncRedis.from_url(url, decode_responses=True), **kwargs)

    async def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> str:
        fields = _encode(payload, dict(headers or {}), self._clock.now_ns())
        raw_id = await self._redis.xadd(topic, fields, maxlen=self._maxlen, approximate=True)
        return raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)

    async def subscribe(
        self,
        pattern: str,
        *,
        group: str,
        consumer: str = "default",
    ) -> RedisStreamsSubscription:
        """Resolve *pattern* to concrete streams and read them as a group.

        Redis Streams has no server-side pattern subscribe, so a trailing ``*``
        is expanded with ``SCAN`` at subscribe time. Streams created afterwards
        are not picked up — for a fixed universe that is fine, and it is why
        symbols are configuration rather than discovery.
        """
        if pattern.endswith("*"):
            streams = [key async for key in self._redis.scan_iter(match=pattern, _type="STREAM")]
        else:
            streams = [pattern]
        subscription = RedisStreamsSubscription(
            self._redis,
            [str(s) for s in streams] or [pattern.rstrip("*")],
            group,
            consumer,
            self._block_ms,
        )
        await subscription._ensure_groups()
        return subscription

    async def close(self) -> None:
        await self._redis.aclose()
