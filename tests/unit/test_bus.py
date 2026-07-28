"""Message bus semantics.

The behaviours tested here are the ones consumers have to be written against:
at-least-once delivery, explicit acks, redelivery after a crash, per-group fan
out, and oldest-first drops under backpressure with a visible counter.

Spec §4.3 is blunt about the consequence of at-least-once — *"같은 체결 이벤트를
두 번 받아도 포지션이 두 배가 되면 안 된다"* — so these tests exist to make sure
the bus really does redeliver, rather than quietly being exactly-once in
practice and letting a non-idempotent consumer ship.
"""

from __future__ import annotations

from atrader.bus.memory import InMemoryBus
from atrader.bus.topics import bar_topic, signal_topic, tick_topic
from atrader.core.clock import SimulatedClock


class TestTopicNaming:
    def test_topic_helpers_match_the_spec(self) -> None:
        assert tick_topic("AAPL") == "md.tick.AAPL"
        assert bar_topic("1m") == "md.bar.1m"
        assert signal_topic("momentum_v3") == "signal.momentum_v3"


class TestDelivery:
    async def test_publish_then_receive(self) -> None:
        bus = InMemoryBus(SimulatedClock())
        subscription = await bus.subscribe("intent.new", group="risk")
        await bus.publish("intent.new", {"symbol": "AAPL"})

        message = await subscription.next(timeout_s=0.5)
        assert message is not None
        assert message.payload == {"symbol": "AAPL"}

    async def test_wildcard_subscription(self) -> None:
        bus = InMemoryBus(SimulatedClock())
        subscription = await bus.subscribe("md.tick.*", group="strategy")
        await bus.publish(tick_topic("AAPL"), {"last": "187.50"})
        await bus.publish(tick_topic("MSFT"), {"last": "410.00"})

        first = await subscription.next(timeout_s=0.5)
        second = await subscription.next(timeout_s=0.5)
        assert first is not None and second is not None
        assert {first.topic, second.topic} == {"md.tick.AAPL", "md.tick.MSFT"}

    async def test_non_matching_topics_are_not_delivered(self) -> None:
        bus = InMemoryBus(SimulatedClock())
        subscription = await bus.subscribe("md.tick.*", group="strategy")
        await bus.publish("fill.new", {"quantity": "100"})
        assert await subscription.next(timeout_s=0.05) is None

    async def test_ordering_is_preserved(self) -> None:
        # Deterministic replay depends on this (spec §2.2).
        bus = InMemoryBus(SimulatedClock())
        subscription = await bus.subscribe("order.update", group="oms")
        for i in range(50):
            await bus.publish("order.update", {"seq": i})

        received = [
            message.payload["seq"]
            for _ in range(50)
            if (message := await subscription.next(timeout_s=0.5)) is not None
        ]
        assert received == list(range(50))

    async def test_each_group_gets_its_own_copy(self) -> None:
        bus = InMemoryBus(SimulatedClock())
        risk = await bus.subscribe("intent.new", group="risk")
        audit = await bus.subscribe("intent.new", group="audit")
        await bus.publish("intent.new", {"symbol": "AAPL"})

        assert await risk.next(timeout_s=0.5) is not None
        assert await audit.next(timeout_s=0.5) is not None

    async def test_members_of_one_group_share_the_stream(self) -> None:
        # Otherwise scaling out a consumer would process every message twice.
        bus = InMemoryBus(SimulatedClock())
        first = await bus.subscribe("intent.new", group="risk", consumer="a")
        second = await bus.subscribe("intent.new", group="risk", consumer="b")
        await bus.publish("intent.new", {"symbol": "AAPL"})

        delivered = [
            message
            for message in (
                await first.next(timeout_s=0.05),
                await second.next(timeout_s=0.05),
            )
            if message is not None
        ]
        assert len(delivered) == 1


class TestAtLeastOnce:
    async def test_unacked_messages_are_pending(self) -> None:
        bus = InMemoryBus(SimulatedClock())
        subscription = await bus.subscribe("fill.new", group="portfolio")
        await bus.publish("fill.new", {"quantity": "100"})

        message = await subscription.next(timeout_s=0.5)
        assert message is not None
        assert subscription.pending_count == 1

        await subscription.ack(message)
        assert subscription.pending_count == 0

    async def test_a_crashed_consumer_gets_its_messages_back(self) -> None:
        # This is the whole reason handlers must be idempotent.
        bus = InMemoryBus(SimulatedClock())
        subscription = await bus.subscribe("fill.new", group="portfolio")
        await bus.publish("fill.new", {"broker_fill_id": "exec-1"})

        first = await subscription.next(timeout_s=0.5)
        assert first is not None  # delivered, then the consumer "crashes"

        assert await subscription.redeliver_pending() == 1
        redelivered = await subscription.next(timeout_s=0.5)
        assert redelivered is not None
        assert redelivered.payload == first.payload
        assert redelivered.delivery_count == 2

    async def test_nack_requeues_at_the_front(self) -> None:
        bus = InMemoryBus(SimulatedClock())
        subscription = await bus.subscribe("fill.new", group="portfolio")
        await bus.publish("fill.new", {"n": 1})
        await bus.publish("fill.new", {"n": 2})

        first = await subscription.next(timeout_s=0.5)
        assert first is not None
        await subscription.nack(first)

        again = await subscription.next(timeout_s=0.5)
        assert again is not None
        assert again.payload == {"n": 1}
        assert again.delivery_count == 2


class TestBackpressure:
    async def test_oldest_messages_are_dropped_and_counted(self) -> None:
        # Spec §5.4: drop under load if you must, but never silently.
        bus = InMemoryBus(SimulatedClock(), max_queue=10)
        subscription = await bus.subscribe("md.tick.*", group="slow")
        for i in range(25):
            await bus.publish(tick_topic("AAPL"), {"seq": i})

        assert bus.dropped == 15
        first = await subscription.next(timeout_s=0.5)
        assert first is not None
        assert first.payload["seq"] == 15  # the oldest 15 were discarded


class TestHeaders:
    async def test_trace_id_propagates_to_the_consumer(self) -> None:
        # Spec §9.2: one trace id from signal generation through to the fill.
        bus = InMemoryBus(SimulatedClock())
        subscription = await bus.subscribe("intent.new", group="risk")
        await bus.publish("intent.new", {"symbol": "AAPL"}, headers={"trace_id": "trace-123"})

        message = await subscription.next(timeout_s=0.5)
        assert message is not None
        assert message.trace_id == "trace-123"


class TestIteration:
    async def test_async_iteration_yields_messages(self) -> None:
        bus = InMemoryBus(SimulatedClock())
        subscription = await bus.subscribe("intent.new", group="risk")
        for i in range(3):
            await bus.publish("intent.new", {"n": i})

        received = []
        async for message in subscription:
            received.append(message.payload["n"])
            await subscription.ack(message)
            if len(received) == 3:
                await subscription.close()
        assert received == [0, 1, 2]
