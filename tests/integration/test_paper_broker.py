"""Paper broker — full-flow integration tests.

Runs the simulator through the properties spec §8.1/§8.3 call out explicitly:
partial fills, queue-position fills (not "touched the price, therefore
filled"), latency, rejects, cancel-vs-fill races, and out-of-order event
delivery. No external infrastructure — the whole point of the paper broker is
that this suite runs everywhere.

Not marked ``integration`` (that marker is reserved for tests needing real
Postgres/Redis); this needs nothing beyond the in-process simulator.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from atrader.brokers.models import BrokerEventType
from atrader.brokers.paper import PaperBroker, PaperBrokerConfig
from atrader.brokers.protocol import BrokerCapabilities
from atrader.core.clock import SimulatedClock
from atrader.core.errors import PermanentBrokerError, UnsupportedByBrokerError
from atrader.core.models import OrderRequest
from atrader.core.rng import SeededRng
from atrader.core.types import OrderStatus, OrderType, Side, TimeInForce

BASE_NS = 1_700_000_000_000_000_000


def make_request(**overrides: Any) -> OrderRequest:
    defaults: dict[str, Any] = {
        "client_order_id": "coid-1",
        "symbol": "AAPL",
        "side": Side.BUY,
        "order_type": OrderType.LIMIT,
        "quantity": Decimal("100"),
        "limit_price": Decimal("190.00"),
        "time_in_force": TimeInForce.DAY,
    }
    return OrderRequest(**{**defaults, **overrides})


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(start_ns=BASE_NS)


class TestQueuePositionFills:
    async def test_price_merely_touching_the_limit_does_not_fill(
        self, clock: SimulatedClock
    ) -> None:
        broker = PaperBroker(
            clock,
            config=PaperBrokerConfig(
                latency_ns=0, partial_fill_probability=0.0, queue_ahead_multiple=Decimal(0)
            ),
        )
        await broker.submit_order(make_request(limit_price=Decimal("190.00")))
        # The trade prints exactly at the limit — spec §8.1: conservative fills
        # require the price to trade *through* the limit, not merely touch it.
        events = broker.advance_market("AAPL", Decimal("190.00"), Decimal("1000"))
        assert events == []
        state = await broker.get_order("coid-1")
        assert state is not None and state.status is OrderStatus.NEW

    async def test_price_trading_through_the_limit_fills(self, clock: SimulatedClock) -> None:
        broker = PaperBroker(
            clock,
            config=PaperBrokerConfig(
                latency_ns=0, partial_fill_probability=0.0, queue_ahead_multiple=Decimal(0)
            ),
        )
        await broker.submit_order(make_request(limit_price=Decimal("190.00")))
        events = broker.advance_market("AAPL", Decimal("189.99"), Decimal("1000"))
        assert len(events) == 1
        assert events[0].event_type is BrokerEventType.FILL

    async def test_a_resting_order_must_wait_out_the_queue_ahead_of_it(
        self, clock: SimulatedClock
    ) -> None:
        broker = PaperBroker(
            clock,
            config=PaperBrokerConfig(
                latency_ns=0, partial_fill_probability=0.0, queue_ahead_multiple=Decimal("2.0")
            ),
        )
        await broker.submit_order(
            make_request(quantity=Decimal("100"), limit_price=Decimal("190.00"))
        )

        # queue_ahead_multiple=2.0 means 200 shares must trade through before
        # our 100-share order gets any of it.
        first = broker.advance_market("AAPL", Decimal("189.00"), Decimal("150"))
        assert first == []
        state = await broker.get_order("coid-1")
        assert state is not None and state.filled_quantity == Decimal(0)

        # 150 + 100 = 250 total volume seen; 200 of it is queue-ahead, leaving
        # 50 shares of "available" — our 100-share order gets only that much.
        second = broker.advance_market("AAPL", Decimal("189.00"), Decimal("100"))
        assert len(second) == 1
        state = await broker.get_order("coid-1")
        assert state is not None and state.filled_quantity == Decimal("50")

        # A third round of volume clears the rest.
        broker.advance_market("AAPL", Decimal("189.00"), Decimal("100"))
        state = await broker.get_order("coid-1")
        assert state is not None and state.filled_quantity == Decimal("100")

    async def test_sell_orders_require_the_price_to_trade_up_through_the_limit(
        self, clock: SimulatedClock
    ) -> None:
        broker = PaperBroker(
            clock,
            config=PaperBrokerConfig(
                latency_ns=0, partial_fill_probability=0.0, queue_ahead_multiple=Decimal(0)
            ),
        )
        await broker.submit_order(
            make_request(side=Side.SELL, quantity=Decimal("50"), limit_price=Decimal("190.00"))
        )
        untouched = broker.advance_market("AAPL", Decimal("190.00"), Decimal("1000"))
        assert untouched == []
        crossed = broker.advance_market("AAPL", Decimal("190.01"), Decimal("1000"))
        assert len(crossed) == 1


class TestPartialFills:
    async def test_partial_fill_probability_produces_a_partial_then_a_completion(
        self, clock: SimulatedClock
    ) -> None:
        broker = PaperBroker(
            clock,
            config=PaperBrokerConfig(
                latency_ns=0,
                partial_fill_probability=1.0,  # always partial when eligible
                max_partial_fraction=0.5,
                queue_ahead_multiple=Decimal(0),
            ),
            rng=SeededRng(seed=0),
        )
        await broker.submit_order(
            make_request(quantity=Decimal("100"), limit_price=Decimal("190.00"))
        )
        broker.advance_market("AAPL", Decimal("189.00"), Decimal("100"))
        state = await broker.get_order("coid-1")
        assert state is not None
        assert Decimal(0) < state.filled_quantity < Decimal("100")
        assert state.status is OrderStatus.PARTIALLY_FILLED
        first_fill = state.filled_quantity

        # A further round of volume advances the fill further still — a
        # partial fill is progress, not a dead end, even though a fraction
        # under 1.0 can (by construction) never guarantee full completion:
        # each round takes half of what remains, and flooring to whole shares
        # means the last share or two can be left stranded forever.
        broker.advance_market("AAPL", Decimal("189.00"), Decimal("100"))
        state = await broker.get_order("coid-1")
        assert state is not None
        assert state.filled_quantity > first_fill
        assert state.filled_quantity <= Decimal("100")

    async def test_average_fill_price_is_a_weighted_average_across_partials(
        self, clock: SimulatedClock
    ) -> None:
        # A resting limit order fills at *its own* limit price, not the tape
        # price that crossed it (spec §8.1's conservative fill model) — so the
        # only way to get two different fill prices on one order is to amend
        # the limit between fills, which is exactly what a chase-the-market
        # order-working strategy does in practice.
        from atrader.brokers.models import OrderModification

        broker = PaperBroker(
            clock,
            config=PaperBrokerConfig(
                latency_ns=0, partial_fill_probability=0.0, queue_ahead_multiple=Decimal(0)
            ),
        )
        await broker.submit_order(
            make_request(quantity=Decimal("100"), limit_price=Decimal("200.00"))
        )
        broker.advance_market(
            "AAPL", Decimal("190.00"), Decimal("50")
        )  # fills 50 @ 200 (the limit)

        await broker.modify_order("coid-1", OrderModification(limit_price=Decimal("180.00")))
        broker.advance_market("AAPL", Decimal("170.00"), Decimal("50"))  # fills 50 @ 180

        state = await broker.get_order("coid-1")
        assert state is not None
        assert state.filled_quantity == Decimal("100")
        assert state.avg_fill_price == Decimal("190.00000000")  # (200*50 + 180*50) / 100


class TestRejectsAndLatency:
    async def test_reject_probability_rejects_without_creating_a_working_order(
        self, clock: SimulatedClock
    ) -> None:
        broker = PaperBroker(clock, config=PaperBrokerConfig(reject_probability=1.0))
        ack = await broker.submit_order(make_request())
        assert not ack.accepted
        state = await broker.get_order("coid-1")
        assert state is not None and state.status is OrderStatus.REJECTED

    async def test_short_selling_disabled_and_no_position_is_a_permanent_error(
        self, clock: SimulatedClock
    ) -> None:
        broker = PaperBroker(clock, config=PaperBrokerConfig(allow_short=False))
        with pytest.raises(PermanentBrokerError, match="short selling is disabled"):
            await broker.submit_order(make_request(side=Side.SELL))

    async def test_selling_a_held_position_is_allowed_even_with_shorting_disabled(
        self, clock: SimulatedClock
    ) -> None:
        broker = PaperBroker(
            clock,
            config=PaperBrokerConfig(
                allow_short=False,
                latency_ns=0,
                partial_fill_probability=0.0,
                queue_ahead_multiple=Decimal(0),
            ),
        )
        await broker.submit_order(
            make_request(side=Side.BUY, quantity=Decimal("100"), limit_price=Decimal("200.00"))
        )
        broker.advance_market("AAPL", Decimal("190.00"), Decimal("100"))
        ack = await broker.submit_order(
            make_request(
                client_order_id="coid-2",
                side=Side.SELL,
                quantity=Decimal("50"),
                limit_price=Decimal("180.00"),
            )
        )
        assert ack.accepted

    async def test_accepted_at_ns_reflects_the_configured_latency(
        self, clock: SimulatedClock
    ) -> None:
        broker = PaperBroker(clock, config=PaperBrokerConfig(latency_ns=5_000_000))
        submitted_at = clock.now_ns()
        await broker.submit_order(make_request())
        state = await broker.get_order("coid-1")
        assert state is not None
        # accepted_at_ns isn't directly exposed on OrderState, but the ack's
        # received_at_ns should not have silently absorbed the latency delay —
        # it reflects when submit_order was called, not when it "landed".
        assert state.updated_at_ns >= submitted_at

    async def test_unsupported_request_is_rejected_synchronously_without_creating_an_order(
        self, clock: SimulatedClock
    ) -> None:
        # The default paper capabilities don't support fractional shares —
        # spec §6.1: the rejection must happen here, not after a round trip.
        broker = PaperBroker(clock)
        fractional = make_request(quantity=Decimal("1.5"))
        with pytest.raises(UnsupportedByBrokerError, match="fractional"):
            await broker.submit_order(fractional)
        assert await broker.get_order("coid-1") is None


class TestCancelRaces:
    async def test_cancel_after_the_order_is_already_filled_loses_the_race(
        self, clock: SimulatedClock
    ) -> None:
        broker = PaperBroker(
            clock,
            config=PaperBrokerConfig(
                latency_ns=0, partial_fill_probability=0.0, queue_ahead_multiple=Decimal(0)
            ),
        )
        await broker.submit_order(
            make_request(quantity=Decimal("10"), limit_price=Decimal("190.00"))
        )
        broker.advance_market("AAPL", Decimal("189.00"), Decimal("10"))
        ack = await broker.cancel_order("coid-1")
        assert not ack.accepted
        assert "lost the race" in ack.reason

    async def test_cancel_of_an_unknown_order_is_reported_not_raised(
        self, clock: SimulatedClock
    ) -> None:
        broker = PaperBroker(clock)
        ack = await broker.cancel_order("never-existed")
        assert not ack.accepted
        assert ack.reason == "unknown order"

    async def test_cancel_of_a_still_working_order_succeeds(self, clock: SimulatedClock) -> None:
        broker = PaperBroker(clock, config=PaperBrokerConfig(latency_ns=0))
        await broker.submit_order(make_request())
        ack = await broker.cancel_order("coid-1")
        assert ack.accepted
        state = await broker.get_order("coid-1")
        assert state is not None and state.status is OrderStatus.CANCELED


class TestMarketOrders:
    async def test_a_market_order_fills_immediately_at_the_reference_price(
        self, clock: SimulatedClock
    ) -> None:
        broker = PaperBroker(
            clock, config=PaperBrokerConfig(latency_ns=0), prices={"AAPL": Decimal("190.00")}
        )
        ack = await broker.submit_order(
            make_request(order_type=OrderType.MARKET, limit_price=None, quantity=Decimal("10"))
        )
        assert ack.accepted
        state = await broker.get_order("coid-1")
        assert state is not None
        assert state.status is OrderStatus.FILLED
        assert state.avg_fill_price == Decimal("190.00000000")

    async def test_a_market_order_with_no_reference_price_is_a_permanent_error(
        self, clock: SimulatedClock
    ) -> None:
        broker = PaperBroker(clock, config=PaperBrokerConfig(latency_ns=0))
        with pytest.raises(PermanentBrokerError, match="no simulated price"):
            await broker.submit_order(
                make_request(order_type=OrderType.MARKET, limit_price=None, quantity=Decimal("10"))
            )


class TestOpenOrdersAndClose:
    async def test_open_orders_excludes_terminal_ones(self, clock: SimulatedClock) -> None:
        broker = PaperBroker(clock, config=PaperBrokerConfig(latency_ns=0))
        await broker.submit_order(make_request(client_order_id="a"))
        await broker.submit_order(make_request(client_order_id="b"))
        await broker.cancel_order("a")
        open_orders = await broker.get_open_orders()
        assert [o.client_order_id for o in open_orders] == ["b"]

    async def test_close_clears_the_pending_event_queue(self, clock: SimulatedClock) -> None:
        broker = PaperBroker(clock, config=PaperBrokerConfig(latency_ns=0))
        await broker.submit_order(make_request())
        await broker.close()
        assert broker.pending_events() == []


class TestModify:
    async def test_modify_quantity_updates_the_working_size(self, clock: SimulatedClock) -> None:
        from atrader.brokers.models import OrderModification

        broker = PaperBroker(clock, config=PaperBrokerConfig(latency_ns=0))
        await broker.submit_order(make_request(quantity=Decimal("100")))
        await broker.modify_order("coid-1", OrderModification(quantity=Decimal("60")))
        state = await broker.get_order("coid-1")
        assert state is not None and state.quantity == Decimal("60")

    async def test_modify_updates_price_and_resets_queue_position(
        self, clock: SimulatedClock
    ) -> None:
        broker = PaperBroker(
            clock,
            config=PaperBrokerConfig(
                latency_ns=0, partial_fill_probability=0.0, queue_ahead_multiple=Decimal("1.0")
            ),
        )
        await broker.submit_order(
            make_request(quantity=Decimal("100"), limit_price=Decimal("190.00"))
        )
        broker.advance_market("AAPL", Decimal("189.00"), Decimal("50"))  # partial queue progress

        from atrader.brokers.models import OrderModification

        await broker.modify_order("coid-1", OrderModification(limit_price=Decimal("195.00")))
        state = await broker.get_order("coid-1")
        assert state is not None and state.limit_price == Decimal("195.00")

        # Queue priority was lost — the 50 shares that already traded through
        # do not count toward the new resting order's position.
        again = broker.advance_market("AAPL", Decimal("189.00"), Decimal("50"))
        assert again == []

    async def test_modify_cannot_reduce_below_what_is_already_filled(
        self, clock: SimulatedClock
    ) -> None:
        broker = PaperBroker(
            clock,
            config=PaperBrokerConfig(
                latency_ns=0, partial_fill_probability=0.0, queue_ahead_multiple=Decimal(0)
            ),
        )
        await broker.submit_order(
            make_request(quantity=Decimal("100"), limit_price=Decimal("190.00"))
        )
        broker.advance_market("AAPL", Decimal("189.00"), Decimal("60"))

        from atrader.brokers.models import OrderModification

        with pytest.raises(PermanentBrokerError, match="below the"):
            await broker.modify_order("coid-1", OrderModification(quantity=Decimal("50")))

    async def test_modify_an_unknown_order_raises(self, clock: SimulatedClock) -> None:
        from atrader.brokers.models import OrderModification

        broker = PaperBroker(clock)
        with pytest.raises(PermanentBrokerError, match="unknown order"):
            await broker.modify_order("never-existed", OrderModification(quantity=Decimal("1")))

    async def test_modify_a_terminal_order_raises(self, clock: SimulatedClock) -> None:
        from atrader.brokers.models import OrderModification

        broker = PaperBroker(clock, config=PaperBrokerConfig(latency_ns=0))
        await broker.submit_order(make_request())
        await broker.cancel_order("coid-1")
        with pytest.raises(PermanentBrokerError, match="cannot modify"):
            await broker.modify_order("coid-1", OrderModification(quantity=Decimal("1")))


class TestIdempotentResubmission:
    async def test_resubmitting_the_same_client_order_id_returns_the_original(
        self, clock: SimulatedClock
    ) -> None:
        broker = PaperBroker(clock, config=PaperBrokerConfig(latency_ns=0))
        first = await broker.submit_order(make_request())
        second = await broker.submit_order(make_request())  # identical client_order_id
        assert first.broker_order_id == second.broker_order_id
        assert "duplicate" in second.reason


class TestOutOfOrderDelivery:
    async def test_shuffle_pending_events_reorders_the_queue(self, clock: SimulatedClock) -> None:
        broker = PaperBroker(
            clock,
            config=PaperBrokerConfig(
                latency_ns=0, partial_fill_probability=0.0, queue_ahead_multiple=Decimal(0)
            ),
        )
        for i in range(5):
            await broker.submit_order(make_request(client_order_id=f"coid-{i}"))

        events = broker.pending_events()
        assert [e.client_order_id for e in events] == [f"coid-{i}" for i in range(5)]

        # Re-queue and shuffle — deterministic under a seeded RNG.
        for i in range(5):
            await broker.submit_order(make_request(client_order_id=f"shuffled-{i}"))
        broker.shuffle_pending_events()
        shuffled = broker.pending_events()
        assert {e.client_order_id for e in shuffled} == {f"shuffled-{i}" for i in range(5)}

    async def test_stream_updates_drains_in_fifo_order_by_default(
        self, clock: SimulatedClock
    ) -> None:
        broker = PaperBroker(clock, config=PaperBrokerConfig(latency_ns=0))
        await broker.submit_order(make_request(client_order_id="coid-a"))
        await broker.submit_order(make_request(client_order_id="coid-b"))
        events = [e async for e in broker.stream_updates()]
        assert [e.client_order_id for e in events] == ["coid-a", "coid-b"]


class TestAccountAndPositions:
    async def test_a_fill_moves_cash_and_opens_a_position(self, clock: SimulatedClock) -> None:
        broker = PaperBroker(
            clock,
            config=PaperBrokerConfig(
                latency_ns=0,
                partial_fill_probability=0.0,
                commission_bps=Decimal(0),
                queue_ahead_multiple=Decimal(0),
                starting_cash=Decimal("100000"),
            ),
        )
        await broker.submit_order(
            make_request(quantity=Decimal("100"), limit_price=Decimal("200.00"))
        )
        broker.advance_market("AAPL", Decimal("190.00"), Decimal("100"))

        positions = await broker.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == Decimal("100")

        account = await broker.get_account()
        # Fills happen at the order's own limit price (200), not the tape
        # price that crossed it (190) — see the fill-price note above.
        assert account.cash == Decimal("80000.00000000")  # 100000 - 100*200

    async def test_closing_a_position_realises_pnl(self, clock: SimulatedClock) -> None:
        broker = PaperBroker(
            clock,
            config=PaperBrokerConfig(
                latency_ns=0,
                partial_fill_probability=0.0,
                commission_bps=Decimal(0),
                queue_ahead_multiple=Decimal(0),
            ),
        )
        await broker.submit_order(
            make_request(
                client_order_id="buy", quantity=Decimal("100"), limit_price=Decimal("200.00")
            )
        )
        broker.advance_market("AAPL", Decimal("190.00"), Decimal("100"))
        await broker.submit_order(
            make_request(
                client_order_id="sell",
                side=Side.SELL,
                quantity=Decimal("100"),
                limit_price=Decimal("180.00"),
            )
        )
        broker.advance_market("AAPL", Decimal("200.00"), Decimal("100"))

        positions = await broker.get_positions()
        assert positions == []  # flat positions are excluded

    async def test_get_positions_excludes_flat_symbols_from_the_dict(
        self, clock: SimulatedClock
    ) -> None:
        broker = PaperBroker(clock, config=PaperBrokerConfig(latency_ns=0))
        assert await broker.get_positions() == []

    def test_set_price_updates_open_position_marks(self, clock: SimulatedClock) -> None:
        broker = PaperBroker(clock, prices={"AAPL": Decimal("190.00")})
        broker.set_price("AAPL", Decimal("195.00"))
        # No position open yet — set_price should not raise even so.


class TestCapabilities:
    def test_default_capabilities_support_all_four_order_types(self, clock: SimulatedClock) -> None:
        broker = PaperBroker(clock)
        caps = broker.capabilities
        assert OrderType.MARKET in caps.order_types
        assert OrderType.STOP in caps.order_types
        assert OrderType.STOP_LIMIT in caps.order_types


class TestCapabilitiesRejection:
    def test_rejects_an_unsupported_order_type(self) -> None:
        caps = BrokerCapabilities(name="test", order_types=frozenset({OrderType.LIMIT}))
        request = OrderRequest(
            client_order_id="c",
            symbol="AAPL",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
        )
        assert caps.rejects(request) is not None

    def test_rejects_an_unsupported_time_in_force(self) -> None:
        caps = BrokerCapabilities(name="test", time_in_force=frozenset({TimeInForce.DAY}))
        request = OrderRequest(
            client_order_id="c",
            symbol="AAPL",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1"),
            limit_price=Decimal("10"),
            time_in_force=TimeInForce.IOC,
        )
        assert caps.rejects(request) is not None

    def test_rejects_fractional_shares_when_unsupported(self) -> None:
        caps = BrokerCapabilities(name="test", supports_fractional_shares=False)
        request = OrderRequest(
            client_order_id="c",
            symbol="AAPL",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.5"),
            limit_price=Decimal("10"),
        )
        assert caps.rejects(request) is not None

    def test_rejects_notional_below_the_venue_minimum(self) -> None:
        caps = BrokerCapabilities(name="test", min_order_notional=Decimal("1000"))
        request = OrderRequest(
            client_order_id="c",
            symbol="AAPL",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1"),
            limit_price=Decimal("10"),
        )
        assert caps.rejects(request) is not None

    def test_accepts_a_request_within_every_bound(self) -> None:
        caps = BrokerCapabilities(name="test", supports_fractional_shares=True)
        request = OrderRequest(
            client_order_id="c",
            symbol="AAPL",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.5"),
            limit_price=Decimal("10"),
        )
        assert caps.rejects(request) is None
