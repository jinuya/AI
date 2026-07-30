"""The backtest broker — spec §2.2 (same BrokerAdapter Protocol, bar-driven)."""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from atrader.backtest.broker import BacktestBroker
from atrader.backtest.cost_model import CostModel
from atrader.backtest.fill_model import ConservativeFillModel
from atrader.brokers.models import BrokerEventType, OrderModification
from atrader.core.clock import SimulatedClock
from atrader.core.errors import PermanentBrokerError, UnsupportedByBrokerError
from atrader.core.models import OrderRequest
from atrader.core.types import OrderStatus, OrderType, Side, TimeInForce
from atrader.marketdata.models import Bar

BASE_NS = 1_700_000_000_000_000_000


def make_bar(*, open_: str, high: str, low: str, close: str, volume: str = "10000") -> Bar:
    return Bar(
        symbol="AAPL",
        interval="1d",
        open_ts=BASE_NS,
        close_ts=BASE_NS + 1,
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal(volume),
        is_final=True,
    )


def make_request(
    *,
    side: Side = Side.BUY,
    order_type: OrderType = OrderType.MARKET,
    limit_price: str | None = None,
) -> OrderRequest:
    return OrderRequest(
        client_order_id=str(uuid4()),
        symbol="AAPL",
        side=side,
        order_type=order_type,
        quantity=Decimal("10"),
        limit_price=Decimal(limit_price) if limit_price else None,
        time_in_force=TimeInForce.DAY,
    )


class TestSubmitAndAdvance:
    async def test_a_fresh_order_does_not_fill_until_advance_bar_is_called(self) -> None:
        broker = BacktestBroker(clock=SimulatedClock(start_ns=BASE_NS))
        request = make_request()
        ack = await broker.submit_order(request)
        assert ack.accepted
        assert ack.status is OrderStatus.NEW
        assert await broker.get_open_orders() != []

    async def test_advance_bar_fills_a_market_order_at_the_open(self) -> None:
        broker = BacktestBroker(clock=SimulatedClock(start_ns=BASE_NS))
        request = make_request()
        await broker.submit_order(request)
        bar = make_bar(open_="100", high="105", low="99", close="102")
        events = broker.advance_bar(bar)
        fills = [e for e in events if e.event_type is BrokerEventType.FILL]
        assert len(fills) == 1
        assert fills[0].price == Decimal("100")
        assert fills[0].quantity == Decimal("10")

    async def test_a_partial_fill_leaves_the_remainder_resting(self) -> None:
        broker = BacktestBroker(
            clock=SimulatedClock(start_ns=BASE_NS),
            fill_model=ConservativeFillModel(max_participation_pct=Decimal("10")),
        )
        request = make_request()
        await broker.submit_order(request)
        bar = make_bar(open_="100", high="105", low="99", close="102", volume="50")
        # cap = 10% of 50 = 5, order wants 10 -> partial fill of 5
        broker.advance_bar(bar)
        state = await broker.get_order(request.client_order_id)
        assert state is not None
        assert state.status is OrderStatus.PARTIALLY_FILLED
        assert state.filled_quantity == Decimal("5")

        # The remainder fills on the next bar.
        bar2 = make_bar(open_="101", high="106", low="100", close="103", volume="50")
        broker.advance_bar(bar2)
        state2 = await broker.get_order(request.client_order_id)
        assert state2 is not None
        assert state2.status is OrderStatus.FILLED

    async def test_a_limit_order_that_never_crosses_stays_open(self) -> None:
        broker = BacktestBroker(clock=SimulatedClock(start_ns=BASE_NS))
        request = make_request(order_type=OrderType.LIMIT, limit_price="50")
        await broker.submit_order(request)
        bar = make_bar(open_="100", high="105", low="99", close="102")
        events = broker.advance_bar(bar)
        assert [e for e in events if e.event_type is BrokerEventType.FILL] == []

    async def test_only_orders_for_the_bars_symbol_are_evaluated(self) -> None:
        broker = BacktestBroker(clock=SimulatedClock(start_ns=BASE_NS))
        request = make_request()
        await broker.submit_order(request)
        other_bar = Bar(
            symbol="MSFT",
            interval="1d",
            open_ts=BASE_NS,
            close_ts=BASE_NS + 1,
            open=Decimal("400"),
            high=Decimal("405"),
            low=Decimal("399"),
            close=Decimal("402"),
            volume=Decimal("10000"),
            is_final=True,
        )
        events = broker.advance_bar(other_bar)
        assert events == []


class TestCapabilities:
    async def test_name_and_capabilities_identify_this_as_the_backtest_broker(self) -> None:
        broker = BacktestBroker(clock=SimulatedClock(start_ns=BASE_NS))
        assert broker.name == "backtest"
        assert broker.capabilities.name == "backtest"

    async def test_an_unsupported_order_type_is_rejected_synchronously(self) -> None:
        broker = BacktestBroker(clock=SimulatedClock(start_ns=BASE_NS))
        request = OrderRequest(
            client_order_id=str(uuid4()),
            symbol="AAPL",
            side=Side.BUY,
            order_type=OrderType.STOP,
            quantity=Decimal("10"),
            stop_price=Decimal("90"),
        )
        with pytest.raises(UnsupportedByBrokerError):
            await broker.submit_order(request)


class TestIdempotency:
    async def test_resubmitting_the_same_client_order_id_returns_the_existing_order(self) -> None:
        broker = BacktestBroker(clock=SimulatedClock(start_ns=BASE_NS))
        request = make_request()
        first = await broker.submit_order(request)
        second = await broker.submit_order(request)
        assert first.broker_order_id == second.broker_order_id
        assert "duplicate" in second.reason


class TestCancel:
    async def test_cancelling_a_working_order_stops_it_from_filling(self) -> None:
        broker = BacktestBroker(clock=SimulatedClock(start_ns=BASE_NS))
        request = make_request()
        await broker.submit_order(request)
        ack = await broker.cancel_order(request.client_order_id)
        assert ack.accepted

        bar = make_bar(open_="100", high="105", low="99", close="102")
        events = broker.advance_bar(bar)
        assert [e for e in events if e.event_type is BrokerEventType.FILL] == []

    async def test_cancelling_an_unknown_order_is_reported_not_raised(self) -> None:
        broker = BacktestBroker(clock=SimulatedClock(start_ns=BASE_NS))
        ack = await broker.cancel_order("nope")
        assert not ack.accepted

    async def test_cancelling_an_already_filled_order_loses_the_race_gracefully(self) -> None:
        broker = BacktestBroker(clock=SimulatedClock(start_ns=BASE_NS))
        request = make_request()
        await broker.submit_order(request)
        broker.advance_bar(make_bar(open_="100", high="105", low="99", close="102"))
        ack = await broker.cancel_order(request.client_order_id)
        assert not ack.accepted
        assert "already" in ack.reason

    async def test_modify_updates_the_resting_orders_limit_price(self) -> None:
        broker = BacktestBroker(clock=SimulatedClock(start_ns=BASE_NS))
        request = make_request(order_type=OrderType.LIMIT, limit_price="50")
        await broker.submit_order(request)
        ack = await broker.modify_order(
            request.client_order_id, OrderModification(limit_price=Decimal("101"))
        )
        assert ack.accepted
        bar = make_bar(open_="100", high="105", low="99", close="102")
        events = broker.advance_bar(bar)
        assert [e for e in events if e.event_type is BrokerEventType.FILL] != []

    async def test_modify_with_no_fields_is_a_no_op_ack(self) -> None:
        broker = BacktestBroker(clock=SimulatedClock(start_ns=BASE_NS))
        request = make_request(order_type=OrderType.LIMIT, limit_price="50")
        await broker.submit_order(request)
        ack = await broker.modify_order(request.client_order_id, OrderModification())
        assert ack.accepted

    async def test_modifying_an_unknown_order_raises(self) -> None:
        broker = BacktestBroker(clock=SimulatedClock(start_ns=BASE_NS))
        with pytest.raises(PermanentBrokerError, match="unknown order"):
            await broker.modify_order("nope", OrderModification(limit_price=Decimal("101")))

    async def test_modifying_a_terminal_order_raises(self) -> None:
        broker = BacktestBroker(clock=SimulatedClock(start_ns=BASE_NS))
        request = make_request()
        await broker.submit_order(request)
        broker.advance_bar(make_bar(open_="100", high="105", low="99", close="102"))
        with pytest.raises(PermanentBrokerError, match="cannot modify"):
            await broker.modify_order(
                request.client_order_id, OrderModification(limit_price=Decimal("101"))
            )

    async def test_reducing_quantity_below_what_already_filled_raises(self) -> None:
        broker = BacktestBroker(
            clock=SimulatedClock(start_ns=BASE_NS),
            fill_model=ConservativeFillModel(max_participation_pct=Decimal("10")),
        )
        request = make_request()
        await broker.submit_order(request)
        broker.advance_bar(make_bar(open_="100", high="105", low="99", close="102", volume="50"))
        state = await broker.get_order(request.client_order_id)
        assert state is not None and state.filled_quantity == Decimal("5")
        with pytest.raises(PermanentBrokerError, match="already filled"):
            await broker.modify_order(
                request.client_order_id, OrderModification(quantity=Decimal("2"))
            )

    async def test_increasing_quantity_on_a_partially_filled_order_succeeds(self) -> None:
        broker = BacktestBroker(
            clock=SimulatedClock(start_ns=BASE_NS),
            fill_model=ConservativeFillModel(max_participation_pct=Decimal("10")),
        )
        request = make_request()
        await broker.submit_order(request)
        broker.advance_bar(make_bar(open_="100", high="105", low="99", close="102", volume="50"))
        ack = await broker.modify_order(
            request.client_order_id, OrderModification(quantity=Decimal("20"))
        )
        assert ack.accepted


class TestAccountAndPositions:
    async def test_a_fill_updates_cash_and_position(self) -> None:
        broker = BacktestBroker(
            clock=SimulatedClock(start_ns=BASE_NS),
            cost_model=CostModel(
                impact_coefficient=Decimal("0"), default_commission_bps=Decimal("0")
            ),
        )
        request = make_request()
        await broker.submit_order(request)
        bar = make_bar(open_="100", high="105", low="99", close="102")
        broker.advance_bar(bar)

        positions = await broker.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == Decimal("10")

        account = await broker.get_account()
        assert account.cash == Decimal("100000") - Decimal("1000")  # 10 shares @ 100

    async def test_flipping_a_long_to_short_realizes_pnl_and_reprices_the_remainder(self) -> None:
        broker = BacktestBroker(
            clock=SimulatedClock(start_ns=BASE_NS),
            cost_model=CostModel(
                impact_coefficient=Decimal("0"), default_commission_bps=Decimal("0")
            ),
        )
        buy = make_request(side=Side.BUY)
        await broker.submit_order(buy)
        broker.advance_bar(make_bar(open_="100", high="105", low="99", close="102"))

        sell = OrderRequest(
            client_order_id=str(uuid4()),
            symbol="AAPL",
            side=Side.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("30"),
        )
        await broker.submit_order(sell)
        broker.advance_bar(make_bar(open_="110", high="112", low="108", close="109"))

        positions = await broker.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == Decimal("-20")
        assert positions[0].avg_price == Decimal("110")
        assert positions[0].realized_pnl == Decimal("100")  # (110-100)*10 closed

    async def test_stream_updates_yields_queued_events_and_then_stops(self) -> None:
        broker = BacktestBroker(clock=SimulatedClock(start_ns=BASE_NS))
        await broker.submit_order(make_request())
        seen = [event async for event in broker.stream_updates()]
        assert len(seen) == 1
        assert seen[0].event_type is BrokerEventType.ORDER_ACCEPTED
        assert broker.pending_events() == []

    async def test_get_order_for_an_unknown_client_order_id_is_none(self) -> None:
        broker = BacktestBroker(clock=SimulatedClock(start_ns=BASE_NS))
        assert await broker.get_order("nope") is None

    async def test_close_clears_the_pending_event_queue(self) -> None:
        broker = BacktestBroker(clock=SimulatedClock(start_ns=BASE_NS))
        request = make_request()
        await broker.submit_order(request)
        assert broker.pending_events() != []
        await broker.submit_order(make_request())
        await broker.close()
        assert broker.pending_events() == []


class TestOpenPositionsAreMarkedToMarket:
    """``last_price`` was written only at fill time, so ``get_account()`` —
    and with it the equity curve, the daily-loss input and the drawdown input
    — stayed frozen at the last traded price. A position could halve while
    reported equity did not move, which is precisely the scenario the loss
    limit and the drawdown breaker exist to catch: neither could ever fire on
    an open position, and every performance metric was computed from a curve
    blind to unrealized P&L.
    """

    async def _long_ten(self) -> BacktestBroker:
        broker = BacktestBroker(clock=SimulatedClock(start_ns=BASE_NS))
        await broker.submit_order(make_request())
        broker.advance_bar(make_bar(open_="100", high="100", low="100", close="100"))
        return broker

    async def test_equity_follows_the_bar_close(self) -> None:
        broker = await self._long_ten()
        opening = (await broker.get_account()).equity

        broker.advance_bar(make_bar(open_="200", high="200", low="200", close="200"))

        assert (await broker.get_account()).equity > opening

    async def test_a_falling_price_reduces_equity(self) -> None:
        broker = await self._long_ten()
        opening = (await broker.get_account()).equity

        broker.advance_bar(make_bar(open_="50", high="50", low="50", close="50"))

        assert (await broker.get_account()).equity < opening

    async def test_the_position_carries_the_latest_mark(self) -> None:
        broker = await self._long_ten()
        broker.advance_bar(make_bar(open_="175", high="175", low="175", close="175"))

        (position,) = await broker.get_positions()
        assert position.last_price == Decimal("175")

    async def test_a_flat_book_is_unaffected(self) -> None:
        broker = BacktestBroker(clock=SimulatedClock(start_ns=BASE_NS))
        before = (await broker.get_account()).equity

        broker.advance_bar(make_bar(open_="500", high="500", low="500", close="500"))

        assert (await broker.get_account()).equity == before
