"""Position book — cost-basis arithmetic (spec §FR-PF-01)."""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from atrader.core.clock import SimulatedClock
from atrader.core.models import Fill
from atrader.core.types import CostBasisMethod, Side
from atrader.portfolio.pnl import mark_to_market
from atrader.portfolio.positions import PositionBook
from atrader.storage.memory import InMemoryPositionStore

BASE_NS = 1_700_000_000_000_000_000


def make_fill(
    *,
    symbol: str = "AAPL",
    side: Side,
    quantity: str,
    price: str,
    at_ns: int = BASE_NS,
) -> Fill:
    return Fill(
        fill_id=uuid4(),
        order_id=uuid4(),
        symbol=symbol,
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        executed_at_ns=at_ns,
    )


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(start_ns=BASE_NS)


@pytest.fixture
def store() -> InMemoryPositionStore:
    return InMemoryPositionStore()


class TestAverageCostBasis:
    def method(self) -> CostBasisMethod:
        return CostBasisMethod.AVERAGE

    def test_opening_a_position_sets_quantity_and_price(
        self, store: InMemoryPositionStore, clock: SimulatedClock
    ) -> None:
        book = PositionBook(store=store, clock=clock, method=self.method())
        result = book.apply_fill(make_fill(side=Side.BUY, quantity="100", price="150"))
        assert result.position.quantity == Decimal("100")
        assert result.position.avg_price == Decimal("150")
        assert result.position.realized_pnl == Decimal("0")
        assert result.closed_quantity == Decimal("0")
        assert result.position.opened_at_ns == BASE_NS

    def test_adding_to_a_long_updates_the_weighted_average(
        self, store: InMemoryPositionStore, clock: SimulatedClock
    ) -> None:
        book = PositionBook(store=store, clock=clock, method=self.method())
        book.apply_fill(make_fill(side=Side.BUY, quantity="100", price="100"))
        result = book.apply_fill(make_fill(side=Side.BUY, quantity="100", price="120"))
        assert result.position.quantity == Decimal("200")
        assert result.position.avg_price == Decimal("110")

    def test_partially_closing_a_long_realizes_pnl_at_the_average_price(
        self, store: InMemoryPositionStore, clock: SimulatedClock
    ) -> None:
        book = PositionBook(store=store, clock=clock, method=self.method())
        book.apply_fill(make_fill(side=Side.BUY, quantity="100", price="100"))
        result = book.apply_fill(make_fill(side=Side.SELL, quantity="40", price="130"))
        assert result.position.quantity == Decimal("60")
        assert result.position.avg_price == Decimal("100")  # cost basis unchanged by a reduce
        assert result.realized_pnl_delta == Decimal("1200")  # (130-100)*40
        assert result.closed_quantity == Decimal("40")

    def test_closing_a_long_exactly_flattens_and_clears_cost_basis(
        self, store: InMemoryPositionStore, clock: SimulatedClock
    ) -> None:
        book = PositionBook(store=store, clock=clock, method=self.method())
        book.apply_fill(make_fill(side=Side.BUY, quantity="100", price="100"))
        result = book.apply_fill(make_fill(side=Side.SELL, quantity="100", price="110"))
        assert result.position.quantity == Decimal("0")
        assert result.position.avg_price == Decimal("0")
        assert result.position.opened_at_ns is None
        assert result.realized_pnl_delta == Decimal("1000")

    def test_flipping_past_flat_realizes_the_close_and_opens_the_other_side(
        self, store: InMemoryPositionStore, clock: SimulatedClock
    ) -> None:
        book = PositionBook(store=store, clock=clock, method=self.method())
        book.apply_fill(make_fill(side=Side.BUY, quantity="100", price="100"))
        result = book.apply_fill(make_fill(side=Side.SELL, quantity="150", price="120"))
        assert result.position.quantity == Decimal("-50")
        assert result.position.avg_price == Decimal("120")  # fresh entry price for the short
        assert result.realized_pnl_delta == Decimal("2000")  # (120-100)*100 closed
        assert result.closed_quantity == Decimal("100")

    def test_short_realized_pnl_uses_entry_minus_exit(
        self, store: InMemoryPositionStore, clock: SimulatedClock
    ) -> None:
        book = PositionBook(store=store, clock=clock, method=self.method())
        book.apply_fill(make_fill(side=Side.SELL, quantity="50", price="100"))
        result = book.apply_fill(make_fill(side=Side.BUY, quantity="50", price="80"))
        assert result.position.quantity == Decimal("0")
        assert result.realized_pnl_delta == Decimal(
            "1000"
        )  # (100-80)*50, a short profits on a drop

    def test_realized_pnl_accumulates_across_multiple_closes(
        self, store: InMemoryPositionStore, clock: SimulatedClock
    ) -> None:
        book = PositionBook(store=store, clock=clock, method=self.method())
        book.apply_fill(make_fill(side=Side.BUY, quantity="100", price="100"))
        book.apply_fill(make_fill(side=Side.SELL, quantity="50", price="110"))
        result = book.apply_fill(make_fill(side=Side.SELL, quantity="50", price="120"))
        assert result.position.realized_pnl == Decimal("500") + Decimal("1000")


class TestFifoCostBasis:
    def method(self) -> CostBasisMethod:
        return CostBasisMethod.FIFO

    def test_closing_consumes_the_oldest_lot_first(
        self, store: InMemoryPositionStore, clock: SimulatedClock
    ) -> None:
        book = PositionBook(store=store, clock=clock, method=self.method())
        book.apply_fill(make_fill(side=Side.BUY, quantity="100", price="100"))
        book.apply_fill(make_fill(side=Side.BUY, quantity="100", price="200"))
        # Closing 100 should realize against the first (cheaper) lot only.
        result = book.apply_fill(make_fill(side=Side.SELL, quantity="100", price="150"))
        assert result.position.quantity == Decimal("100")
        assert result.position.avg_price == Decimal("200")  # only the second lot remains
        assert result.realized_pnl_delta == Decimal("5000")  # (150-100)*100

    def test_closing_exactly_flattens_and_clears_the_lot_queue(
        self, store: InMemoryPositionStore, clock: SimulatedClock
    ) -> None:
        book = PositionBook(store=store, clock=clock, method=self.method())
        book.apply_fill(make_fill(side=Side.BUY, quantity="100", price="100"))
        result = book.apply_fill(make_fill(side=Side.SELL, quantity="100", price="110"))
        assert result.position.quantity == Decimal("0")
        assert result.position.avg_price == Decimal("0")
        assert result.position.opened_at_ns is None
        assert result.realized_pnl_delta == Decimal("1000")

        # The lot queue was actually cleared, not just reporting zero by
        # coincidence: a fresh buy must start a brand new cost basis.
        reopened = book.apply_fill(make_fill(side=Side.BUY, quantity="10", price="500"))
        assert reopened.position.avg_price == Decimal("500")

    def test_a_close_spanning_two_lots_blends_their_prices(
        self, store: InMemoryPositionStore, clock: SimulatedClock
    ) -> None:
        book = PositionBook(store=store, clock=clock, method=self.method())
        book.apply_fill(make_fill(side=Side.BUY, quantity="50", price="100"))
        book.apply_fill(make_fill(side=Side.BUY, quantity="50", price="200"))
        result = book.apply_fill(make_fill(side=Side.SELL, quantity="75", price="150"))
        # 50 shares @100 + 25 shares @200 close: (150-100)*50 + (150-200)*25 = 2500 - 1250
        assert result.realized_pnl_delta == Decimal("1250")
        assert result.position.quantity == Decimal("25")
        assert result.position.avg_price == Decimal("200")

    def test_flipping_past_flat_opens_a_fresh_lot_on_the_other_side(
        self, store: InMemoryPositionStore, clock: SimulatedClock
    ) -> None:
        book = PositionBook(store=store, clock=clock, method=self.method())
        book.apply_fill(make_fill(side=Side.BUY, quantity="100", price="100"))
        result = book.apply_fill(make_fill(side=Side.SELL, quantity="150", price="120"))
        assert result.position.quantity == Decimal("-50")
        assert result.position.avg_price == Decimal("120")
        assert result.closed_quantity == Decimal("100")

    def test_rebuild_replays_fills_in_order_and_reaches_the_same_state(
        self, store: InMemoryPositionStore, clock: SimulatedClock
    ) -> None:
        fills = [
            make_fill(side=Side.BUY, quantity="50", price="100", at_ns=BASE_NS),
            make_fill(side=Side.BUY, quantity="50", price="200", at_ns=BASE_NS + 1),
            make_fill(side=Side.SELL, quantity="75", price="150", at_ns=BASE_NS + 2),
        ]
        book = PositionBook(store=store, clock=clock, method=self.method())
        for fill in fills:
            book.apply_fill(fill)
        expected = store.get("AAPL")

        fresh_store = InMemoryPositionStore()
        fresh_book = PositionBook(store=fresh_store, clock=clock, method=self.method())
        fresh_book.rebuild(fills)

        assert fresh_store.get("AAPL") == expected


class TestAClosedPositionCarriesNoUnrealizedPnl:
    """`mark_to_market` skips flat positions, so a mark left on a position
    that later closes was never cleared. Every consumer reporting
    realized + unrealized (``net_pnl_report``, ``Runtime.pnl_snapshot``) then
    counted the whole round trip twice: once as realized, once as an
    unrealized figure for a position that no longer exists.
    """

    def test_flattening_clears_the_stale_mark(
        self, store: InMemoryPositionStore, clock: SimulatedClock
    ) -> None:
        book = PositionBook(store=store, clock=clock, method=CostBasisMethod.AVERAGE)
        book.apply_fill(make_fill(side=Side.BUY, quantity="100", price="100"))
        mark_to_market(store, {"AAPL": Decimal("110")}, now_ns=BASE_NS)
        assert store.get("AAPL").unrealized_pnl == Decimal("1000")  # type: ignore[union-attr]

        book.apply_fill(make_fill(side=Side.SELL, quantity="100", price="110"))

        closed = store.get("AAPL")
        assert closed is not None
        assert closed.realized_pnl == Decimal("1000")
        assert closed.unrealized_pnl == Decimal("0")
        assert closed.realized_pnl + closed.unrealized_pnl == Decimal("1000")

    def test_reopening_does_not_inherit_the_previous_cycles_mark(
        self, store: InMemoryPositionStore, clock: SimulatedClock
    ) -> None:
        book = PositionBook(store=store, clock=clock, method=CostBasisMethod.AVERAGE)
        book.apply_fill(make_fill(side=Side.BUY, quantity="100", price="100"))
        mark_to_market(store, {"AAPL": Decimal("110")}, now_ns=BASE_NS)
        book.apply_fill(make_fill(side=Side.SELL, quantity="100", price="110"))

        book.apply_fill(make_fill(side=Side.BUY, quantity="50", price="120"))

        reopened = store.get("AAPL")
        assert reopened is not None
        assert reopened.unrealized_pnl == Decimal("0")


class TestRebuildResetsTheStoreNotJustTheLots:
    """`rebuild` is the startup-recovery path (spec §10.4). Clearing only the
    in-memory lots left the *stored* position in place, so replaying fills
    against a persistent store applied each one on top of state that already
    contained it.
    """

    FILLS = (
        ("BUY", "100", "100"),
        ("SELL", "40", "110"),
    )

    def _replay(self, book: PositionBook) -> None:
        book.rebuild(
            [
                make_fill(side=Side[s], quantity=q, price=p, at_ns=BASE_NS + i)
                for i, (s, q, p) in enumerate(self.FILLS)
            ]
        )

    @pytest.mark.parametrize("method", [CostBasisMethod.AVERAGE, CostBasisMethod.FIFO])
    def test_replaying_onto_a_populated_store_is_idempotent(
        self, store: InMemoryPositionStore, clock: SimulatedClock, method: CostBasisMethod
    ) -> None:
        first = PositionBook(store=store, clock=clock, method=method)
        self._replay(first)
        after_first = store.get("AAPL")
        assert after_first is not None

        # A fresh book over the same (persistent) store — restart recovery.
        second = PositionBook(store=store, clock=clock, method=method)
        self._replay(second)

        after_second = store.get("AAPL")
        assert after_second is not None
        assert after_second.quantity == after_first.quantity == Decimal("60")
        assert after_second.realized_pnl == after_first.realized_pnl

    @pytest.mark.parametrize("method", [CostBasisMethod.AVERAGE, CostBasisMethod.FIFO])
    def test_quantity_still_equals_the_signed_sum_of_fills(
        self, store: InMemoryPositionStore, clock: SimulatedClock, method: CostBasisMethod
    ) -> None:
        """The invariant the property suite pins — it must survive a replay."""
        PositionBook(store=store, clock=clock, method=method).apply_fill(
            make_fill(side=Side.BUY, quantity="100", price="100")
        )
        self._replay(PositionBook(store=store, clock=clock, method=method))

        position = store.get("AAPL")
        assert position is not None
        assert position.quantity == Decimal("60")  # 100 - 40
