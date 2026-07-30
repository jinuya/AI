"""Net P&L rollup — spec §FR-PF-02."""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

from atrader.config.schema import InstrumentSpec
from atrader.core.models import Fill, Position
from atrader.core.types import Side
from atrader.portfolio.pnl import BorrowCostModel, PnLReport, mark_to_market, net_pnl_report
from atrader.storage.memory import InMemoryPositionStore

BASE_NS = 1_700_000_000_000_000_000


def make_fill(*, commission: str = "0", tax: str = "0") -> Fill:
    return Fill(
        fill_id=uuid4(),
        order_id=uuid4(),
        symbol="AAPL",
        side=Side.BUY,
        quantity=Decimal("10"),
        price=Decimal("100"),
        commission=Decimal(commission),
        tax=Decimal(tax),
        executed_at_ns=BASE_NS,
    )


class TestMarkToMarket:
    def test_refreshes_last_price_and_unrealized_pnl_for_a_long(self) -> None:
        store = InMemoryPositionStore()
        store.upsert(Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("100")))
        updated = mark_to_market(store, {"AAPL": Decimal("110")}, now_ns=BASE_NS)
        assert len(updated) == 1
        assert updated[0].last_price == Decimal("110")
        assert updated[0].unrealized_pnl == Decimal("1000")

    def test_unrealized_pnl_on_a_short_is_negative_when_price_rises(self) -> None:
        store = InMemoryPositionStore()
        store.upsert(Position(symbol="AAPL", quantity=Decimal("-100"), avg_price=Decimal("100")))
        updated = mark_to_market(store, {"AAPL": Decimal("110")}, now_ns=BASE_NS)
        assert updated[0].unrealized_pnl == Decimal("-1000")

    def test_flat_positions_are_skipped(self) -> None:
        store = InMemoryPositionStore()
        store.upsert(Position(symbol="AAPL", quantity=Decimal("0")))
        assert mark_to_market(store, {"AAPL": Decimal("110")}, now_ns=BASE_NS) == []

    def test_a_symbol_with_no_fresh_price_keeps_its_previous_mark(self) -> None:
        store = InMemoryPositionStore()
        store.upsert(
            Position(
                symbol="AAPL",
                quantity=Decimal("100"),
                avg_price=Decimal("100"),
                last_price=Decimal("105"),
            )
        )
        assert mark_to_market(store, {}, now_ns=BASE_NS) == []
        assert store.get("AAPL").last_price == Decimal("105")  # type: ignore[union-attr]


class TestBorrowCostModel:
    def test_long_positions_accrue_no_borrow_cost(self) -> None:
        model = BorrowCostModel()
        position = Position(symbol="AAPL", quantity=Decimal("100"), avg_price=Decimal("100"))
        spec = InstrumentSpec(symbol="AAPL", borrow_bps_annual=Decimal("25"))
        assert model.daily_cost(position, spec) == Decimal("0")

    def test_short_positions_accrue_the_annual_rate_prorated_daily(self) -> None:
        model = BorrowCostModel()
        position = Position(
            symbol="AAPL", quantity=Decimal("-1000"), avg_price=Decimal("100"), last_price=None
        )
        spec = InstrumentSpec(symbol="AAPL", borrow_bps_annual=Decimal("365"))
        # exposure = 1000*100 = 100000; annual_rate = 365bps = 3.65%; /365 days = 0.01%/day
        expected = Decimal("100000") * Decimal("0.0365") / Decimal(365)
        assert model.daily_cost(position, spec) == expected

    def test_no_instrument_spec_means_no_borrow_cost(self) -> None:
        model = BorrowCostModel()
        position = Position(symbol="AAPL", quantity=Decimal("-100"), avg_price=Decimal("100"))
        assert model.daily_cost(position, None) == Decimal("0")


class TestNetPnLReport:
    def instruments(self, spec: InstrumentSpec | None) -> object:
        def lookup(symbol: str) -> InstrumentSpec | None:
            return spec if spec is not None and spec.symbol == symbol else None

        return lookup

    def test_net_pnl_subtracts_commission_and_tax_from_trading_pnl(self) -> None:
        position = Position(
            symbol="AAPL",
            quantity=Decimal("0"),
            realized_pnl=Decimal("1000"),
            unrealized_pnl=Decimal("0"),
        )
        fills = [make_fill(commission="5", tax="2"), make_fill(commission="5", tax="2")]
        report = net_pnl_report(
            [position],
            fills,
            instrument_of=self.instruments(None),
            as_of_ns=BASE_NS,
        )
        assert report.commission == Decimal("10")
        assert report.tax == Decimal("4")
        assert report.gross_trading_pnl == Decimal("1000")
        assert report.net_pnl == Decimal("1000") - Decimal("10") - Decimal("4")

    def test_borrow_cost_reduces_net_pnl_for_short_positions(self) -> None:
        spec = InstrumentSpec(symbol="AAPL", currency="USD", borrow_bps_annual=Decimal("365"))
        position = Position(
            symbol="AAPL",
            quantity=Decimal("-1000"),
            avg_price=Decimal("100"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("0"),
        )
        report = net_pnl_report(
            [position], [], instrument_of=self.instruments(spec), as_of_ns=BASE_NS
        )
        assert report.borrow_cost > Decimal("0")
        assert report.net_pnl == -report.borrow_cost

    def test_same_currency_positions_have_zero_fx_adjustment(self) -> None:
        spec = InstrumentSpec(symbol="AAPL", currency="USD")
        position = Position(
            symbol="AAPL",
            quantity=Decimal("0"),
            realized_pnl=Decimal("500"),
            unrealized_pnl=Decimal("0"),
        )
        report = net_pnl_report(
            [position],
            [],
            instrument_of=self.instruments(spec),
            as_of_ns=BASE_NS,
            base_currency="USD",
        )
        assert report.fx_adjustment == Decimal("0")

    def test_foreign_currency_pnl_is_converted_and_the_fx_component_isolated(self) -> None:
        spec = InstrumentSpec(symbol="SAP", currency="EUR")
        position = Position(
            symbol="SAP",
            quantity=Decimal("0"),
            realized_pnl=Decimal("100"),
            unrealized_pnl=Decimal("0"),
        )
        report = net_pnl_report(
            [position],
            [],
            instrument_of=self.instruments(spec),
            as_of_ns=BASE_NS,
            base_currency="USD",
            fx_rates={"EUR": Decimal("1.10")},
        )
        assert report.realized_pnl == Decimal("110")
        assert report.fx_adjustment == Decimal("10")  # 100 * (1.10 - 1)
        assert report.per_symbol_pnl["SAP"] == Decimal("110")

    def test_an_untracked_currency_is_treated_as_parity(self) -> None:
        spec = InstrumentSpec(symbol="TSE", currency="JPY")
        position = Position(
            symbol="TSE",
            quantity=Decimal("0"),
            realized_pnl=Decimal("100"),
            unrealized_pnl=Decimal("0"),
        )
        report = net_pnl_report(
            [position], [], instrument_of=self.instruments(spec), as_of_ns=BASE_NS, fx_rates={}
        )
        assert report.realized_pnl == Decimal("100")
        assert report.fx_adjustment == Decimal("0")


class TestFxIsDisclosedNotAddedTwice:
    """``realized_pnl``/``unrealized_pnl`` are already converted at the FX
    rate, so ``gross_trading_pnl`` contains the currency component.
    ``fx_adjustment`` reports how much of it came from the currency rather
    than the trade — adding it to ``net_pnl`` counted that component twice.
    """

    def test_net_pnl_matches_what_the_account_actually_received(self) -> None:
        report = PnLReport(
            as_of_ns=0,
            realized_pnl=Decimal("108"),  # 100 EUR at 1.08
            unrealized_pnl=Decimal("0"),
            commission=Decimal("0"),
            tax=Decimal("0"),
            borrow_cost=Decimal("0"),
            fx_adjustment=Decimal("8"),
        )
        assert report.gross_trading_pnl == Decimal("108")
        assert report.net_pnl == Decimal("108")

    def test_costs_still_come_off(self) -> None:
        report = PnLReport(
            as_of_ns=0,
            realized_pnl=Decimal("108"),
            unrealized_pnl=Decimal("0"),
            commission=Decimal("3"),
            tax=Decimal("2"),
            borrow_cost=Decimal("1"),
            fx_adjustment=Decimal("8"),
        )
        assert report.net_pnl == Decimal("102")

    def test_a_base_currency_position_is_unaffected(self) -> None:
        report = PnLReport(
            as_of_ns=0,
            realized_pnl=Decimal("100"),
            unrealized_pnl=Decimal("50"),
            commission=Decimal("0"),
            tax=Decimal("0"),
            borrow_cost=Decimal("0"),
            fx_adjustment=Decimal("0"),
        )
        assert report.net_pnl == Decimal("150")
