"""Multi-strategy intent netting — spec §FR-STR-04."""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

from atrader.core.clock import SimulatedClock
from atrader.core.ids import DeterministicIdGenerator
from atrader.core.models import Position, TradingIntent
from atrader.core.types import Side, TargetType, Urgency
from atrader.portfolio.netting import NETTED_STRATEGY_ID, net_intents

BASE_NS = 1_700_000_000_000_000_000


def make_intent(
    *,
    strategy_id: str,
    symbol: str = "AAPL",
    side: Side,
    target_type: TargetType = TargetType.SHARES,
    target_value: Decimal,
    urgency: Urgency = Urgency.NORMAL,
    confidence: Decimal = Decimal("1"),
) -> TradingIntent:
    return TradingIntent(
        intent_id=uuid4(),
        strategy_id=strategy_id,
        symbol=symbol,
        side=side,
        target_type=target_type,
        target_value=target_value,
        urgency=urgency,
        confidence=confidence,
        created_at_ns=BASE_NS,
    )


_DEFAULT_PRICES = {"AAPL": Decimal("100")}


def net(
    intents: list[TradingIntent],
    *,
    positions: dict[str, Position] | None = None,
    prices: dict[str, Decimal] | None = None,
    equity: Decimal = Decimal("100000"),
    deadband_pct: Decimal = Decimal("0"),
) -> tuple[list[TradingIntent], list]:
    clock = SimulatedClock(start_ns=BASE_NS)
    ids = DeterministicIdGenerator(clock, seed=1)
    return net_intents(
        intents,
        positions=positions or {},
        prices=_DEFAULT_PRICES if prices is None else prices,
        equity=equity,
        ids=ids,
        clock=clock,
        deadband_pct=deadband_pct,
    )


class TestSingleStrategyPassthrough:
    def test_a_lone_intent_for_a_symbol_passes_through_unchanged(self) -> None:
        intent = make_intent(strategy_id="sma", side=Side.BUY, target_value=Decimal("100"))
        netted, conflicts = net([intent])
        assert netted == [intent]
        assert conflicts == []

    def test_different_symbols_never_conflict(self) -> None:
        a = make_intent(strategy_id="sma", symbol="AAPL", side=Side.BUY, target_value=Decimal("10"))
        b = make_intent(strategy_id="llm", symbol="MSFT", side=Side.SELL, target_value=Decimal("5"))
        netted, conflicts = net([a, b], prices={"AAPL": Decimal("100"), "MSFT": Decimal("400")})
        assert set(netted) == {a, b}
        assert conflicts == []


class TestNetting:
    def test_two_equal_and_opposite_intents_cancel_completely(self) -> None:
        a = make_intent(strategy_id="sma", side=Side.BUY, target_value=Decimal("100"))
        b = make_intent(strategy_id="llm", side=Side.SELL, target_value=Decimal("100"))
        netted, conflicts = net([a, b])
        assert netted == []
        assert len(conflicts) == 1
        assert conflicts[0].net_shares == Decimal("0")
        assert conflicts[0].gross_shares == Decimal("200")
        assert conflicts[0].cancelled_shares == Decimal("200")

    def test_partially_opposing_intents_net_to_the_difference(self) -> None:
        a = make_intent(strategy_id="sma", side=Side.BUY, target_value=Decimal("100"))
        b = make_intent(strategy_id="llm", side=Side.SELL, target_value=Decimal("40"))
        netted, conflicts = net([a, b])
        assert len(netted) == 1
        assert netted[0].side is Side.BUY
        assert netted[0].target_value == Decimal("60")
        assert netted[0].strategy_id == NETTED_STRATEGY_ID
        assert conflicts[0].net_shares == Decimal("60")

    def test_same_direction_intents_add_up(self) -> None:
        a = make_intent(strategy_id="sma", side=Side.BUY, target_value=Decimal("50"))
        b = make_intent(strategy_id="llm", side=Side.BUY, target_value=Decimal("30"))
        netted, _ = net([a, b])
        assert netted[0].side is Side.BUY
        assert netted[0].target_value == Decimal("80")

    def test_the_synthesized_intent_drops_stop_loss_and_limit_price(self) -> None:
        a = TradingIntent(
            intent_id=uuid4(),
            strategy_id="sma",
            symbol="AAPL",
            side=Side.BUY,
            target_type=TargetType.SHARES,
            target_value=Decimal("100"),
            limit_price=Decimal("99"),
            stop_loss=Decimal("90"),
            created_at_ns=BASE_NS,
        )
        b = make_intent(strategy_id="llm", side=Side.BUY, target_value=Decimal("20"))
        netted, _ = net([a, b])
        assert netted[0].limit_price is None
        assert netted[0].stop_loss is None

    def test_urgency_takes_the_most_aggressive_contributor(self) -> None:
        a = make_intent(
            strategy_id="sma", side=Side.BUY, target_value=Decimal("50"), urgency=Urgency.PASSIVE
        )
        b = make_intent(
            strategy_id="llm",
            side=Side.BUY,
            target_value=Decimal("30"),
            urgency=Urgency.AGGRESSIVE,
        )
        netted, _ = net([a, b])
        assert netted[0].urgency is Urgency.AGGRESSIVE

    def test_confidence_is_weighted_by_contributed_size(self) -> None:
        a = make_intent(
            strategy_id="sma",
            side=Side.BUY,
            target_value=Decimal("90"),
            confidence=Decimal("1.0"),
        )
        b = make_intent(
            strategy_id="llm",
            side=Side.BUY,
            target_value=Decimal("10"),
            confidence=Decimal("0.5"),
        )
        netted, _ = net([a, b])
        # (90*1.0 + 10*0.5) / 100 = 0.95
        assert netted[0].confidence == Decimal("0.95")

    def test_a_near_cancellation_within_the_deadband_produces_nothing(self) -> None:
        a = make_intent(strategy_id="sma", side=Side.BUY, target_value=Decimal("101"))
        b = make_intent(strategy_id="llm", side=Side.SELL, target_value=Decimal("100"))
        # net = 1 share at $100 = $100 notional, equity = $100000 -> 0.1% of equity
        netted, conflicts = net([a, b], deadband_pct=Decimal("0.5"))
        assert netted == []
        assert conflicts[0].net_shares == Decimal("1")

    def test_three_strategies_net_together(self) -> None:
        a = make_intent(strategy_id="sma", side=Side.BUY, target_value=Decimal("100"))
        b = make_intent(strategy_id="llm", side=Side.SELL, target_value=Decimal("30"))
        c = make_intent(strategy_id="mean_reversion", side=Side.SELL, target_value=Decimal("20"))
        netted, conflicts = net([a, b, c])
        assert netted[0].target_value == Decimal("50")
        assert netted[0].side is Side.BUY
        assert conflicts[0].gross_shares == Decimal("150")


class TestTargetWeightNetting:
    def test_target_weight_converts_against_the_current_position(self) -> None:
        # Strategy wants 20% of equity in AAPL; equity=100000, price=100 -> target 200 shares.
        # We already hold 50 -> desired delta = +150.
        a = make_intent(
            strategy_id="sma",
            side=Side.BUY,
            target_type=TargetType.TARGET_WEIGHT,
            target_value=Decimal("0.2"),
        )
        b = make_intent(strategy_id="llm", side=Side.SELL, target_value=Decimal("50"))
        netted, _ = net(
            [a, b],
            positions={"AAPL": Position(symbol="AAPL", quantity=Decimal("50"))},
        )
        assert netted[0].side is Side.BUY
        assert netted[0].target_value == Decimal("100")  # 150 - 50

    def test_notional_intents_are_converted_using_the_price(self) -> None:
        # $5000 buy at $100/share = 50 shares.
        a = make_intent(
            strategy_id="sma",
            side=Side.BUY,
            target_type=TargetType.NOTIONAL,
            target_value=Decimal("5000"),
        )
        b = make_intent(strategy_id="llm", side=Side.SELL, target_value=Decimal("20"))
        netted, _ = net([a, b], prices={"AAPL": Decimal("100")})
        assert netted[0].side is Side.BUY
        assert netted[0].target_value == Decimal("30")  # 50 - 20


class TestMissingPrice:
    def test_a_notional_intent_with_no_price_contributes_nothing(self) -> None:
        a = make_intent(
            strategy_id="sma",
            side=Side.BUY,
            target_type=TargetType.NOTIONAL,
            target_value=Decimal("5000"),
        )
        b = make_intent(strategy_id="llm", side=Side.SELL, target_value=Decimal("20"))
        netted, conflicts = net([a, b], prices={})
        # Only b's SHARES delta (-20) counts; a contributes zero without a price.
        assert netted[0].side is Side.SELL
        assert netted[0].target_value == Decimal("20")
        assert conflicts[0].net_shares == Decimal("-20")

    def test_the_deadband_is_zero_when_the_symbol_has_no_price(self) -> None:
        a = make_intent(strategy_id="sma", side=Side.BUY, target_value=Decimal("1"))
        b = make_intent(strategy_id="llm", side=Side.SELL, target_value=Decimal("1"))
        # Net is exactly zero regardless of price, so this exercises the "no
        # price" deadband path (share_deadband stays zero) without depending
        # on rounding.
        netted, conflicts = net([a, b], prices={}, deadband_pct=Decimal("50"))
        assert netted == []
        assert conflicts[0].net_shares == Decimal("0")


class TestAbsoluteTargetsChargeThePositionOnce:
    """``TARGET_WEIGHT`` states an absolute target, so converting it to a
    share delta subtracts what is already held. Summing several such deltas
    subtracted the position once *per intent*, and the surplus flipped the
    netted order against every strategy that contributed to it.
    """

    def weight(self, strategy_id: str, target: str) -> TradingIntent:
        return make_intent(
            strategy_id=strategy_id,
            side=Side.BUY,
            target_type=TargetType.TARGET_WEIGHT,
            target_value=Decimal(target),
        )

    def test_two_long_targets_do_not_produce_a_short(self) -> None:
        """Both strategies want to be long 10% of a book already long 25%.
        The netted order must trim toward the targets, never cross zero."""
        positions = {"AAPL": Position(symbol="AAPL", quantity=Decimal("250"))}
        netted, _ = net([self.weight("a", "0.10"), self.weight("b", "0.10")], positions=positions)

        (order,) = netted
        resulting = Decimal("250") + order.target_value * order.side.sign
        assert resulting > 0, "netted order flipped the book short"
        assert resulting == Decimal("200"), "20% — the two 10% targets, summed"

    def test_targets_that_together_exceed_the_holding_still_buy(self) -> None:
        positions = {"AAPL": Position(symbol="AAPL", quantity=Decimal("100"))}
        netted, _ = net([self.weight("a", "0.10"), self.weight("b", "0.10")], positions=positions)

        (order,) = netted
        assert order.side is Side.BUY
        assert Decimal("100") + order.target_value * order.side.sign == Decimal("200")

    def test_a_single_weight_intent_is_unchanged(self) -> None:
        """One intent is passed through untouched — the correction only
        applies when several absolute targets are being combined."""
        positions = {"AAPL": Position(symbol="AAPL", quantity=Decimal("250"))}
        netted, _ = net([self.weight("a", "0.10")], positions=positions)

        (passed_through,) = netted
        assert passed_through.target_type is TargetType.TARGET_WEIGHT
        assert passed_through.target_value == Decimal("0.10")

    def test_a_flat_book_nets_the_targets_directly(self) -> None:
        netted, _ = net([self.weight("a", "0.10"), self.weight("b", "0.05")])

        (order,) = netted
        assert order.side is Side.BUY
        assert order.target_value == Decimal("150")  # 15% of 100k at 100

    def test_share_deltas_are_not_affected(self) -> None:
        """SHARES intents are already deltas; nothing is charged to them."""
        positions = {"AAPL": Position(symbol="AAPL", quantity=Decimal("250"))}
        netted, _ = net(
            [
                make_intent(strategy_id="a", side=Side.BUY, target_value=Decimal("30")),
                make_intent(strategy_id="b", side=Side.BUY, target_value=Decimal("20")),
            ],
            positions=positions,
        )

        (order,) = netted
        assert order.side is Side.BUY
        assert order.target_value == Decimal("50")
