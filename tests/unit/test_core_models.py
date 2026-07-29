"""The validators on the core records — spec §4.5, §FR-STR-03, §FR-EXE-01.

These validators are the last structural gate before an intent reaches the
risk engine and an order reaches a broker adapter, and every one of their
*rejection* paths was unexercised: the suite constructed these models 27 times
and every construction was a happy path. A validator with no test pinning it
shut is a validator that can be inverted, weakened, or half-written without
anything going red.

That is not hypothetical here. Writing this file is what surfaced an
asymmetry in :meth:`OrderRequest._prices_match_the_order_type`: a stray
``limit_price`` on a non-limit order was rejected, but a stray ``stop_price``
on a non-stop order was accepted and would have been handed to a broker
adapter, which might honour it and execute something other than what the risk
engine approved. The matrix below is written to walk *both* directions of
every rule for exactly that reason.

Two invariants deserve naming, because getting either backwards is expensive
rather than merely wrong:

* **TARGET_WEIGHT is a fraction, not a percent.** ``0.1`` means 10%. Accepting
  ``10`` would size a position at 1000% of equity, and ``models.py`` is the
  only place in the codebase that bounds this value.
* **A stop must not already be triggered at entry.** A BUY whose stop_loss sits
  at or above its limit price fires the moment it fills.
"""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from atrader.core.models import OrderRequest, TradingIntent
from atrader.core.types import OrderType, Side, TargetType, TimeInForce

#: order type -> the price fields it is allowed (and required) to carry.
_REQUIRED_PRICES: dict[OrderType, tuple[str, ...]] = {
    OrderType.LIMIT: ("limit_price",),
    OrderType.STOP: ("stop_price",),
    OrderType.STOP_LIMIT: ("limit_price", "stop_price"),
    OrderType.MARKET: (),
}


def intent(**overrides: object) -> TradingIntent:
    base: dict[str, object] = {
        "intent_id": uuid4(),
        "strategy_id": "test",
        "symbol": "AAPL",
        "side": Side.BUY,
        "target_type": TargetType.SHARES,
        "target_value": Decimal("10"),
        "created_at_ns": 1,
    }
    return TradingIntent(**{**base, **overrides})  # type: ignore[arg-type]


def request(**overrides: object) -> OrderRequest:
    base: dict[str, object] = {
        "client_order_id": str(uuid4()),
        "symbol": "AAPL",
        "side": Side.BUY,
        "quantity": Decimal("10"),
        "order_type": OrderType.MARKET,
        "time_in_force": TimeInForce.DAY,
    }
    return OrderRequest(**{**base, **overrides})  # type: ignore[arg-type]


class TestTargetWeightIsAFraction:
    """``0.1`` means 10%, not 10. This bound is the only one in the codebase:
    ``risk/engine.py`` multiplies equity by ``target_value`` directly, and the
    only downstream backstop is ``max_position_pct``, a different invariant.
    """

    @pytest.mark.parametrize("weight", ["0", "0.1", "1", "-0.5", "-1"])
    def test_a_fraction_inside_the_unit_interval_is_accepted(self, weight: str) -> None:
        assert intent(target_type=TargetType.TARGET_WEIGHT, target_value=Decimal(weight))

    @pytest.mark.parametrize("weight", ["10", "1.5", "-1.5", "100"])
    def test_anything_outside_it_is_refused(self, weight: str) -> None:
        """A percent typed where a fraction belongs is a 100x sizing error."""
        with pytest.raises(ValidationError, match="fraction"):
            intent(target_type=TargetType.TARGET_WEIGHT, target_value=Decimal(weight))

    def test_the_bound_does_not_apply_to_share_counts(self) -> None:
        assert intent(target_type=TargetType.SHARES, target_value=Decimal("1000"))


class TestIntentPricesMustBePositive:
    @pytest.mark.parametrize("field", ["limit_price", "stop_loss", "take_profit"])
    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_a_non_positive_price_is_refused(self, field: str, value: str) -> None:
        with pytest.raises(ValidationError, match="positive"):
            intent(**{field: Decimal(value)})

    @pytest.mark.parametrize("field", ["limit_price", "stop_loss", "take_profit"])
    def test_a_positive_price_is_accepted(self, field: str) -> None:
        assert intent(**{field: Decimal("100")})


class TestAStopMustNotFireOnEntry:
    """A stop on the wrong side of the entry price triggers the instant the
    order fills. Nothing downstream re-checks this: ``strategy/llm/guards.py``
    validates that a stop is positive and inside a price band, but never
    compares it against the limit.
    """

    @pytest.mark.parametrize("stop", ["100", "101"])
    def test_a_buy_stop_at_or_above_the_limit_is_refused(self, stop: str) -> None:
        with pytest.raises(ValidationError, match="triggers on entry"):
            intent(side=Side.BUY, limit_price=Decimal("100"), stop_loss=Decimal(stop))

    def test_a_buy_stop_below_the_limit_is_fine(self) -> None:
        assert intent(side=Side.BUY, limit_price=Decimal("100"), stop_loss=Decimal("99"))

    @pytest.mark.parametrize("stop", ["100", "99"])
    def test_a_sell_stop_at_or_below_the_limit_is_refused(self, stop: str) -> None:
        with pytest.raises(ValidationError, match="triggers on entry"):
            intent(side=Side.SELL, limit_price=Decimal("100"), stop_loss=Decimal(stop))

    def test_a_sell_stop_above_the_limit_is_fine(self) -> None:
        assert intent(side=Side.SELL, limit_price=Decimal("100"), stop_loss=Decimal("101"))

    def test_a_stop_without_a_limit_price_has_nothing_to_contradict(self) -> None:
        """A market entry has no limit to compare against, so the rule cannot
        apply — and must not fire spuriously."""
        assert intent(side=Side.BUY, stop_loss=Decimal("99"))


class TestOrderRequestPriceMatrix:
    """Walked in both directions for every rule.

    The one-directional version of this test is what let a missing
    ``stop_price`` guard survive: checking only that required prices are
    present would have passed while a forbidden one sailed through.
    """

    def _prices(self, order_type: OrderType) -> dict[str, Decimal]:
        return {name: Decimal("100") for name in _REQUIRED_PRICES[order_type]}

    @pytest.mark.parametrize("order_type", list(_REQUIRED_PRICES))
    def test_an_order_type_with_exactly_its_own_prices_is_accepted(
        self, order_type: OrderType
    ) -> None:
        assert request(order_type=order_type, **self._prices(order_type))

    @pytest.mark.parametrize("order_type", [OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT])
    @pytest.mark.parametrize("omitted", ["limit_price", "stop_price"])
    def test_a_missing_required_price_is_refused(self, order_type: OrderType, omitted: str) -> None:
        prices = self._prices(order_type)
        if omitted not in prices:
            pytest.skip(f"{order_type} does not require {omitted}")
        del prices[omitted]
        with pytest.raises(ValidationError, match=f"requires a {omitted}"):
            request(order_type=order_type, **prices)

    @pytest.mark.parametrize("order_type", list(_REQUIRED_PRICES))
    @pytest.mark.parametrize("extra", ["limit_price", "stop_price"])
    def test_a_price_the_order_type_does_not_use_is_refused(
        self, order_type: OrderType, extra: str
    ) -> None:
        """The direction that was missing for ``stop_price``. This request is
        handed to a broker adapter as-is; a broker that honours a stray price
        executes something the risk engine never approved."""
        if extra in _REQUIRED_PRICES[order_type]:
            pytest.skip(f"{order_type} legitimately carries {extra}")
        prices = self._prices(order_type)
        prices[extra] = Decimal("100")
        with pytest.raises(ValidationError, match=f"must not carry a {extra}"):
            request(order_type=order_type, **prices)

    @pytest.mark.parametrize("field", ["limit_price", "stop_price"])
    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_a_non_positive_price_is_refused(self, field: str, value: str) -> None:
        order_type = OrderType.LIMIT if field == "limit_price" else OrderType.STOP
        with pytest.raises(ValidationError, match="must be positive"):
            request(order_type=order_type, **{field: Decimal(value)})
