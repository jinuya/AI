"""Execution algorithms and routing — spec §FR-EXE-02, §FR-EXE-03.

ADV의 1%를 초과하는 주문은 반드시 분할 실행해야 한다.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest

from atrader.config.schema import ExecutionConfig, OrderLimits
from atrader.core.clock import SimulatedClock
from atrader.core.ids import DeterministicIdGenerator
from atrader.core.models import Order
from atrader.core.types import OrderStatus, OrderType, Side
from atrader.execution.algos import Slice, SliceSchedule
from atrader.execution.algos.pov import POVExecutor
from atrader.execution.algos.twap import plan_twap
from atrader.execution.algos.vwap import plan_vwap
from atrader.execution.router import AlgoChoice, Router, needs_split

BASE_NS = 1_700_000_000_000_000_000


def make_order(**overrides: Any) -> Order:
    defaults: dict[str, Any] = {
        "order_id": uuid4(),
        "client_order_id": str(uuid4()),
        "strategy_id": "test",
        "symbol": "AAPL",
        "side": Side.BUY,
        "order_type": OrderType.LIMIT,
        "quantity": Decimal("10000"),
        "limit_price": Decimal("190.00"),
        "status": OrderStatus.PENDING_NEW,
        "risk_check_id": uuid4(),
        "created_at_ns": BASE_NS,
        "updated_at_ns": BASE_NS,
    }
    return Order(**{**defaults, **overrides})


class TestSliceSchedule:
    def test_slices_must_sum_to_the_total(self) -> None:
        with pytest.raises(ValueError, match="sum to"):
            SliceSchedule(Decimal(100), (Slice(0, 0.0, Decimal(40)),))

    def test_offsets_must_be_non_decreasing(self) -> None:
        with pytest.raises(ValueError, match="non-decreasing"):
            SliceSchedule(Decimal(100), (Slice(0, 10.0, Decimal(50)), Slice(1, 5.0, Decimal(50))))

    def test_sequence_numbers_must_have_no_gaps(self) -> None:
        with pytest.raises(ValueError, match="no gaps"):
            SliceSchedule(Decimal(100), (Slice(0, 0.0, Decimal(50)), Slice(2, 1.0, Decimal(50))))

    def test_a_negative_or_zero_slice_quantity_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            Slice(0, 0.0, Decimal(0))

    def test_a_negative_sequence_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="sequence"):
            Slice(-1, 0.0, Decimal(1))

    def test_a_negative_offset_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="offset_seconds"):
            Slice(0, -1.0, Decimal(1))

    def test_is_empty(self) -> None:
        assert SliceSchedule(Decimal(0), ()).is_empty
        assert not SliceSchedule(Decimal(1), (Slice(0, 0.0, Decimal(1)),)).is_empty


class TestTwap:
    def test_splits_into_equal_slices(self) -> None:
        schedule = plan_twap(Decimal(1000), duration_seconds=100, slice_count=10)
        assert schedule.slice_count == 10
        assert all(s.quantity == Decimal(100) for s in schedule.slices)
        assert schedule.total_quantity == Decimal(1000)

    def test_slices_are_evenly_spaced(self) -> None:
        schedule = plan_twap(Decimal(1000), duration_seconds=100, slice_count=10)
        offsets = [s.offset_seconds for s in schedule.slices]
        assert offsets == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]

    def test_remainder_is_folded_into_the_last_slice(self) -> None:
        schedule = plan_twap(Decimal(1000), duration_seconds=90, slice_count=3, lot_size=Decimal(1))
        # 1000 / 3 = 333.33 -> floors to 333 each, remainder 1 goes to the last.
        assert [s.quantity for s in schedule.slices] == [Decimal(333), Decimal(333), Decimal(334)]

    def test_a_single_slice_when_the_lot_size_would_zero_out_the_split(self) -> None:
        schedule = plan_twap(Decimal(5), duration_seconds=100, slice_count=10, lot_size=Decimal(1))
        assert schedule.slice_count == 1
        assert schedule.slices[0].quantity == Decimal(5)

    def test_single_slice_count_has_no_spacing(self) -> None:
        schedule = plan_twap(Decimal(100), duration_seconds=100, slice_count=1)
        assert schedule.slices[0].offset_seconds == 0.0

    @pytest.mark.parametrize(
        ("total", "duration", "count"),
        [(Decimal(0), 100, 5), (Decimal(100), -1, 5), (Decimal(100), 100, 0)],
    )
    def test_rejects_invalid_inputs(self, total: Decimal, duration: float, count: int) -> None:
        with pytest.raises(ValueError):
            plan_twap(total, duration_seconds=duration, slice_count=count)


class TestVwap:
    def test_splits_proportional_to_the_curve(self) -> None:
        curve = [Decimal(1), Decimal(2), Decimal(1)]
        schedule = plan_vwap(
            Decimal(400), volume_curve=curve, bucket_seconds=60, lot_size=Decimal(1)
        )
        assert [s.quantity for s in schedule.slices] == [Decimal(100), Decimal(200), Decimal(100)]

    def test_curve_need_not_sum_to_one(self) -> None:
        curve = [Decimal(10), Decimal(30)]  # 25% / 75%
        schedule = plan_vwap(
            Decimal(400), volume_curve=curve, bucket_seconds=60, lot_size=Decimal(1)
        )
        assert [s.quantity for s in schedule.slices] == [Decimal(100), Decimal(300)]

    def test_zero_weight_buckets_get_no_slice(self) -> None:
        curve = [Decimal(1), Decimal(0), Decimal(1)]
        schedule = plan_vwap(
            Decimal(200), volume_curve=curve, bucket_seconds=60, lot_size=Decimal(1)
        )
        assert schedule.slice_count == 2

    def test_sequence_numbers_have_no_gaps_after_dropping_zero_buckets(self) -> None:
        curve = [Decimal(1), Decimal(0), Decimal(1)]
        schedule = plan_vwap(
            Decimal(200), volume_curve=curve, bucket_seconds=60, lot_size=Decimal(1)
        )
        assert [s.sequence for s in schedule.slices] == [0, 1]

    def test_rounding_remainder_uses_largest_remainder_method(self) -> None:
        # 100 split 3 ways evenly: 33.33 each. Largest-remainder gives one
        # bucket an extra share rather than dumping all rounding on the last.
        curve = [Decimal(1), Decimal(1), Decimal(1)]
        schedule = plan_vwap(
            Decimal(100), volume_curve=curve, bucket_seconds=60, lot_size=Decimal(1)
        )
        total = sum((s.quantity for s in schedule.slices), Decimal(0))
        assert total == Decimal(100)

    def test_total_always_matches_even_under_heavy_rounding(self) -> None:
        curve = [Decimal(str(x)) for x in (1, 3, 7, 2, 9, 4)]
        schedule = plan_vwap(
            Decimal(137), volume_curve=curve, bucket_seconds=60, lot_size=Decimal(1)
        )
        assert sum((s.quantity for s in schedule.slices), Decimal(0)) == Decimal(137)

    def test_rejects_non_positive_total_quantity(self) -> None:
        with pytest.raises(ValueError, match="total_quantity"):
            plan_vwap(Decimal(0), volume_curve=[Decimal(1)], bucket_seconds=60)

    def test_rejects_a_negative_bucket_duration(self) -> None:
        with pytest.raises(ValueError, match="bucket_seconds"):
            plan_vwap(Decimal(100), volume_curve=[Decimal(1)], bucket_seconds=-1)

    def test_rejects_an_empty_curve(self) -> None:
        with pytest.raises(ValueError, match="at least one bucket"):
            plan_vwap(Decimal(100), volume_curve=[], bucket_seconds=60)

    def test_rejects_a_negative_weight(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            plan_vwap(Decimal(100), volume_curve=[Decimal(-1)], bucket_seconds=60)

    def test_rejects_a_curve_that_is_entirely_zero(self) -> None:
        with pytest.raises(ValueError, match="positive weight"):
            plan_vwap(Decimal(100), volume_curve=[Decimal(0), Decimal(0)], bucket_seconds=60)

    def test_a_tiny_order_over_many_buckets_still_sums_correctly(self) -> None:
        # 3 shares over 20 equal-weight buckets: every bucket floors to zero,
        # so the largest-remainder pass hands one whole share each to the
        # first three buckets rather than losing the order to rounding.
        curve = [Decimal(1)] * 20
        schedule = plan_vwap(Decimal(3), volume_curve=curve, bucket_seconds=60, lot_size=Decimal(1))
        assert sum((s.quantity for s in schedule.slices), Decimal(0)) == Decimal(3)
        assert all(s.quantity == Decimal(1) for s in schedule.slices)
        assert schedule.slice_count == 3


class TestPov:
    def test_participation_rate_scales_the_slice(self) -> None:
        pov = POVExecutor(total_quantity=Decimal(1000), participation_rate_pct=Decimal(10))
        slice_qty = pov.on_market_volume(Decimal(1000))
        assert slice_qty == Decimal(100)
        assert pov.sent_quantity == Decimal(100)

    def test_unspent_entitlement_carries_forward(self) -> None:
        # 10% of 5 shares is 0.5 — below a whole lot, so nothing is sent yet;
        # the entitlement must still be there next time.
        pov = POVExecutor(
            total_quantity=Decimal(100), participation_rate_pct=Decimal(10), lot_size=Decimal(1)
        )
        first = pov.on_market_volume(Decimal(5))
        assert first == Decimal(0)
        second = pov.on_market_volume(Decimal(5))  # now 1.0 entitled
        assert second == Decimal(1)

    def test_never_exceeds_the_remaining_quantity(self) -> None:
        pov = POVExecutor(total_quantity=Decimal(10), participation_rate_pct=Decimal(100))
        slice_qty = pov.on_market_volume(Decimal(1_000_000))
        assert slice_qty == Decimal(10)
        assert pov.is_complete

    def test_max_slice_quantity_caps_a_single_burst(self) -> None:
        pov = POVExecutor(
            total_quantity=Decimal(1000),
            participation_rate_pct=Decimal(100),
            max_slice_quantity=Decimal(50),
        )
        slice_qty = pov.on_market_volume(Decimal(1000))
        assert slice_qty == Decimal(50)
        assert pov.remaining_quantity == Decimal(950)

    def test_min_slice_quantity_holds_back_small_slices(self) -> None:
        pov = POVExecutor(
            total_quantity=Decimal(1000),
            participation_rate_pct=Decimal(1),
            min_slice_quantity=Decimal(50),
            lot_size=Decimal(1),
        )
        slice_qty = pov.on_market_volume(Decimal(100))  # entitled to 1 share, below the minimum
        assert slice_qty == Decimal(0)
        assert pov.volume_seen == Decimal(100)

    def test_min_slice_quantity_is_bypassed_when_it_would_exceed_the_remainder(self) -> None:
        # A 5-share order at 100% participation: the entitled slice (5) is
        # below the 50-share minimum, but it is also literally the entire
        # remaining order — the minimum must not block it forever.
        pov = POVExecutor(
            total_quantity=Decimal(5),
            participation_rate_pct=Decimal(100),
            min_slice_quantity=Decimal(50),
            lot_size=Decimal(1),
        )
        slice_qty = pov.on_market_volume(Decimal(5))
        assert slice_qty == Decimal(5)
        assert pov.is_complete

    def test_on_market_volume_after_completion_returns_zero(self) -> None:
        pov = POVExecutor(total_quantity=Decimal(10), participation_rate_pct=Decimal(100))
        pov.on_market_volume(Decimal(100))
        assert pov.is_complete
        assert pov.on_market_volume(Decimal(1000)) == Decimal(0)

    def test_finish_remaining_releases_the_rest_and_completes(self) -> None:
        pov = POVExecutor(total_quantity=Decimal(100), participation_rate_pct=Decimal(10))
        pov.on_market_volume(Decimal(100))  # sends 10
        remainder = pov.finish_remaining()
        assert remainder == Decimal(90)
        assert pov.is_complete
        assert pov.finish_remaining() == Decimal(0)  # nothing left the second time

    def test_rejects_invalid_construction(self) -> None:
        with pytest.raises(ValueError, match="total_quantity"):
            POVExecutor(total_quantity=Decimal(0), participation_rate_pct=Decimal(10))
        with pytest.raises(ValueError, match="participation_rate_pct"):
            POVExecutor(total_quantity=Decimal(10), participation_rate_pct=Decimal(0))
        with pytest.raises(ValueError, match="lot_size"):
            POVExecutor(
                total_quantity=Decimal(10), participation_rate_pct=Decimal(10), lot_size=Decimal(0)
            )
        with pytest.raises(ValueError, match="min_slice_quantity"):
            POVExecutor(
                total_quantity=Decimal(10),
                participation_rate_pct=Decimal(10),
                min_slice_quantity=Decimal(-1),
            )
        with pytest.raises(ValueError, match="max_slice_quantity"):
            POVExecutor(
                total_quantity=Decimal(10),
                participation_rate_pct=Decimal(10),
                max_slice_quantity=Decimal(0),
            )

    def test_rejects_negative_traded_quantity(self) -> None:
        pov = POVExecutor(total_quantity=Decimal(10), participation_rate_pct=Decimal(10))
        with pytest.raises(ValueError, match="non-negative"):
            pov.on_market_volume(Decimal(-1))


class TestNeedsSplit:
    def test_below_threshold_does_not_split(self) -> None:
        assert not needs_split(Decimal(100), Decimal(100_000), threshold_pct=Decimal(1))

    def test_above_threshold_splits(self) -> None:
        assert needs_split(Decimal(2000), Decimal(100_000), threshold_pct=Decimal(1))

    def test_missing_adv_never_splits(self) -> None:
        assert not needs_split(Decimal(1_000_000), None, threshold_pct=Decimal(1))

    def test_zero_adv_never_splits(self) -> None:
        assert not needs_split(Decimal(1_000_000), Decimal(0), threshold_pct=Decimal(1))


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(start_ns=BASE_NS)


@pytest.fixture
def ids(clock: SimulatedClock) -> DeterministicIdGenerator:
    return DeterministicIdGenerator(clock, seed=3)


@pytest.fixture
def router(ids: DeterministicIdGenerator) -> Router:
    return Router(ids=ids, order_limits=OrderLimits(), execution=ExecutionConfig())


class TestRouter:
    def test_small_order_relative_to_adv_goes_direct(self, router: Router) -> None:
        order = make_order(quantity=Decimal(100))
        plan = router.route(order, adv=Decimal(1_000_000))
        assert plan.algo is AlgoChoice.DIRECT
        assert not plan.is_split
        assert plan.orders_to_submit_now == (order,)

    def test_large_order_relative_to_adv_is_split(self, router: Router) -> None:
        order = make_order(quantity=Decimal(50_000))  # 5% of a 1M ADV
        plan = router.route(order, adv=Decimal(1_000_000))
        assert plan.is_split
        assert plan.algo is AlgoChoice.TWAP  # the configured default
        assert plan.schedule is not None
        assert sum((c.quantity for c in plan.children), Decimal(0)) == order.quantity

    def test_children_carry_the_same_risk_check_and_parent_intent(self, router: Router) -> None:
        parent_intent = uuid4()
        order = make_order(quantity=Decimal(50_000), parent_intent_id=parent_intent)
        plan = router.route(order, adv=Decimal(1_000_000))
        for child in plan.children:
            assert child.risk_check_id == order.risk_check_id
            assert child.parent_intent_id == parent_intent
            assert child.order_id != order.order_id
            assert child.client_order_id != order.client_order_id
            assert child.status == order.status
            assert child.filled_quantity == Decimal(0)

    def test_a_volume_curve_selects_vwap_even_when_default_is_twap(self, router: Router) -> None:
        order = make_order(quantity=Decimal(50_000))
        curve = [Decimal(1), Decimal(2), Decimal(1)]
        plan = router.route(order, adv=Decimal(1_000_000), volume_curve=curve)
        assert plan.algo is AlgoChoice.VWAP

    def test_explicit_algo_overrides_the_default(self, ids: DeterministicIdGenerator) -> None:
        router = Router(
            ids=ids,
            order_limits=OrderLimits(),
            execution=ExecutionConfig(default_algo="twap"),
        )
        order = make_order(quantity=Decimal(50_000))
        plan = router.route(order, adv=Decimal(1_000_000), algo=AlgoChoice.DIRECT)
        assert plan.algo is AlgoChoice.DIRECT
        # DIRECT is chosen explicitly even though the order would otherwise split.

    def test_vwap_without_a_curve_raises(self, ids: DeterministicIdGenerator) -> None:
        router = Router(
            ids=ids,
            order_limits=OrderLimits(),
            execution=ExecutionConfig(default_algo="vwap"),
        )
        order = make_order(quantity=Decimal(50_000))
        with pytest.raises(ValueError, match="volume_curve"):
            router.route(order, adv=Decimal(1_000_000))

    def test_orders_to_submit_now_is_only_the_zero_offset_slices(self, router: Router) -> None:
        order = make_order(quantity=Decimal(50_000))
        plan = router.route(order, adv=Decimal(1_000_000))
        immediate = plan.orders_to_submit_now
        assert len(immediate) == 1  # only the first TWAP slice starts at t=0


class TestPovRouting:
    def test_pov_plan_has_no_immediate_orders(self, ids: DeterministicIdGenerator) -> None:
        router = Router(
            ids=ids, order_limits=OrderLimits(), execution=ExecutionConfig(default_algo="pov")
        )
        order = make_order(quantity=Decimal(50_000))
        plan = router.route(order, adv=Decimal(1_000_000))
        assert plan.algo is AlgoChoice.POV
        assert plan.pov is not None
        assert plan.orders_to_submit_now == ()

    def test_pov_child_produces_orders_as_volume_prints(
        self, ids: DeterministicIdGenerator
    ) -> None:
        router = Router(
            ids=ids,
            order_limits=OrderLimits(),
            execution=ExecutionConfig(default_algo="pov", pov_participation_rate_pct=Decimal(10)),
        )
        order = make_order(quantity=Decimal(50_000))
        plan = router.route(order, adv=Decimal(1_000_000))

        child = router.pov_child(plan, Decimal(10_000))
        assert child is not None
        assert child.quantity == Decimal(1000)
        assert child.risk_check_id == order.risk_check_id

    def test_pov_child_returns_none_below_the_minimum(self, ids: DeterministicIdGenerator) -> None:
        router = Router(
            ids=ids,
            order_limits=OrderLimits(),
            execution=ExecutionConfig(default_algo="pov", pov_participation_rate_pct=Decimal(1)),
        )
        order = make_order(quantity=Decimal(50_000))
        plan = router.route(order, adv=Decimal(1_000_000))
        child = router.pov_child(plan, Decimal(1))
        assert child is None

    def test_pov_child_on_a_non_pov_plan_raises(self, router: Router) -> None:
        order = make_order(quantity=Decimal(100))
        plan = router.route(order, adv=Decimal(1_000_000))  # DIRECT
        with pytest.raises(ValueError, match="non-POV"):
            router.pov_child(plan, Decimal(100))

    def test_pov_finish_releases_the_remainder(self, ids: DeterministicIdGenerator) -> None:
        router = Router(
            ids=ids, order_limits=OrderLimits(), execution=ExecutionConfig(default_algo="pov")
        )
        order = make_order(quantity=Decimal(50_000))
        plan = router.route(order, adv=Decimal(1_000_000))
        router.pov_child(plan, Decimal(10_000))
        final = router.pov_finish(plan)
        assert final is not None
        assert plan.pov is not None and plan.pov.is_complete

    def test_pov_finish_on_a_completed_plan_returns_none(
        self, ids: DeterministicIdGenerator
    ) -> None:
        router = Router(
            ids=ids,
            order_limits=OrderLimits(),
            execution=ExecutionConfig(default_algo="pov", pov_participation_rate_pct=Decimal(100)),
        )
        order = make_order(quantity=Decimal(50_000))
        plan = router.route(order, adv=Decimal(1_000_000))
        router.pov_child(plan, Decimal(1_000_000))  # completes it
        assert router.pov_finish(plan) is None

    def test_pov_finish_on_a_non_pov_plan_raises(self, router: Router) -> None:
        order = make_order(quantity=Decimal(100))
        plan = router.route(order, adv=Decimal(1_000_000))
        with pytest.raises(ValueError, match="non-POV"):
            router.pov_finish(plan)
