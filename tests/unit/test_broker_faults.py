"""Fault injection — spec §8.3 (카오스 테스트).

Covers the fault-decision machinery itself (:class:`FaultInjector`,
:class:`FaultProfile`, :class:`ScriptedFaults`) and :class:`FaultyBroker`'s
wrapping behaviour. The scenario that matters most — a submission that lands
and only the response is lost — is exercised end-to-end against
:class:`~atrader.execution.idempotency.submit_with_retry` and the OMS in
``tests/chaos/``; this file is about the injector and wrapper in isolation.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from atrader.brokers.faults import (
    Fault,
    FaultInjector,
    FaultProfile,
    FaultyBroker,
    ScriptedFaults,
    replay_events,
)
from atrader.brokers.models import BrokerEvent, BrokerEventType, OrderModification
from atrader.brokers.paper import PaperBroker, PaperBrokerConfig
from atrader.core.clock import SimulatedClock
from atrader.core.errors import (
    AmbiguousBrokerError,
    PermanentBrokerError,
    TransientBrokerError,
)
from atrader.core.models import OrderRequest
from atrader.core.rng import SeededRng
from atrader.core.types import OrderType, Side, TimeInForce

BASE_NS = 1_700_000_000_000_000_000


def make_request(**overrides: object) -> OrderRequest:
    defaults: dict[str, object] = {
        "client_order_id": "coid-1",
        "symbol": "AAPL",
        "side": Side.BUY,
        "order_type": OrderType.LIMIT,
        "quantity": Decimal("10"),
        "limit_price": Decimal("190.00"),
        "time_in_force": TimeInForce.DAY,
    }
    return OrderRequest(**{**defaults, **overrides})  # type: ignore[arg-type]


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(start_ns=BASE_NS)


@pytest.fixture
def paper(clock: SimulatedClock) -> PaperBroker:
    return PaperBroker(
        clock,
        config=PaperBrokerConfig(
            latency_ns=0, partial_fill_probability=0.0, reject_probability=0.0
        ),
        prices={"AAPL": Decimal("190.00")},
    )


class TestFaultProfile:
    def test_probabilities_over_100_percent_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="100%"):
            FaultProfile(timeout_pct=Decimal(60), transient_pct=Decimal(60))

    def test_duplicate_event_pct_out_of_range_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate_event_pct"):
            FaultProfile(duplicate_event_pct=Decimal(150))

    def test_drop_event_pct_out_of_range_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="drop_event_pct"):
            FaultProfile(drop_event_pct=Decimal(-1))

    def test_flaky_and_hostile_profiles_are_valid(self) -> None:
        assert FaultProfile.flaky() is not None
        assert FaultProfile.hostile() is not None


class TestScriptedFaults:
    def test_faults_are_consumed_in_order(self) -> None:
        scripted = ScriptedFaults.of(Fault.TIMEOUT, Fault.AMBIGUOUS)
        assert scripted.next_fault() == Fault.TIMEOUT
        assert scripted.next_fault() == Fault.AMBIGUOUS
        assert scripted.exhausted

    def test_exhausted_script_returns_none(self) -> None:
        scripted = ScriptedFaults.of(Fault.TIMEOUT)
        scripted.next_fault()
        assert scripted.next_fault() == Fault.NONE
        assert scripted.next_fault() == Fault.NONE

    def test_reset_replays_from_the_start(self) -> None:
        scripted = ScriptedFaults.of(Fault.TIMEOUT, Fault.AMBIGUOUS)
        scripted.next_fault()
        scripted.reset()
        assert not scripted.exhausted
        assert scripted.next_fault() == Fault.TIMEOUT


class TestFaultInjector:
    def test_scripted_takes_precedence_over_probabilities(self) -> None:
        injector = FaultInjector(
            profile=FaultProfile(timeout_pct=Decimal(100)),  # would always fire
            scripted=ScriptedFaults.of(Fault.AMBIGUOUS),
        )
        assert injector.next_fault() == Fault.AMBIGUOUS  # scripted wins
        assert injector.next_fault() == Fault.NONE  # script exhausted; probabilities ignored too

    def test_call_count_increments_on_every_draw(self) -> None:
        injector = FaultInjector(scripted=ScriptedFaults.of(Fault.NONE, Fault.NONE))
        injector.next_fault()
        injector.next_fault()
        assert injector.call_count == 2

    def test_faults_fired_tallies_by_kind(self) -> None:
        injector = FaultInjector(
            scripted=ScriptedFaults.of(Fault.TIMEOUT, Fault.TIMEOUT, Fault.NONE)
        )
        injector.next_fault()
        injector.next_fault()
        injector.next_fault()
        assert injector.faults_fired == {Fault.TIMEOUT: 2}

    def test_probability_draw_is_deterministic_for_a_seed(self) -> None:
        profile = FaultProfile(timeout_pct=Decimal(50))
        a = FaultInjector(profile=profile, rng=SeededRng(seed=1))
        b = FaultInjector(profile=profile, rng=SeededRng(seed=1))
        draws_a = [a.next_fault() for _ in range(20)]
        draws_b = [b.next_fault() for _ in range(20)]
        assert draws_a == draws_b

    def test_probability_draw_always_none_at_zero_pct(self) -> None:
        injector = FaultInjector(profile=FaultProfile(), rng=SeededRng(seed=0))
        assert all(injector.next_fault() == Fault.NONE for _ in range(20))

    def test_roll_pct_is_false_at_zero(self) -> None:
        injector = FaultInjector(rng=SeededRng(seed=0))
        assert not injector.roll_pct(Decimal(0))

    def test_roll_pct_is_deterministic_for_a_seed(self) -> None:
        a = FaultInjector(rng=SeededRng(seed=5))
        b = FaultInjector(rng=SeededRng(seed=5))
        assert [a.roll_pct(Decimal(50)) for _ in range(10)] == [
            b.roll_pct(Decimal(50)) for _ in range(10)
        ]


class TestFaultyBrokerSubmit:
    async def test_no_fault_delegates_normally(self, paper: PaperBroker) -> None:
        faulty = FaultyBroker(inner=paper, injector=FaultInjector(scripted=ScriptedFaults.of()))
        ack = await faulty.submit_order(make_request())
        assert ack.accepted

    async def test_timeout_never_reaches_the_inner_broker(self, paper: PaperBroker) -> None:
        faulty = FaultyBroker(
            inner=paper, injector=FaultInjector(scripted=ScriptedFaults.of(Fault.TIMEOUT))
        )
        with pytest.raises(TimeoutError):
            await faulty.submit_order(make_request())
        assert await paper.get_order("coid-1") is None  # never actually placed

    async def test_transient_never_reaches_the_inner_broker(self, paper: PaperBroker) -> None:
        faulty = FaultyBroker(
            inner=paper, injector=FaultInjector(scripted=ScriptedFaults.of(Fault.TRANSIENT))
        )
        with pytest.raises(TransientBrokerError):
            await faulty.submit_order(make_request())
        assert await paper.get_order("coid-1") is None

    async def test_permanent_never_reaches_the_inner_broker(self, paper: PaperBroker) -> None:
        faulty = FaultyBroker(
            inner=paper, injector=FaultInjector(scripted=ScriptedFaults.of(Fault.PERMANENT))
        )
        with pytest.raises(PermanentBrokerError):
            await faulty.submit_order(make_request())
        assert await paper.get_order("coid-1") is None

    async def test_ambiguous_order_actually_lands_before_the_error(
        self, paper: PaperBroker
    ) -> None:
        """The dangerous case this whole module exists to reproduce."""
        faulty = FaultyBroker(
            inner=paper, injector=FaultInjector(scripted=ScriptedFaults.of(Fault.AMBIGUOUS))
        )
        with pytest.raises(AmbiguousBrokerError) as exc_info:
            await faulty.submit_order(make_request())
        assert exc_info.value.client_order_id == "coid-1"
        # Unlike TIMEOUT/TRANSIENT/PERMANENT, the order really was placed.
        assert await paper.get_order("coid-1") is not None


class TestFaultyBrokerCancelAndModify:
    async def test_cancel_timeout_never_reaches_the_inner_broker(self, paper: PaperBroker) -> None:
        await paper.submit_order(make_request())
        faulty = FaultyBroker(
            inner=paper, injector=FaultInjector(scripted=ScriptedFaults.of(Fault.TIMEOUT))
        )
        with pytest.raises(TimeoutError):
            await faulty.cancel_order("coid-1")
        state = await paper.get_order("coid-1")
        assert state is not None and state.status.is_open  # cancel never landed

    async def test_cancel_ambiguous_lands_before_the_error(self, paper: PaperBroker) -> None:
        await paper.submit_order(make_request())
        faulty = FaultyBroker(
            inner=paper, injector=FaultInjector(scripted=ScriptedFaults.of(Fault.AMBIGUOUS))
        )
        with pytest.raises(AmbiguousBrokerError):
            await faulty.cancel_order("coid-1")
        state = await paper.get_order("coid-1")
        assert state is not None and not state.status.is_open  # it really was canceled

    async def test_cancel_transient_never_reaches_the_inner_broker(
        self, paper: PaperBroker
    ) -> None:
        await paper.submit_order(make_request())
        faulty = FaultyBroker(
            inner=paper, injector=FaultInjector(scripted=ScriptedFaults.of(Fault.TRANSIENT))
        )
        with pytest.raises(TransientBrokerError):
            await faulty.cancel_order("coid-1")

    async def test_cancel_with_no_fault_delegates(self, paper: PaperBroker) -> None:
        await paper.submit_order(make_request())
        faulty = FaultyBroker(inner=paper, injector=FaultInjector(scripted=ScriptedFaults.of()))
        ack = await faulty.cancel_order("coid-1")
        assert ack.accepted

    async def test_modify_timeout_never_reaches_the_inner_broker(self, paper: PaperBroker) -> None:
        await paper.submit_order(make_request())
        faulty = FaultyBroker(
            inner=paper, injector=FaultInjector(scripted=ScriptedFaults.of(Fault.TIMEOUT))
        )
        with pytest.raises(TimeoutError):
            await faulty.modify_order("coid-1", OrderModification(quantity=Decimal("5")))

    async def test_modify_transient_never_reaches_the_inner_broker(
        self, paper: PaperBroker
    ) -> None:
        await paper.submit_order(make_request())
        faulty = FaultyBroker(
            inner=paper, injector=FaultInjector(scripted=ScriptedFaults.of(Fault.TRANSIENT))
        )
        with pytest.raises(TransientBrokerError):
            await faulty.modify_order("coid-1", OrderModification(quantity=Decimal("5")))

    async def test_modify_with_no_fault_delegates(self, paper: PaperBroker) -> None:
        await paper.submit_order(make_request())
        faulty = FaultyBroker(inner=paper, injector=FaultInjector(scripted=ScriptedFaults.of()))
        ack = await faulty.modify_order("coid-1", OrderModification(quantity=Decimal("5")))
        assert ack.accepted


class TestFaultyBrokerQueries:
    async def test_get_order_fails_transiently_without_touching_the_inner_broker(
        self, paper: PaperBroker
    ) -> None:
        await paper.submit_order(make_request())
        faulty = FaultyBroker(
            inner=paper, injector=FaultInjector(scripted=ScriptedFaults.of(Fault.TRANSIENT))
        )
        with pytest.raises(TransientBrokerError):
            await faulty.get_order("coid-1")

    async def test_get_order_with_no_fault_delegates(self, paper: PaperBroker) -> None:
        await paper.submit_order(make_request())
        faulty = FaultyBroker(inner=paper, injector=FaultInjector(scripted=ScriptedFaults.of()))
        assert await faulty.get_order("coid-1") is not None

    async def test_get_open_orders_can_fail(self, paper: PaperBroker) -> None:
        faulty = FaultyBroker(
            inner=paper, injector=FaultInjector(scripted=ScriptedFaults.of(Fault.TIMEOUT))
        )
        with pytest.raises(TransientBrokerError):
            await faulty.get_open_orders()

    async def test_get_open_orders_with_no_fault_delegates(self, paper: PaperBroker) -> None:
        faulty = FaultyBroker(inner=paper, injector=FaultInjector(scripted=ScriptedFaults.of()))
        assert await faulty.get_open_orders() == []

    async def test_get_positions_can_fail(self, paper: PaperBroker) -> None:
        faulty = FaultyBroker(
            inner=paper, injector=FaultInjector(scripted=ScriptedFaults.of(Fault.TIMEOUT))
        )
        with pytest.raises(TransientBrokerError):
            await faulty.get_positions()

    async def test_get_positions_with_no_fault_delegates(self, paper: PaperBroker) -> None:
        faulty = FaultyBroker(inner=paper, injector=FaultInjector(scripted=ScriptedFaults.of()))
        assert await faulty.get_positions() == []

    async def test_get_account_can_fail(self, paper: PaperBroker) -> None:
        faulty = FaultyBroker(
            inner=paper, injector=FaultInjector(scripted=ScriptedFaults.of(Fault.TIMEOUT))
        )
        with pytest.raises(TransientBrokerError):
            await faulty.get_account()

    async def test_get_account_with_no_fault_delegates(self, paper: PaperBroker) -> None:
        faulty = FaultyBroker(inner=paper, injector=FaultInjector(scripted=ScriptedFaults.of()))
        account = await faulty.get_account()
        assert account.cash == Decimal("100000.00000000")


class TestFaultyBrokerPassthrough:
    def test_name_and_capabilities_are_delegated(self, paper: PaperBroker) -> None:
        faulty = FaultyBroker(inner=paper, injector=FaultInjector())
        assert faulty.name == "faulty:paper"
        assert faulty.capabilities == paper.capabilities

    async def test_close_delegates(self, paper: PaperBroker) -> None:
        faulty = FaultyBroker(inner=paper, injector=FaultInjector())
        await faulty.close()  # no exception


class TestCorruptedStream:
    async def test_no_corruption_passes_events_through_unchanged(self, paper: PaperBroker) -> None:
        await paper.submit_order(make_request())
        faulty = FaultyBroker(inner=paper, injector=FaultInjector(profile=FaultProfile()))
        events = [e async for e in faulty.stream_updates()]
        assert len(events) == 1
        assert events[0].event_type is BrokerEventType.ORDER_ACCEPTED

    async def test_duplicate_event_pct_redelivers(self, paper: PaperBroker) -> None:
        await paper.submit_order(make_request())
        faulty = FaultyBroker(
            inner=paper,
            injector=FaultInjector(profile=FaultProfile(duplicate_event_pct=Decimal(100))),
        )
        events = [e async for e in faulty.stream_updates()]
        assert len(events) == 2
        assert events[0] == events[1]

    async def test_drop_event_pct_discards(self, paper: PaperBroker) -> None:
        await paper.submit_order(make_request())
        faulty = FaultyBroker(
            inner=paper, injector=FaultInjector(profile=FaultProfile(drop_event_pct=Decimal(100)))
        )
        events = [e async for e in faulty.stream_updates()]
        assert events == []

    async def test_reorder_events_swaps_adjacent_pairs(self, paper: PaperBroker) -> None:
        # Two events: ORDER_ACCEPTED then FILL (a market order fills on submit).
        await paper.submit_order(make_request(order_type=OrderType.MARKET, limit_price=None))
        faulty = FaultyBroker(
            inner=paper, injector=FaultInjector(profile=FaultProfile(reorder_events=True))
        )
        events = [e async for e in faulty.stream_updates()]
        assert [e.event_type for e in events] == [
            BrokerEventType.FILL,
            BrokerEventType.ORDER_ACCEPTED,
        ]

    async def test_reorder_with_an_odd_event_count_flushes_the_held_event(
        self, paper: PaperBroker
    ) -> None:
        await paper.submit_order(make_request())  # one event only: ORDER_ACCEPTED
        faulty = FaultyBroker(
            inner=paper, injector=FaultInjector(profile=FaultProfile(reorder_events=True))
        )
        events = [e async for e in faulty.stream_updates()]
        assert len(events) == 1  # held-back event still gets flushed at the end

    async def test_disconnect_after_n_events_raises(self, paper: PaperBroker) -> None:
        await paper.submit_order(make_request(client_order_id="coid-1"))
        await paper.submit_order(make_request(client_order_id="coid-2"))
        faulty = FaultyBroker(
            inner=paper,
            injector=FaultInjector(profile=FaultProfile(disconnect_after_events=1)),
        )
        received: list[BrokerEvent] = []
        with pytest.raises(TransientBrokerError, match="connection dropped"):
            async for event in faulty.stream_updates():
                received.append(event)
        assert len(received) == 1


class TestReplayEvents:
    def test_duplicates_every_event(self) -> None:
        events = [
            BrokerEvent(event_type=BrokerEventType.FILL, client_order_id="a"),
            BrokerEvent(event_type=BrokerEventType.FILL, client_order_id="b"),
        ]
        doubled = replay_events(events)
        assert len(doubled) == 4
        assert doubled[0] == doubled[1]
        assert doubled[2] == doubled[3]

    def test_times_parameter_controls_the_multiplier(self) -> None:
        events = [BrokerEvent(event_type=BrokerEventType.FILL, client_order_id="a")]
        assert len(replay_events(events, times=3)) == 3
