"""Trace propagation, signal to fill — spec §9.2."""

from __future__ import annotations

from atrader.core.clock import SimulatedClock
from atrader.core.ids import DeterministicIdGenerator
from atrader.monitoring.logging import current_trace_id
from atrader.monitoring.tracing import bind_trace_id, start_trace, trace_context


class TestStartTrace:
    def test_mints_an_id_from_the_given_generator(self) -> None:
        clock = SimulatedClock(start_ns=0)
        ids = DeterministicIdGenerator(clock, seed=1)
        trace_id = start_trace(ids)
        assert isinstance(trace_id, str)
        assert trace_id

    def test_same_seed_and_sequence_mints_the_same_id(self) -> None:
        clock = SimulatedClock(start_ns=0)
        first = start_trace(DeterministicIdGenerator(clock, seed=7))
        second = start_trace(DeterministicIdGenerator(SimulatedClock(start_ns=0), seed=7))
        assert first == second


class TestPropagation:
    def test_trace_context_binds_for_the_duration_of_the_block(self) -> None:
        clock = SimulatedClock(start_ns=0)
        ids = DeterministicIdGenerator(clock, seed=1)
        trace_id = start_trace(ids)

        assert current_trace_id() is None
        with trace_context(trace_id):
            assert current_trace_id() == trace_id
        assert current_trace_id() is None

    def test_bind_trace_id_sets_it_outside_a_context_manager(self) -> None:
        clock = SimulatedClock(start_ns=0)
        trace_id = start_trace(DeterministicIdGenerator(clock, seed=2))
        try:
            bind_trace_id(trace_id)
            assert current_trace_id() == trace_id
        finally:
            bind_trace_id(None)
