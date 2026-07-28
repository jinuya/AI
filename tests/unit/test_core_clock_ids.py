"""Clock and ID generation — the two injection points that make replay possible."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from atrader.core.clock import (
    NS_PER_SECOND,
    Clock,
    SimulatedClock,
    SystemClock,
    datetime_to_ns,
    ns_to_datetime,
)
from atrader.core.ids import (
    DeterministicIdGenerator,
    IdGenerator,
    SystemIdGenerator,
    compose_uuid7,
)


class TestSystemClock:
    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(SystemClock(), Clock)

    def test_reports_utc(self) -> None:
        now = SystemClock().now()
        assert now.tzinfo is not None
        assert now.utcoffset() == timedelta(0)

    def test_monotonic_never_goes_backwards(self) -> None:
        clock = SystemClock()
        first = clock.monotonic_ns()
        second = clock.monotonic_ns()
        assert second >= first


class TestSimulatedClock:
    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(SimulatedClock(), Clock)

    def test_starts_where_told_and_does_not_move_on_its_own(self) -> None:
        clock = SimulatedClock(start_ns=1_700_000_000 * NS_PER_SECOND)
        assert clock.now_ns() == clock.now_ns()

    def test_advance(self) -> None:
        clock = SimulatedClock(start_ns=1000)
        clock.advance(500)
        assert clock.now_ns() == 1500
        assert clock.monotonic_ns() == 500

    def test_advance_seconds(self) -> None:
        clock = SimulatedClock()
        clock.advance_seconds(1.5)
        assert clock.now_ns() == 1_500_000_000

    def test_rejects_negative_advance(self) -> None:
        with pytest.raises(ValueError, match="negative delta"):
            SimulatedClock().advance(-1)

    def test_rejects_negative_start(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            SimulatedClock(start_ns=-1)

    def test_set_to_moves_forward(self) -> None:
        clock = SimulatedClock(start_ns=1000)
        clock.set_to(5000)
        assert clock.now_ns() == 5000
        assert clock.monotonic_ns() == 4000

    def test_set_to_refuses_to_rewind(self) -> None:
        # Out-of-order replay events are a bug in the recording, not something
        # to paper over by letting time run backwards.
        clock = SimulatedClock(start_ns=5000)
        with pytest.raises(ValueError, match="cannot set clock backwards"):
            clock.set_to(1000)


class TestTimestampConversion:
    def test_round_trip(self) -> None:
        original = datetime(2026, 7, 28, 14, 30, 0, tzinfo=UTC)
        assert ns_to_datetime(datetime_to_ns(original)) == original

    def test_naive_datetime_is_rejected(self) -> None:
        # An implicit timezone is how a feed ends up eight hours out.
        with pytest.raises(ValueError, match="naive datetime rejected"):
            datetime_to_ns(datetime(2026, 7, 28, 14, 30, 0))  # noqa: DTZ001


class TestUuid7Layout:
    def test_version_and_variant_bits(self) -> None:
        uid = compose_uuid7(0x0123456789AB, 0xABC, 0x1234567890ABCDE)
        assert uid.version == 7
        assert (uid.int >> 62) & 0b11 == 0b10

    def test_timestamp_is_recoverable(self) -> None:
        unix_ms = 1_753_000_000_000
        uid = compose_uuid7(unix_ms, 0, 0)
        assert uid.int >> 80 == unix_ms

    def test_ids_sort_by_time(self) -> None:
        early = compose_uuid7(1_000_000_000_000, 0xFFF, (1 << 62) - 1)
        late = compose_uuid7(1_000_000_000_001, 0, 0)
        assert early.int < late.int


class TestSystemIdGenerator:
    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(SystemIdGenerator(SystemClock()), IdGenerator)

    def test_ids_are_unique(self) -> None:
        gen = SystemIdGenerator(SystemClock())
        ids = {gen.new_id() for _ in range(1000)}
        assert len(ids) == 1000

    def test_ids_are_version_7(self) -> None:
        assert SystemIdGenerator(SystemClock()).new_id().version == 7


class TestDeterministicIdGenerator:
    def test_same_seed_and_clock_produce_the_same_sequence(self) -> None:
        # This is what makes byte-for-byte replay comparison possible.
        def run() -> list[str]:
            gen = DeterministicIdGenerator(
                SimulatedClock(start_ns=1_700_000_000 * NS_PER_SECOND), seed=42
            )
            return [str(gen.new_id()) for _ in range(10)]

        assert run() == run()

    def test_different_seeds_diverge(self) -> None:
        clock = SimulatedClock(start_ns=1_700_000_000 * NS_PER_SECOND)
        a = DeterministicIdGenerator(clock, seed=1).new_id()
        b = DeterministicIdGenerator(clock, seed=2).new_id()
        assert a != b

    def test_ids_are_unique_within_a_run(self) -> None:
        gen = DeterministicIdGenerator(SimulatedClock(), seed=7)
        ids = {gen.new_id() for _ in range(5000)}
        assert len(ids) == 5000

    def test_reset_restarts_the_sequence(self) -> None:
        gen = DeterministicIdGenerator(SimulatedClock(), seed=3)
        first = [gen.new_id() for _ in range(5)]
        gen.reset()
        assert [gen.new_id() for _ in range(5)] == first

    def test_ids_are_version_7(self) -> None:
        assert DeterministicIdGenerator(SimulatedClock()).new_id().version == 7
