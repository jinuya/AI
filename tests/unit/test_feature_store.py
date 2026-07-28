"""Point-in-time feature store — spec §8.2, look-ahead prevention."""

from __future__ import annotations

from decimal import Decimal

import pytest

from atrader.features.store import FeatureStore, FeatureValue

BASE_NS = 1_700_000_000_000_000_000


def make_value(*, at_ns: int, value: str = "1") -> FeatureValue:
    return FeatureValue(
        symbol="AAPL", name="sma_20", value=Decimal(value), valid_from_ns=at_ns, version=1
    )


class TestPublishOrdering:
    def test_publishing_out_of_order_is_rejected(self) -> None:
        store = FeatureStore()
        store.publish(make_value(at_ns=BASE_NS + 100))
        with pytest.raises(ValueError, match="out of order"):
            store.publish(make_value(at_ns=BASE_NS))

    def test_equal_timestamps_are_accepted(self) -> None:
        store = FeatureStore()
        store.publish(make_value(at_ns=BASE_NS, value="1"))
        store.publish(make_value(at_ns=BASE_NS, value="2"))  # same tick, e.g. a correction
        assert store.latest("AAPL", "sma_20").value == Decimal("2")  # type: ignore[union-attr]


class TestAsof:
    def test_nothing_published_returns_none(self) -> None:
        store = FeatureStore()
        assert store.asof("AAPL", "sma_20", BASE_NS) is None

    def test_a_query_before_the_first_publish_returns_none(self) -> None:
        store = FeatureStore()
        store.publish(make_value(at_ns=BASE_NS + 1000))
        assert store.asof("AAPL", "sma_20", BASE_NS) is None

    def test_a_query_at_exactly_valid_from_ns_sees_the_value(self) -> None:
        store = FeatureStore()
        store.publish(make_value(at_ns=BASE_NS))
        result = store.asof("AAPL", "sma_20", BASE_NS)
        assert result is not None
        assert result.value == Decimal("1")

    def test_a_later_query_sees_the_most_recent_value_not_in_the_future(self) -> None:
        store = FeatureStore()
        store.publish(make_value(at_ns=BASE_NS, value="1"))
        store.publish(make_value(at_ns=BASE_NS + 100, value="2"))
        store.publish(make_value(at_ns=BASE_NS + 200, value="3"))

        assert store.asof("AAPL", "sma_20", BASE_NS + 150).value == Decimal("2")  # type: ignore[union-attr]

    def test_this_is_the_look_ahead_guarantee_itself(self) -> None:
        """A value published *after* the query time must never be visible."""
        store = FeatureStore()
        store.publish(make_value(at_ns=BASE_NS + 1_000_000, value="999"))
        assert store.asof("AAPL", "sma_20", BASE_NS) is None

    def test_different_symbols_and_names_are_independent(self) -> None:
        store = FeatureStore()
        store.publish(make_value(at_ns=BASE_NS, value="1"))
        store.publish(
            FeatureValue(
                symbol="MSFT", name="sma_20", value=Decimal("9"), valid_from_ns=BASE_NS, version=1
            )
        )
        assert store.asof("AAPL", "sma_20", BASE_NS).value == Decimal("1")  # type: ignore[union-attr]
        assert store.asof("MSFT", "sma_20", BASE_NS).value == Decimal("9")  # type: ignore[union-attr]


class TestHistory:
    def test_history_returns_every_published_value_in_order(self) -> None:
        store = FeatureStore()
        store.publish(make_value(at_ns=BASE_NS, value="1"))
        store.publish(make_value(at_ns=BASE_NS + 1, value="2"))
        result = store.history("AAPL", "sma_20")
        assert [v.value for v in result] == [Decimal("1"), Decimal("2")]

    def test_history_of_an_unknown_key_is_empty(self) -> None:
        store = FeatureStore()
        assert store.history("AAPL", "sma_20") == ()
