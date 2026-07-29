"""The Redis Streams wire format — spec §4.3.

``atrader.bus.redis_streams`` is excluded from the default suite because it
needs a real server, and that exclusion had swallowed its two *pure* functions
along with everything else: ``_encode``/``_decode`` are the actual contract
between a publisher and a subscriber, they need no Redis at all, and they had
never been executed.

The reason to pin them here rather than leave them to an integration run is
that a round-trip defect in this pair would be invisible in every test written
against the in-memory bus and wrong in production only — the exact failure
shape the rest of this codebase spends its effort designing out.

One divergence is documented rather than fixed below: **payload values do not
survive as Python objects**. The in-memory bus stores the payload dict by
reference, so a ``Decimal`` stays a ``Decimal``; this one serializes with
``default=str``, so it comes back a ``str``. Nothing in the system publishes
to the bus yet, so this is a trap laid for whoever wires it up rather than a
live bug — and a trap is worth a test that names it.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import pytest

from atrader.bus.redis_streams import _decode, _encode

TOPIC = "md.bars.1m"
PUBLISHED_AT = 1_700_000_000_123_456_789


def round_trip(
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    *,
    message_id: str = "1700000000000-0",
    delivery_count: int = 1,
):  # type: ignore[no-untyped-def]
    fields = _encode(payload, headers or {}, PUBLISHED_AT)
    return _decode(TOPIC, message_id, fields, delivery_count)


class TestRoundTrip:
    def test_a_plain_payload_survives_intact(self) -> None:
        message = round_trip({"symbol": "AAPL", "count": 3, "final": True})
        assert message.payload == {"symbol": "AAPL", "count": 3, "final": True}

    def test_the_topic_and_id_come_from_the_caller_not_the_fields(self) -> None:
        """Redis owns the message id; it is not part of the encoded body."""
        message = round_trip({}, message_id="1700000000999-7")
        assert message.topic == TOPIC
        assert message.message_id == "1700000000999-7"

    def test_the_publish_timestamp_survives_full_nanosecond_precision(self) -> None:
        """Encoded as a string, not a JSON number: 1.7e18 exceeds what a
        double can hold exactly, so a JSON number would quietly round the
        timestamp."""
        message = round_trip({})
        assert message.published_at_ns == PUBLISHED_AT

    def test_headers_survive(self) -> None:
        message = round_trip({}, {"trace_id": "abc-123", "source": "sim"})
        assert message.headers == {"trace_id": "abc-123", "source": "sim"}
        assert message.trace_id == "abc-123"

    def test_the_delivery_count_comes_from_the_caller(self) -> None:
        """Redelivery is Redis's bookkeeping, not the publisher's."""
        assert round_trip({}, delivery_count=4).delivery_count == 4

    def test_nested_structures_survive(self) -> None:
        payload = {"bars": [{"o": 1, "c": 2}, {"o": 2, "c": 3}], "meta": {"n": 2}}
        assert round_trip(payload).payload == payload


class TestEncodingIsDeterministic:
    def test_keys_are_sorted_so_the_same_payload_encodes_identically(self) -> None:
        """Two dicts that differ only in insertion order must produce the same
        bytes — otherwise a recorded stream could not be byte-compared, which
        is what acceptance criterion #3 does."""
        first = _encode({"b": 2, "a": 1}, {"y": "2", "x": "1"}, PUBLISHED_AT)
        second = _encode({"a": 1, "b": 2}, {"x": "1", "y": "2"}, PUBLISHED_AT)
        assert first == second

    def test_every_encoded_field_is_a_string(self) -> None:
        """Redis stream fields are strings; handing it an int would fail at
        the client boundary rather than here."""
        fields = _encode({"n": 1}, {"h": "v"}, PUBLISHED_AT)
        assert all(isinstance(value, str) for value in fields.values())
        assert set(fields) == {"payload", "headers", "published_at_ns"}


class TestDecodingIsForgivingAboutMissingFields:
    """A stream entry written by an older publisher, or truncated, should not
    take the consumer down — an unparseable message is a poison pill that
    would be redelivered forever."""

    def test_an_entry_with_no_fields_decodes_to_an_empty_message(self) -> None:
        message = _decode(TOPIC, "1-0", {}, 1)
        assert message.payload == {}
        assert message.headers == {}
        assert message.published_at_ns == 0

    def test_a_missing_payload_alone_still_yields_the_headers(self) -> None:
        message = _decode(TOPIC, "1-0", {"headers": json.dumps({"trace_id": "t"})}, 1)
        assert message.payload == {}
        assert message.trace_id == "t"


class TestPayloadValuesLoseTheirPythonTypes:
    """The divergence from the in-memory bus, pinned so it is discovered here
    rather than in production.

    The in-memory bus keeps the payload dict by reference — a ``Decimal`` in,
    a ``Decimal`` out. This backend serializes with ``default=str``, so any
    value JSON cannot represent natively arrives as a string. A consumer that
    does arithmetic on ``payload["price"]`` therefore passes every in-memory
    test and raises ``TypeError`` against Redis.
    """

    def test_a_decimal_arrives_as_a_string(self) -> None:
        message = round_trip({"price": Decimal("100.50")})
        assert message.payload["price"] == "100.50"
        assert not isinstance(message.payload["price"], Decimal)

    def test_the_value_is_preserved_exactly_even_though_the_type_is_not(self) -> None:
        """``default=str`` rather than ``float``: the digits survive, so a
        consumer that reconstructs the Decimal loses nothing."""
        original = Decimal("12345.67890123")
        message = round_trip({"price": original})
        assert Decimal(message.payload["price"]) == original

    @pytest.mark.parametrize("value", ["0.1", "0.2", "1e-8", "99999999999.99999999"])
    def test_no_float_rounding_is_introduced_on_the_way_through(self, value: str) -> None:
        original = Decimal(value)
        assert Decimal(round_trip({"v": original}).payload["v"]) == original

    def test_json_native_types_keep_their_types(self) -> None:
        """Only what JSON cannot express goes through ``str`` — ints, floats,
        bools, None and strings are unaffected."""
        payload = {"i": 7, "f": 1.5, "b": False, "n": None, "s": "x"}
        assert round_trip(payload).payload == payload
