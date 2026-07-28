"""Audit log hash chaining — acceptance criterion #9.

The property under test is narrow and important: if anyone edits, reorders,
deletes or inserts a record after the fact, :func:`verify_chain` says so and
names the first bad sequence number. Without that, a post-mortem is reading a
document it cannot trust.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from atrader.audit.hashchain import (
    GENESIS_HASH,
    AuditRecord,
    canonical_json,
    compute_hash,
    verify_chain,
)
from atrader.audit.logger import AuditEvent, AuditLogger, InMemoryAuditSink
from atrader.config.secrets import Secret
from atrader.core.clock import SimulatedClock
from atrader.storage.sql.repositories import create_storage


@pytest.fixture
def logger() -> AuditLogger:
    return AuditLogger(InMemoryAuditSink(), SimulatedClock(start_ns=1_000))


class TestCanonicalJson:
    def test_key_order_does_not_change_the_bytes(self) -> None:
        # A hash over a non-canonical encoding only verifies where it was written.
        assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})

    def test_decimals_keep_their_exact_digits(self) -> None:
        assert b'"0.10000000"' in canonical_json({"x": Decimal("0.10000000")})

    def test_uuid_and_bytes_are_serialisable(self) -> None:
        payload = {"id": UUID(int=1), "digest": b"\xde\xad"}
        assert b"dead" in canonical_json(payload)

    def test_unserialisable_values_are_rejected_loudly(self) -> None:
        with pytest.raises(TypeError, match="not serialisable"):
            canonical_json({"fn": object()})


class TestChainConstruction:
    def test_first_record_links_to_genesis(self, logger: AuditLogger) -> None:
        record = logger.append(AuditEvent.SYSTEM_STARTED)
        assert record.seq == 1
        assert record.prev_hash == GENESIS_HASH

    def test_each_record_links_to_the_previous(self, logger: AuditLogger) -> None:
        first = logger.append(AuditEvent.INTENT_CREATED)
        second = logger.append(AuditEvent.ORDER_REQUESTED)
        assert second.prev_hash == first.hash
        assert second.seq == first.seq + 1

    def test_a_populated_chain_verifies(self, logger: AuditLogger) -> None:
        for i in range(50):
            logger.append(AuditEvent.RISK_CHECK_PASSED, actor="momentum_v3", payload={"i": i})
        result = logger.verify()
        assert result.valid
        assert result.records_checked == 50

    def test_an_empty_chain_verifies(self, logger: AuditLogger) -> None:
        assert logger.verify().valid

    def test_restart_resumes_the_existing_chain(self) -> None:
        # A restart must not begin a second, unverifiable segment.
        sink = InMemoryAuditSink()
        clock = SimulatedClock(start_ns=1_000)

        first = AuditLogger(sink, clock)
        first.append(AuditEvent.SYSTEM_STARTED)
        first.append(AuditEvent.INTENT_CREATED)

        resumed = AuditLogger(sink, clock)
        third = resumed.append(AuditEvent.SYSTEM_STOPPED)

        assert third.seq == 3
        assert resumed.verify().valid


class TestTamperDetection:
    def _chain(self) -> list[AuditRecord]:
        """A valid five-record chain, for the tamper cases below to corrupt."""
        sink = InMemoryAuditSink()
        logger = AuditLogger(sink, SimulatedClock(start_ns=1_000))
        for i in range(5):
            logger.append(AuditEvent.ORDER_REQUESTED, payload={"quantity": i})
        assert logger.verify().valid
        return sink.read_all()

    def test_edited_payload_is_detected(self) -> None:
        records = self._chain()
        # Someone quietly changes a recorded order size after the fact.
        records[2] = AuditRecord(
            seq=records[2].seq,
            event_type=records[2].event_type,
            actor=records[2].actor,
            payload={"quantity": 9999},
            created_at_ns=records[2].created_at_ns,
            prev_hash=records[2].prev_hash,
            hash=records[2].hash,
        )
        result = verify_chain(records)
        assert not result.valid
        assert result.first_bad_seq == 3
        assert result.reason is not None
        assert "does not match its hash" in result.reason

    def test_deleted_record_is_detected(self) -> None:
        records = self._chain()
        del records[2]
        result = verify_chain(records)
        assert not result.valid
        assert result.reason is not None
        assert "sequence gap" in result.reason

    def test_reordered_records_are_detected(self) -> None:
        records = self._chain()
        records[1], records[2] = records[2], records[1]
        assert not verify_chain(records).valid

    def test_a_record_rehashed_to_look_valid_still_breaks_the_link(self) -> None:
        # A tamperer who recomputes the edited record's own hash still cannot
        # fix the *next* record, which committed to the old hash.
        records = self._chain()
        tampered_payload = {"quantity": 9999}
        rehashed = AuditRecord(
            seq=records[2].seq,
            event_type=records[2].event_type,
            actor=records[2].actor,
            payload=tampered_payload,
            created_at_ns=records[2].created_at_ns,
            prev_hash=records[2].prev_hash,
            hash=compute_hash(
                seq=records[2].seq,
                event_type=records[2].event_type,
                actor=records[2].actor,
                payload=tampered_payload,
                created_at_ns=records[2].created_at_ns,
                prev_hash=records[2].prev_hash,
            ),
        )
        records[2] = rehashed
        assert rehashed.is_self_consistent()

        result = verify_chain(records)
        assert not result.valid
        assert result.first_bad_seq == 4
        assert result.reason is not None
        assert "broken link" in result.reason


class TestSecretRedaction:
    def test_secrets_never_enter_the_log(self, logger: AuditLogger) -> None:
        # The audit log is kept for seven years and cannot be edited without
        # breaking the chain, so a key written into it is a key you cannot remove.
        record = logger.append(
            AuditEvent.LLM_REQUEST,
            payload={
                "api_key": "sk-ant-abcdefghijklmnop",
                "note": "authorization: Bearer abcdefghijklmnopqrstuvwxyz",
                "wrapped": Secret("ANTHROPIC_API_KEY", "sk-ant-secret-value"),
            },
        )
        rendered = canonical_json(record.payload).decode()
        assert "abcdefghijklmnop" not in rendered
        assert "sk-ant-secret-value" not in rendered
        assert "REDACTED" in rendered

    def test_nested_secrets_are_redacted(self, logger: AuditLogger) -> None:
        record = logger.append(
            AuditEvent.CONFIG_CHANGED,
            payload={"broker": {"credentials": {"api_key": "sk-ant-nested-key-value"}}},
        )
        assert "sk-ant-nested-key-value" not in canonical_json(record.payload).decode()


class TestSqlAuditSink:
    def test_chain_survives_persistence_and_reopening(self, tmp_path: Path) -> None:
        dsn = f"sqlite+pysqlite:///{tmp_path / 'audit.db'}"
        clock = SimulatedClock(start_ns=1_000)

        storage = create_storage(dsn)
        logger = AuditLogger(storage.audit, clock)
        for i in range(10):
            logger.append(AuditEvent.FILL_RECEIVED, payload={"i": i, "price": Decimal("187.50")})
        assert logger.verify().valid
        storage.close()

        reopened = create_storage(dsn)
        try:
            resumed = AuditLogger(reopened.audit, clock)
            assert resumed.verify().valid
            assert resumed.next_seq == 11
            # And a new record continues the same chain rather than forking it.
            assert resumed.append(AuditEvent.SYSTEM_STOPPED).seq == 11
            assert resumed.verify().valid
        finally:
            reopened.close()
