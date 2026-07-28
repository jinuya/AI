"""Tamper-evident hash chaining for the audit log.

Spec §4.5:

    감사 로그에 해시 체인을 거는 이유는 사후에 로그를 고칠 수 없게 하기 위해서다.
    규제 대응에서도 그렇고, 무엇보다 사고 조사할 때 로그를 믿을 수 있어야 한다.

Each record hashes its own content together with the previous record's hash, so
editing or deleting anything in the middle invalidates every hash after it.
:func:`verify_chain` reports the first index where that happens — acceptance
criterion #9.

Serialisation is canonical (sorted keys, no insignificant whitespace, ``Decimal``
rendered exactly) because a hash over a non-canonical encoding verifies only on
the machine that wrote it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

__all__ = [
    "GENESIS_HASH",
    "AuditRecord",
    "ChainVerification",
    "canonical_json",
    "compute_hash",
    "dump_records_jsonl",
    "load_records_jsonl",
    "record_from_jsonable",
    "record_to_jsonable",
    "records_from_json_bytes",
    "records_to_json_bytes",
    "verify_chain",
]

#: ``prev_hash`` of the first record. 32 zero bytes, matching the digest width.
GENESIS_HASH = b"\x00" * 32


def _default(value: object) -> str | list[Any]:
    if isinstance(value, Decimal):
        # str() keeps the exact digits; float() would not.
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, frozenset | set):
        return sorted(str(item) for item in value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"{type(value).__name__} is not serialisable in an audit payload")


def canonical_json(payload: object) -> bytes:
    """Serialise deterministically: sorted keys, compact separators, UTF-8.

    Two processes must produce identical bytes for identical content, otherwise
    the chain cannot be verified anywhere but where it was written.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_default,
    ).encode("utf-8")


def compute_hash(
    *,
    seq: int,
    event_type: str,
    actor: str,
    payload: dict[str, Any],
    created_at_ns: int,
    prev_hash: bytes,
) -> bytes:
    """SHA-256 over the record's content plus the previous record's hash."""
    digest = hashlib.sha256()
    digest.update(prev_hash)
    digest.update(
        canonical_json(
            {
                "seq": seq,
                "event_type": event_type,
                "actor": actor,
                "created_at_ns": created_at_ns,
                "payload": payload,
            }
        )
    )
    return digest.digest()


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One immutable entry in the audit log (spec §4.5 ``audit_log`` table)."""

    seq: int
    event_type: str
    actor: str
    """``strategy_id`` | ``user_id`` | ``'system'``."""
    payload: dict[str, Any]
    created_at_ns: int
    prev_hash: bytes
    hash: bytes

    def recompute_hash(self) -> bytes:
        return compute_hash(
            seq=self.seq,
            event_type=self.event_type,
            actor=self.actor,
            payload=self.payload,
            created_at_ns=self.created_at_ns,
            prev_hash=self.prev_hash,
        )

    def is_self_consistent(self) -> bool:
        return self.recompute_hash() == self.hash


@dataclass(frozen=True, slots=True)
class ChainVerification:
    """Outcome of verifying a chain."""

    valid: bool
    records_checked: int
    first_bad_seq: int | None = None
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.valid


def verify_chain(records: list[AuditRecord]) -> ChainVerification:
    """Verify hashes and linkage across an ordered run of records.

    Checks three things, in the order that makes a failure easiest to diagnose:
    contiguous sequence numbers, each record's own hash, and each record's link
    to its predecessor.
    """
    if not records:
        return ChainVerification(valid=True, records_checked=0)

    expected_prev = records[0].prev_hash
    expected_seq = records[0].seq

    for index, record in enumerate(records):
        if record.seq != expected_seq:
            return ChainVerification(
                valid=False,
                records_checked=index,
                first_bad_seq=record.seq,
                reason=(
                    f"sequence gap: expected seq {expected_seq}, found {record.seq}. "
                    "A missing record is as much a break as an edited one."
                ),
            )
        if record.prev_hash != expected_prev:
            return ChainVerification(
                valid=False,
                records_checked=index,
                first_bad_seq=record.seq,
                reason=(
                    f"broken link at seq {record.seq}: prev_hash "
                    f"{record.prev_hash.hex()[:16]}... does not match the previous record's "
                    f"hash {expected_prev.hex()[:16]}..."
                ),
            )
        if not record.is_self_consistent():
            return ChainVerification(
                valid=False,
                records_checked=index,
                first_bad_seq=record.seq,
                reason=(
                    f"content at seq {record.seq} does not match its hash — the record was "
                    "modified after it was written"
                ),
            )
        expected_prev = record.hash
        expected_seq = record.seq + 1

    return ChainVerification(valid=True, records_checked=len(records))


# ---------------------------------------------------------------------------
# Offline export/import — spec §9.3 seven-year retention, verified out of band
# from a live process (``atrader verify-audit``).
# ---------------------------------------------------------------------------


def record_to_jsonable(record: AuditRecord) -> dict[str, Any]:
    return {
        "seq": record.seq,
        "event_type": record.event_type,
        "actor": record.actor,
        "payload": record.payload,
        "created_at_ns": record.created_at_ns,
        "prev_hash": record.prev_hash.hex(),
        "hash": record.hash.hex(),
    }


def record_from_jsonable(data: dict[str, Any]) -> AuditRecord:
    return AuditRecord(
        seq=data["seq"],
        event_type=data["event_type"],
        actor=data["actor"],
        payload=data["payload"],
        created_at_ns=data["created_at_ns"],
        prev_hash=bytes.fromhex(data["prev_hash"]),
        hash=bytes.fromhex(data["hash"]),
    )


def dump_records_jsonl(records: Iterable[AuditRecord], path: Path) -> int:
    """Write records as one JSON object per line, in order.

    The same ``default=`` normalisation :func:`canonical_json` uses (Decimal
    to its exact string, UUID to its string form, etc.) is applied here too —
    a payload round-tripped through this file hashes identically to the
    original, which is what makes :func:`load_records_jsonl` output usable
    with :func:`verify_chain` at all.
    """
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record_to_jsonable(record), sort_keys=True, ensure_ascii=False, default=_default
                )
            )
            handle.write("\n")
            count += 1
    return count


def load_records_jsonl(path: Path) -> list[AuditRecord]:
    """Load records written by :func:`dump_records_jsonl`, in file order."""
    records: list[AuditRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(record_from_jsonable(json.loads(line)))
    return records


def records_to_json_bytes(records: Iterable[AuditRecord]) -> bytes:
    """Render records as one JSON array — the wire shape for ``GET /audit``.

    Same normalisation as :func:`dump_records_jsonl`; the counterpart is
    :func:`records_from_json_bytes`, used by ``atrader verify-audit`` when it
    fetches from a running process instead of an offline export.
    """
    return json.dumps(
        [record_to_jsonable(record) for record in records],
        sort_keys=True,
        ensure_ascii=False,
        default=_default,
    ).encode("utf-8")


def records_from_json_bytes(data: bytes) -> list[AuditRecord]:
    return [record_from_jsonable(item) for item in json.loads(data)]
