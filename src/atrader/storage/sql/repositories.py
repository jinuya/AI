"""SQL-backed repositories.

Runs on SQLite (default, no infrastructure needed) and PostgreSQL (production).
The same code serves both; only the DSN changes.

Each write commits on its own. That is deliberate rather than lazy: an order or
fill that exists in memory but not on disk is exactly the state that makes a
crash unrecoverable, and reconciliation (spec §FR-EXE-05) can only compare
against what was actually durable.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, Row, create_engine, delete, select
from sqlalchemy.exc import IntegrityError

from atrader.audit.hashchain import AuditRecord, canonical_json
from atrader.core.models import Fill, Order, Position, TradingIntent
from atrader.core.types import OrderStatus
from atrader.storage.sql.schema import (
    METADATA,
    audit_log,
    fills,
    intents,
    orders,
    positions,
)

__all__ = [
    "SqlAuditSink",
    "SqlFillStore",
    "SqlIntentStore",
    "SqlOrderStore",
    "SqlPositionStore",
    "SqlStorage",
    "create_storage",
]

_OPEN_STATUS_VALUES = tuple(status.value for status in OrderStatus if status.is_open)


def _row_to_dict(row: Row[Any]) -> dict[str, Any]:
    # ._mapping is SQLAlchemy's documented Row -> Mapping accessor, not private API.
    return dict(row._mapping)


class SqlOrderStore:
    __slots__ = ("_engine",)

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def upsert(self, order: Order) -> None:
        values = order.model_dump()
        values["order_id"] = str(order.order_id)
        values["parent_intent_id"] = (
            None if order.parent_intent_id is None else str(order.parent_intent_id)
        )
        values["risk_check_id"] = None if order.risk_check_id is None else str(order.risk_check_id)
        values["side"] = order.side.value
        values["order_type"] = order.order_type.value
        values["time_in_force"] = order.time_in_force.value
        values["status"] = order.status.value

        with self._engine.begin() as conn:
            existing = conn.execute(
                select(orders.c.order_id).where(orders.c.order_id == str(order.order_id))
            ).first()
            if existing is None:
                try:
                    conn.execute(orders.insert().values(**values))
                except IntegrityError as exc:
                    raise ValueError(
                        f"client_order_id {order.client_order_id!r} is already in use — "
                        "reusing one would break the idempotency guarantee"
                    ) from exc
            else:
                conn.execute(
                    orders.update().where(orders.c.order_id == str(order.order_id)).values(**values)
                )

    def _select(self, whereclause: Any) -> Order | None:
        with self._engine.connect() as conn:
            row = conn.execute(select(orders).where(whereclause)).first()
        return None if row is None else Order.model_validate(_row_to_dict(row))

    def get(self, order_id: UUID) -> Order | None:
        return self._select(orders.c.order_id == str(order_id))

    def get_by_client_order_id(self, client_order_id: str) -> Order | None:
        return self._select(orders.c.client_order_id == client_order_id)

    def open_orders(self) -> list[Order]:
        with self._engine.connect() as conn:
            rows = conn.execute(select(orders).where(orders.c.status.in_(_OPEN_STATUS_VALUES)))
            return [Order.model_validate(_row_to_dict(row)) for row in rows]

    def all_orders(self) -> list[Order]:
        with self._engine.connect() as conn:
            rows = conn.execute(select(orders).order_by(orders.c.created_at_ns))
            return [Order.model_validate(_row_to_dict(row)) for row in rows]


class SqlFillStore:
    __slots__ = ("_engine",)

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def append(self, fill: Fill) -> bool:
        values = fill.model_dump()
        values["fill_id"] = str(fill.fill_id)
        values["order_id"] = str(fill.order_id)
        values["side"] = fill.side.value
        try:
            with self._engine.begin() as conn:
                conn.execute(fills.insert().values(**values))
        except IntegrityError:
            # UNIQUE violation on fill_id or broker_fill_id: this execution was
            # already recorded. Redelivery is normal on an at-least-once bus.
            return False
        return True

    def get(self, fill_id: UUID) -> Fill | None:
        with self._engine.connect() as conn:
            row = conn.execute(select(fills).where(fills.c.fill_id == str(fill_id))).first()
        return None if row is None else Fill.model_validate(_row_to_dict(row))

    def for_order(self, order_id: UUID) -> list[Fill]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(fills)
                .where(fills.c.order_id == str(order_id))
                .order_by(fills.c.executed_at_ns)
            )
            return [Fill.model_validate(_row_to_dict(row)) for row in rows]

    def all_fills(self) -> list[Fill]:
        with self._engine.connect() as conn:
            rows = conn.execute(select(fills).order_by(fills.c.executed_at_ns))
            return [Fill.model_validate(_row_to_dict(row)) for row in rows]


class SqlIntentStore:
    __slots__ = ("_engine",)

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def append(self, intent: TradingIntent) -> None:
        values = intent.model_dump()
        values["intent_id"] = str(intent.intent_id)
        values["side"] = intent.side.value
        values["target_type"] = intent.target_type.value
        values["urgency"] = intent.urgency.value
        values["time_in_force"] = intent.time_in_force.value
        with self._engine.begin() as conn:
            conn.execute(delete(intents).where(intents.c.intent_id == str(intent.intent_id)))
            conn.execute(intents.insert().values(**values))

    def get(self, intent_id: UUID) -> TradingIntent | None:
        with self._engine.connect() as conn:
            row = conn.execute(select(intents).where(intents.c.intent_id == str(intent_id))).first()
        return None if row is None else TradingIntent.model_validate(_row_to_dict(row))

    def all_intents(self) -> list[TradingIntent]:
        with self._engine.connect() as conn:
            rows = conn.execute(select(intents).order_by(intents.c.created_at_ns))
            return [TradingIntent.model_validate(_row_to_dict(row)) for row in rows]


class SqlPositionStore:
    __slots__ = ("_engine",)

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def upsert(self, position: Position) -> None:
        values = position.model_dump()
        with self._engine.begin() as conn:
            conn.execute(delete(positions).where(positions.c.symbol == position.symbol))
            conn.execute(positions.insert().values(**values))

    def get(self, symbol: str) -> Position | None:
        with self._engine.connect() as conn:
            row = conn.execute(select(positions).where(positions.c.symbol == symbol)).first()
        return None if row is None else Position.model_validate(_row_to_dict(row))

    def all_positions(self) -> list[Position]:
        with self._engine.connect() as conn:
            rows = conn.execute(select(positions).order_by(positions.c.symbol))
            return [Position.model_validate(_row_to_dict(row)) for row in rows]


class SqlAuditSink:
    """Durable :class:`~atrader.audit.logger.AuditSink`.

    Spec §9.3 asks for WORM storage in production. This does not provide that on
    its own — it is an ordinary table — but the hash chain means tampering is
    *detectable* even when the storage layer permits it, which is the property
    an incident review actually depends on.
    """

    __slots__ = ("_engine",)

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def append(self, record: AuditRecord) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                audit_log.insert().values(
                    seq=record.seq,
                    event_type=record.event_type,
                    actor=record.actor,
                    # The same serializer the hash was computed with. `default=str`
                    # renders bytes as "b'\\x01'" and a set as "{'a'}",
                    # neither of which reads back as what was hashed — so an
                    # untouched record failed verify_chain, indistinguishable
                    # from real tampering.
                    payload=canonical_json(record.payload).decode("utf-8"),
                    prev_hash=record.prev_hash,
                    hash=record.hash,
                    created_at_ns=record.created_at_ns,
                )
            )

    def _to_record(self, row: Row[Any]) -> AuditRecord:
        data = _row_to_dict(row)
        return AuditRecord(
            seq=int(data["seq"]),
            event_type=str(data["event_type"]),
            actor=str(data["actor"]),
            payload=json.loads(data["payload"]),
            created_at_ns=int(data["created_at_ns"]),
            prev_hash=bytes(data["prev_hash"] or b""),
            hash=bytes(data["hash"]),
        )

    def read_all(self) -> list[AuditRecord]:
        with self._engine.connect() as conn:
            rows = conn.execute(select(audit_log).order_by(audit_log.c.seq))
            return [self._to_record(row) for row in rows]

    def last(self) -> AuditRecord | None:
        with self._engine.connect() as conn:
            row = conn.execute(select(audit_log).order_by(audit_log.c.seq.desc()).limit(1)).first()
        return None if row is None else self._to_record(row)


class SqlStorage:
    """Bundle of SQL repositories sharing one engine."""

    __slots__ = ("_audit", "_engine", "_fills", "_intents", "_orders", "_positions")

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._orders = SqlOrderStore(engine)
        self._fills = SqlFillStore(engine)
        self._intents = SqlIntentStore(engine)
        self._positions = SqlPositionStore(engine)
        self._audit = SqlAuditSink(engine)

    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def orders(self) -> SqlOrderStore:
        return self._orders

    @property
    def fills(self) -> SqlFillStore:
        return self._fills

    @property
    def intents(self) -> SqlIntentStore:
        return self._intents

    @property
    def positions(self) -> SqlPositionStore:
        return self._positions

    @property
    def audit(self) -> SqlAuditSink:
        return self._audit

    def close(self) -> None:
        self._engine.dispose()


def create_storage(dsn: str, *, echo: bool = False, create_tables: bool = True) -> SqlStorage:
    """Open a connection and, by default, ensure the schema exists.

    Production uses Alembic migrations instead of ``create_tables`` — spec
    §10.2 requires migrations to be backward-compatible so a rollback does not
    strand the previous image.
    """
    engine = create_engine(dsn, echo=echo, future=True)
    if engine.dialect.name == "sqlite":
        # Without this, a concurrent reader and writer on the same file will
        # deadlock the reconciliation loop against the order writer.
        with engine.begin() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            conn.exec_driver_sql("PRAGMA foreign_keys=ON")
    if create_tables:
        METADATA.create_all(engine)
    return SqlStorage(engine)
