"""SQLAlchemy Core schema — spec §4.5.

Money is ``NUMERIC(20,8)`` on PostgreSQL, exactly as the spec requires. On
SQLite there is no native decimal type and SQLAlchemy's ``Numeric`` round-trips
through ``float`` there, which would reintroduce precisely the drift §4.4 warns
about — so a ``TypeDecorator`` stores the digits as text on that dialect
instead. The one thing this gives up is arithmetic and range comparison on money
columns in SQL; that is fine here, because money is aggregated in Python where
it stays :class:`~decimal.Decimal` end to end.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Index,
    LargeBinary,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    TypeDecorator,
)
from sqlalchemy.engine import Dialect

from atrader.core.money import quantize

__all__ = ["METADATA", "MONEY", "audit_log", "fills", "intents", "orders", "positions"]


class DecimalAsText(TypeDecorator[Decimal]):
    """Store a ``Decimal`` as its exact textual form.

    Used only on SQLite. ``str(Decimal)`` round-trips without loss, which is the
    property that matters; the trade-off is that ordering is lexicographic, so
    nothing in this codebase sorts or range-filters on a money column in SQL.
    """

    impl = String(48)
    cache_ok = True

    def process_bind_param(self, value: Decimal | None, dialect: Dialect) -> str | None:
        return None if value is None else str(quantize(value))

    def process_result_value(self, value: str | None, dialect: Dialect) -> Decimal | None:
        return None if value is None else Decimal(value)


#: ``NUMERIC(20,8)`` on real databases, exact text on SQLite.
MONEY: Any = Numeric(20, 8, asdecimal=True).with_variant(DecimalAsText(), "sqlite")

METADATA = MetaData()

intents = Table(
    "intents",
    METADATA,
    Column("intent_id", String(36), primary_key=True),
    Column("strategy_id", String(64), nullable=False),
    Column("symbol", String(32), nullable=False),
    Column("side", String(8), nullable=False),
    Column("target_type", String(16), nullable=False),
    Column("target_value", MONEY, nullable=False),
    Column("urgency", String(16), nullable=False),
    Column("limit_price", MONEY),
    Column("time_in_force", String(8), nullable=False),
    Column("stop_loss", MONEY),
    Column("take_profit", MONEY),
    Column("valid_until_ns", BigInteger),
    Column("confidence", MONEY, nullable=False),
    Column("rationale", Text, nullable=False, default=""),
    Column("created_at_ns", BigInteger, nullable=False),
)

orders = Table(
    "orders",
    METADATA,
    Column("order_id", String(36), primary_key=True),
    # The idempotency key (spec §FR-EXE-04). UNIQUE is load-bearing: it is what
    # makes a retry-after-timeout impossible to turn into a duplicate order.
    Column("client_order_id", String(64), nullable=False, unique=True),
    Column("broker_order_id", String(64)),
    Column("parent_intent_id", String(36)),
    Column("strategy_id", String(64), nullable=False),
    Column("symbol", String(32), nullable=False),
    Column("side", String(8), nullable=False),
    Column("order_type", String(16), nullable=False),
    Column("quantity", MONEY, nullable=False),
    Column("limit_price", MONEY),
    Column("stop_price", MONEY),
    Column("time_in_force", String(8), nullable=False),
    Column("status", String(24), nullable=False),
    Column("filled_quantity", MONEY, nullable=False, default=Decimal(0)),
    Column("avg_fill_price", MONEY),
    # Which risk evaluation admitted this order — the audit link back to §7.2.
    Column("risk_check_id", String(36)),
    Column("reduce_only", Boolean, nullable=False, default=False),
    Column("created_at_ns", BigInteger, nullable=False),
    Column("updated_at_ns", BigInteger, nullable=False),
    CheckConstraint("quantity > 0", name="qty_positive"),
)

# Spec §4.5 indexes the open statuses; reconciliation reads this set every 30s.
Index(
    "idx_orders_status",
    orders.c.status,
    sqlite_where=orders.c.status.in_(("PENDING_NEW", "NEW", "PARTIALLY_FILLED")),
    postgresql_where=orders.c.status.in_(("PENDING_NEW", "NEW", "PARTIALLY_FILLED")),
)
Index("idx_orders_symbol", orders.c.symbol)

fills = Table(
    "fills",
    METADATA,
    Column("fill_id", String(36), primary_key=True),
    Column("order_id", String(36), nullable=False),
    # UNIQUE so a redelivered fill event cannot be applied twice (spec §4.3).
    Column("broker_fill_id", String(64), unique=True),
    Column("symbol", String(32), nullable=False),
    Column("side", String(8), nullable=False),
    Column("quantity", MONEY, nullable=False),
    Column("price", MONEY, nullable=False),
    Column("commission", MONEY, nullable=False, default=Decimal(0)),
    Column("tax", MONEY, nullable=False, default=Decimal(0)),
    Column("executed_at_ns", BigInteger, nullable=False),
)
Index("idx_fills_order", fills.c.order_id)

positions = Table(
    "positions",
    METADATA,
    Column("symbol", String(32), primary_key=True),
    Column("quantity", MONEY, nullable=False),
    Column("avg_price", MONEY, nullable=False),
    Column("realized_pnl", MONEY, nullable=False),
    Column("unrealized_pnl", MONEY, nullable=False),
    Column("last_price", MONEY),
    Column("opened_at_ns", BigInteger),
    Column("updated_at_ns", BigInteger, nullable=False),
)

audit_log = Table(
    "audit_log",
    METADATA,
    Column("seq", BigInteger, primary_key=True, autoincrement=False),
    Column("event_type", String(64), nullable=False),
    Column("actor", String(64), nullable=False),
    Column("payload", Text, nullable=False),
    Column("prev_hash", LargeBinary),
    Column("hash", LargeBinary, nullable=False),
    Column("created_at_ns", BigInteger, nullable=False),
)
Index("idx_audit_event_type", audit_log.c.event_type)
Index("idx_audit_created_at", audit_log.c.created_at_ns)
