"""Persistence.

Spec §4.4 splits storage three ways because each is good at something different.
This package covers the transactional store (orders, fills, positions, audit log)
behind a Protocol, with an in-memory implementation for tests and a SQL one that
runs on both SQLite and PostgreSQL.
"""

from __future__ import annotations
