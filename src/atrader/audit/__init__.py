"""Tamper-evident audit log.

Spec §4.5: each row carries the hash of the previous row, so the log cannot be
quietly edited after the fact. Matters for regulators, matters more when you are
reconstructing an incident and need to trust what you are reading.
"""

from __future__ import annotations
