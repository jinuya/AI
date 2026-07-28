"""Market data ingestion, normalisation and quality control.

Spec §3.1. Order of operations is fixed: ingest -> validate -> normalise ->
store -> publish. Data that has not been validated must never reach a strategy.
"""

from __future__ import annotations
