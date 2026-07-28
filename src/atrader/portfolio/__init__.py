"""Positions, P&L, netting, rebalancing and margin.

Spec §3.5. P&L is net of commission, tax, borrow cost and FX — a strategy that
looks profitable gross and loses money net is a common outcome, not an exotic one.
"""

from __future__ import annotations
