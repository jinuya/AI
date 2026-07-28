"""Backtesting, cost modelling and deterministic replay.

Spec §8.1: strategy code is *literally the same* in backtest and live trading.
Only the data source and the execution layer differ. If that stops being true,
the backtest stops being evidence of anything.
"""

from __future__ import annotations
