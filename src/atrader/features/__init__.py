"""Indicator computation and the point-in-time feature store.

Spec §FR-SIG-02 is the load-bearing part: querying at time T must return only
what was actually knowable at T. Without that, look-ahead bias makes a backtest
look far better than the strategy really is.
"""

from __future__ import annotations
