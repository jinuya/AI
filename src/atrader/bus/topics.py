"""Topic names — spec §4.3.

A fixed vocabulary rather than ad-hoc strings: a publisher and a subscriber that
disagree about a topic name fail silently, and "the strategy stopped receiving
bars" is an unpleasant thing to debug at 09:31.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "FILL_NEW",
    "INTENT_NEW",
    "MD_BAR",
    "MD_TICK",
    "ORDER_REQUEST",
    "ORDER_UPDATE",
    "RISK_ALERT",
    "SIGNAL",
    "SYSTEM_HEALTH",
    "bar_topic",
    "signal_topic",
    "tick_topic",
]

MD_TICK: Final = "md.tick"
MD_BAR: Final = "md.bar"
SIGNAL: Final = "signal"
INTENT_NEW: Final = "intent.new"
ORDER_REQUEST: Final = "order.request"
ORDER_UPDATE: Final = "order.update"
FILL_NEW: Final = "fill.new"
RISK_ALERT: Final = "risk.alert"
SYSTEM_HEALTH: Final = "system.health"


def tick_topic(symbol: str) -> str:
    """``md.tick.{symbol}``"""
    return f"{MD_TICK}.{symbol}"


def bar_topic(interval: str) -> str:
    """``md.bar.{interval}``"""
    return f"{MD_BAR}.{interval}"


def signal_topic(strategy_id: str) -> str:
    """``signal.{strategy_id}``"""
    return f"{SIGNAL}.{strategy_id}"
