"""Broker adapters.

Spec §6.1: every broker hides behind one Protocol so strategy code never knows
which venue it is trading. ``BrokerCapabilities`` lets the adapter reject an
unsupported order immediately instead of paying a round trip to find out.
"""

from __future__ import annotations
