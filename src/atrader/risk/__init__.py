"""The risk engine — the mandatory gate on the order path.

Spec §7.1: this is the only component that turns a ``TradingIntent`` into an
``Order``. There is no bypass. If it cannot answer, orders are rejected; if its
configuration cannot be loaded, the system refuses to boot. Fail-closed.
"""

from __future__ import annotations
