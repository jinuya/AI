"""Trading strategies.

Strategies emit :class:`~atrader.strategy.intent.TradingIntent` — never orders.
They cannot import the broker adapter, the execution layer, or the risk engine;
that is enforced statically (spec §7.1, acceptance criterion #1).
"""

from __future__ import annotations
