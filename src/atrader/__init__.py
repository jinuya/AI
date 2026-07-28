"""AI trading agent.

Safety-first automated trading system. The architectural invariant that
everything else follows from (spec §7.1):

    The AI model never calls the broker API. Strategies emit schema-constrained
    *intents*; a deterministic risk engine sitting behind them decides whether
    an order is actually sent.

That boundary is enforced statically — see the import-linter contracts in
``pyproject.toml`` and ``tests/unit/test_import_boundaries.py``.
"""

from __future__ import annotations

__version__ = "0.1.0"
