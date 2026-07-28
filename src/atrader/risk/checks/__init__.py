"""Pre-trade checks (spec §7.2).

Fifteen checks, evaluated in order, first failure wins. Each is a pure function
of (intent, snapshot, limits) so it can be tested in isolation and reasoned about
without starting the system.
"""

from __future__ import annotations
