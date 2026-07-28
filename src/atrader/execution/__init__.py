"""Order management, execution algorithms and reconciliation.

Spec §3.4. Everything here assumes the network will fail at the worst moment:
orders carry client-generated IDs, an unanswered request is resolved by querying
rather than resending, and local state is continuously reconciled against the
broker.
"""

from __future__ import annotations
