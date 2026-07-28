"""Message bus abstraction.

Spec §4.3: Redis Streams by default, NATS JetStream swappable behind a thin
layer. Delivery is at-least-once, so **every consumer must be idempotent** —
receiving the same fill twice must not double the position.
"""

from __future__ import annotations
