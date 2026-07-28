"""Configuration and secrets.

Spec §4.6: strategy parameters and risk limits live in version-controlled YAML,
not in code. Risk limit changes go through a reviewed PR — a limit you can raise
at runtime is not a limit.
"""

from __future__ import annotations
