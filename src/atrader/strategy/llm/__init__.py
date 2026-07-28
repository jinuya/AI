"""LLM agent strategy and its safety layer (spec §7.7).

The model produces a schema-constrained intent and nothing else. Symbol
whitelisting, numeric sanity checks, confidence scaling and prompt-injection
isolation all sit here — but the real defence is the deterministic risk engine
downstream, not any of this.
"""

from __future__ import annotations
