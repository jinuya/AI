"""Feed adapters.

Every exchange names its fields differently; the adapter layer converts them to
the internal schema so nothing downstream knows which venue it is reading.
"""

from __future__ import annotations
