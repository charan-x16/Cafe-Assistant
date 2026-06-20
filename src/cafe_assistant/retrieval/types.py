"""Implementation module for types.
Contains typed helpers used by the cafe assistant backend runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Container for search hit behavior and data."""
    item_id: int
    score: float
    source: str
    rank: int
    kind: str = "legacy"
