from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchHit:
    item_id: int
    score: float
    source: str
    rank: int
