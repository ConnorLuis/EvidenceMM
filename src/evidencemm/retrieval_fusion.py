from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores = defaultdict(float)

    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] += 1.0 / (k + rank)

    return sorted(
        scores.items(),
        key=lambda x: (-x[1], x[0]),
    )
