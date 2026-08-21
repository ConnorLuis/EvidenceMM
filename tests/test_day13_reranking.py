from __future__ import annotations

import pytest

from evidencemm.hybrid_candidate_union import (
    HybridCandidate,
)
from evidencemm.reranking import (
    rank_reranker_scores,
)


def candidate(
    pool_index: int,
    page: int,
) -> HybridCandidate:
    return HybridCandidate(
        pool_index=pool_index,
        source_id="manual",
        page_number=page,
        from_bm25=True,
        bm25_rank=pool_index,
        bm25_score=float(
            10 - pool_index
        ),
        from_dense=True,
        dense_rank=pool_index,
        dense_score=(
            1.0
            - pool_index / 10.0
        ),
    )


def test_reranker_orders_descending_score():
    hits = rank_reranker_scores(
        pool=[
            candidate(1, 3),
            candidate(2, 4),
            candidate(3, 5),
        ],
        scores=[
            0.1,
            0.9,
            0.5,
        ],
        top_k=3,
    )
    assert [
        hit.page_number
        for hit in hits
    ] == [4, 5, 3]


def test_reranker_tie_breaks_by_page():
    hits = rank_reranker_scores(
        pool=[
            candidate(1, 5),
            candidate(2, 3),
        ],
        scores=[
            0.8,
            0.8,
        ],
        top_k=2,
    )
    assert [
        hit.page_number
        for hit in hits
    ] == [3, 5]


def test_reranker_preserves_branch_trace():
    hits = rank_reranker_scores(
        pool=[
            candidate(1, 3)
        ],
        scores=[0.7],
        top_k=1,
    )
    hit = hits[0]
    assert hit.bm25_rank == 1
    assert hit.dense_rank == 1
    assert hit.candidate_pool_index == 1


def test_reranker_rejects_score_count_mismatch():
    with pytest.raises(
        ValueError,
        match="counts differ",
    ):
        rank_reranker_scores(
            pool=[
                candidate(1, 3)
            ],
            scores=[],
            top_k=1,
        )


def test_reranker_respects_top_k():
    hits = rank_reranker_scores(
        pool=[
            candidate(1, 3),
            candidate(2, 4),
            candidate(3, 5),
        ],
        scores=[
            0.3,
            0.2,
            0.1,
        ],
        top_k=2,
    )
    assert len(hits) == 2
