from __future__ import annotations

import pytest

from evidencemm.dense_retrieval import (
    DensePageHit,
)
from evidencemm.hybrid_candidate_union import (
    branch_overlap_count,
    build_candidate_union,
    candidate_pool_contains_gold,
)
from evidencemm.text_retrieval import (
    RetrievalHit,
)


def bm25(
    rank: int,
    page: int,
    score: float,
) -> RetrievalHit:
    return RetrievalHit(
        rank=rank,
        score=score,
        source_id="manual",
        page_number=page,
        text_preview="x",
    )


def dense(
    rank: int,
    page: int,
    score: float,
) -> DensePageHit:
    return DensePageHit(
        rank=rank,
        score=score,
        source_id="manual",
        page_number=page,
    )


def test_union_deduplicates_pages():
    pool = build_candidate_union(
        bm25_hits=[
            bm25(1, 3, 3.0),
            bm25(2, 4, 2.0),
        ],
        dense_hits=[
            dense(1, 3, 0.9),
            dense(2, 5, 0.8),
        ],
    )
    assert [
        row.page_number
        for row in pool
    ] == [3, 4, 5]


def test_union_preserves_both_branch_traces():
    pool = build_candidate_union(
        bm25_hits=[bm25(1, 3, 3.0)],
        dense_hits=[dense(1, 3, 0.9)],
    )
    row = pool[0]
    assert row.from_bm25 is True
    assert row.bm25_rank == 1
    assert row.bm25_score == 3.0
    assert row.from_dense is True
    assert row.dense_rank == 1
    assert row.dense_score == 0.9


def test_union_supports_bm25_only_candidate():
    pool = build_candidate_union(
        bm25_hits=[bm25(1, 4, 2.0)],
        dense_hits=[dense(1, 5, 0.8)],
    )
    row = next(
        item
        for item in pool
        if item.page_number == 4
    )
    assert row.from_bm25 is True
    assert row.from_dense is False


def test_union_supports_dense_only_candidate():
    pool = build_candidate_union(
        bm25_hits=[bm25(1, 4, 2.0)],
        dense_hits=[dense(1, 5, 0.8)],
    )
    row = next(
        item
        for item in pool
        if item.page_number == 5
    )
    assert row.from_bm25 is False
    assert row.from_dense is True


def test_union_rejects_non_contiguous_ranks():
    with pytest.raises(
        ValueError,
        match="contiguous",
    ):
        build_candidate_union(
            bm25_hits=[
                bm25(2, 3, 3.0)
            ],
            dense_hits=[],
        )


def test_candidate_pool_contains_gold():
    pool = build_candidate_union(
        bm25_hits=[bm25(1, 3, 3.0)],
        dense_hits=[dense(1, 5, 0.8)],
    )
    assert candidate_pool_contains_gold(
        pool=pool,
        gold_pages={
            ("manual", 5),
        },
    )


def test_branch_overlap_count():
    pool = build_candidate_union(
        bm25_hits=[
            bm25(1, 3, 3.0),
            bm25(2, 4, 2.0),
        ],
        dense_hits=[
            dense(1, 3, 0.9),
            dense(2, 5, 0.8),
        ],
    )
    assert branch_overlap_count(pool) == 1
