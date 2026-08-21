from __future__ import annotations

import pytest

from evidencemm.canonical_document_retriever import (
    CANONICAL_HYBRID_RETRIEVER_NAME,
    ranked_candidates_from_reranker,
)
from evidencemm.reranking import RerankedPage
from evidencemm.schemas import (
    EvidenceRef,
    SourceType,
)
from evidencemm.unified_evidence import (
    DocumentPagePayload,
    EvidenceProvenance,
    UnifiedEvidenceItem,
    UnifiedEvidenceKind,
)


SHA = "a" * 64


def item(page: int) -> UnifiedEvidenceItem:
    return UnifiedEvidenceItem(
        evidence_id=f"doc:manual:p{page}",
        kind=(
            UnifiedEvidenceKind.DOCUMENT_PAGE
        ),
        refs=[
            EvidenceRef(
                source_id="manual",
                source_type=SourceType.PDF,
                page_number=page,
            )
        ],
        provenance=EvidenceProvenance(
            source_id="manual",
            source_type=SourceType.PDF,
            manifest_path="manual.json",
            canonical_sha256=SHA,
        ),
        payload=DocumentPagePayload(
            page_number=page,
            text_sha256=SHA,
            char_count=4,
            text_excerpt="text",
        ),
    )


def hit(
    *,
    rank: int,
    page: int,
    score: float,
) -> RerankedPage:
    return RerankedPage(
        rank=rank,
        source_id="manual",
        page_number=page,
        reranker_score=score,
        candidate_pool_index=rank,
        from_bm25=True,
        bm25_rank=rank,
        bm25_score=1.0,
        from_dense=True,
        dense_rank=rank,
        dense_score=0.5,
    )


def test_ranked_candidates_preserve_reranker_order():
    candidates = ranked_candidates_from_reranker(
        reranked_hits=[
            hit(rank=1, page=3, score=5.0),
            hit(rank=2, page=4, score=4.0),
        ],
        item_by_page={
            3: item(3),
            4: item(4),
        },
    )

    assert [
        candidate.item.payload.page_number
        for candidate in candidates
    ] == [3, 4]


def test_ranked_candidates_use_canonical_name():
    candidates = ranked_candidates_from_reranker(
        reranked_hits=[
            hit(rank=1, page=3, score=5.0)
        ],
        item_by_page={3: item(3)},
    )

    assert (
        candidates[0].retriever_name
        == CANONICAL_HYBRID_RETRIEVER_NAME
    )


def test_ranked_candidates_preserve_raw_reranker_score():
    candidates = ranked_candidates_from_reranker(
        reranked_hits=[
            hit(rank=1, page=3, score=5.25)
        ],
        item_by_page={3: item(3)},
    )

    assert candidates[0].raw_score == 5.25


def test_ranked_candidates_reject_missing_page_item():
    with pytest.raises(
        ValueError,
        match="missing canonical evidence item",
    ):
        ranked_candidates_from_reranker(
            reranked_hits=[
                hit(
                    rank=1,
                    page=3,
                    score=5.0,
                )
            ],
            item_by_page={},
        )


def test_ranked_candidates_reject_non_contiguous_rank():
    with pytest.raises(
        ValueError,
        match="contiguous",
    ):
        ranked_candidates_from_reranker(
            reranked_hits=[
                hit(
                    rank=2,
                    page=3,
                    score=5.0,
                )
            ],
            item_by_page={
                3: item(3)
            },
        )
