from __future__ import annotations

from evidencemm.hybrid_retrieval import (
    RankedPage,
    fuse_rrf,
    rrf_contribution,
)


def page(page_number: int, rank: int) -> RankedPage:
    return RankedPage(
        source_id="manual",
        page_number=page_number,
        rank=rank,
    )


def test_rrf_contribution():
    assert rrf_contribution(
        rank=1,
        rrf_k=60,
        weight=1.0,
    ) == 1.0 / 61.0


def test_rrf_rewards_agreement_between_modalities():
    fused = fuse_rrf(
        text_hits=[
            page(3, 1),
            page(4, 2),
        ],
        vision_hits=[
            page(4, 1),
            page(3, 5),
        ],
        top_k=2,
        rrf_k=60,
    )

    assert fused[0].page_number == 4


def test_rrf_handles_page_missing_from_one_modality():
    fused = fuse_rrf(
        text_hits=[page(3, 1)],
        vision_hits=[page(4, 1)],
        top_k=2,
        rrf_k=60,
    )

    assert {hit.page_number for hit in fused} == {3, 4}


def test_rrf_is_deterministic_for_equal_scores():
    fused = fuse_rrf(
        text_hits=[
            page(3, 1),
            page(4, 2),
        ],
        vision_hits=[
            page(4, 1),
            page(3, 2),
        ],
        top_k=2,
        rrf_k=60,
    )

    assert [hit.page_number for hit in fused] == [3, 4]
