from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

from evidencemm.dense_retrieval import DensePageHit
from evidencemm.text_retrieval import RetrievalHit


@dataclass(frozen=True)
class HybridCandidate:
    pool_index: int
    source_id: str
    page_number: int
    from_bm25: bool
    bm25_rank: int | None
    bm25_score: float | None
    from_dense: bool
    dense_rank: int | None
    dense_score: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_ranked_hits(
    *,
    hits,
    name: str,
) -> None:
    seen: set[tuple[str, int]] = set()
    for expected_rank, hit in enumerate(
        hits,
        start=1,
    ):
        if hit.rank != expected_rank:
            raise ValueError(
                f"{name} ranks must be contiguous from 1"
            )
        key = (
            hit.source_id,
            hit.page_number,
        )
        if key in seen:
            raise ValueError(
                f"{name} contains duplicate page: {key}"
            )
        seen.add(key)

        score = float(
            hit.score
        )
        if not isfinite(score):
            raise ValueError(
                f"{name} score must be finite"
            )


def build_candidate_union(
    *,
    bm25_hits: list[RetrievalHit],
    dense_hits: list[DensePageHit],
) -> list[HybridCandidate]:
    if not bm25_hits and not dense_hits:
        raise ValueError(
            "candidate union requires at least one branch hit"
        )

    _validate_ranked_hits(
        hits=bm25_hits,
        name="bm25",
    )
    _validate_ranked_hits(
        hits=dense_hits,
        name="dense",
    )

    bm25_by_key = {
        (
            hit.source_id,
            hit.page_number,
        ): hit
        for hit in bm25_hits
    }
    dense_by_key = {
        (
            hit.source_id,
            hit.page_number,
        ): hit
        for hit in dense_hits
    }

    # This order is deliberately lexical, not a relevance fusion.
    # Relevance ranking is delegated to the cross-encoder reranker.
    keys = sorted(
        set(bm25_by_key)
        | set(dense_by_key)
    )

    pool: list[HybridCandidate] = []
    for pool_index, key in enumerate(
        keys,
        start=1,
    ):
        bm25 = bm25_by_key.get(key)
        dense = dense_by_key.get(key)

        pool.append(
            HybridCandidate(
                pool_index=pool_index,
                source_id=key[0],
                page_number=key[1],
                from_bm25=bm25 is not None,
                bm25_rank=(
                    bm25.rank
                    if bm25 is not None
                    else None
                ),
                bm25_score=(
                    float(bm25.score)
                    if bm25 is not None
                    else None
                ),
                from_dense=dense is not None,
                dense_rank=(
                    dense.rank
                    if dense is not None
                    else None
                ),
                dense_score=(
                    float(dense.score)
                    if dense is not None
                    else None
                ),
            )
        )

    return pool


def candidate_pool_contains_gold(
    *,
    pool: list[HybridCandidate],
    gold_pages: set[tuple[str, int]],
) -> bool:
    if not gold_pages:
        raise ValueError(
            "gold_pages must not be empty"
        )
    return any(
        (
            candidate.source_id,
            candidate.page_number,
        )
        in gold_pages
        for candidate in pool
    )


def branch_overlap_count(
    pool: list[HybridCandidate],
) -> int:
    return sum(
        candidate.from_bm25
        and candidate.from_dense
        for candidate in pool
    )
