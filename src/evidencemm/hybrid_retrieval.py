from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RankedPage:
    source_id: str
    page_number: int
    rank: int
    raw_score: float | None = None


@dataclass(frozen=True)
class HybridHit:
    rank: int
    source_id: str
    page_number: int
    rrf_score: float
    text_rank: int | None
    vision_rank: int | None
    text_contribution: float
    vision_contribution: float

    def to_dict(self) -> dict:
        return asdict(self)


def rrf_contribution(
    *,
    rank: int | None,
    rrf_k: int,
    weight: float,
) -> float:
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")
    if weight < 0:
        raise ValueError("weight must be >= 0")
    if rank is None:
        return 0.0
    if rank < 1:
        raise ValueError("rank must be >= 1")
    return weight / (rrf_k + rank)


def fuse_rrf(
    *,
    text_hits: list[RankedPage],
    vision_hits: list[RankedPage],
    top_k: int,
    rrf_k: int = 60,
    text_weight: float = 1.0,
    vision_weight: float = 1.0,
) -> list[HybridHit]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    text_by_key = {
        (hit.source_id, hit.page_number): hit
        for hit in text_hits
    }
    vision_by_key = {
        (hit.source_id, hit.page_number): hit
        for hit in vision_hits
    }

    keys = set(text_by_key) | set(vision_by_key)
    rows = []

    for key in keys:
        text = text_by_key.get(key)
        vision = vision_by_key.get(key)
        text_rank = text.rank if text else None
        vision_rank = vision.rank if vision else None

        text_part = rrf_contribution(
            rank=text_rank,
            rrf_k=rrf_k,
            weight=text_weight,
        )
        vision_part = rrf_contribution(
            rank=vision_rank,
            rrf_k=rrf_k,
            weight=vision_weight,
        )

        rows.append(
            {
                "source_id": key[0],
                "page_number": key[1],
                "rrf_score": text_part + vision_part,
                "text_rank": text_rank,
                "vision_rank": vision_rank,
                "text_contribution": text_part,
                "vision_contribution": vision_part,
            }
        )

    rows.sort(
        key=lambda row: (
            -row["rrf_score"],
            row["text_rank"]
            if row["text_rank"] is not None
            else 10**9,
            row["vision_rank"]
            if row["vision_rank"] is not None
            else 10**9,
            row["source_id"],
            row["page_number"],
        )
    )

    return [
        HybridHit(
            rank=rank,
            source_id=row["source_id"],
            page_number=row["page_number"],
            rrf_score=row["rrf_score"],
            text_rank=row["text_rank"],
            vision_rank=row["vision_rank"],
            text_contribution=row["text_contribution"],
            vision_contribution=row["vision_contribution"],
        )
        for rank, row in enumerate(rows[:top_k], start=1)
    ]
