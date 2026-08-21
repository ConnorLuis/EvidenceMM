from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import torch

from evidencemm.hybrid_candidate_union import (
    HybridCandidate,
)


@dataclass(frozen=True)
class RerankedPage:
    rank: int
    source_id: str
    page_number: int
    reranker_score: float
    candidate_pool_index: int
    from_bm25: bool
    bm25_rank: int | None
    bm25_score: float | None
    from_dense: bool
    dense_rank: int | None
    dense_score: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def load_reranker(
    *,
    model_name: str,
    device: str = "cuda:0",
):
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Day 13 reranker baseline requires CUDA"
        )

    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )
    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            model_name,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )

    load_sec = time.perf_counter() - started
    return model, tokenizer, load_sec


def score_query_passages(
    *,
    model,
    tokenizer,
    query: str,
    passages: list[str],
    batch_size: int,
    max_length: int,
) -> tuple[list[float], float]:
    if not query.strip():
        raise ValueError(
            "query must not be blank"
        )
    if not passages:
        raise ValueError(
            "passages must not be empty"
        )
    if batch_size < 1:
        raise ValueError(
            "batch_size must be >= 1"
        )
    if max_length < 1:
        raise ValueError(
            "max_length must be >= 1"
        )

    scores: list[float] = []
    started = time.perf_counter()

    for offset in range(
        0,
        len(passages),
        batch_size,
    ):
        batch_passages = passages[
            offset : offset + batch_size
        ]
        pairs = [
            [query, passage]
            for passage in batch_passages
        ]
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            logits = model(
                **inputs,
                return_dict=True,
            ).logits

        batch_scores = (
            logits
            .view(-1)
            .detach()
            .float()
            .cpu()
            .tolist()
        )
        scores.extend(
            float(score)
            for score in batch_scores
        )

    if model.device.type == "cuda":
        torch.cuda.synchronize()

    return (
        scores,
        time.perf_counter() - started,
    )


def rank_reranker_scores(
    *,
    pool: list[HybridCandidate],
    scores: list[float],
    top_k: int,
) -> list[RerankedPage]:
    if len(pool) != len(scores):
        raise ValueError(
            "pool and reranker score counts differ"
        )
    if top_k < 1:
        raise ValueError(
            "top_k must be >= 1"
        )
    if not pool:
        raise ValueError(
            "pool must not be empty"
        )
    if not all(
        torch.isfinite(
            torch.tensor(score)
        ).item()
        for score in scores
    ):
        raise ValueError(
            "reranker scores must be finite"
        )

    order = sorted(
        range(len(pool)),
        key=lambda index: (
            -scores[index],
            pool[index].source_id,
            pool[index].page_number,
        ),
    )[: min(top_k, len(pool))]

    return [
        RerankedPage(
            rank=rank,
            source_id=pool[index].source_id,
            page_number=pool[index].page_number,
            reranker_score=float(
                scores[index]
            ),
            candidate_pool_index=(
                pool[index].pool_index
            ),
            from_bm25=pool[index].from_bm25,
            bm25_rank=pool[index].bm25_rank,
            bm25_score=pool[index].bm25_score,
            from_dense=pool[index].from_dense,
            dense_rank=pool[index].dense_rank,
            dense_score=pool[index].dense_score,
        )
        for rank, index in enumerate(
            order,
            start=1,
        )
    ]
