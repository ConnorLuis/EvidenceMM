from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import torch
import torch.nn.functional as F

from evidencemm.text_retrieval import PageDocument


@dataclass(frozen=True)
class DensePageHit:
    rank: int
    score: float
    source_id: str
    page_number: int

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_dense_embeddings(
    embeddings: torch.Tensor,
) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings must be two-dimensional"
        )
    if embeddings.shape[0] < 1:
        raise ValueError(
            "embeddings must contain at least one row"
        )
    if not torch.isfinite(embeddings).all():
        raise ValueError(
            "embeddings must contain only finite values"
        )

    norms = torch.linalg.vector_norm(
        embeddings.float(),
        dim=1,
    )
    if torch.any(norms <= 0):
        raise ValueError(
            "dense embeddings must have non-zero norm"
        )

    return F.normalize(
        embeddings.float(),
        p=2,
        dim=1,
    )


def load_dense_encoder(
    *,
    model_name: str,
    device: str = "cuda:0",
):
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Day 13 dense retrieval baseline requires CUDA"
        )

    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )
    model = (
        AutoModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )

    load_sec = time.perf_counter() - started
    return model, tokenizer, load_sec


def encode_dense_texts(
    *,
    model,
    tokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int,
) -> tuple[torch.Tensor, float]:
    if not texts:
        raise ValueError(
            "texts must not be empty"
        )
    if batch_size < 1:
        raise ValueError(
            "batch_size must be >= 1"
        )
    if max_length < 1:
        raise ValueError(
            "max_length must be >= 1"
        )

    chunks = []
    started = time.perf_counter()

    for offset in range(
        0,
        len(texts),
        batch_size,
    ):
        batch_texts = texts[
            offset : offset + batch_size
        ]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            outputs = model(**encoded)

        # BGE-M3's sentence-transformers pooling config uses CLS pooling.
        pooled = outputs.last_hidden_state[:, 0]
        chunks.append(
            pooled.detach().float().cpu()
        )

    if model.device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - started
    embeddings = normalize_dense_embeddings(
        torch.cat(chunks, dim=0)
    )
    return embeddings, elapsed


def score_dense_documents(
    *,
    query_embedding: torch.Tensor,
    document_embeddings: torch.Tensor,
) -> torch.Tensor:
    query = normalize_dense_embeddings(
        query_embedding
    )
    documents = normalize_dense_embeddings(
        document_embeddings
    )

    if query.shape[0] != 1:
        raise ValueError(
            "query_embedding must contain exactly one row"
        )
    if query.shape[1] != documents.shape[1]:
        raise ValueError(
            "query/document embedding dimensions differ"
        )

    return (
        query
        @ documents.transpose(0, 1)
    )[0]


def rank_dense_scores(
    *,
    scores: torch.Tensor,
    documents: list[PageDocument],
    top_k: int,
) -> list[DensePageHit]:
    if scores.ndim != 1:
        raise ValueError(
            "scores must be one-dimensional"
        )
    if len(documents) != scores.numel():
        raise ValueError(
            "document count must match score count"
        )
    if top_k < 1:
        raise ValueError(
            "top_k must be >= 1"
        )

    order = sorted(
        range(scores.numel()),
        key=lambda index: (
            -float(scores[index]),
            documents[index].source_id,
            documents[index].page_number,
        ),
    )[: min(top_k, scores.numel())]

    return [
        DensePageHit(
            rank=rank,
            score=float(scores[index]),
            source_id=documents[index].source_id,
            page_number=documents[index].page_number,
        )
        for rank, index in enumerate(
            order,
            start=1,
        )
    ]


def first_relevant_rank(
    *,
    ranked_pages: list[tuple[str, int]],
    gold_pages: set[tuple[str, int]],
) -> int | None:
    return next(
        (
            rank
            for rank, page in enumerate(
                ranked_pages,
                start=1,
            )
            if page in gold_pages
        ),
        None,
    )
