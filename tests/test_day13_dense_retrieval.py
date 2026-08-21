from __future__ import annotations

import pytest
import torch

from evidencemm.dense_retrieval import (
    first_relevant_rank,
    normalize_dense_embeddings,
    rank_dense_scores,
    score_dense_documents,
)
from evidencemm.text_retrieval import PageDocument


def docs() -> list[PageDocument]:
    return [
        PageDocument.from_text(
            source_id="manual",
            page_number=1,
            text="alpha",
        ),
        PageDocument.from_text(
            source_id="manual",
            page_number=2,
            text="beta",
        ),
        PageDocument.from_text(
            source_id="manual",
            page_number=3,
            text="gamma",
        ),
    ]


def test_normalize_dense_embeddings_unit_norm():
    embeddings = torch.tensor(
        [
            [3.0, 4.0],
            [0.0, 2.0],
        ]
    )
    normalized = normalize_dense_embeddings(
        embeddings
    )
    norms = torch.linalg.vector_norm(
        normalized,
        dim=1,
    )
    assert torch.allclose(
        norms,
        torch.ones_like(norms),
    )


def test_normalize_dense_embeddings_rejects_zero_vector():
    with pytest.raises(
        ValueError,
        match="non-zero norm",
    ):
        normalize_dense_embeddings(
            torch.tensor([[0.0, 0.0]])
        )


def test_score_dense_documents_is_cosine_after_normalization():
    scores = score_dense_documents(
        query_embedding=torch.tensor(
            [[1.0, 0.0]]
        ),
        document_embeddings=torch.tensor(
            [
                [2.0, 0.0],
                [0.0, 3.0],
            ]
        ),
    )
    assert torch.allclose(
        scores,
        torch.tensor([1.0, 0.0]),
    )


def test_score_dense_documents_rejects_multi_query():
    with pytest.raises(
        ValueError,
        match="exactly one row",
    ):
        score_dense_documents(
            query_embedding=torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
            document_embeddings=torch.tensor(
                [[1.0, 0.0]]
            ),
        )


def test_rank_dense_scores_orders_descending():
    hits = rank_dense_scores(
        scores=torch.tensor(
            [0.1, 0.9, 0.5]
        ),
        documents=docs(),
        top_k=3,
    )
    assert [
        hit.page_number
        for hit in hits
    ] == [2, 3, 1]
    assert [
        hit.rank
        for hit in hits
    ] == [1, 2, 3]


def test_rank_dense_scores_tie_breaks_by_page():
    hits = rank_dense_scores(
        scores=torch.tensor(
            [0.8, 0.8, 0.1]
        ),
        documents=docs(),
        top_k=2,
    )
    assert [
        hit.page_number
        for hit in hits
    ] == [1, 2]


def test_first_relevant_rank():
    rank = first_relevant_rank(
        ranked_pages=[
            ("manual", 5),
            ("manual", 3),
            ("manual", 4),
        ],
        gold_pages={
            ("manual", 3),
        },
    )
    assert rank == 2
