from __future__ import annotations

import torch

from evidencemm.visual_retrieval import rank_scores


def test_visual_rank_scores_descending():
    result = rank_scores(
        torch.tensor([0.2, 0.9, 0.4]),
        top_k=3,
    )

    assert [index for index, _ in result] == [1, 2, 0]


def test_visual_rank_scores_caps_top_k():
    result = rank_scores(
        torch.tensor([0.2, 0.9]),
        top_k=5,
    )

    assert len(result) == 2
