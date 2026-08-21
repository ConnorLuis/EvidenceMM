from __future__ import annotations

from evidencemm.ranking_analysis import (
    bm25_term_contributions,
    robot_top_tie_summary,
)
from evidencemm.robot_candidate_retrieval import (
    RobotSignalRetrievalHit,
)
from evidencemm.text_retrieval import (
    BM25Index,
    PageDocument,
)


def docs() -> list[PageDocument]:
    return [
        PageDocument.from_text(
            source_id="manual",
            page_number=1,
            text="alpha beta",
        ),
        PageDocument.from_text(
            source_id="manual",
            page_number=2,
            text="alpha gamma gamma",
        ),
    ]


def robot_hit(
    rank: int,
    frame: int,
    score: float,
) -> RobotSignalRetrievalHit:
    return RobotSignalRetrievalHit(
        rank=rank,
        frame_index=frame,
        timestamp_sec=frame / 10.0,
        raw_score=score,
        signal_scores={
            "action": score
        },
    )


def test_bm25_contributions_reproduce_score():
    documents = docs()
    index = BM25Index(documents)
    expected = index.score(
        "gamma"
    )[1]

    total, _ = (
        bm25_term_contributions(
            documents=documents,
            query="gamma",
            source_id="manual",
            page_number=2,
        )
    )
    assert abs(
        total - expected
    ) < 1e-12


def test_bm25_contributions_are_sorted():
    _, rows = (
        bm25_term_contributions(
            documents=docs(),
            query="alpha gamma",
            source_id="manual",
            page_number=2,
        )
    )
    assert rows
    assert all(
        rows[index].contribution
        >= rows[index + 1].contribution
        for index in range(
            len(rows) - 1
        )
    )


def test_robot_top_tie_summary_counts_exact_ties():
    summary = robot_top_tie_summary(
        hits=[
            robot_hit(1, 155, 1.5),
            robot_hit(2, 156, 1.5),
            robot_hit(3, 381, 1.5),
            robot_hit(4, 500, 1.0),
        ],
        selected_k=2,
    )
    assert (
        summary[
            "exact_top_score_tie_count"
        ]
        == 3
    )


def test_robot_top_tie_summary_separates_near_equal_scores():
    summary = robot_top_tie_summary(
        hits=[
            robot_hit(1, 155, 1.5),
            robot_hit(
                2,
                156,
                1.5 - 5e-13,
            ),
            robot_hit(3, 381, 1.0),
        ],
        selected_k=2,
        near_tolerance=1e-12,
    )
    assert (
        summary[
            "exact_top_score_tie_count"
        ]
        == 1
    )
    assert (
        summary[
            "near_top_score_count"
        ]
        == 2
    )


def test_robot_top_tie_summary_selects_lower_ranked_frames():
    summary = robot_top_tie_summary(
        hits=[
            robot_hit(1, 155, 1.5),
            robot_hit(2, 156, 1.5),
            robot_hit(3, 381, 1.5),
        ],
        selected_k=2,
    )
    assert summary[
        "selected_frames"
    ] == [155, 156]
