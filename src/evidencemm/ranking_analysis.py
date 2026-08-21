from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math

from evidencemm.robot_candidate_retrieval import (
    RobotSignalRetrievalHit,
)
from evidencemm.text_retrieval import (
    PageDocument,
    tokenize_mixed,
)


@dataclass(frozen=True)
class BM25TermContribution:
    term: str
    query_frequency: int
    document_frequency: int
    document_term_frequency: int
    idf: float
    contribution: float

    def to_dict(self) -> dict:
        return asdict(self)


def bm25_term_contributions(
    *,
    documents: list[PageDocument],
    query: str,
    source_id: str,
    page_number: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[
    float,
    list[BM25TermContribution],
]:
    if not documents:
        raise ValueError(
            "documents must not be empty"
        )

    target_index = next(
        (
            index
            for index, document
            in enumerate(documents)
            if (
                document.source_id
                == source_id
                and document.page_number
                == page_number
            )
        ),
        None,
    )
    if target_index is None:
        raise ValueError(
            "target page not found in corpus"
        )

    tokenized = [
        tokenize_mixed(document.text)
        for document in documents
    ]
    document_lengths = [
        len(tokens)
        for tokens in tokenized
    ]
    avgdl = (
        sum(document_lengths)
        / len(document_lengths)
    )

    document_frequency: Counter[str] = (
        Counter()
    )
    for tokens in tokenized:
        document_frequency.update(
            set(tokens)
        )

    query_counts = Counter(
        tokenize_mixed(query)
    )
    target_counts = Counter(
        tokenized[target_index]
    )
    target_length = (
        document_lengths[target_index]
    )
    n = len(documents)

    rows: list[BM25TermContribution] = []
    total = 0.0

    for term, query_frequency in (
        query_counts.items()
    ):
        tf = target_counts.get(
            term,
            0,
        )
        if tf == 0:
            continue

        df = document_frequency.get(
            term,
            0,
        )
        idf = math.log(
            1.0
            + (n - df + 0.5)
            / (df + 0.5)
        )

        numerator = (
            tf
            * (k1 + 1.0)
        )
        denominator = (
            tf
            + k1
            * (
                1.0
                - b
                + b
                * target_length
                / avgdl
            )
        )
        one_occurrence = (
            idf
            * numerator
            / denominator
        )
        contribution = (
            one_occurrence
            * query_frequency
        )
        total += contribution

        rows.append(
            BM25TermContribution(
                term=term,
                query_frequency=query_frequency,
                document_frequency=df,
                document_term_frequency=tf,
                idf=idf,
                contribution=contribution,
            )
        )

    rows.sort(
        key=lambda row: (
            -row.contribution,
            row.term,
        )
    )
    return total, rows


def robot_top_tie_summary(
    *,
    hits: list[RobotSignalRetrievalHit],
    selected_k: int,
    tolerance: float = 1e-12,
) -> dict:
    if not hits:
        raise ValueError(
            "robot hits must not be empty"
        )
    if selected_k < 1:
        raise ValueError(
            "selected_k must be >= 1"
        )
    if selected_k > len(hits):
        raise ValueError(
            "selected_k exceeds available hits"
        )

    top_score = hits[0].raw_score
    tied = [
        hit
        for hit in hits
        if abs(
            hit.raw_score
            - top_score
        )
        <= tolerance
    ]

    selected = hits[:selected_k]

    return {
        "top_score": top_score,
        "top_score_tie_count": len(tied),
        "top_score_tied_frames": [
            hit.frame_index
            for hit in tied
        ],
        "selected_k": selected_k,
        "selected_frames": [
            hit.frame_index
            for hit in selected
        ],
        "selection_rule": (
            "raw_score descending; "
            "frame_index ascending on ties"
        ),
        "selected_hits": [
            hit.to_dict()
            for hit in selected
        ],
    }
