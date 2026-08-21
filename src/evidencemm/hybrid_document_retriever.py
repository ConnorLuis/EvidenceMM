from __future__ import annotations

from evidencemm.evidence_retriever import EvidenceCandidate
from evidencemm.retrieval_fusion import reciprocal_rank_fusion


class HybridDocumentRetriever:
    """
    Day15 canonical document retriever.

    Combines sparse and dense retrieval outputs while preserving
    EvidenceCandidate contract.
    """

    def __init__(self, bm25_retriever=None, dense_retriever=None):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever

    def retrieve(self, query: str, top_k: int = 5) -> list[EvidenceCandidate]:
        bm25 = []
        dense = []

        if self.bm25_retriever:
            bm25 = self.bm25_retriever.retrieve(query, top_k)

        if self.dense_retriever:
            dense = self.dense_retriever.retrieve(query, top_k)

        bm25_ids = [x.evidence_id for x in bm25]
        dense_ids = [x.evidence_id for x in dense]

        fused = reciprocal_rank_fusion(
            [bm25_ids, dense_ids]
        )

        lookup = {}
        for item in bm25 + dense:
            lookup[item.evidence_id] = item

        results = []
        for evidence_id, score in fused[:top_k]:
            item = lookup[evidence_id]
            results.append(
                EvidenceCandidate(
                    evidence_id=item.evidence_id,
                    domain="document",
                    score=score,
                    metadata={
                        **item.metadata,
                        "fusion": "rrf",
                    },
                )
            )

        return results
