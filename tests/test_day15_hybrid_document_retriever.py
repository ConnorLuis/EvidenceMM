from evidencemm.evidence_retriever import EvidenceCandidate
from evidencemm.hybrid_document_retriever import HybridDocumentRetriever


class FakeRetriever:
    def __init__(self, items):
        self.items = items

    def retrieve(self, query, top_k):
        return self.items[:top_k]


def test_hybrid_returns_candidates():
    retriever = HybridDocumentRetriever(
        FakeRetriever([
            EvidenceCandidate("p3", "document", 1.0, {})
        ]),
        FakeRetriever([
            EvidenceCandidate("p4", "document", 1.0, {})
        ]),
    )

    result = retriever.retrieve("voltage")

    assert len(result) == 2
    assert result[0].domain == "document"


def test_hybrid_is_deterministic():
    retriever = HybridDocumentRetriever(
        FakeRetriever([
            EvidenceCandidate("p3", "document", 1.0, {})
        ]),
        FakeRetriever([
            EvidenceCandidate("p3", "document", 1.0, {})
        ]),
    )

    assert retriever.retrieve("x")[0].evidence_id == "p3"
