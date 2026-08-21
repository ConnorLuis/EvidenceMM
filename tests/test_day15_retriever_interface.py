from evidencemm.evidence_retriever import EvidenceCandidate

def test_evidence_candidate_contract():
    item = EvidenceCandidate(
        evidence_id="doc:test:p1",
        domain="document",
        score=1.0,
        metadata={"page": 1},
    )
    assert item.domain == "document"
