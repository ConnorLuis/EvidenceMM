from __future__ import annotations

import pytest

from evidencemm.retrieval import (
    DEFAULT_TOP_K,
    EvidenceRetriever,
    RetrievalContractError,
    normalize_query,
    retrieve_with_contract,
    validate_retrieved_bundle,
    validate_top_k,
)
from evidencemm.schemas import EvidenceRef, SourceType
from evidencemm.unified_evidence import (
    DocumentPagePayload,
    EvidenceProvenance,
    UnifiedEvidenceBundle,
    UnifiedEvidenceItem,
    UnifiedEvidenceKind,
)


SHA = "a" * 64


def document_item(
    page_number: int,
) -> UnifiedEvidenceItem:
    return UnifiedEvidenceItem(
        evidence_id=f"doc:manual:p{page_number}",
        kind=UnifiedEvidenceKind.DOCUMENT_PAGE,
        refs=[
            EvidenceRef(
                source_id="manual",
                source_type=SourceType.PDF,
                page_number=page_number,
            )
        ],
        provenance=EvidenceProvenance(
            source_id="manual",
            source_type=SourceType.PDF,
            manifest_path=(
                "data/manifests/sources/manual.json"
            ),
            canonical_sha256=SHA,
        ),
        payload=DocumentPagePayload(
            page_number=page_number,
            text_sha256=SHA,
            char_count=10,
            text_excerpt=f"page {page_number}",
        ),
    )


def bundle(
    *,
    question: str,
    item_count: int = 1,
) -> UnifiedEvidenceBundle:
    return UnifiedEvidenceBundle(
        bundle_id="retrieval-test",
        question=question,
        items=[
            document_item(page_number)
            for page_number in range(
                1,
                item_count + 1,
            )
        ],
    )


class FakeRetriever:
    def __init__(
        self,
        result: UnifiedEvidenceBundle,
    ) -> None:
        self.result = result
        self.calls: list[tuple[str, int]] = []

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> UnifiedEvidenceBundle:
        self.calls.append((query, top_k))
        return self.result


class BadRetriever:
    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ):
        return {"query": query, "top_k": top_k}


def test_default_top_k_is_frozen_to_five():
    assert DEFAULT_TOP_K == 5


def test_query_is_trimmed_before_retrieval():
    assert normalize_query("  servo voltage  ") == (
        "servo voltage"
    )


def test_blank_query_is_rejected():
    with pytest.raises(
        RetrievalContractError,
        match="non-whitespace",
    ):
        normalize_query("   ")


def test_top_k_must_be_positive_integer():
    assert validate_top_k(5) == 5

    with pytest.raises(RetrievalContractError):
        validate_top_k(0)

    with pytest.raises(TypeError):
        validate_top_k(True)


def test_fake_retriever_satisfies_runtime_protocol():
    retriever = FakeRetriever(
        bundle(question="query")
    )
    assert isinstance(
        retriever,
        EvidenceRetriever,
    )


def test_contract_wrapper_returns_only_unified_bundle():
    retriever = FakeRetriever(
        bundle(question="servo voltage")
    )

    result = retrieve_with_contract(
        retriever,
        query="  servo voltage  ",
    )

    assert isinstance(
        result,
        UnifiedEvidenceBundle,
    )
    assert retriever.calls == [
        ("servo voltage", 5)
    ]


def test_bundle_question_must_match_query():
    with pytest.raises(
        RetrievalContractError,
        match="bundle.question",
    ):
        validate_retrieved_bundle(
            query="servo voltage",
            top_k=5,
            bundle=bundle(
                question="different query"
            ),
        )


def test_bundle_item_count_cannot_exceed_top_k():
    with pytest.raises(
        RetrievalContractError,
        match="exceeds top_k",
    ):
        validate_retrieved_bundle(
            query="query",
            top_k=5,
            bundle=bundle(
                question="query",
                item_count=6,
            ),
        )


def test_non_bundle_retriever_output_is_rejected():
    with pytest.raises(
        TypeError,
        match="UnifiedEvidenceBundle",
    ):
        retrieve_with_contract(
            BadRetriever(),
            query="query",
        )
