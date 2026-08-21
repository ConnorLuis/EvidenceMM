from __future__ import annotations

from typing import Protocol, runtime_checkable

from evidencemm.unified_evidence import UnifiedEvidenceBundle


DEFAULT_TOP_K = 5


class RetrievalContractError(ValueError):
    """Raised when a retriever violates the Day 12 boundary contract."""


def normalize_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError("query must be a string")

    normalized = query.strip()
    if not normalized:
        raise RetrievalContractError(
            "query must contain non-whitespace text"
        )
    return normalized


def validate_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k < 1:
        raise RetrievalContractError(
            "top_k must be >= 1"
        )
    return top_k


@runtime_checkable
class EvidenceRetriever(Protocol):
    """Minimal retrieval boundary used by EvidenceMM generation.

    Implementations may use BM25, dense retrieval, visual retrieval,
    temporal retrieval, or a future external index. Those details stay
    behind this protocol.

    The only production-facing output is a validated
    ``UnifiedEvidenceBundle``.
    """

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> UnifiedEvidenceBundle:
        ...


def validate_retrieved_bundle(
    *,
    query: str,
    top_k: int,
    bundle: UnifiedEvidenceBundle,
) -> UnifiedEvidenceBundle:
    normalized_query = normalize_query(query)
    validate_top_k(top_k)

    if not isinstance(bundle, UnifiedEvidenceBundle):
        raise TypeError(
            "retriever must return UnifiedEvidenceBundle"
        )

    if bundle.question != normalized_query:
        raise RetrievalContractError(
            "bundle.question must equal the normalized retrieval query"
        )

    if len(bundle.items) > top_k:
        raise RetrievalContractError(
            "retrieved evidence item count exceeds top_k"
        )

    return bundle


def retrieve_with_contract(
    retriever: EvidenceRetriever,
    *,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> UnifiedEvidenceBundle:
    """Invoke one retriever while enforcing the stable Day 12 boundary."""

    normalized_query = normalize_query(query)
    validated_top_k = validate_top_k(top_k)

    bundle = retriever.retrieve(
        normalized_query,
        validated_top_k,
    )

    return validate_retrieved_bundle(
        query=normalized_query,
        top_k=validated_top_k,
        bundle=bundle,
    )
