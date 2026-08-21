from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from evidencemm.dense_retrieval import (
    encode_dense_texts,
    load_dense_encoder,
    rank_dense_scores,
    score_dense_documents,
)
from evidencemm.document_candidate_retrieval import (
    DocumentBM25CandidateRetriever,
)
from evidencemm.hybrid_candidate_union import (
    build_candidate_union,
)
from evidencemm.reranking import (
    RerankedPage,
    load_reranker,
    rank_reranker_scores,
    score_query_passages,
)
from evidencemm.retrieval import (
    normalize_query,
    validate_top_k,
)
from evidencemm.retrieval_ranking import (
    RankedEvidenceCandidate,
    RetrievalDomain,
)


CANONICAL_HYBRID_RETRIEVER_NAME = (
    "document_bm25_bge_m3_reranker_v1"
)


@dataclass(frozen=True)
class CanonicalDocumentTrace:
    query: str
    branch_top_k: int
    final_top_k: int
    bm25_hits: list[dict]
    dense_hits: list[dict]
    candidate_union: list[dict]
    reranked_hits: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def ranked_candidates_from_reranker(
    *,
    reranked_hits: list[RerankedPage],
    item_by_page: dict[int, object],
) -> list[RankedEvidenceCandidate]:
    candidates: list[RankedEvidenceCandidate] = []

    for expected_rank, hit in enumerate(
        reranked_hits,
        start=1,
    ):
        if hit.rank != expected_rank:
            raise ValueError(
                "reranker ranks must be contiguous from 1"
            )
        if hit.page_number not in item_by_page:
            raise ValueError(
                f"missing canonical evidence item for page "
                f"{hit.page_number}"
            )

        candidates.append(
            RankedEvidenceCandidate(
                domain=RetrievalDomain.DOCUMENT,
                retriever_name=(
                    CANONICAL_HYBRID_RETRIEVER_NAME
                ),
                rank=hit.rank,
                raw_score=hit.reranker_score,
                item=item_by_page[hit.page_number],
            )
        )

    return candidates


class CanonicalHybridDocumentRetriever:
    """Day15 canonical document retriever.

    Reuses the frozen Day13 stack:
    BM25 Top-K + BGE-M3 Top-K -> candidate union -> cross-encoder reranker.

    It returns the Day12 RankedEvidenceCandidate contract so the existing
    cross-domain fixed-quota bundle and grounded generation path can be reused
    without modifying frozen Day11-Day14 modules.
    """

    def __init__(
        self,
        *,
        project_root: str | Path,
        source_manifest_path: str | Path,
        visual_manifest_path: str | Path,
        hybrid_config_path: str | Path = (
            "configs/day13_hybrid_retrieval.yaml"
        ),
    ) -> None:
        self.project_root = Path(project_root).resolve()

        config_path = Path(hybrid_config_path)
        if not config_path.is_absolute():
            config_path = self.project_root / config_path
        self.config = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )

        self.branch_top_k = int(
            self.config["branch_top_k"]
        )
        if self.branch_top_k != 5:
            raise ValueError(
                "canonical Day15 document retrieval requires "
                "the frozen Day13 branch_top_k=5"
            )

        self.base = DocumentBM25CandidateRetriever(
            project_root=self.project_root,
            source_manifest_path=source_manifest_path,
            visual_manifest_path=visual_manifest_path,
        )

        self.last_trace: CanonicalDocumentTrace | None = None

        self._dense_model = None
        self._dense_tokenizer = None
        self._reranker = None
        self._reranker_tokenizer = None

    def _release_dense(self) -> None:
        self._dense_model = None
        self._dense_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _release_reranker(self) -> None:
        self._reranker = None
        self._reranker_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_models(self) -> None:
        self._release_dense()
        self._release_reranker()

    def search(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[RankedEvidenceCandidate]:
        normalized_query = normalize_query(query)
        final_top_k = validate_top_k(top_k)

        if final_top_k > self.branch_top_k:
            raise ValueError(
                "canonical document final top_k cannot exceed "
                "the frozen branch_top_k"
            )

        bm25_hits = self.base.index.search(
            normalized_query,
            top_k=self.branch_top_k,
        )

        (
            self._dense_model,
            self._dense_tokenizer,
            _,
        ) = load_dense_encoder(
            model_name=self.config[
                "dense_model_name"
            ],
            device=self.config["device"],
        )

        document_embeddings, _ = encode_dense_texts(
            model=self._dense_model,
            tokenizer=self._dense_tokenizer,
            texts=[
                document.text
                for document in self.base.documents
            ],
            batch_size=int(
                self.config["dense_batch_size"]
            ),
            max_length=int(
                self.config["dense_max_length"]
            ),
        )
        query_embedding, _ = encode_dense_texts(
            model=self._dense_model,
            tokenizer=self._dense_tokenizer,
            texts=[normalized_query],
            batch_size=1,
            max_length=int(
                self.config["dense_max_length"]
            ),
        )
        dense_scores = score_dense_documents(
            query_embedding=query_embedding,
            document_embeddings=document_embeddings,
        )
        dense_hits = rank_dense_scores(
            scores=dense_scores,
            documents=self.base.documents,
            top_k=self.branch_top_k,
        )

        del document_embeddings
        del query_embedding
        self._release_dense()

        pool = build_candidate_union(
            bm25_hits=bm25_hits,
            dense_hits=dense_hits,
        )

        (
            self._reranker,
            self._reranker_tokenizer,
            _,
        ) = load_reranker(
            model_name=self.config[
                "reranker_model_name"
            ],
            device=self.config["device"],
        )

        document_by_key = {
            (
                document.source_id,
                document.page_number,
            ): document
            for document in self.base.documents
        }
        passages = [
            document_by_key[
                (
                    candidate.source_id,
                    candidate.page_number,
                )
            ].text
            for candidate in pool
        ]
        reranker_scores, _ = score_query_passages(
            model=self._reranker,
            tokenizer=self._reranker_tokenizer,
            query=normalized_query,
            passages=passages,
            batch_size=int(
                self.config["reranker_batch_size"]
            ),
            max_length=int(
                self.config["reranker_max_length"]
            ),
        )
        reranked_hits = rank_reranker_scores(
            pool=pool,
            scores=reranker_scores,
            top_k=final_top_k,
        )

        self._release_reranker()

        # The frozen Day12 retriever remains the canonical builder for
        # DocumentPagePayload / EvidenceRef / provenance. Search all bound
        # pages only to obtain those canonical evidence items; its ranking is
        # not used for final Day15 ordering.
        all_items = self.base.search(
            normalized_query,
            top_k=len(self.base.documents),
        )
        item_by_page = {
            candidate.item.payload.page_number: candidate.item
            for candidate in all_items
        }

        candidates = ranked_candidates_from_reranker(
            reranked_hits=reranked_hits,
            item_by_page=item_by_page,
        )

        self.last_trace = CanonicalDocumentTrace(
            query=normalized_query,
            branch_top_k=self.branch_top_k,
            final_top_k=final_top_k,
            bm25_hits=[
                hit.to_dict()
                for hit in bm25_hits
            ],
            dense_hits=[
                hit.to_dict()
                for hit in dense_hits
            ],
            candidate_union=[
                candidate.to_dict()
                for candidate in pool
            ],
            reranked_hits=[
                hit.to_dict()
                for hit in reranked_hits
            ],
        )

        return candidates
