from __future__ import annotations

from pathlib import Path

from evidencemm.data_binding import sha256_file
from evidencemm.pdf_corpus import (
    load_source_manifest,
    standardize_pdf_pages,
)
from evidencemm.retrieval import (
    normalize_query,
    validate_top_k,
)
from evidencemm.retrieval_ranking import (
    RankedEvidenceCandidate,
    RetrievalDomain,
)
from evidencemm.schemas import EvidenceRef, SourceType
from evidencemm.text_retrieval import BM25Index
from evidencemm.unified_evidence import (
    DocumentPagePayload,
    EvidenceProvenance,
    UnifiedEvidenceItem,
    UnifiedEvidenceKind,
)
from evidencemm.visual_corpus import load_visual_manifest


DOCUMENT_RETRIEVER_NAME = "document_bm25_v1"


def _stored_path(
    path: Path,
    *,
    project_root: Path,
) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(
            project_root.resolve()
        ).as_posix()
    except ValueError:
        return str(resolved)


class DocumentBM25CandidateRetriever:
    """Build real PDF-page evidence candidates from the frozen BM25 baseline."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        source_manifest_path: str | Path,
        visual_manifest_path: str | Path,
    ) -> None:
        self.project_root = Path(project_root).resolve()

        manifest_path = Path(source_manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = (
                self.project_root / manifest_path
            )
        self.source_manifest_path = (
            manifest_path.resolve()
        )
        self.manifest = load_source_manifest(
            self.source_manifest_path
        )
        if self.manifest.source_type != SourceType.PDF:
            raise ValueError(
                "document BM25 retriever requires PDF source"
            )

        self.documents = standardize_pdf_pages(
            self.manifest,
            project_root=self.project_root,
        )
        self.index = BM25Index(self.documents)
        self.document_by_page = {
            document.page_number: document
            for document in self.documents
        }

        visual_path = Path(visual_manifest_path)
        if not visual_path.is_absolute():
            visual_path = self.project_root / visual_path
        self.visual_manifest_path = visual_path.resolve()

        visual_pages = load_visual_manifest(
            self.visual_manifest_path
        )
        self.visual_by_page = {}

        for page in visual_pages:
            if page.source_id != self.manifest.source_id:
                raise ValueError(
                    "visual page source_id differs from PDF source"
                )
            image_path = (
                self.project_root / page.image_path
            )
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if sha256_file(image_path) != page.image_sha256:
                raise ValueError(
                    "rendered PDF page image SHA256 mismatch"
                )
            self.visual_by_page[page.page_number] = page

        expected_pages = set(self.document_by_page)
        if set(self.visual_by_page) != expected_pages:
            raise ValueError(
                "visual page manifest does not cover PDF pages exactly"
            )

    def search(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[RankedEvidenceCandidate]:
        normalized_query = normalize_query(query)
        validated_top_k = validate_top_k(top_k)

        hits = self.index.search(
            normalized_query,
            top_k=validated_top_k,
        )

        candidates: list[RankedEvidenceCandidate] = []
        for hit in hits:
            document = self.document_by_page[
                hit.page_number
            ]
            visual_page = self.visual_by_page[
                hit.page_number
            ]

            item = UnifiedEvidenceItem(
                evidence_id=(
                    f"doc:{self.manifest.source_id}:"
                    f"p{hit.page_number}"
                ),
                kind=UnifiedEvidenceKind.DOCUMENT_PAGE,
                refs=[
                    EvidenceRef(
                        source_id=self.manifest.source_id,
                        source_type=SourceType.PDF,
                        page_number=hit.page_number,
                    )
                ],
                provenance=EvidenceProvenance(
                    source_id=self.manifest.source_id,
                    source_type=SourceType.PDF,
                    manifest_path=_stored_path(
                        self.source_manifest_path,
                        project_root=self.project_root,
                    ),
                    canonical_sha256=self.manifest.sha256,
                    supporting_sha256={
                        "rendered_page_image": (
                            visual_page.image_sha256
                        )
                    },
                ),
                payload=DocumentPagePayload(
                    page_number=hit.page_number,
                    text_sha256=document.text_sha256,
                    char_count=document.char_count,
                    text_excerpt=document.text,
                    page_image_path=visual_page.image_path,
                ),
            )

            candidates.append(
                RankedEvidenceCandidate(
                    domain=RetrievalDomain.DOCUMENT,
                    retriever_name=DOCUMENT_RETRIEVER_NAME,
                    rank=hit.rank,
                    raw_score=hit.score,
                    item=item,
                )
            )

        return candidates
