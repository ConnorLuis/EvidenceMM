from __future__ import annotations

import json
from pathlib import Path

import pymupdf

from evidencemm.data_binding import sha256_file
from evidencemm.schemas import SourceManifest, SourceType
from evidencemm.text_retrieval import PageDocument


def load_source_manifest(
    path: str | Path,
) -> SourceManifest:
    manifest_path = Path(path)
    return SourceManifest.model_validate_json(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )


def resolve_source_path(
    manifest: SourceManifest,
    *,
    project_root: str | Path,
) -> Path:
    path = Path(manifest.local_path)
    if not path.is_absolute():
        path = (
            Path(project_root)
            / path
        )
    return path.resolve()


def standardize_pdf_pages(
    manifest: SourceManifest,
    *,
    project_root: str | Path,
) -> list[PageDocument]:
    if manifest.source_type != SourceType.PDF:
        raise ValueError(
            "standardize_pdf_pages requires a PDF source"
        )

    pdf_path = resolve_source_path(
        manifest,
        project_root=project_root,
    )
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    actual_sha = sha256_file(pdf_path)
    if actual_sha != manifest.sha256:
        raise ValueError(
            "source bytes do not match manifest SHA256"
        )

    documents: list[PageDocument] = []

    with pymupdf.open(pdf_path) as pdf:
        if manifest.page_count != len(pdf):
            raise ValueError(
                "manifest page_count does not match PDF"
            )

        for page_index, page in enumerate(pdf):
            documents.append(
                PageDocument.from_text(
                    source_id=manifest.source_id,
                    page_number=page_index + 1,
                    text=page.get_text("text"),
                )
            )

    return documents
