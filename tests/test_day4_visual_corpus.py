from __future__ import annotations

import pymupdf

from evidencemm.data_binding import sha256_file
from evidencemm.schemas import SourceManifest, SourceType
from evidencemm.visual_corpus import (
    load_visual_manifest,
    render_pdf_pages,
    save_visual_manifest,
)


def test_render_pdf_pages_preserves_one_based_pages(tmp_path):
    repo = tmp_path / "repo"
    raw = repo / "data" / "raw"
    raw.mkdir(parents=True)
    pdf_path = raw / "sample.pdf"

    doc = pymupdf.open()
    for text in ["one", "two"]:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(pdf_path)
    doc.close()

    manifest = SourceManifest(
        source_id="sample_pdf",
        source_type=SourceType.PDF,
        local_path="data/raw/sample.pdf",
        sha256=sha256_file(pdf_path),
        size_bytes=pdf_path.stat().st_size,
        mime_type="application/pdf",
        added_at="2026-08-20T00:00:00Z",
        page_count=2,
    )

    pages = render_pdf_pages(
        manifest,
        project_root=repo,
        output_dir="data/processed/pages",
        render_dpi=144,
    )

    assert [p.page_number for p in pages] == [1, 2]
    assert all(p.width_px > 0 for p in pages)
    assert all(p.height_px > 0 for p in pages)
    assert all(len(p.image_sha256) == 64 for p in pages)


def test_visual_manifest_round_trip(tmp_path):
    repo = tmp_path / "repo"
    raw = repo / "data" / "raw"
    raw.mkdir(parents=True)
    pdf_path = raw / "sample.pdf"

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "hello")
    doc.save(pdf_path)
    doc.close()

    manifest = SourceManifest(
        source_id="sample_pdf",
        source_type=SourceType.PDF,
        local_path="data/raw/sample.pdf",
        sha256=sha256_file(pdf_path),
        size_bytes=pdf_path.stat().st_size,
        mime_type="application/pdf",
        added_at="2026-08-20T00:00:00Z",
        page_count=1,
    )

    pages = render_pdf_pages(
        manifest,
        project_root=repo,
        output_dir="data/processed/pages",
        render_dpi=144,
    )

    output = repo / "data/processed/pages.jsonl"
    save_visual_manifest(pages, output)

    assert load_visual_manifest(output) == pages
