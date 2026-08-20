from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf
from PIL import Image

from evidencemm.data_binding import sha256_file
from evidencemm.pdf_corpus import resolve_source_path
from evidencemm.schemas import SourceManifest, SourceType


@dataclass(frozen=True)
class VisualPage:
    source_id: str
    page_number: int
    image_path: str
    image_sha256: str
    width_px: int
    height_px: int
    render_dpi: int

    def to_dict(self) -> dict:
        return asdict(self)


def render_pdf_pages(
    manifest: SourceManifest,
    *,
    project_root: str | Path,
    output_dir: str | Path,
    render_dpi: int,
) -> list[VisualPage]:
    if manifest.source_type != SourceType.PDF:
        raise ValueError("render_pdf_pages requires a PDF source")
    if render_dpi < 72:
        raise ValueError("render_dpi must be >= 72")

    root = Path(project_root).resolve()
    pdf_path = resolve_source_path(
        manifest,
        project_root=root,
    )

    if sha256_file(pdf_path) != manifest.sha256:
        raise ValueError(
            "source bytes do not match SourceManifest SHA256"
        )

    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)

    pages: list[VisualPage] = []
    scale = render_dpi / 72.0

    with pymupdf.open(pdf_path) as pdf:
        if len(pdf) != manifest.page_count:
            raise ValueError(
                "manifest page_count does not match PDF"
            )

        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            image_path = (
                output
                / f"{manifest.source_id}_page_{page_number:04d}.png"
            )

            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(scale, scale),
                alpha=False,
            )
            pix.save(image_path)

            with Image.open(image_path) as image:
                width, height = image.size

            try:
                stored_path = image_path.relative_to(root).as_posix()
            except ValueError:
                stored_path = str(image_path.resolve())

            pages.append(
                VisualPage(
                    source_id=manifest.source_id,
                    page_number=page_number,
                    image_path=stored_path,
                    image_sha256=sha256_file(image_path),
                    width_px=width,
                    height_px=height,
                    render_dpi=render_dpi,
                )
            )

    return pages


def save_visual_manifest(
    pages: list[VisualPage],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        json.dumps(page.to_dict(), ensure_ascii=False)
        for page in pages
    ]
    output.write_text(
        "\n".join(rows) + ("\n" if rows else ""),
        encoding="utf-8",
        newline="\n",
    )
    return output


def load_visual_manifest(
    path: str | Path,
) -> list[VisualPage]:
    return [
        VisualPage(**json.loads(line))
        for line in Path(path).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
