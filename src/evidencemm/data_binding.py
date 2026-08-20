from __future__ import annotations

import hashlib
import mimetypes
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
from PIL import Image

from evidencemm.schemas import SourceManifest, SourceType


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def bind_source(
    *,
    source_id: str,
    source_type: SourceType,
    path: str | Path,
    origin_uri: str | None = None,
    project_root: str | Path | None = None,
) -> SourceManifest:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    local_path = str(resolved)

    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
        try:
            local_path = resolved.relative_to(root).as_posix()
        except ValueError:
            pass

    kwargs: dict = {
        "source_id": source_id,
        "source_type": source_type,
        "local_path": local_path,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
        "mime_type": mimetypes.guess_type(resolved.name)[0],
        "origin_uri": origin_uri,
        "added_at": datetime.now(timezone.utc),
        "metadata": {},
    }

    if source_type == SourceType.PDF:
        with pymupdf.open(resolved) as doc:
            kwargs["page_count"] = len(doc)

    elif source_type == SourceType.IMAGE:
        with Image.open(resolved) as image:
            kwargs["width_px"] = image.width
            kwargs["height_px"] = image.height
            kwargs["metadata"] = {
                "image_mode": image.mode,
                "image_format": image.format,
            }

    else:
        raise ValueError(
            "Day2 binder currently supports only pdf and image sources"
        )

    return SourceManifest(**kwargs)
