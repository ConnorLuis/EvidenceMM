from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidencemm.data_binding import sha256_file


class CameraSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera: Literal["front", "wrist"]
    frame_count: int = Field(gt=0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    transform: str = Field(min_length=1)


class EpisodeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "evidencemm_robot_sequence_episode_v1"
    ] = "evidencemm_robot_sequence_episode_v1"

    episode_id: str = Field(min_length=1)
    canonical_source_type: Literal[
        "sample_synchronized_image_sequence"
    ] = "sample_synchronized_image_sequence"

    source_schema_version: str = Field(min_length=1)
    source_script_version: str = Field(min_length=1)
    task: str = Field(min_length=1)

    frame_count: int = Field(gt=0)
    nominal_hz: float = Field(gt=0)
    actual_record_span_seconds: float = Field(gt=0)
    timestamp_source: Literal["samples_csv.elapsed_ns"]

    metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    samples_csv_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    episode_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    cameras: list[CameraSpec]
    source_checks_overall_pass: bool

    @model_validator(mode="after")
    def validate_cameras(self):
        names = [camera.camera for camera in self.cameras]
        if names != ["front", "wrist"]:
            raise ValueError(
                "camera order must be ['front', 'wrist']"
            )
        for camera in self.cameras:
            if camera.frame_count != self.frame_count:
                raise ValueError(
                    "camera frame_count must match episode frame_count"
                )
        return self


class FrameRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    camera: Literal["front", "wrist"]
    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0)
    sample_timestamp_ns: int = Field(gt=0)
    source_timestamp_ns: int = Field(gt=0)
    source_age_ms: float = Field(ge=0)
    camera_sequence: int = Field(ge=0)
    image_relpath: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)


@dataclass(frozen=True)
class SampleRow:
    frame_index: int
    sample_timestamp_ns: int
    elapsed_ns: int
    front_sequence: int
    front_source_timestamp_ns: int
    front_age_ms: float
    front_image: str
    wrist_sequence: int
    wrist_source_timestamp_ns: int
    wrist_age_ms: float
    wrist_image: str


REQUIRED_COLUMNS = {
    "frame_index",
    "sample_timestamp_ns",
    "elapsed_ns",
    "front_sequence",
    "front_source_timestamp_ns",
    "front_age_ms",
    "front_image",
    "wrist_sequence",
    "wrist_source_timestamp_ns",
    "wrist_age_ms",
    "wrist_image",
}


def load_source_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_sample_rows(path: Path) -> list[SampleRow]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError(
                "samples.csv missing required columns: "
                + repr(sorted(missing))
            )

        rows = [
            SampleRow(
                frame_index=int(row["frame_index"]),
                sample_timestamp_ns=int(row["sample_timestamp_ns"]),
                elapsed_ns=int(row["elapsed_ns"]),
                front_sequence=int(row["front_sequence"]),
                front_source_timestamp_ns=int(
                    row["front_source_timestamp_ns"]
                ),
                front_age_ms=float(row["front_age_ms"]),
                front_image=row["front_image"],
                wrist_sequence=int(row["wrist_sequence"]),
                wrist_source_timestamp_ns=int(
                    row["wrist_source_timestamp_ns"]
                ),
                wrist_age_ms=float(row["wrist_age_ms"]),
                wrist_image=row["wrist_image"],
            )
            for row in reader
        ]

    if not rows:
        raise ValueError("samples.csv contains no data rows")

    if [row.frame_index for row in rows] != list(range(len(rows))):
        raise ValueError(
            "frame_index must be contiguous and zero-based"
        )

    elapsed = [row.elapsed_ns for row in rows]
    if any(
        later <= earlier
        for earlier, later in zip(elapsed, elapsed[1:])
    ):
        raise ValueError("elapsed_ns must be strictly monotonic")

    return rows


def inspect_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def build_frame_records(
    *,
    episode_dir: Path,
    episode_id: str,
    rows: list[SampleRow],
) -> list[FrameRecord]:
    records: list[FrameRecord] = []

    for row in rows:
        for camera in ("front", "wrist"):
            if camera == "front":
                relpath = row.front_image
                sequence = row.front_sequence
                source_timestamp_ns = row.front_source_timestamp_ns
                age_ms = row.front_age_ms
            else:
                relpath = row.wrist_image
                sequence = row.wrist_sequence
                source_timestamp_ns = row.wrist_source_timestamp_ns
                age_ms = row.wrist_age_ms

            expected_relpath = (
                f"{camera}/{row.frame_index:06d}.jpg"
            )
            if relpath != expected_relpath:
                raise ValueError(
                    f"{camera} image path mismatch at frame "
                    f"{row.frame_index}: {relpath!r} != "
                    f"{expected_relpath!r}"
                )

            image_path = episode_dir / relpath
            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            width, height = inspect_image(image_path)

            records.append(
                FrameRecord(
                    episode_id=episode_id,
                    camera=camera,
                    frame_index=row.frame_index,
                    timestamp_sec=row.elapsed_ns / 1e9,
                    sample_timestamp_ns=row.sample_timestamp_ns,
                    source_timestamp_ns=source_timestamp_ns,
                    source_age_ms=age_ms,
                    camera_sequence=sequence,
                    image_relpath=relpath,
                    image_sha256=sha256_file(image_path),
                    width_px=width,
                    height_px=height,
                )
            )

    return records


def canonical_episode_hash(
    *,
    metadata_sha256: str,
    samples_csv_sha256: str,
    records: list[FrameRecord],
) -> str:
    lines = [
        f"metadata:{metadata_sha256}",
        f"samples:{samples_csv_sha256}",
    ]
    ordered = sorted(
        records,
        key=lambda item: (
            item.camera,
            item.frame_index,
        ),
    )
    lines.extend(
        (
            f"{record.camera}:"
            f"{record.frame_index}:"
            f"{record.image_sha256}"
        )
        for record in ordered
    )
    payload = "\n".join(lines) + "\n"
    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def save_frame_records(
    records: list[FrameRecord],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps(
                record.model_dump(),
                ensure_ascii=False,
            )
            for record in records
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_frame_records(path: Path) -> list[FrameRecord]:
    return [
        FrameRecord.model_validate(json.loads(line))
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
