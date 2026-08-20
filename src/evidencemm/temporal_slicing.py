from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidencemm.temporal_evidence import EpisodeManifest, FrameRecord


class CameraMidpointEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera: Literal["front", "wrist"]
    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0)
    source_timestamp_ns: int = Field(gt=0)
    source_age_ms: float = Field(ge=0)
    camera_sequence: int = Field(ge=0)
    image_relpath: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)


class TemporalSlice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slice_group_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)

    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)

    start_frame_index: int = Field(ge=0)
    end_frame_index_exclusive: int = Field(gt=0)

    midpoint_target_sec: float = Field(ge=0)
    midpoint_frame_index: int = Field(ge=0)
    midpoint_timestamp_sec: float = Field(ge=0)
    midpoint_error_ms: float = Field(ge=0)

    cameras: list[CameraMidpointEvidence]

    @model_validator(mode="after")
    def validate_contract(self):
        if self.end_sec <= self.start_sec:
            raise ValueError("slice interval must have positive duration")
        if self.end_frame_index_exclusive <= self.start_frame_index:
            raise ValueError("frame interval must have positive length")
        if not (
            self.start_sec
            <= self.midpoint_timestamp_sec
            < self.end_sec
        ):
            raise ValueError("midpoint sample timestamp must lie inside slice")
        if not (
            self.start_frame_index
            <= self.midpoint_frame_index
            < self.end_frame_index_exclusive
        ):
            raise ValueError("midpoint frame index must lie inside slice")

        camera_names = [item.camera for item in self.cameras]
        if camera_names != ["front", "wrist"]:
            raise ValueError("slice cameras must be ordered front, wrist")

        for item in self.cameras:
            if item.frame_index != self.midpoint_frame_index:
                raise ValueError(
                    "camera midpoint frame must match shared frame index"
                )
            if abs(item.timestamp_sec - self.midpoint_timestamp_sec) > 1e-12:
                raise ValueError(
                    "camera midpoint timestamp must match shared timestamp"
                )

        return self


def _group_records(
    records: list[FrameRecord],
) -> dict[int, dict[str, FrameRecord]]:
    grouped: dict[int, dict[str, FrameRecord]] = {}

    for record in records:
        camera_map = grouped.setdefault(record.frame_index, {})
        if record.camera in camera_map:
            raise ValueError(
                f"duplicate {record.camera} record at frame {record.frame_index}"
            )
        camera_map[record.camera] = record

    if not grouped:
        raise ValueError("no FrameRecords supplied")

    expected_indices = list(range(min(grouped), max(grouped) + 1))
    if sorted(grouped) != expected_indices:
        raise ValueError("FrameRecord indices are not contiguous")

    for index, camera_map in grouped.items():
        if set(camera_map) != {"front", "wrist"}:
            raise ValueError(f"incomplete camera pair at frame {index}")

        front_ts = camera_map["front"].timestamp_sec
        wrist_ts = camera_map["wrist"].timestamp_sec
        if abs(front_ts - wrist_ts) > 1e-12:
            raise ValueError(f"pair timestamp mismatch at frame {index}")

    return grouped


def build_temporal_slices(
    *,
    manifest: EpisodeManifest,
    records: list[FrameRecord],
    slice_duration_sec: float,
) -> list[TemporalSlice]:
    if slice_duration_sec <= 0:
        raise ValueError("slice_duration_sec must be positive")

    grouped = _group_records(records)

    if len(grouped) != manifest.frame_count:
        raise ValueError("FrameRecord pair count does not match manifest")

    timestamps = {
        index: camera_map["front"].timestamp_sec
        for index, camera_map in grouped.items()
    }

    max_timestamp = max(timestamps.values())
    slices: list[TemporalSlice] = []

    slice_index = 0
    start_sec = 0.0

    while start_sec <= max_timestamp + 1e-12:
        end_sec = start_sec + slice_duration_sec

        member_indices = [
            index
            for index, timestamp in timestamps.items()
            if start_sec <= timestamp < end_sec
        ]
        if not member_indices:
            raise ValueError(
                "temporal window contains no samples: "
                f"[{start_sec}, {end_sec})"
            )

        member_indices.sort()

        midpoint_target = start_sec + slice_duration_sec / 2.0
        midpoint_index = min(
            member_indices,
            key=lambda index: (
                abs(timestamps[index] - midpoint_target),
                index,
            ),
        )

        midpoint_timestamp = timestamps[midpoint_index]
        camera_map = grouped[midpoint_index]

        camera_evidence = []
        for camera in ("front", "wrist"):
            record = camera_map[camera]
            camera_evidence.append(
                CameraMidpointEvidence(
                    camera=camera,
                    frame_index=record.frame_index,
                    timestamp_sec=record.timestamp_sec,
                    source_timestamp_ns=record.source_timestamp_ns,
                    source_age_ms=record.source_age_ms,
                    camera_sequence=record.camera_sequence,
                    image_relpath=record.image_relpath,
                    image_sha256=record.image_sha256,
                    width_px=record.width_px,
                    height_px=record.height_px,
                )
            )

        slices.append(
            TemporalSlice(
                slice_group_id=(
                    f"{manifest.episode_id}_slice_{slice_index:04d}"
                ),
                episode_id=manifest.episode_id,
                start_sec=start_sec,
                end_sec=end_sec,
                start_frame_index=member_indices[0],
                end_frame_index_exclusive=member_indices[-1] + 1,
                midpoint_target_sec=midpoint_target,
                midpoint_frame_index=midpoint_index,
                midpoint_timestamp_sec=midpoint_timestamp,
                midpoint_error_ms=(
                    abs(midpoint_timestamp - midpoint_target) * 1000.0
                ),
                cameras=camera_evidence,
            )
        )

        slice_index += 1
        start_sec = end_sec

    return slices


def save_temporal_slices(
    slices: list[TemporalSlice],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps(item.model_dump(), ensure_ascii=False)
            for item in slices
        )
        + ("\n" if slices else ""),
        encoding="utf-8",
        newline="\n",
    )


def load_temporal_slices(path: Path) -> list[TemporalSlice]:
    return [
        TemporalSlice.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
