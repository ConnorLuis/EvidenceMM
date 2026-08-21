from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidencemm.temporal_evidence import FrameRecord
from evidencemm.temporal_slicing import TemporalSlice


FROZEN_K_VALUES = (1, 2, 3)
TARGET_RULE = "uniform_interior_quantiles"
NEAREST_SAMPLE_TIE_BREAK = "lower_frame_index"
SELECTION_SIGNALS = "timestamp_only"
DUPLICATE_POLICY = "reject"


class CameraKeyframeEvidence(BaseModel):
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


class TemporalKeyframe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyframe_rank: int = Field(ge=1)
    target_fraction: float = Field(gt=0.0, lt=1.0)
    target_timestamp_sec: float = Field(ge=0.0)

    selected_frame_index: int = Field(ge=0)
    selected_timestamp_sec: float = Field(ge=0.0)
    selection_error_ms: float = Field(ge=0.0)

    cameras: list[CameraKeyframeEvidence]

    @model_validator(mode="after")
    def validate_camera_pair(self):
        names = [item.camera for item in self.cameras]
        if names != ["front", "wrist"]:
            raise ValueError(
                "keyframe cameras must be ordered front, wrist"
            )
        for item in self.cameras:
            if item.frame_index != self.selected_frame_index:
                raise ValueError(
                    "camera frame must match selected shared frame"
                )
            if (
                abs(
                    item.timestamp_sec
                    - self.selected_timestamp_sec
                )
                > 1e-12
            ):
                raise ValueError(
                    "camera timestamp must match selected timestamp"
                )
        return self


class MultiKeyframeSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slice_group_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)

    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(gt=0.0)
    start_frame_index: int = Field(ge=0)
    end_frame_index_exclusive: int = Field(gt=0)

    k: int = Field(ge=1)
    shared_sample_budget: int = Field(ge=1)
    evidence_image_budget: int = Field(ge=2)

    keyframes: list[TemporalKeyframe]

    @model_validator(mode="after")
    def validate_contract(self):
        if self.k not in FROZEN_K_VALUES:
            raise ValueError(
                f"k must be one of {FROZEN_K_VALUES}"
            )
        if self.end_sec <= self.start_sec:
            raise ValueError(
                "selection interval must have positive duration"
            )
        if (
            self.end_frame_index_exclusive
            <= self.start_frame_index
        ):
            raise ValueError(
                "frame interval must have positive length"
            )
        if self.shared_sample_budget != self.k:
            raise ValueError(
                "shared sample budget must equal k"
            )
        if self.evidence_image_budget != self.k * 2:
            raise ValueError(
                "evidence image budget must equal 2*k"
            )
        if len(self.keyframes) != self.k:
            raise ValueError(
                "keyframe count must equal k"
            )

        ranks = [item.keyframe_rank for item in self.keyframes]
        if ranks != list(range(1, self.k + 1)):
            raise ValueError(
                "keyframe ranks must be contiguous from 1"
            )

        indices = [
            item.selected_frame_index
            for item in self.keyframes
        ]
        if len(set(indices)) != len(indices):
            raise ValueError(
                "frozen duplicate policy rejects repeated "
                "selected frame indices"
            )

        for item in self.keyframes:
            if not (
                self.start_frame_index
                <= item.selected_frame_index
                < self.end_frame_index_exclusive
            ):
                raise ValueError(
                    "selected frame must lie inside frozen slice"
                )
            if not (
                self.start_sec
                <= item.selected_timestamp_sec
                < self.end_sec
            ):
                raise ValueError(
                    "selected timestamp must lie inside frozen slice"
                )

        return self


def interior_quantile_fractions(k: int) -> list[float]:
    if k not in FROZEN_K_VALUES:
        raise ValueError(
            f"k must be one of {FROZEN_K_VALUES}"
        )
    return [
        rank / (k + 1)
        for rank in range(1, k + 1)
    ]


def _group_records(
    records: list[FrameRecord],
) -> dict[int, dict[str, FrameRecord]]:
    grouped: dict[int, dict[str, FrameRecord]] = {}

    for record in records:
        camera_map = grouped.setdefault(
            record.frame_index,
            {},
        )
        if record.camera in camera_map:
            raise ValueError(
                f"duplicate {record.camera} record at "
                f"frame {record.frame_index}"
            )
        camera_map[record.camera] = record

    if not grouped:
        raise ValueError("no FrameRecords supplied")

    for index, camera_map in grouped.items():
        if set(camera_map) != {"front", "wrist"}:
            raise ValueError(
                f"incomplete camera pair at frame {index}"
            )
        if (
            abs(
                camera_map["front"].timestamp_sec
                - camera_map["wrist"].timestamp_sec
            )
            > 1e-12
        ):
            raise ValueError(
                f"pair timestamp mismatch at frame {index}"
            )

    return grouped


def _camera_evidence(
    camera_map: dict[str, FrameRecord],
) -> list[CameraKeyframeEvidence]:
    result = []
    for camera in ("front", "wrist"):
        record = camera_map[camera]
        result.append(
            CameraKeyframeEvidence(
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
    return result


def build_multi_keyframe_selection(
    *,
    records: list[FrameRecord],
    temporal_slice: TemporalSlice,
    k: int,
) -> MultiKeyframeSelection:
    fractions = interior_quantile_fractions(k)
    grouped = _group_records(records)

    member_indices = list(
        range(
            temporal_slice.start_frame_index,
            temporal_slice.end_frame_index_exclusive,
        )
    )
    if not member_indices:
        raise ValueError(
            "temporal slice contains no candidate frames"
        )

    missing = [
        index
        for index in member_indices
        if index not in grouped
    ]
    if missing:
        raise ValueError(
            "missing FrameRecord pairs inside temporal slice: "
            + repr(missing)
        )

    timestamps = {
        index: grouped[index]["front"].timestamp_sec
        for index in member_indices
    }

    duration = (
        temporal_slice.end_sec
        - temporal_slice.start_sec
    )
    keyframes: list[TemporalKeyframe] = []

    for rank, fraction in enumerate(
        fractions,
        start=1,
    ):
        target = (
            temporal_slice.start_sec
            + fraction * duration
        )
        selected_index = min(
            member_indices,
            key=lambda index: (
                abs(timestamps[index] - target),
                index,
            ),
        )
        selected_timestamp = timestamps[selected_index]

        keyframes.append(
            TemporalKeyframe(
                keyframe_rank=rank,
                target_fraction=fraction,
                target_timestamp_sec=target,
                selected_frame_index=selected_index,
                selected_timestamp_sec=selected_timestamp,
                selection_error_ms=(
                    abs(selected_timestamp - target)
                    * 1000.0
                ),
                cameras=_camera_evidence(
                    grouped[selected_index]
                ),
            )
        )

    return MultiKeyframeSelection(
        slice_group_id=temporal_slice.slice_group_id,
        episode_id=temporal_slice.episode_id,
        start_sec=temporal_slice.start_sec,
        end_sec=temporal_slice.end_sec,
        start_frame_index=(
            temporal_slice.start_frame_index
        ),
        end_frame_index_exclusive=(
            temporal_slice.end_frame_index_exclusive
        ),
        k=k,
        shared_sample_budget=k,
        evidence_image_budget=k * 2,
        keyframes=keyframes,
    )
