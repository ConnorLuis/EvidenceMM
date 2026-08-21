from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidencemm.data_binding import sha256_file
from evidencemm.temporal_evidence import EpisodeManifest, FrameRecord
from evidencemm.temporal_slicing import TemporalSlice


MOTION_RESIZE_WIDTH = 160
MOTION_RESIZE_HEIGHT = 120
MOTION_DIFFERENCE = "mean_absolute_pixel_difference"
MOTION_FUSION = "max"
MOTION_TIE_BREAK = "lower_frame_index"


class MotionFrameScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0)
    front_motion_score: float = Field(ge=0)
    wrist_motion_score: float = Field(ge=0)
    fused_motion_score: float = Field(ge=0)


class CameraMotionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera: Literal["front", "wrist"]
    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0)
    source_timestamp_ns: int = Field(gt=0)
    source_age_ms: float = Field(ge=0)
    camera_sequence: int = Field(ge=0)
    image_relpath: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    motion_score: float = Field(ge=0)


class MotionAwareSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slice_group_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    start_frame_index: int = Field(ge=0)
    end_frame_index_exclusive: int = Field(gt=0)
    selected_frame_index: int = Field(ge=0)
    selected_timestamp_sec: float = Field(ge=0)
    front_motion_score: float = Field(ge=0)
    wrist_motion_score: float = Field(ge=0)
    fused_motion_score: float = Field(ge=0)
    cameras: list[CameraMotionEvidence]

    @model_validator(mode="after")
    def validate_contract(self):
        if self.end_sec <= self.start_sec:
            raise ValueError("selection interval must have positive duration")
        if not (
            self.start_frame_index
            <= self.selected_frame_index
            < self.end_frame_index_exclusive
        ):
            raise ValueError("selected frame must lie inside slice")
        if not self.start_sec <= self.selected_timestamp_sec < self.end_sec:
            raise ValueError("selected timestamp must lie inside slice")
        if [item.camera for item in self.cameras] != ["front", "wrist"]:
            raise ValueError("selection cameras must be ordered front, wrist")
        for item in self.cameras:
            if item.frame_index != self.selected_frame_index:
                raise ValueError("camera evidence frame mismatch")
            if abs(item.timestamp_sec - self.selected_timestamp_sec) > 1e-12:
                raise ValueError("camera evidence timestamp mismatch")
        if abs(
            self.fused_motion_score
            - max(self.front_motion_score, self.wrist_motion_score)
        ) > 1e-12:
            raise ValueError("fused score must equal max camera score")
        return self


def apply_camera_transform(image: Image.Image, transform: str) -> Image.Image:
    if transform == "none":
        return image
    if transform == "ccw90":
        return image.transpose(Image.Transpose.ROTATE_90)
    raise ValueError(f"unsupported camera transform: {transform!r}")


def preprocess_motion_image(
    path: Path,
    *,
    transform: str,
    resize_width: int = MOTION_RESIZE_WIDTH,
    resize_height: int = MOTION_RESIZE_HEIGHT,
) -> np.ndarray:
    if resize_width <= 0 or resize_height <= 0:
        raise ValueError("motion resize dimensions must be positive")
    with Image.open(path) as image:
        oriented = apply_camera_transform(image, transform)
        gray = oriented.convert("L")
        resized = gray.resize(
            (resize_width, resize_height),
            resample=Image.Resampling.BILINEAR,
        )
        return np.asarray(resized, dtype=np.float32)


def mean_absolute_pixel_difference(
    previous: np.ndarray,
    current: np.ndarray,
) -> float:
    if previous.shape != current.shape:
        raise ValueError("motion arrays must have identical shape")
    if previous.ndim != 2 or current.ndim != 2:
        raise ValueError("motion arrays must be grayscale 2-D arrays")
    return float(np.mean(np.abs(current - previous)))


def fuse_motion_scores(front_score: float, wrist_score: float) -> float:
    if front_score < 0 or wrist_score < 0:
        raise ValueError("motion scores must be non-negative")
    return max(front_score, wrist_score)


def select_motion_frame(scores: list[MotionFrameScore]) -> MotionFrameScore:
    if not scores:
        raise ValueError("no motion scores supplied")
    return min(
        scores,
        key=lambda item: (-item.fused_motion_score, item.frame_index),
    )


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
    for index, camera_map in grouped.items():
        if set(camera_map) != {"front", "wrist"}:
            raise ValueError(f"incomplete camera pair at frame {index}")
    return grouped


def _verify_source_image(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"source image SHA256 mismatch for {path}: "
            f"{actual} != {expected_sha256}"
        )


def score_motion_window(
    *,
    episode_dir: Path,
    manifest: EpisodeManifest,
    records: list[FrameRecord],
    temporal_slice: TemporalSlice,
    resize_width: int = MOTION_RESIZE_WIDTH,
    resize_height: int = MOTION_RESIZE_HEIGHT,
    verify_source_sha256: bool = True,
) -> list[MotionFrameScore]:
    grouped = _group_records(records)
    transforms = {item.camera: item.transform for item in manifest.cameras}
    member_indices = list(
        range(
            temporal_slice.start_frame_index,
            temporal_slice.end_frame_index_exclusive,
        )
    )
    if not member_indices:
        raise ValueError("temporal slice contains no frame indices")
    for index in member_indices:
        if index not in grouped:
            raise ValueError(f"missing FrameRecord pair at frame {index}")

    first_index = member_indices[0]
    predecessor_index = first_index - 1 if first_index > min(grouped) else None
    per_camera_scores: dict[str, dict[int, float]] = {
        "front": {},
        "wrist": {},
    }

    for camera in ("front", "wrist"):
        previous_array: np.ndarray | None = None
        if predecessor_index is not None:
            predecessor = grouped[predecessor_index][camera]
            path = episode_dir / predecessor.image_relpath
            if verify_source_sha256:
                _verify_source_image(path, predecessor.image_sha256)
            previous_array = preprocess_motion_image(
                path,
                transform=transforms[camera],
                resize_width=resize_width,
                resize_height=resize_height,
            )

        for index in member_indices:
            record = grouped[index][camera]
            path = episode_dir / record.image_relpath
            if verify_source_sha256:
                _verify_source_image(path, record.image_sha256)
            current_array = preprocess_motion_image(
                path,
                transform=transforms[camera],
                resize_width=resize_width,
                resize_height=resize_height,
            )
            score = (
                0.0
                if previous_array is None
                else mean_absolute_pixel_difference(
                    previous_array,
                    current_array,
                )
            )
            per_camera_scores[camera][index] = score
            previous_array = current_array

    output = []
    for index in member_indices:
        front = grouped[index]["front"]
        wrist = grouped[index]["wrist"]
        if abs(front.timestamp_sec - wrist.timestamp_sec) > 1e-12:
            raise ValueError(f"pair timestamp mismatch at frame {index}")
        timestamp_sec = front.timestamp_sec
        if not (
            temporal_slice.start_sec
            <= timestamp_sec
            < temporal_slice.end_sec
        ):
            raise ValueError(f"frame {index} timestamp lies outside slice")
        front_score = per_camera_scores["front"][index]
        wrist_score = per_camera_scores["wrist"][index]
        output.append(
            MotionFrameScore(
                frame_index=index,
                timestamp_sec=timestamp_sec,
                front_motion_score=front_score,
                wrist_motion_score=wrist_score,
                fused_motion_score=fuse_motion_scores(
                    front_score,
                    wrist_score,
                ),
            )
        )
    return output


def build_motion_aware_selection(
    *,
    episode_dir: Path,
    manifest: EpisodeManifest,
    records: list[FrameRecord],
    temporal_slice: TemporalSlice,
    resize_width: int = MOTION_RESIZE_WIDTH,
    resize_height: int = MOTION_RESIZE_HEIGHT,
    verify_source_sha256: bool = True,
) -> tuple[MotionAwareSelection, list[MotionFrameScore]]:
    scores = score_motion_window(
        episode_dir=episode_dir,
        manifest=manifest,
        records=records,
        temporal_slice=temporal_slice,
        resize_width=resize_width,
        resize_height=resize_height,
        verify_source_sha256=verify_source_sha256,
    )
    selected = select_motion_frame(scores)
    grouped = _group_records(records)
    camera_scores = {
        "front": selected.front_motion_score,
        "wrist": selected.wrist_motion_score,
    }
    cameras = []
    for camera in ("front", "wrist"):
        record = grouped[selected.frame_index][camera]
        cameras.append(
            CameraMotionEvidence(
                camera=camera,
                frame_index=record.frame_index,
                timestamp_sec=record.timestamp_sec,
                source_timestamp_ns=record.source_timestamp_ns,
                source_age_ms=record.source_age_ms,
                camera_sequence=record.camera_sequence,
                image_relpath=record.image_relpath,
                image_sha256=record.image_sha256,
                motion_score=camera_scores[camera],
            )
        )

    return (
        MotionAwareSelection(
            slice_group_id=temporal_slice.slice_group_id,
            episode_id=temporal_slice.episode_id,
            start_sec=temporal_slice.start_sec,
            end_sec=temporal_slice.end_sec,
            start_frame_index=temporal_slice.start_frame_index,
            end_frame_index_exclusive=(
                temporal_slice.end_frame_index_exclusive
            ),
            selected_frame_index=selected.frame_index,
            selected_timestamp_sec=selected.timestamp_sec,
            front_motion_score=selected.front_motion_score,
            wrist_motion_score=selected.wrist_motion_score,
            fused_motion_score=selected.fused_motion_score,
            cameras=cameras,
        ),
        scores,
    )
