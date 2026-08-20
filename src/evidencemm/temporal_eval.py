from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidencemm.temporal_evidence import FrameRecord
from evidencemm.temporal_slicing import TemporalSlice


class TemporalEventAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    description: str = Field(min_length=1)
    status: Literal["draft", "verified"]

    start_frame_index: int | None = Field(
        default=None,
        ge=0,
    )
    end_frame_index_inclusive: int | None = Field(
        default=None,
        ge=0,
    )

    evidence_cameras: list[
        Literal["front", "wrist"]
    ]
    notes: str = ""

    @model_validator(mode="after")
    def validate_verified_interval(self):
        if self.status == "verified":
            if (
                self.start_frame_index is None
                or self.end_frame_index_inclusive is None
            ):
                raise ValueError(
                    "verified annotation requires frame interval"
                )
            if (
                self.end_frame_index_inclusive
                < self.start_frame_index
            ):
                raise ValueError(
                    "verified event end must be >= start"
                )

        if not self.evidence_cameras:
            raise ValueError(
                "at least one evidence camera is required"
            )

        return self


def load_annotations(
    path: Path,
) -> list[TemporalEventAnnotation]:
    return [
        TemporalEventAnnotation.model_validate(
            json.loads(line)
        )
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def save_annotations(
    annotations: list[TemporalEventAnnotation],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        "\n".join(
            json.dumps(
                item.model_dump(),
                ensure_ascii=False,
            )
            for item in annotations
        )
        + ("\n" if annotations else ""),
        encoding="utf-8",
        newline="\n",
    )


def sample_timestamps(
    records: list[FrameRecord],
) -> dict[int, float]:
    front = {
        item.frame_index: item.timestamp_sec
        for item in records
        if item.camera == "front"
    }
    wrist = {
        item.frame_index: item.timestamp_sec
        for item in records
        if item.camera == "wrist"
    }

    if front.keys() != wrist.keys():
        raise ValueError(
            "front/wrist frame-index sets differ"
        )

    for index in front:
        if abs(front[index] - wrist[index]) > 1e-12:
            raise ValueError(
                f"sample timestamp mismatch at frame {index}"
            )

    return front


def evaluate_event_coverage(
    *,
    annotations: list[TemporalEventAnnotation],
    records: list[FrameRecord],
    slices: list[TemporalSlice],
) -> dict:
    verified = [
        item
        for item in annotations
        if item.status == "verified"
    ]
    if not verified:
        raise ValueError(
            "no verified temporal annotations"
        )

    timestamps = sample_timestamps(records)

    rows = []
    covered_count = 0

    for event in verified:
        assert event.start_frame_index is not None
        assert event.end_frame_index_inclusive is not None

        start = event.start_frame_index
        end = event.end_frame_index_inclusive

        if start not in timestamps or end not in timestamps:
            raise ValueError(
                f"{event.event_id} frame bounds "
                "outside episode"
            )

        start_sec = timestamps[start]
        end_sec = timestamps[end]
        center_sec = (
            start_sec + end_sec
        ) / 2.0

        hits = [
            item
            for item in slices
            if (
                start
                <= item.midpoint_frame_index
                <= end
            )
        ]
        covered = bool(hits)
        if covered:
            covered_count += 1

        closest = min(
            slices,
            key=lambda item: (
                abs(
                    item.midpoint_timestamp_sec
                    - center_sec
                ),
                item.midpoint_frame_index,
            ),
        )

        rows.append(
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "description": event.description,
                "start_frame_index": start,
                "end_frame_index_inclusive": end,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": (
                    end_sec - start_sec
                ),
                "event_center_sec": center_sec,
                "covered": covered,
                "covering_slice_ids": [
                    item.slice_group_id
                    for item in hits
                ],
                "covering_midpoint_frames": [
                    item.midpoint_frame_index
                    for item in hits
                ],
                "closest_midpoint_slice_id":
                    closest.slice_group_id,
                "closest_midpoint_frame":
                    closest.midpoint_frame_index,
                "closest_midpoint_sec":
                    closest.midpoint_timestamp_sec,
                "closest_midpoint_to_event_center_ms":
                    abs(
                        closest.midpoint_timestamp_sec
                        - center_sec
                    )
                    * 1000.0,
                "evidence_cameras":
                    event.evidence_cameras,
            }
        )

    return {
        "verified_events": len(verified),
        "covered_events": covered_count,
        "event_coverage": (
            covered_count / len(verified)
        ),
        "events": rows,
    }
