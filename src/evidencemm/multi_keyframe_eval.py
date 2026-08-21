from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidencemm.multi_keyframe_selection import MultiKeyframeSelection
from evidencemm.temporal_evidence import FrameRecord
from evidencemm.temporal_eval import (
    TemporalEventAnnotation,
    sample_timestamps,
)


class MultiKeyframeCoverageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: int = Field(ge=1)
    verified_events: int = Field(gt=0)
    covered_events: int = Field(ge=0)
    event_coverage: float = Field(ge=0.0, le=1.0)

    window_count: int = Field(gt=0)
    shared_sample_budget: int = Field(gt=0)
    evidence_image_budget: int = Field(gt=0)

    events: list[dict]
    selected_frames: list[dict]

    @model_validator(mode="after")
    def validate_budget(self):
        if self.shared_sample_budget != self.window_count * self.k:
            raise ValueError(
                "shared sample budget must equal window_count * k"
            )
        if self.evidence_image_budget != self.shared_sample_budget * 2:
            raise ValueError(
                "evidence image budget must equal shared samples * 2"
            )
        return self


def evaluate_multi_keyframe_event_coverage(
    *,
    annotations: list[TemporalEventAnnotation],
    records: list[FrameRecord],
    selections: list[MultiKeyframeSelection],
) -> MultiKeyframeCoverageResult:
    verified = [
        item
        for item in annotations
        if item.status == "verified"
    ]
    if not verified:
        raise ValueError("no verified temporal annotations")
    if not selections:
        raise ValueError("no multi-keyframe selections")

    k_values = {item.k for item in selections}
    if len(k_values) != 1:
        raise ValueError(
            "all selections in one evaluation must use the same k"
        )
    k = next(iter(k_values))

    slice_ids = [item.slice_group_id for item in selections]
    if len(set(slice_ids)) != len(slice_ids):
        raise ValueError(
            "duplicate multi-keyframe selection slice_group_id"
        )

    timestamps = sample_timestamps(records)

    flattened = [
        (selection, keyframe)
        for selection in selections
        for keyframe in selection.keyframes
    ]

    rows = []
    covered_count = 0

    for event in verified:
        assert event.start_frame_index is not None
        assert event.end_frame_index_inclusive is not None

        start = event.start_frame_index
        end = event.end_frame_index_inclusive

        if start not in timestamps or end not in timestamps:
            raise ValueError(
                f"{event.event_id} frame bounds outside episode"
            )

        start_sec = timestamps[start]
        end_sec = timestamps[end]
        center_sec = (start_sec + end_sec) / 2.0

        hits = [
            (selection, keyframe)
            for selection, keyframe in flattened
            if start <= keyframe.selected_frame_index <= end
        ]
        covered = bool(hits)
        if covered:
            covered_count += 1

        closest_selection, closest_keyframe = min(
            flattened,
            key=lambda pair: (
                abs(
                    pair[1].selected_timestamp_sec
                    - center_sec
                ),
                pair[1].selected_frame_index,
                pair[1].keyframe_rank,
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
                "duration_sec": end_sec - start_sec,
                "event_center_sec": center_sec,
                "covered": covered,
                "covering_slice_ids": [
                    selection.slice_group_id
                    for selection, _keyframe in hits
                ],
                "covering_selected_frames": [
                    keyframe.selected_frame_index
                    for _selection, keyframe in hits
                ],
                "closest_selected_slice_id": (
                    closest_selection.slice_group_id
                ),
                "closest_selected_keyframe_rank": (
                    closest_keyframe.keyframe_rank
                ),
                "closest_selected_frame": (
                    closest_keyframe.selected_frame_index
                ),
                "closest_selected_sec": (
                    closest_keyframe.selected_timestamp_sec
                ),
                "closest_selected_to_event_center_ms": (
                    abs(
                        closest_keyframe.selected_timestamp_sec
                        - center_sec
                    )
                    * 1000.0
                ),
                "evidence_cameras": event.evidence_cameras,
            }
        )

    selected_rows = [
        {
            "slice_group_id": selection.slice_group_id,
            "k": selection.k,
            "keyframe_rank": keyframe.keyframe_rank,
            "target_fraction": keyframe.target_fraction,
            "target_timestamp_sec": keyframe.target_timestamp_sec,
            "selected_frame_index": (
                keyframe.selected_frame_index
            ),
            "selected_timestamp_sec": (
                keyframe.selected_timestamp_sec
            ),
            "selection_error_ms": (
                keyframe.selection_error_ms
            ),
        }
        for selection in selections
        for keyframe in selection.keyframes
    ]

    shared_budget = len(selected_rows)

    return MultiKeyframeCoverageResult(
        k=k,
        verified_events=len(verified),
        covered_events=covered_count,
        event_coverage=covered_count / len(verified),
        window_count=len(selections),
        shared_sample_budget=shared_budget,
        evidence_image_budget=shared_budget * 2,
        events=rows,
        selected_frames=selected_rows,
    )
