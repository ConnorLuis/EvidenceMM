from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidencemm.state_action_selection import StateActionAwareSelection
from evidencemm.temporal_evidence import FrameRecord
from evidencemm.temporal_eval import (
    TemporalEventAnnotation,
    sample_timestamps,
)


class StateActionCoverageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verified_events: int = Field(gt=0)
    covered_events: int = Field(ge=0)
    event_coverage: float = Field(ge=0.0, le=1.0)
    shared_sample_budget: int = Field(gt=0)
    evidence_image_budget: int = Field(gt=0)
    events: list[dict]
    selected_frames: list[dict]

    @model_validator(mode="after")
    def validate_budget(self):
        if self.evidence_image_budget != self.shared_sample_budget * 2:
            raise ValueError(
                "evidence image budget must be exactly two cameras "
                "per shared sample"
            )
        return self


def evaluate_state_action_event_coverage(
    *,
    annotations: list[TemporalEventAnnotation],
    records: list[FrameRecord],
    selections: list[StateActionAwareSelection],
) -> StateActionCoverageResult:
    verified = [
        item
        for item in annotations
        if item.status == "verified"
    ]
    if not verified:
        raise ValueError("no verified temporal annotations")
    if not selections:
        raise ValueError("no state/action-aware selections")

    timestamps = sample_timestamps(records)

    slice_ids = [item.slice_group_id for item in selections]
    if len(set(slice_ids)) != len(slice_ids):
        raise ValueError(
            "duplicate state/action selection slice_group_id"
        )

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
            item
            for item in selections
            if start <= item.selected_frame_index <= end
        ]
        covered = bool(hits)
        if covered:
            covered_count += 1

        closest = min(
            selections,
            key=lambda item: (
                abs(item.selected_timestamp_sec - center_sec),
                item.selected_frame_index,
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
                    item.slice_group_id
                    for item in hits
                ],
                "covering_selected_frames": [
                    item.selected_frame_index
                    for item in hits
                ],
                "closest_selected_slice_id": closest.slice_group_id,
                "closest_selected_frame": closest.selected_frame_index,
                "closest_selected_sec": closest.selected_timestamp_sec,
                "closest_selected_to_event_center_ms": (
                    abs(
                        closest.selected_timestamp_sec
                        - center_sec
                    )
                    * 1000.0
                ),
                "evidence_cameras": event.evidence_cameras,
            }
        )

    selected_rows = [
        {
            "slice_group_id": item.slice_group_id,
            "start_sec": item.start_sec,
            "end_sec": item.end_sec,
            "selected_frame_index": item.selected_frame_index,
            "selected_timestamp_sec": item.selected_timestamp_sec,
            "state_change_rms": item.state_change_rms,
            "action_change_rms": item.action_change_rms,
            "fused_state_action_score": (
                item.fused_state_action_score
            ),
            "dominant_change_channel": (
                "state"
                if item.state_change_rms > item.action_change_rms
                else "action"
                if item.action_change_rms > item.state_change_rms
                else "tie"
            ),
            "tracking_gap_rms": item.tracking_gap_rms,
        }
        for item in selections
    ]

    return StateActionCoverageResult(
        verified_events=len(verified),
        covered_events=covered_count,
        event_coverage=covered_count / len(verified),
        shared_sample_budget=len(selections),
        evidence_image_budget=len(selections) * 2,
        events=rows,
        selected_frames=selected_rows,
    )
