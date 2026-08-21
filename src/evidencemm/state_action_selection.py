from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidencemm.temporal_slicing import TemporalSlice


JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

STATE_SIGNAL = "observation"
ACTION_SIGNAL = "action"
STATE_CHANGE = "adjacent_rms"
ACTION_CHANGE = "adjacent_rms"
STATE_ACTION_FUSION = "max"
STATE_ACTION_TIE_BREAK = "lower_frame_index"
TRACKING_GAP_ROLE = "diagnostic_only"


class JointVector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shoulder_pan: float
    shoulder_lift: float
    elbow_flex: float
    wrist_flex: float
    wrist_roll: float
    gripper: float

    def ordered_values(self) -> list[float]:
        return [
            float(getattr(self, joint))
            for joint in JOINT_ORDER
        ]


class StateActionSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0)
    observation: JointVector
    action: JointVector
    tracking_error: JointVector


class StateActionFrameScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0)

    state_change_rms: float = Field(ge=0)
    action_change_rms: float = Field(ge=0)
    fused_state_action_score: float = Field(ge=0)
    tracking_gap_rms: float = Field(ge=0)

    observation: JointVector
    action: JointVector
    state_delta: JointVector
    action_delta: JointVector
    tracking_error: JointVector


class StateActionAwareSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slice_group_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    start_frame_index: int = Field(ge=0)
    end_frame_index_exclusive: int = Field(gt=0)

    selected_frame_index: int = Field(ge=0)
    selected_timestamp_sec: float = Field(ge=0)

    state_change_rms: float = Field(ge=0)
    action_change_rms: float = Field(ge=0)
    fused_state_action_score: float = Field(ge=0)
    tracking_gap_rms: float = Field(ge=0)

    observation: JointVector
    action: JointVector
    state_delta: JointVector
    action_delta: JointVector
    tracking_error: JointVector

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
        expected = max(self.state_change_rms, self.action_change_rms)
        if abs(self.fused_state_action_score - expected) > 1e-12:
            raise ValueError(
                "fused state/action score must equal max change score"
            )
        return self


def _float(row: dict[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"non-finite value in {name}")
    return value


def _joint_vector(
    row: dict[str, str],
    prefix: str,
) -> JointVector:
    return JointVector(
        **{
            joint: _float(row, f"{prefix}_{joint}")
            for joint in JOINT_ORDER
        }
    )


def validate_source_semantics(metadata_path: Path) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    semantics = metadata.get("semantics")
    if not isinstance(semantics, dict):
        raise ValueError("metadata missing semantics mapping")

    observation = str(semantics.get("observation", ""))
    action = str(semantics.get("action", ""))

    if "Follower Present_Position" not in observation:
        raise ValueError(
            "metadata observation semantics do not identify "
            "Follower Present_Position"
        )
    if "before the current action write" not in observation:
        raise ValueError(
            "metadata observation semantics do not identify "
            "pre-action timing"
        )
    if "Final absolute Goal_Position actually sent" not in action:
        raise ValueError(
            "metadata action semantics do not identify the final "
            "Goal_Position sent to the follower"
        )
    if "rate limiting" not in action:
        raise ValueError(
            "metadata action semantics do not identify post-limit action"
        )


def load_state_action_samples(
    path: Path,
    *,
    verify_tracking_error: bool = True,
) -> list[StateActionSample]:
    required = {"frame_index", "elapsed_ns"}
    for prefix in ("observation", "action", "tracking_error"):
        required.update(
            f"{prefix}_{joint}"
            for joint in JOINT_ORDER
        )

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError(
                "samples.csv missing state/action columns: "
                + repr(sorted(missing))
            )
        raw_rows = list(reader)

    if not raw_rows:
        raise ValueError("samples.csv contains no rows")

    samples: list[StateActionSample] = []
    for row in raw_rows:
        observation = _joint_vector(row, "observation")
        action = _joint_vector(row, "action")
        tracking_error = _joint_vector(row, "tracking_error")

        if verify_tracking_error:
            for joint in JOINT_ORDER:
                expected = abs(
                    float(getattr(action, joint))
                    - float(getattr(observation, joint))
                )
                actual = float(getattr(tracking_error, joint))
                if abs(actual - expected) > 1e-9:
                    raise ValueError(
                        "tracking_error invariant mismatch at "
                        f"frame {row['frame_index']} joint {joint}"
                    )

        samples.append(
            StateActionSample(
                frame_index=int(row["frame_index"]),
                timestamp_sec=int(row["elapsed_ns"]) / 1e9,
                observation=observation,
                action=action,
                tracking_error=tracking_error,
            )
        )

    indices = [sample.frame_index for sample in samples]
    if indices != list(range(len(samples))):
        raise ValueError(
            "state/action frame_index must be contiguous and zero-based"
        )

    timestamps = [sample.timestamp_sec for sample in samples]
    if any(
        later <= earlier
        for earlier, later in zip(timestamps, timestamps[1:])
    ):
        raise ValueError(
            "state/action timestamps must be strictly monotonic"
        )

    return samples


def rms(values: list[float]) -> float:
    if not values:
        raise ValueError("RMS requires at least one value")
    return math.sqrt(
        sum(value * value for value in values) / len(values)
    )


def subtract_vectors(
    current: JointVector,
    previous: JointVector,
) -> JointVector:
    return JointVector(
        **{
            joint: (
                float(getattr(current, joint))
                - float(getattr(previous, joint))
            )
            for joint in JOINT_ORDER
        }
    )


def zero_vector() -> JointVector:
    return JointVector(
        **{
            joint: 0.0
            for joint in JOINT_ORDER
        }
    )


def vector_rms(vector: JointVector) -> float:
    return rms(vector.ordered_values())


def fuse_state_action_changes(
    state_change_rms: float,
    action_change_rms: float,
) -> float:
    if state_change_rms < 0 or action_change_rms < 0:
        raise ValueError("change scores must be non-negative")
    return max(state_change_rms, action_change_rms)


def score_state_action_sample(
    *,
    current: StateActionSample,
    previous: StateActionSample | None,
) -> StateActionFrameScore:
    if previous is None:
        state_delta = zero_vector()
        action_delta = zero_vector()
    else:
        if current.frame_index != previous.frame_index + 1:
            raise ValueError(
                "state/action scoring requires adjacent frame indices"
            )
        state_delta = subtract_vectors(
            current.observation,
            previous.observation,
        )
        action_delta = subtract_vectors(
            current.action,
            previous.action,
        )

    state_change_rms = vector_rms(state_delta)
    action_change_rms = vector_rms(action_delta)

    tracking_gap = subtract_vectors(
        current.action,
        current.observation,
    )

    return StateActionFrameScore(
        frame_index=current.frame_index,
        timestamp_sec=current.timestamp_sec,
        state_change_rms=state_change_rms,
        action_change_rms=action_change_rms,
        fused_state_action_score=fuse_state_action_changes(
            state_change_rms,
            action_change_rms,
        ),
        tracking_gap_rms=vector_rms(tracking_gap),
        observation=current.observation,
        action=current.action,
        state_delta=state_delta,
        action_delta=action_delta,
        tracking_error=current.tracking_error,
    )


def select_state_action_frame(
    scores: list[StateActionFrameScore],
) -> StateActionFrameScore:
    if not scores:
        raise ValueError("no state/action scores supplied")

    return min(
        scores,
        key=lambda item: (
            -item.fused_state_action_score,
            item.frame_index,
        ),
    )


def score_state_action_window(
    *,
    samples: list[StateActionSample],
    temporal_slice: TemporalSlice,
) -> list[StateActionFrameScore]:
    by_index = {
        sample.frame_index: sample
        for sample in samples
    }

    member_indices = list(
        range(
            temporal_slice.start_frame_index,
            temporal_slice.end_frame_index_exclusive,
        )
    )
    if not member_indices:
        raise ValueError("temporal slice contains no state/action samples")

    scores: list[StateActionFrameScore] = []
    for index in member_indices:
        if index not in by_index:
            raise ValueError(
                f"missing state/action sample at frame {index}"
            )

        current = by_index[index]
        previous = (
            by_index.get(index - 1)
            if index > 0
            else None
        )

        if not (
            temporal_slice.start_sec
            <= current.timestamp_sec
            < temporal_slice.end_sec
        ):
            raise ValueError(
                f"frame {index} timestamp lies outside frozen slice"
            )

        scores.append(
            score_state_action_sample(
                current=current,
                previous=previous,
            )
        )

    return scores


def build_state_action_selection(
    *,
    samples: list[StateActionSample],
    temporal_slice: TemporalSlice,
) -> tuple[StateActionAwareSelection, list[StateActionFrameScore]]:
    scores = score_state_action_window(
        samples=samples,
        temporal_slice=temporal_slice,
    )
    selected = select_state_action_frame(scores)

    return (
        StateActionAwareSelection(
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
            state_change_rms=selected.state_change_rms,
            action_change_rms=selected.action_change_rms,
            fused_state_action_score=(
                selected.fused_state_action_score
            ),
            tracking_gap_rms=selected.tracking_gap_rms,
            observation=selected.observation,
            action=selected.action,
            state_delta=selected.state_delta,
            action_delta=selected.action_delta,
            tracking_error=selected.tracking_error,
        ),
        scores,
    )
