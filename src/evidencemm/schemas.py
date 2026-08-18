from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class SourceType(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    VIDEO = "video"
    ROBOT_STATE = "robot_state"
    ROBOT_ACTION = "robot_action"


class NormalizedBBox(BaseModel):
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_order(self) -> "NormalizedBBox":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bbox must satisfy x1 < x2 and y1 < y2")
        return self


class EvidenceRef(BaseModel):
    source_id: str
    source_type: SourceType
    page_number: int | None = Field(default=None, ge=1)
    time_start_sec: float | None = Field(default=None, ge=0.0)
    time_end_sec: float | None = Field(default=None, ge=0.0)
    frame_index: int | None = Field(default=None, ge=0)
    camera: Literal["wrist", "front"] | None = None
    bbox: NormalizedBBox | None = None
    note: str | None = None

    @model_validator(mode="after")
    def validate_time_range(self) -> "EvidenceRef":
        if (
            self.time_start_sec is not None
            and self.time_end_sec is not None
            and self.time_end_sec < self.time_start_sec
        ):
            raise ValueError("time_end_sec must be >= time_start_sec")
        return self


class RobotEpisodeManifest(BaseModel):
    episode_id: str
    task: str
    wrist_video: str
    front_video: str
    state_file: str
    action_file: str | None = None
    success: bool | None = None
    failure_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalCase(BaseModel):
    case_id: str
    question: str
    input_ids: list[str]
    answerable: bool | None = None
    expected_answer: str | None = None
    expected_evidence: list[EvidenceRef] = Field(default_factory=list)
    annotation_status: Literal["draft", "verified"] = "draft"
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def verified_requires_ground_truth(self) -> "EvalCase":
        if self.annotation_status == "verified":
            if self.answerable is None:
                raise ValueError("verified case requires answerable label")
            if self.answerable and not self.expected_answer:
                raise ValueError("answerable verified case requires expected_answer")
        return self


class BaselineRecord(BaseModel):
    model_name: str
    question: str
    input_files: list[str]
    response: str
    latency_sec: float = Field(ge=0.0)
    peak_gpu_memory_mb: float | None = Field(default=None, ge=0.0)
