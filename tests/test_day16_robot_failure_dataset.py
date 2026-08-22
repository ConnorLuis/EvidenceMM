import pytest
from pydantic import ValidationError

from evidencemm.robot_failure_dataset import (
    AnomalyEvent,
    AnomalyReviewCase,
    CausalDiagnosis,
    ObservedFailureMode,
    ReviewStatus,
)


def test_day16_final_contract_is_event_level():
    review = AnomalyReviewCase(
        review_id="r1",
        episode_id="ep1",
        task_success=True,
        original_failure_reason="第一次抓取掉落，之后又碰到方块",
        events=[
            AnomalyEvent(
                event_id="ep1_event_01",
                observed_failure_mode=ObservedFailureMode.GRASP_DROP,
            ),
            AnomalyEvent(
                event_id="ep1_event_02",
                observed_failure_mode=ObservedFailureMode.POST_PLACE_COLLISION,
            ),
        ],
        diagnostic_manifest_path="m.json",
        diagnostic_frames_path="f.jsonl",
    )
    assert len(review.events) == 2
    assert review.observed_failure_modes == [
        ObservedFailureMode.GRASP_DROP,
        ObservedFailureMode.POST_PLACE_COLLISION,
    ]
    assert review.all_events_draft is True
    assert review.all_causal_diagnoses_unset is True


def test_verified_event_requires_interval_diagnosis_confidence_and_support():
    with pytest.raises(
        ValidationError,
        match="failure_interval",
    ):
        AnomalyEvent(
            event_id="ep1_event_01",
            observed_failure_mode=ObservedFailureMode.GRASP_DROP,
            event_status=ReviewStatus.VERIFIED,
            causal_diagnosis=CausalDiagnosis.GRIPPER_CLOSE_TIMING,
            confidence=0.9,
        )
