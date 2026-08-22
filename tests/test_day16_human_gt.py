import pytest

from evidencemm.day16_human_gt import (
    HUMAN_REVIEWER,
    HumanGTValidationError,
    REVIEW_UNRESOLVED,
    REVIEW_VERIFIED,
    classify_review_disposition,
)


def verified_event():
    return {
        "event_id": "ep_event_01",
        "event_status": "verified",
        "failure_interval": {
            "start_frame": 10,
            "end_frame": 12,
            "start_sec": 1.0,
            "end_sec": 1.2,
        },
        "causal_diagnosis": (
            "insufficient_evidence"
        ),
        "supporting_robot_refs": [
            {
                "source_id": "ep",
            }
        ],
        "counterevidence_robot_refs": [],
        "confidence": 0.9,
    }


def unresolved_event():
    return {
        "event_id": "ep_event_01",
        "event_status": "draft",
        "failure_interval": None,
        "causal_diagnosis": None,
        "supporting_robot_refs": [],
        "counterevidence_robot_refs": [],
        "confidence": None,
    }


def test_verified_event_is_promoted():
    disposition = (
        classify_review_disposition(
            verified_event(),
            reviewer=HUMAN_REVIEWER,
            notes="human reviewed",
        )
    )

    assert disposition == REVIEW_VERIFIED


def test_reviewed_unresolved_event_is_explicit():
    disposition = (
        classify_review_disposition(
            unresolved_event(),
            reviewer=HUMAN_REVIEWER,
            notes=(
                "event unresolved after "
                "human evidence review"
            ),
        )
    )

    assert (
        disposition
        == REVIEW_UNRESOLVED
    )


def test_unreviewed_draft_is_rejected():
    with pytest.raises(
        HumanGTValidationError
    ):
        classify_review_disposition(
            unresolved_event(),
            reviewer=None,
            notes=None,
        )
