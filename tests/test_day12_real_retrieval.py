from __future__ import annotations

from evidencemm.robot_candidate_retrieval import (
    parse_robot_signal_query,
    rank_robot_signal_samples,
    score_robot_signal_sample,
)
from evidencemm.state_action_selection import JointVector, StateActionSample


def vector(
    *,
    shoulder_pan: float = 0.0,
    shoulder_lift: float = 0.0,
    elbow_flex: float = 0.0,
    wrist_flex: float = 0.0,
    wrist_roll: float = 0.0,
    gripper: float = 0.0,
) -> JointVector:
    return JointVector(
        shoulder_pan=shoulder_pan,
        shoulder_lift=shoulder_lift,
        elbow_flex=elbow_flex,
        wrist_flex=wrist_flex,
        wrist_roll=wrist_roll,
        gripper=gripper,
    )


def sample(
    frame_index: int,
    *,
    observation: JointVector | None = None,
    action: JointVector | None = None,
    tracking_error: JointVector | None = None,
) -> StateActionSample:
    return StateActionSample(
        frame_index=frame_index,
        timestamp_sec=float(frame_index),
        observation=observation or vector(),
        action=action or vector(),
        tracking_error=tracking_error or vector(),
    )


def test_profile_uses_exact_canonical_gripper_action_terms():
    profile = parse_robot_signal_query("robot gripper action")
    assert profile.joints == ("gripper",)
    assert profile.signals == ("action",)
    assert profile.explicit_joint_terms is True
    assert profile.explicit_signal_terms is True


def test_profile_does_not_inject_chinese_synonyms():
    profile = parse_robot_signal_query("机器人夹爪动作")
    assert profile.joints == (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
    assert profile.signals == ("observation", "action")
    assert profile.explicit_joint_terms is False
    assert profile.explicit_signal_terms is False


def test_gripper_action_query_ignores_larger_other_joint_change():
    samples = [
        sample(0),
        sample(
            1,
            action=vector(
                shoulder_pan=100.0,
                gripper=5.0,
            ),
        ),
        sample(
            2,
            action=vector(
                shoulder_pan=0.0,
                gripper=1.0,
            ),
        ),
    ]

    _, hits = rank_robot_signal_samples(
        samples=samples,
        query="gripper action",
        top_k=3,
    )

    assert hits[0].frame_index == 1
    assert hits[0].raw_score == 5.0


def test_shoulder_action_query_can_select_other_frame():
    samples = [
        sample(0),
        sample(
            1,
            action=vector(
                shoulder_pan=1.0,
                gripper=20.0,
            ),
        ),
        sample(
            2,
            action=vector(
                shoulder_pan=10.0,
                gripper=20.0,
            ),
        ),
    ]

    _, hits = rank_robot_signal_samples(
        samples=samples,
        query="shoulder_pan action",
        top_k=1,
    )

    assert hits[0].frame_index == 2
    assert hits[0].raw_score == 9.0


def test_tracking_error_query_uses_current_gap_not_delta():
    current = sample(
        1,
        tracking_error=vector(gripper=7.0),
    )
    profile = parse_robot_signal_query(
        "gripper tracking_error"
    )

    raw_score, scores = score_robot_signal_sample(
        current=current,
        previous=sample(0),
        profile=profile,
    )

    assert raw_score == 7.0
    assert scores == {"tracking_error": 7.0}


def test_equal_scores_break_tie_by_lower_frame_index():
    samples = [
        sample(0),
        sample(
            1,
            action=vector(gripper=5.0),
        ),
        sample(
            2,
            action=vector(gripper=0.0),
        ),
    ]

    _, hits = rank_robot_signal_samples(
        samples=samples,
        query="gripper action",
        top_k=2,
    )

    assert [hit.frame_index for hit in hits] == [1, 2]
    assert hits[0].raw_score == hits[1].raw_score
