from __future__ import annotations

from evidencemm.state_action_selection import (
    JointVector,
    StateActionFrameScore,
    StateActionSample,
    fuse_state_action_changes,
    score_state_action_sample,
    select_state_action_frame,
    subtract_vectors,
    vector_rms,
)


def vector(value: float) -> JointVector:
    return JointVector(
        shoulder_pan=value,
        shoulder_lift=value,
        elbow_flex=value,
        wrist_flex=value,
        wrist_roll=value,
        gripper=value,
    )


def sample(
    frame_index: int,
    observation: float,
    action: float,
) -> StateActionSample:
    error = abs(action - observation)
    return StateActionSample(
        frame_index=frame_index,
        timestamp_sec=float(frame_index),
        observation=vector(observation),
        action=vector(action),
        tracking_error=vector(error),
    )


def test_joint_vector_uses_frozen_six_joint_order():
    item = JointVector(
        shoulder_pan=1.0,
        shoulder_lift=2.0,
        elbow_flex=3.0,
        wrist_flex=4.0,
        wrist_roll=5.0,
        gripper=6.0,
    )
    assert item.ordered_values() == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ]


def test_vector_rms_has_no_joint_weighting():
    assert vector_rms(vector(3.0)) == 3.0


def test_state_action_fusion_uses_max_change():
    assert fuse_state_action_changes(0.2, 0.7) == 0.7
    assert fuse_state_action_changes(0.9, 0.4) == 0.9


def test_adjacent_state_action_score_is_exact():
    previous = sample(0, observation=1.0, action=2.0)
    current = sample(1, observation=4.0, action=6.0)

    result = score_state_action_sample(
        current=current,
        previous=previous,
    )

    assert result.state_change_rms == 3.0
    assert result.action_change_rms == 4.0
    assert result.fused_state_action_score == 4.0
    assert result.tracking_gap_rms == 2.0


def test_first_episode_sample_has_zero_change():
    result = score_state_action_sample(
        current=sample(0, observation=5.0, action=8.0),
        previous=None,
    )
    assert result.state_change_rms == 0.0
    assert result.action_change_rms == 0.0
    assert result.fused_state_action_score == 0.0
    assert result.tracking_gap_rms == 3.0


def test_state_action_selection_tie_breaks_to_lower_frame():
    scores = [
        StateActionFrameScore(
            frame_index=11,
            timestamp_sec=0.7,
            state_change_rms=0.8,
            action_change_rms=0.2,
            fused_state_action_score=0.8,
            tracking_gap_rms=1.0,
            observation=vector(1.0),
            action=vector(2.0),
            state_delta=vector(0.8),
            action_delta=vector(0.2),
            tracking_error=vector(1.0),
        ),
        StateActionFrameScore(
            frame_index=10,
            timestamp_sec=0.6,
            state_change_rms=0.1,
            action_change_rms=0.8,
            fused_state_action_score=0.8,
            tracking_gap_rms=1.0,
            observation=vector(1.0),
            action=vector(2.0),
            state_delta=vector(0.1),
            action_delta=vector(0.8),
            tracking_error=vector(1.0),
        ),
    ]
    assert select_state_action_frame(scores).frame_index == 10
