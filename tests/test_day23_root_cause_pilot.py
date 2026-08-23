\
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from evidencemm.root_cause_pilot import (
    PilotRecord,
    derive_motion_and_gripper_proxy,
    split_semicolon,
)


def _sample(frame, arm=0.0, gripper=0.0):
    action = SimpleNamespace(
        shoulder_pan=arm,
        shoulder_lift=arm,
        elbow_flex=arm,
        wrist_flex=arm,
        wrist_roll=arm,
        gripper=gripper,
    )
    return SimpleNamespace(
        frame_index=frame,
        timestamp_sec=frame / 15.0,
        action=action,
    )


def _record(**updates):
    base = dict(
        schema_version="evidencemm_day23_pilot_record_v1",
        pilot_row_id="p23_g01_target",
        pilot_group_id="p23_g01",
        sequence_order=2,
        intensity_rank=1,
        intensity_label="mild",
        pilot_role="controlled_cause",
        planned_physical_cause="target_offset_or_perception",
        planned_intervention_type="object_target_pose_offset",
        episode_id="20260823_000001",
        raw_episode_relpath="20260823_000001",
        recorder_overall_pass=True,
        failed_checks=(),
        task_success=False,
        intervention_predeclared=True,
        intervention_applied=True,
        single_primary_intervention=True,
        parameter_direction="right",
        parameter_value=10.0,
        parameter_unit="mm",
        changed_factor_observable=True,
        observable_modalities=("front", "wrist"),
        gripper_transition_verified_as_grasp_close=None,
        safety_abort=False,
        hardware_fault=False,
        operator_notes="pilot",
    )
    base.update(updates)
    return PilotRecord(**base)


def test_split_semicolon():
    assert split_semicolon("front;wrist; action ") == (
        "front", "wrist", "action"
    )


def test_target_record_accepts_positive_mm():
    record = _record()
    assert record.parameter_value == 10.0


def test_target_record_rejects_non_mm():
    with pytest.raises(ValidationError, match="unit must be mm"):
        _record(parameter_unit="deg")


def test_control_rejects_intervention():
    with pytest.raises(ValidationError, match="clean control cannot apply"):
        _record(
            pilot_row_id="p23_g01_clean",
            sequence_order=1,
            intensity_rank=0,
            intensity_label="control",
            pilot_role="clean_control",
            planned_physical_cause="none_clean",
            planned_intervention_type="none",
            intervention_predeclared=False,
            intervention_applied=True,
            parameter_direction="",
            parameter_value=None,
            parameter_unit="",
        )


def test_pass_record_rejects_failed_checks():
    with pytest.raises(ValidationError, match="PASS recorder"):
        _record(failed_checks=("wrist_fps",))


def test_gripper_requires_early_or_late():
    with pytest.raises(ValidationError, match="early or late"):
        _record(
            pilot_row_id="p23_g01_gripper",
            sequence_order=3,
            planned_physical_cause="gripper_close_timing",
            planned_intervention_type="manual_gripper_close_timing_shift",
            parameter_direction="random",
            parameter_value=None,
            parameter_unit="",
            gripper_transition_verified_as_grasp_close=True,
        )


def test_gripper_numeric_is_derived_not_manual():
    with pytest.raises(ValidationError, match="derived from samples"):
        _record(
            pilot_row_id="p23_g01_gripper",
            sequence_order=3,
            planned_physical_cause="gripper_close_timing",
            planned_intervention_type="manual_gripper_close_timing_shift",
            parameter_direction="early",
            parameter_value=5.0,
            parameter_unit="frames",
            gripper_transition_verified_as_grasp_close=True,
        )


def test_proxy_detects_motion_and_gripper_transition():
    samples = []
    for frame in range(100):
        arm = 0.0 if frame < 20 else 5.0
        gripper = 0.0 if frame < 40 else 10.0
        samples.append(_sample(frame, arm=arm, gripper=gripper))
    proxy = derive_motion_and_gripper_proxy(
        samples,
        stable_prefix_frames=10,
        arm_motion_rms_threshold_deg=2.0,
        gripper_major_transition_min_deg=3.0,
        gripper_major_transition_fraction_of_range=0.2,
        sustain_frames=3,
    )
    assert proxy["motion_start_frame"] == 20
    assert proxy["gripper_major_transition_frame"] == 40
    assert proxy["gripper_phase_frames_from_motion_start"] == 20


def test_proxy_can_return_none_when_no_motion():
    samples = [_sample(frame) for frame in range(60)]
    proxy = derive_motion_and_gripper_proxy(
        samples,
        stable_prefix_frames=10,
        arm_motion_rms_threshold_deg=2.0,
        gripper_major_transition_min_deg=3.0,
        gripper_major_transition_fraction_of_range=0.2,
        sustain_frames=3,
    )
    assert proxy["motion_start_frame"] is None
    assert proxy["gripper_major_transition_frame"] is None
    assert proxy["gripper_phase_frames_from_motion_start"] is None
