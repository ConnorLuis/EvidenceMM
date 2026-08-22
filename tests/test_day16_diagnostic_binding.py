from __future__ import annotations

from evidencemm.diagnostic_episode_binding import (
    DiagnosticBindingError,
    diagnostic_binding_allowed,
    validate_diagnostic_recording_acceptance,
)
from evidencemm.robot_failure_dataset import (
    AuditCategory,
    ObservedFailureMode,
    SourceAuditRecord,
)


def audit(
    *,
    category,
    technical,
    eligible,
    training,
):
    return SourceAuditRecord(
        episode_id="ep1",
        technical_valid=technical,
        task_success=True,
        demo_quality_valid=False,
        valid_for_training=training,
        original_failure_reason=(
            "方块抓取时掉落"
            if category
            == AuditCategory.OPERATION_ANOMALY
            else "wrist_duplicate_ratio: FAIL"
        ),
        audit_category=category,
        operation_anomaly=(
            category
            == AuditCategory.OPERATION_ANOMALY
        ),
        observed_failure_modes=(
            [
                ObservedFailureMode.GRASP_DROP
            ]
            if category
            == AuditCategory.OPERATION_ANOMALY
            else []
        ),
        diagnostic_eligible=eligible,
        exclusion_reason=(
            None
            if category
            != AuditCategory.TECHNICAL_EXCLUSION
            else "technical"
        ),
        raw_episode_dir="/tmp/ep1",
        raw_episode_dir_exists=True,
        metadata_exists=True,
        samples_csv_exists=True,
        front_dir_exists=True,
        wrist_dir_exists=True,
    )


def test_diagnostic_binding_accepts_operation_anomaly_not_for_training():
    item = audit(
        category=(
            AuditCategory.OPERATION_ANOMALY
        ),
        technical=True,
        eligible=True,
        training=False,
    )
    assert (
        diagnostic_binding_allowed(item)
        is True
    )


def test_diagnostic_binding_rejects_technical_exclusion():
    item = audit(
        category=(
            AuditCategory.TECHNICAL_EXCLUSION
        ),
        technical=False,
        eligible=False,
        training=False,
    )
    assert (
        diagnostic_binding_allowed(item)
        is False
    )



def test_completed_invalid_cleanup_only_is_accepted_for_diagnosis():
    metadata = {
        "status": "completed_invalid",
        "completed": True,
        "aborted": False,
        "error": None,
        "checks": {
            "normal_recording_completion": True,
            "sample_count_exact": True,
            "csv_row_count_exact": True,
            "front_image_count_exact": True,
            "wrist_image_count_exact": True,
            "timestamp_strictly_monotonic": True,
            "front_duplicate_ratio": True,
            "wrist_duplicate_ratio": True,
            "cleanup_home": False,
            "cleanup_park": True,
        },
    }

    failed = (
        validate_diagnostic_recording_acceptance(
            metadata
        )
    )

    assert failed == (
        "cleanup_home",
    )


def test_diagnostic_gate_rejects_camera_quality_failure():
    metadata = {
        "status": "completed_invalid",
        "completed": True,
        "aborted": False,
        "error": None,
        "checks": {
            "normal_recording_completion": True,
            "wrist_duplicate_ratio": False,
            "cleanup_home": True,
        },
    }

    import pytest

    with pytest.raises(
        DiagnosticBindingError,
        match="wrist_duplicate_ratio",
    ):
        validate_diagnostic_recording_acceptance(
            metadata
        )


def test_diagnostic_gate_rejects_aborted_recording():
    metadata = {
        "status": "aborted",
        "completed": True,
        "aborted": True,
        "error": None,
        "checks": {
            "normal_recording_completion": True,
        },
    }

    import pytest

    with pytest.raises(
        DiagnosticBindingError,
        match="must not be aborted",
    ):
        validate_diagnostic_recording_acceptance(
            metadata
        )


def test_diagnostic_gate_rejects_incomplete_recording():
    metadata = {
        "status": "recording",
        "completed": False,
        "aborted": False,
        "error": None,
        "checks": {},
    }

    import pytest

    with pytest.raises(
        DiagnosticBindingError,
        match="must be completed",
    ):
        validate_diagnostic_recording_acceptance(
            metadata
        )
