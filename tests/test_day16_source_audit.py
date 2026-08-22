from __future__ import annotations

import csv

import pytest
from pydantic import ValidationError

from evidencemm.robot_failure_dataset import (
    AnomalyReviewCase,
    AuditCategory,
    CausalDiagnosis,
    ObservedFailureMode,
    ReviewStatus,
    SourceAuditRecord,
    TrainingManifestRow,
    audit_training_row,
    build_anomaly_review_case,
    classify_training_row,
    load_training_manifest,
    required_anomaly_source_presence_complete,
    summarize_source_audit,
)


KNOWN_FAILURE_ROWS = [
    ("111613", True, True, False, False, "第一次抓取掉落，放入目标区后松开夹爪移出目标区碰到方块"),
    ("112058", True, True, False, False, "放入目标区后松开夹爪移出目标区碰到方块"),
    ("112633", True, True, False, False, "放入目标区后松开夹爪移出目标区碰到方块"),
    ("112859", True, True, False, False, "放入目标区后松开夹爪移出目标区碰到方块"),
    ("135518", True, True, False, False, "夹起放下过快，导致后续暂停太长"),
    ("140119", True, True, False, False, "夹起时推动方块"),
    ("141416", True, True, False, False, "方块在目标区上方掉落"),
    ("141657", True, True, False, False, "放入目标区后松开夹爪移出目标区碰到方块"),
    ("153125", False, False, False, False, "Follower突然断电一般，垂落"),
    ("154459", False, True, True, False, "wrist_duplicate_ratio: FAIL 和 cleanup_home: FAIL"),
    ("155139", True, True, False, False, "方块抓取时掉落"),
    ("155524", True, True, True, False, "wrist_duplicate_ratio: FAIL"),
    ("155814", False, False, False, False, "Follower突然断电一般，垂落"),
    ("161422", False, True, True, False, "wrist_duplicate_ratio: FAIL"),
    ("161647", False, True, True, False, "wrist_duplicate_ratio: FAIL"),
]


def row(
    episode_id,
    technical=True,
    task_success=True,
    demo_quality=True,
    training=True,
    reason=None,
):
    return TrainingManifestRow(
        episode_id=episode_id,
        technical_valid=technical,
        task_success=task_success,
        demo_quality_valid=demo_quality,
        valid_for_training=training,
        failure_reason=reason,
    )


def test_gb18030_training_manifest_reader(tmp_path):
    path = tmp_path / "training_manifest.csv"
    with path.open("w", encoding="gb18030", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "episode_id",
            "technical_valid",
            "task_success",
            "demo_quality_valid",
            "valid_for_training",
            "failure_reason",
            "notes",
        ])
        writer.writerow([
            "ep1",
            "True",
            "True",
            "False",
            "False",
            "方块抓取时掉落",
            "",
        ])

    rows = load_training_manifest(path, encoding="gb18030")
    assert len(rows) == 1
    assert rows[0].failure_reason == "方块抓取时掉落"


def test_current_75_row_audit_distribution(tmp_path):
    rows = [row(f"clean_{i:02d}") for i in range(60)]
    rows.extend(
        row(
            episode_id,
            technical,
            task_success,
            demo_quality,
            training,
            reason,
        )
        for (
            episode_id,
            technical,
            task_success,
            demo_quality,
            training,
            reason,
        ) in KNOWN_FAILURE_ROWS
    )

    records = [
        audit_training_row(item, dataset_root=tmp_path)
        for item in rows
    ]
    summary = summarize_source_audit(records)

    assert summary["total_rows"] == 75
    assert summary["categories"] == {
        "clean_reference_candidate": 60,
        "demo_quality_only": 1,
        "operation_anomaly": 8,
        "technical_exclusion": 6,
    }


def test_operation_anomaly_is_independent_of_task_success():
    item = row(
        "ep1",
        task_success=True,
        demo_quality=False,
        training=False,
        reason="方块抓取时掉落",
    )
    category, modes, eligible, _ = classify_training_row(item)

    assert category == AuditCategory.OPERATION_ANOMALY
    assert eligible is True
    assert ObservedFailureMode.GRASP_DROP in modes


def test_original_failure_reason_is_preserved(tmp_path):
    reason = "方块抓取时掉落"
    record = audit_training_row(
        row("ep1", demo_quality=False, training=False, reason=reason),
        dataset_root=tmp_path,
    )
    assert record.original_failure_reason == reason


def test_mixed_observed_failure_reason_has_two_modes(tmp_path):
    record = audit_training_row(
        row(
            "ep1",
            demo_quality=False,
            training=False,
            reason="第一次抓取掉落，放入目标区后松开夹爪移出目标区碰到方块",
        ),
        dataset_root=tmp_path,
    )
    assert record.observed_failure_modes == [
        ObservedFailureMode.GRASP_DROP,
        ObservedFailureMode.POST_PLACE_COLLISION,
    ]


def test_multi_event_review_case_preserves_two_events(tmp_path):
    record = audit_training_row(
        row(
            "ep1",
            demo_quality=False,
            training=False,
            reason="第一次抓取掉落，放入目标区后松开夹爪移出目标区碰到方块",
        ),
        dataset_root=tmp_path,
    )
    review = build_anomaly_review_case(
        record,
        diagnostic_manifest_root="m",
        diagnostic_processed_root="p",
    )
    assert len(review.events) == 2
    assert [
        event.observed_failure_mode
        for event in review.events
    ] == [
        ObservedFailureMode.GRASP_DROP,
        ObservedFailureMode.POST_PLACE_COLLISION,
    ]
    assert all(
        event.causal_diagnosis is None
        for event in review.events
    )
    assert all(
        event.event_status == ReviewStatus.DRAFT
        for event in review.events
    )


def test_technical_marker_wins_even_if_technical_valid_true():
    item = row(
        "ep1",
        technical=True,
        task_success=True,
        demo_quality=True,
        training=False,
        reason="wrist_duplicate_ratio: FAIL",
    )
    category, _, eligible, _ = classify_training_row(item)
    assert category == AuditCategory.TECHNICAL_EXCLUSION
    assert eligible is False


def test_demo_quality_only_is_not_operation_anomaly():
    item = row(
        "ep1",
        demo_quality=False,
        training=False,
        reason="夹起放下过快，导致后续暂停太长",
    )
    category, modes, _, _ = classify_training_row(item)
    assert category == AuditCategory.DEMO_QUALITY_ONLY
    assert modes == []


def test_unknown_reason_requires_manual_review():
    item = row("ep1", reason="此前未见的新问题")
    category, _, _, _ = classify_training_row(item)
    assert category == AuditCategory.MANUAL_REVIEW_REQUIRED


def test_missing_samples_in_technical_exclusion_does_not_fail_required_diagnostic_gate():
    anomaly = SourceAuditRecord(
        episode_id="anomaly",
        technical_valid=True,
        task_success=True,
        demo_quality_valid=False,
        valid_for_training=False,
        original_failure_reason="方块抓取时掉落",
        audit_category=AuditCategory.OPERATION_ANOMALY,
        operation_anomaly=True,
        observed_failure_modes=[ObservedFailureMode.GRASP_DROP],
        diagnostic_eligible=True,
        raw_episode_dir="/tmp/anomaly",
        raw_episode_dir_exists=True,
        metadata_exists=True,
        samples_csv_exists=True,
        front_dir_exists=True,
        wrist_dir_exists=True,
    )
    technical = SourceAuditRecord(
        episode_id="technical",
        technical_valid=False,
        task_success=False,
        demo_quality_valid=False,
        valid_for_training=False,
        original_failure_reason="Follower突然断电一般，垂落",
        audit_category=AuditCategory.TECHNICAL_EXCLUSION,
        operation_anomaly=False,
        observed_failure_modes=[],
        diagnostic_eligible=False,
        exclusion_reason="Follower突然断电一般，垂落",
        raw_episode_dir="/tmp/technical",
        raw_episode_dir_exists=True,
        metadata_exists=True,
        samples_csv_exists=False,
        front_dir_exists=True,
        wrist_dir_exists=True,
    )
    assert required_anomaly_source_presence_complete(
        [anomaly, technical]
    ) is True


def test_missing_samples_in_operation_anomaly_fails_required_diagnostic_gate():
    anomaly = SourceAuditRecord(
        episode_id="anomaly",
        technical_valid=True,
        task_success=True,
        demo_quality_valid=False,
        valid_for_training=False,
        original_failure_reason="方块抓取时掉落",
        audit_category=AuditCategory.OPERATION_ANOMALY,
        operation_anomaly=True,
        observed_failure_modes=[ObservedFailureMode.GRASP_DROP],
        diagnostic_eligible=True,
        raw_episode_dir="/tmp/anomaly",
        raw_episode_dir_exists=True,
        metadata_exists=True,
        samples_csv_exists=False,
        front_dir_exists=True,
        wrist_dir_exists=True,
    )
    assert required_anomaly_source_presence_complete([anomaly]) is False
