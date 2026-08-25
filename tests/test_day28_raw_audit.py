from __future__ import annotations
from pathlib import Path
from evidencemm.day24_target_collection import TechnicalAudit
from evidencemm.day28_raw_audit import (
    EXPECTED_ANONYMOUS_TECHNICAL_EXCLUSION_TREE_SHA256,
    EXPECTED_REGISTERED_RAW_SET_SHA256,
    EpisodeFingerprint,
    FORBIDDEN_MANIFEST_FIELDS,
    MANIFEST_FIELDS,
    aggregate_content_set_sha256,
    aggregate_registered_set_sha256,
    classify_registered_records,
    fingerprint_episode_tree,
    manifest_fields_are_label_safe,
    manifest_row,
)

def test_manifest_schema_contains_no_admin_labels():
    assert manifest_fields_are_label_safe(MANIFEST_FIELDS)
    assert not (set(MANIFEST_FIELDS) & FORBIDDEN_MANIFEST_FIELDS)

def test_registered_aggregate_sha_is_order_stable():
    a = {"20260824_000002": "b" * 64, "20260824_000001": "a" * 64}
    b = {"20260824_000001": "a" * 64, "20260824_000002": "b" * 64}
    assert aggregate_registered_set_sha256(a) == aggregate_registered_set_sha256(b)

def test_content_set_sha_is_order_stable():
    a = ["a" * 64, "b" * 64, "c" * 64]
    assert aggregate_content_set_sha256(a) == aggregate_content_set_sha256(list(reversed(a)))

def test_tree_fingerprint_changes_with_filename(tmp_path: Path):
    ep = tmp_path / "episode"; ep.mkdir(); (ep / "front").mkdir()
    (ep / "metadata.json").write_text("{}", encoding="utf-8")
    (ep / "samples.csv").write_text("x\n1\n", encoding="utf-8")
    (ep / "front" / "000001.jpg").write_bytes(b"abc")
    first = fingerprint_episode_tree(ep)
    (ep / "front" / "000001.jpg").rename(ep / "front" / "000002.jpg")
    second = fingerprint_episode_tree(ep)
    assert first.tree_sha256 != second.tree_sha256

def test_tree_fingerprint_changes_with_content(tmp_path: Path):
    ep = tmp_path / "episode"; ep.mkdir()
    (ep / "metadata.json").write_text("{}", encoding="utf-8")
    (ep / "samples.csv").write_text("x\n1\n", encoding="utf-8")
    first = fingerprint_episode_tree(ep)
    (ep / "samples.csv").write_text("x\n2\n", encoding="utf-8")
    assert first.tree_sha256 != fingerprint_episode_tree(ep).tree_sha256

def test_manifest_row_is_source_only():
    record = {
        "episode_id": "20260824_999999", "raw_episode_relpath": "20260824_999999",
        "pair_group_id": "must_not_leak", "plan_row_id": "must_not_leak",
        "selected_canonical": "true", "experimental_valid": "true", "task_success": "false",
    }
    audit = TechnicalAudit(
        episode_id="20260824_999999", raw_episode_relpath="20260824_999999",
        recorder_script_version="episode_recorder_v7", technical_valid=True,
        recorder_overall_pass=True, failed_checks=[], sample_count=900,
        csv_row_count=900, front_image_count=900, wrist_image_count=900,
        duration_seconds=59.93, max_tracking_error=10.0,
    )
    fp = EpisodeFingerprint(1802, 123, "a" * 64, "b" * 64, "c" * 64)
    row = manifest_row(record=record, audit=audit, fingerprint=fp)
    assert tuple(row.keys()) == MANIFEST_FIELDS
    for forbidden in ("pair_group_id", "plan_row_id", "selected_canonical", "experimental_valid", "task_success"):
        assert forbidden not in row

def test_registered_record_classification_expected_shape():
    rows = []
    for i in range(90):
        rows.append({
            "episode_id": f"c{i:03d}", "raw_episode_relpath": f"c{i:03d}",
            "selected_canonical": "true", "technical_valid": "true",
            "experimental_valid": "true", "exclusion_reason": "",
        })
    for i in range(2):
        rows.append({
            "episode_id": f"x{i:03d}", "raw_episode_relpath": f"x{i:03d}",
            "selected_canonical": "false", "technical_valid": "true",
            "experimental_valid": "false", "exclusion_reason": "retained_noncanonical",
        })
    got = classify_registered_records(rows)
    assert got["registered_attempt_count"] == 92
    assert got["registered_canonical_count"] == 90
    assert got["registered_noncanonical_count"] == 2
    assert got["registered_technical_exclusion_count"] == 0
    assert got["registered_experimental_exclusion_count"] == 2
    assert got["duplicate_episode_id_count"] == 0
    assert got["duplicate_relpath_count"] == 0

def test_frozen_source_hash_constants_are_sha256():
    assert len(EXPECTED_REGISTERED_RAW_SET_SHA256) == 64
    int(EXPECTED_REGISTERED_RAW_SET_SHA256, 16)
    assert len(EXPECTED_ANONYMOUS_TECHNICAL_EXCLUSION_TREE_SHA256) == 64
    int(EXPECTED_ANONYMOUS_TECHNICAL_EXCLUSION_TREE_SHA256, 16)
