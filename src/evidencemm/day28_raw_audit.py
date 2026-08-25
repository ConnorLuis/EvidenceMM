from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from evidencemm.day24_target_collection import (
    DAY22_COLLECTION_PLAN_SHA256,
    TechnicalAudit,
    audit_episode,
    file_sha256,
    parse_bool,
)

FROZEN_DAY27_COMMIT = "eaa29a3ebc9f41fa26ffa6de3291c6a28d93a4cd"
EXPECTED_REGISTERED_ATTEMPTS = 92
EXPECTED_REGISTERED_CANONICAL = 90
EXPECTED_REGISTERED_NONCANONICAL = 2
EXPECTED_REGISTERED_TECHNICAL_EXCLUSIONS = 0
EXPECTED_REGISTERED_EXPERIMENTAL_EXCLUSIONS = 2
EXPECTED_RAW_ROOT_DIRECTORIES = 93
EXPECTED_ANONYMOUS_TECHNICAL_EXCLUSIONS = 1
EXPECTED_EPISODE_FILE_COUNT = 1802
EXPECTED_SAMPLE_COUNT = 900
EXPECTED_RECORDER_SCRIPT_VERSION = "episode_recorder_v7"
EXPECTED_REGISTERED_RAW_SET_SHA256 = "5ec4c38b8c4653781b77b9237951fbfb330541cbd7d607159fcf52e90c621a81"
EXPECTED_ANONYMOUS_TECHNICAL_EXCLUSION_TREE_SHA256 = "ecec47d923656166e26fbd5e46c0c3b54c91702cdc55142e917137fac4175191"
EXPECTED_ANONYMOUS_FAILED_CHECKS = (
    "recorder:cleanup_home",
    "recorder:wrist_age",
    "recorder:wrist_duplicate_ratio",
    "recorder:wrist_fps",
    "recorder_overall_pass",
)
DEFAULT_RECORD_PATHS = (
    Path("data/protocol/day24_target_collection_records.csv"),
    Path("data/protocol/day25_gripper_collection_records.csv"),
    Path("data/protocol/day26_trajectory_collection_records.csv"),
    Path("data/protocol/day27_insufficient_evidence_collection_records.csv"),
)
MANIFEST_FIELDS = (
    "schema_version",
    "episode_id",
    "raw_episode_relpath",
    "recorder_script_version",
    "technical_valid",
    "recorder_overall_pass",
    "failed_checks",
    "sample_count",
    "csv_row_count",
    "front_image_count",
    "wrist_image_count",
    "duration_seconds",
    "max_tracking_error",
    "file_count",
    "total_bytes",
    "metadata_sha256",
    "samples_sha256",
    "episode_tree_sha256",
)
FORBIDDEN_MANIFEST_FIELDS = {
    "pair_group_id", "plan_row_id", "slot_role", "planned_physical_cause",
    "planned_intervention_type", "intervention_type", "intervention_parameters",
    "intervention_applied", "selected_canonical", "experimental_valid", "task_success",
    "physical_cause_gt", "evidence_answerability_gt", "diagnostic_decision_gt",
    "operator_notes", "human_review_notes", "supporting_ground_truth_refs",
}


@dataclass(frozen=True)
class EpisodeFingerprint:
    file_count: int
    total_bytes: int
    metadata_sha256: str
    samples_sha256: str
    tree_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_episode_tree(episode_dir: Path) -> EpisodeFingerprint:
    metadata_path = episode_dir / "metadata.json"
    samples_path = episode_dir / "samples.csv"
    if not metadata_path.is_file():
        raise ValueError("episode missing metadata.json")
    if not samples_path.is_file():
        raise ValueError("episode missing samples.csv")
    files = sorted(
        (p for p in episode_dir.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(episode_dir).as_posix(),
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        rel = path.relative_to(episode_dir).as_posix()
        size = path.stat().st_size
        total_bytes += size
        digest.update(rel.encode("utf-8")); digest.update(b"\0")
        digest.update(str(size).encode("ascii")); digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return EpisodeFingerprint(
        file_count=len(files), total_bytes=total_bytes,
        metadata_sha256=sha256_file(metadata_path),
        samples_sha256=sha256_file(samples_path),
        tree_sha256=digest.hexdigest(),
    )


def aggregate_registered_set_sha256(episode_tree_digests: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for episode_id in sorted(episode_tree_digests):
        digest.update(episode_id.encode("ascii")); digest.update(b"\0")
        digest.update(episode_tree_digests[episode_id].encode("ascii")); digest.update(b"\n")
    return digest.hexdigest()


def aggregate_content_set_sha256(tree_digests: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for tree_sha in sorted(tree_digests):
        digest.update(tree_sha.encode("ascii")); digest.update(b"\n")
    return digest.hexdigest()


def load_records(record_paths: Iterable[Path] = DEFAULT_RECORD_PATHS) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in record_paths:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def classify_registered_records(records: list[dict[str, str]]) -> dict[str, Any]:
    ids = [r["episode_id"].strip() for r in records]
    rels = [r["raw_episode_relpath"].strip() for r in records]
    canonical = [r for r in records if parse_bool(r.get("selected_canonical"))]
    noncanonical = [r for r in records if not parse_bool(r.get("selected_canonical"))]
    tech_excl = [r for r in records if not parse_bool(r.get("technical_valid"))]
    exp_excl = [r for r in records if not parse_bool(r.get("experimental_valid"))]
    return {
        "registered_attempt_count": len(records),
        "registered_unique_episode_id_count": len(set(ids)),
        "registered_unique_relpath_count": len(set(rels)),
        "registered_canonical_count": len(canonical),
        "registered_noncanonical_count": len(noncanonical),
        "registered_technical_exclusion_count": len(tech_excl),
        "registered_experimental_exclusion_count": len(exp_excl),
        "duplicate_episode_id_count": len(ids) - len(set(ids)),
        "duplicate_relpath_count": len(rels) - len(set(rels)),
        "registered_exclusion_reason_counts": dict(Counter(r.get("exclusion_reason", "") for r in records)),
    }


def manifest_fields_are_label_safe(fields: Iterable[str]) -> bool:
    return not bool(set(fields) & FORBIDDEN_MANIFEST_FIELDS)


def _bool_text(v: bool) -> str:
    return "true" if v else "false"


def manifest_row(*, record: dict[str, str], audit: TechnicalAudit, fingerprint: EpisodeFingerprint) -> dict[str, str]:
    return {
        "schema_version": "evidencemm_day28_registered_source_manifest_v1",
        "episode_id": record["episode_id"].strip(),
        "raw_episode_relpath": record["raw_episode_relpath"].strip(),
        "recorder_script_version": audit.recorder_script_version,
        "technical_valid": _bool_text(audit.technical_valid),
        "recorder_overall_pass": _bool_text(audit.recorder_overall_pass),
        "failed_checks": ";".join(audit.failed_checks),
        "sample_count": str(audit.sample_count),
        "csv_row_count": str(audit.csv_row_count),
        "front_image_count": str(audit.front_image_count),
        "wrist_image_count": str(audit.wrist_image_count),
        "duration_seconds": "" if audit.duration_seconds is None else f"{audit.duration_seconds:.9f}",
        "max_tracking_error": "" if audit.max_tracking_error is None else f"{audit.max_tracking_error:.6f}",
        "file_count": str(fingerprint.file_count),
        "total_bytes": str(fingerprint.total_bytes),
        "metadata_sha256": fingerprint.metadata_sha256,
        "samples_sha256": fingerprint.samples_sha256,
        "episode_tree_sha256": fingerprint.tree_sha256,
    }


def compare_record_to_fresh_audit(*, record: dict[str, str], audit: TechnicalAudit) -> list[str]:
    errors: list[str] = []
    if audit.episode_id != record["episode_id"].strip(): errors.append("episode_id")
    if audit.raw_episode_relpath != record["raw_episode_relpath"].strip(): errors.append("raw_episode_relpath")
    if audit.recorder_script_version != record["recorder_script_version"].strip(): errors.append("recorder_script_version")
    if audit.technical_valid != parse_bool(record.get("technical_valid")): errors.append("technical_valid")
    if audit.recorder_overall_pass != parse_bool(record.get("recorder_overall_pass")): errors.append("recorder_overall_pass")
    expected_failed = sorted(x for x in record.get("failed_checks", "").split(";") if x)
    if sorted(audit.failed_checks) != expected_failed: errors.append("failed_checks")
    return errors


def _audit_registered_sources(*, records: list[dict[str, str]], raw_root: Path):
    manifest: list[dict[str, str]] = []
    tree_digests: dict[str, str] = {}
    errors: list[str] = []
    for record in records:
        eid = record["episode_id"].strip(); rel = record["raw_episode_relpath"].strip(); ep = raw_root / rel
        if not ep.is_dir():
            errors.append(f"registered raw missing:{eid}"); continue
        audit = audit_episode(ep)
        mismatch = compare_record_to_fresh_audit(record=record, audit=audit)
        if mismatch: errors.append(f"record/raw mismatch:{eid}:{','.join(mismatch)}")
        if not audit.technical_valid: errors.append(f"registered fresh technical invalid:{eid}")
        if audit.recorder_script_version != EXPECTED_RECORDER_SCRIPT_VERSION: errors.append(f"registered recorder version drift:{eid}")
        if audit.csv_row_count != EXPECTED_SAMPLE_COUNT: errors.append(f"registered csv count drift:{eid}")
        if audit.front_image_count != EXPECTED_SAMPLE_COUNT: errors.append(f"registered front count drift:{eid}")
        if audit.wrist_image_count != EXPECTED_SAMPLE_COUNT: errors.append(f"registered wrist count drift:{eid}")
        fp = fingerprint_episode_tree(ep)
        if fp.file_count != EXPECTED_EPISODE_FILE_COUNT: errors.append(f"registered file count drift:{eid}")
        manifest.append(manifest_row(record=record, audit=audit, fingerprint=fp))
        tree_digests[eid] = fp.tree_sha256
    return manifest, tree_digests, errors


def _anonymous_technical_exclusion_audit(*, raw_root: Path, registered_relpaths: set[str], registered_tree_digests: set[str]):
    errors: list[str] = []
    raw_dirs = {p.name for p in raw_root.iterdir() if p.is_dir()}
    anonymous_dirs = sorted(raw_dirs - registered_relpaths)
    result: dict[str, Any] = {
        "count": len(anonymous_dirs), "identity_recorded": False, "raw_relpath_recorded": False,
        "technical_valid": None, "recorder_overall_pass": None, "recorder_script_version": "",
        "failed_checks": [], "csv_row_count": 0, "front_image_count": 0, "wrist_image_count": 0,
        "file_count": 0, "total_bytes": 0, "tree_sha256": "", "tree_sha256_matches_expected": False,
        "duplicates_registered_tree": False,
    }
    if len(anonymous_dirs) != EXPECTED_ANONYMOUS_TECHNICAL_EXCLUSIONS:
        errors.append(f"anonymous technical exclusion directory count mismatch:{len(anonymous_dirs)}")
        return result, errors
    # Intentionally do not expose or persist the directory name.
    ep = raw_root / anonymous_dirs[0]
    audit = audit_episode(ep); fp = fingerprint_episode_tree(ep)
    result.update({
        "technical_valid": audit.technical_valid,
        "recorder_overall_pass": audit.recorder_overall_pass,
        "recorder_script_version": audit.recorder_script_version,
        "failed_checks": sorted(audit.failed_checks),
        "csv_row_count": audit.csv_row_count,
        "front_image_count": audit.front_image_count,
        "wrist_image_count": audit.wrist_image_count,
        "file_count": fp.file_count,
        "total_bytes": fp.total_bytes,
        "tree_sha256": fp.tree_sha256,
        "tree_sha256_matches_expected": fp.tree_sha256 == EXPECTED_ANONYMOUS_TECHNICAL_EXCLUSION_TREE_SHA256,
        "duplicates_registered_tree": fp.tree_sha256 in registered_tree_digests,
    })
    if audit.technical_valid: errors.append("anonymous exclusion unexpectedly technical-valid")
    if audit.recorder_overall_pass: errors.append("anonymous exclusion recorder unexpectedly PASS")
    if audit.recorder_script_version != EXPECTED_RECORDER_SCRIPT_VERSION: errors.append("anonymous exclusion recorder version drift")
    if sorted(audit.failed_checks) != sorted(EXPECTED_ANONYMOUS_FAILED_CHECKS): errors.append("anonymous exclusion failed-check set drift")
    if audit.csv_row_count != EXPECTED_SAMPLE_COUNT: errors.append("anonymous exclusion csv row count drift")
    if audit.front_image_count != EXPECTED_SAMPLE_COUNT: errors.append("anonymous exclusion front image count drift")
    if audit.wrist_image_count != EXPECTED_SAMPLE_COUNT: errors.append("anonymous exclusion wrist image count drift")
    if fp.file_count != EXPECTED_EPISODE_FILE_COUNT: errors.append("anonymous exclusion file count drift")
    if fp.tree_sha256 != EXPECTED_ANONYMOUS_TECHNICAL_EXCLUSION_TREE_SHA256: errors.append("anonymous exclusion tree SHA256 drift")
    if fp.tree_sha256 in registered_tree_digests: errors.append("anonymous exclusion duplicates registered tree")
    return result, errors


def analyze_raw_audit(*, raw_root: Path, record_paths: Iterable[Path] = DEFAULT_RECORD_PATHS,
                      day22_plan: Path = Path("data/protocol/day22_root_cause_collection_plan.csv")):
    records = load_records(record_paths); counts = classify_registered_records(records); errors: list[str] = []
    day22_sha = file_sha256(day22_plan)
    if day22_sha != DAY22_COLLECTION_PLAN_SHA256: errors.append("frozen Day22 collection-plan SHA256 mismatch")
    expected_pairs = {
        "registered_attempt_count": EXPECTED_REGISTERED_ATTEMPTS,
        "registered_unique_episode_id_count": EXPECTED_REGISTERED_ATTEMPTS,
        "registered_unique_relpath_count": EXPECTED_REGISTERED_ATTEMPTS,
        "registered_canonical_count": EXPECTED_REGISTERED_CANONICAL,
        "registered_noncanonical_count": EXPECTED_REGISTERED_NONCANONICAL,
        "registered_technical_exclusion_count": EXPECTED_REGISTERED_TECHNICAL_EXCLUSIONS,
        "registered_experimental_exclusion_count": EXPECTED_REGISTERED_EXPERIMENTAL_EXCLUSIONS,
        "duplicate_episode_id_count": 0, "duplicate_relpath_count": 0,
    }
    for k, v in expected_pairs.items():
        if counts.get(k) != v: errors.append(f"registered count mismatch:{k}")
    registered_relpaths = {r["raw_episode_relpath"].strip() for r in records}
    raw_dirs = {p.name for p in raw_root.iterdir() if p.is_dir()}
    missing_registered = registered_relpaths - raw_dirs
    manifest, tree_digests, reg_errors = _audit_registered_sources(records=records, raw_root=raw_root)
    errors.extend(reg_errors)
    registered_sha = aggregate_registered_set_sha256(tree_digests)
    if registered_sha != EXPECTED_REGISTERED_RAW_SET_SHA256: errors.append("registered raw-set SHA256 drift")
    anonymous, anon_errors = _anonymous_technical_exclusion_audit(
        raw_root=raw_root, registered_relpaths=registered_relpaths, registered_tree_digests=set(tree_digests.values()))
    errors.extend(anon_errors)
    all_tree = list(tree_digests.values()) + ([anonymous["tree_sha256"]] if anonymous.get("tree_sha256") else [])
    content_set_sha = aggregate_content_set_sha256(all_tree)
    complete = (
        len(raw_dirs) == EXPECTED_RAW_ROOT_DIRECTORIES and not missing_registered and
        len(manifest) == EXPECTED_REGISTERED_ATTEMPTS and
        anonymous["count"] == EXPECTED_ANONYMOUS_TECHNICAL_EXCLUSIONS and not errors
    )
    analysis = {
        "schema_version": "evidencemm_day28_raw_audit_analysis_v1",
        "status": "complete" if complete else "not_ready",
        "provenance": {
            "frozen_day22_collection_plan_sha256": DAY22_COLLECTION_PLAN_SHA256,
            "observed_day22_collection_plan_sha256": day22_sha,
            "frozen_day27_commit": FROZEN_DAY27_COMMIT,
        },
        "registered_collection": counts,
        "fresh_registered_source_audit": {
            "audited_count": len(manifest),
            "fresh_technical_pass_count": sum(1 for r in manifest if r["technical_valid"] == "true"),
            "fresh_technical_fail_count": sum(1 for r in manifest if r["technical_valid"] != "true"),
            "record_vs_raw_mismatch_count": sum(1 for e in errors if e.startswith("record/raw mismatch:")),
            "file_count_distribution": dict(Counter(r["file_count"] for r in manifest)),
            "csv_row_count_distribution": dict(Counter(r["csv_row_count"] for r in manifest)),
            "front_image_count_distribution": dict(Counter(r["front_image_count"] for r in manifest)),
            "wrist_image_count_distribution": dict(Counter(r["wrist_image_count"] for r in manifest)),
            "recorder_script_version_distribution": dict(Counter(r["recorder_script_version"] for r in manifest)),
        },
        "raw_root_inventory": {
            "directory_count": len(raw_dirs),
            "registered_directory_count": len(raw_dirs & registered_relpaths),
            "missing_registered_directory_count": len(missing_registered),
            "anonymous_technical_exclusion_directory_count": anonymous["count"],
            "non_directory_root_entry_count": sum(1 for p in raw_root.iterdir() if not p.is_dir()),
        },
        "anonymous_technical_exclusion": anonymous,
        "source_binding": {
            "manifest_row_count": len(manifest),
            "manifest_fields": list(MANIFEST_FIELDS),
            "manifest_admin_labels_present": not manifest_fields_are_label_safe(MANIFEST_FIELDS),
            "registered_raw_set_sha256": registered_sha,
            "registered_raw_set_sha256_matches_expected": registered_sha == EXPECTED_REGISTERED_RAW_SET_SHA256,
            "anonymous_technical_exclusion_tree_sha256": anonymous.get("tree_sha256", ""),
            "raw_root_content_set_sha256": content_set_sha,
            "binding_identity_rule": "episode_id+relative_path+file_names+file_sizes+file_bytes;anonymous technical exclusion bound by content hash only",
        },
        "raw_attempt_retention_completeness_asserted": complete,
        "eligible_target_episode_count": EXPECTED_REGISTERED_CANONICAL,
        "ground_truth_materialized_on_day28": False,
        "answerability_prejudged_on_day28": False,
        "future_split_materialized": False,
        "errors": errors,
    }
    return manifest, analysis


def validate_final_analysis(analysis: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    def req(cond: bool, msg: str):
        if not cond: errors.append(msg)
    req(analysis.get("status") == "complete", "status must be complete")
    p = analysis.get("provenance") or {}
    req(p.get("frozen_day22_collection_plan_sha256") == DAY22_COLLECTION_PLAN_SHA256, "frozen Day22 SHA mismatch")
    req(p.get("observed_day22_collection_plan_sha256") == DAY22_COLLECTION_PLAN_SHA256, "observed Day22 SHA mismatch")
    req(p.get("frozen_day27_commit") == FROZEN_DAY27_COMMIT, "Day27 provenance commit mismatch")
    r = analysis.get("registered_collection") or {}
    req(r.get("registered_attempt_count") == 92, "registered attempts must be 92")
    req(r.get("registered_canonical_count") == 90, "registered canonical must be 90")
    req(r.get("registered_noncanonical_count") == 2, "registered noncanonical must be 2")
    req(r.get("registered_technical_exclusion_count") == 0, "registered technical exclusions must be 0")
    req(r.get("registered_experimental_exclusion_count") == 2, "registered experimental exclusions must be 2")
    req(r.get("duplicate_episode_id_count") == 0, "registered episode IDs must be unique")
    req(r.get("duplicate_relpath_count") == 0, "registered relpaths must be unique")
    f = analysis.get("fresh_registered_source_audit") or {}
    req(f.get("audited_count") == 92, "fresh audit must cover 92")
    req(f.get("fresh_technical_pass_count") == 92, "fresh technical PASS must be 92")
    req(f.get("fresh_technical_fail_count") == 0, "fresh registered technical FAIL must be 0")
    req(f.get("record_vs_raw_mismatch_count") == 0, "record/raw mismatch must be 0")
    req(f.get("file_count_distribution") == {"1802": 92}, "registered file-count distribution mismatch")
    req(f.get("csv_row_count_distribution") == {"900": 92}, "registered CSV-row distribution mismatch")
    req(f.get("front_image_count_distribution") == {"900": 92}, "registered front-image distribution mismatch")
    req(f.get("wrist_image_count_distribution") == {"900": 92}, "registered wrist-image distribution mismatch")
    req(f.get("recorder_script_version_distribution") == {"episode_recorder_v7": 92}, "registered recorder-version distribution mismatch")
    inv = analysis.get("raw_root_inventory") or {}
    req(inv.get("directory_count") == 93, "raw-root directory count must be 93")
    req(inv.get("registered_directory_count") == 92, "raw-root registered directory count must be 92")
    req(inv.get("missing_registered_directory_count") == 0, "missing registered raw directories must be 0")
    req(inv.get("anonymous_technical_exclusion_directory_count") == 1, "anonymous technical exclusion count must be 1")
    a = analysis.get("anonymous_technical_exclusion") or {}
    req(a.get("count") == 1, "anonymous technical exclusion count mismatch")
    req(a.get("identity_recorded") is False, "anonymous identity must not be recorded")
    req(a.get("raw_relpath_recorded") is False, "anonymous raw path must not be recorded")
    req(a.get("technical_valid") is False, "anonymous exclusion must be technical-invalid")
    req(a.get("recorder_overall_pass") is False, "anonymous recorder must be FAIL")
    req(a.get("recorder_script_version") == "episode_recorder_v7", "anonymous recorder version mismatch")
    req(sorted(a.get("failed_checks") or []) == sorted(EXPECTED_ANONYMOUS_FAILED_CHECKS), "anonymous failed checks mismatch")
    req(a.get("csv_row_count") == 900 and a.get("front_image_count") == 900 and a.get("wrist_image_count") == 900, "anonymous counts mismatch")
    req(a.get("file_count") == 1802, "anonymous file count mismatch")
    req(a.get("tree_sha256") == EXPECTED_ANONYMOUS_TECHNICAL_EXCLUSION_TREE_SHA256, "anonymous tree SHA mismatch")
    req(a.get("tree_sha256_matches_expected") is True, "anonymous tree SHA match flag must be true")
    req(a.get("duplicates_registered_tree") is False, "anonymous tree must not duplicate registered source")
    b = analysis.get("source_binding") or {}
    req(b.get("manifest_row_count") == 92, "source manifest must contain 92 rows")
    req(b.get("manifest_admin_labels_present") is False, "source manifest must not embed admin labels")
    req(b.get("registered_raw_set_sha256") == EXPECTED_REGISTERED_RAW_SET_SHA256, "registered raw-set SHA mismatch")
    req(b.get("registered_raw_set_sha256_matches_expected") is True, "registered raw-set SHA match flag must be true")
    req(b.get("anonymous_technical_exclusion_tree_sha256") == EXPECTED_ANONYMOUS_TECHNICAL_EXCLUSION_TREE_SHA256, "anonymous source binding mismatch")
    req(analysis.get("raw_attempt_retention_completeness_asserted") is True, "retention completeness must be asserted")
    req(analysis.get("eligible_target_episode_count") == 90, "eligible target count must be 90")
    req(analysis.get("ground_truth_materialized_on_day28") is False, "Day28 must not materialize GT")
    req(analysis.get("answerability_prejudged_on_day28") is False, "Day28 must not prejudge answerability")
    req(analysis.get("future_split_materialized") is False, "Day28 must not materialize split")
    req(not analysis.get("errors"), "Day28 analysis errors must be empty")
    return errors


def write_csv_lf(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_outputs(*, manifest: list[dict[str, str]], analysis: dict[str, Any], manifest_path: Path,
                  analysis_json_path: Path, analysis_csv_path: Path) -> None:
    if not manifest_fields_are_label_safe(MANIFEST_FIELDS):
        raise ValueError("source manifest fields contain forbidden admin labels")
    write_csv_lf(manifest_path, manifest, MANIFEST_FIELDS)
    analysis_json_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    flat = [
        ("status", analysis["status"]),
        ("registered_attempt_count", analysis["registered_collection"]["registered_attempt_count"]),
        ("registered_canonical_count", analysis["registered_collection"]["registered_canonical_count"]),
        ("registered_noncanonical_count", analysis["registered_collection"]["registered_noncanonical_count"]),
        ("registered_experimental_exclusion_count", analysis["registered_collection"]["registered_experimental_exclusion_count"]),
        ("fresh_technical_pass_count", analysis["fresh_registered_source_audit"]["fresh_technical_pass_count"]),
        ("raw_root_directory_count", analysis["raw_root_inventory"]["directory_count"]),
        ("anonymous_technical_exclusion_count", analysis["anonymous_technical_exclusion"]["count"]),
        ("registered_raw_set_sha256", analysis["source_binding"]["registered_raw_set_sha256"]),
        ("anonymous_technical_exclusion_tree_sha256", analysis["source_binding"]["anonymous_technical_exclusion_tree_sha256"]),
        ("raw_root_content_set_sha256", analysis["source_binding"]["raw_root_content_set_sha256"]),
        ("raw_attempt_retention_completeness_asserted", str(analysis["raw_attempt_retention_completeness_asserted"]).lower()),
        ("eligible_target_episode_count", analysis["eligible_target_episode_count"]),
        ("ground_truth_materialized_on_day28", str(analysis["ground_truth_materialized_on_day28"]).lower()),
        ("answerability_prejudged_on_day28", str(analysis["answerability_prejudged_on_day28"]).lower()),
        ("future_split_materialized", str(analysis["future_split_materialized"]).lower()),
    ]
    write_csv_lf(analysis_csv_path, [{"metric": k, "value": v} for k, v in flat], ("metric", "value"))
