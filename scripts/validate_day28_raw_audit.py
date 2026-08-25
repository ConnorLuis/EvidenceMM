#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from evidencemm.day28_raw_audit import DEFAULT_RECORD_PATHS, MANIFEST_FIELDS, analyze_raw_audit, manifest_fields_are_label_safe, validate_final_analysis

DEFAULT_RAW_ROOT = Path("/mnt/c/Users/Administrator/projects/embodied-agent-arm/outputs/episodes_root_cause_v2_final")

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    p.add_argument("--day22-plan", type=Path, default=Path("data/protocol/day22_root_cause_collection_plan.csv"))
    p.add_argument("--manifest", type=Path, default=Path("data/protocol/day28_registered_source_manifest.csv"))
    p.add_argument("--analysis", type=Path, default=Path("data/protocol/day28_raw_audit_analysis.json"))
    return p.parse_args()

def load_manifest(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as h:
        r = csv.DictReader(h); return list(r), tuple(r.fieldnames or [])

def main():
    a = args(); errors = []
    stored_analysis = json.loads(a.analysis.read_text(encoding="utf-8"))
    stored_manifest, stored_fields = load_manifest(a.manifest)
    if stored_fields != tuple(MANIFEST_FIELDS): errors.append("stored source-manifest field order/schema mismatch")
    if not manifest_fields_are_label_safe(stored_fields): errors.append("stored source manifest embeds forbidden admin labels")
    fresh_manifest, fresh_analysis = analyze_raw_audit(raw_root=a.raw_root, record_paths=DEFAULT_RECORD_PATHS, day22_plan=a.day22_plan)
    errors.extend(validate_final_analysis(stored_analysis)); errors.extend(validate_final_analysis(fresh_analysis))
    if stored_manifest != fresh_manifest: errors.append("stored source manifest does not exactly match fresh raw audit")
    if stored_analysis != fresh_analysis: errors.append("stored Day28 analysis does not exactly match fresh raw audit")
    if errors:
        print("===== DAY28 RAW AUDIT: NOT READY =====")
        for item in errors: print("FAIL:", item)
        return 1
    print("===== DAY28 RAW AUDIT: PASS =====")
    print("frozen_day22_plan_sha256: PASS")
    print("frozen_day27_commit_provenance: PASS")
    print("registered_collection_attempts: 92/92 PASS")
    print("fresh_registered_technical_audit: 92/92 PASS")
    print("registered_canonical_sources: 90/90 PASS")
    print("registered_noncanonical_retained: 2/2 PASS")
    print("registered_experimental_exclusions: 2/2 PASS")
    print("anonymous_technical_exclusion: 1/1 PASS")
    print("raw_root_directories_accounted: 93/93 PASS")
    print("registered_raw_set_sha256: PASS")
    print("anonymous_technical_exclusion_tree_sha256: PASS")
    print("source_manifest_admin_labels_present: false PASS")
    print("raw_attempt_retention_completeness_asserted: true PASS")
    print("ground_truth_materialized_on_day28: false PASS")
    print("answerability_prejudged_on_day28: false PASS")
    print("future_split_materialized: false PASS")
    return 0

if __name__ == "__main__": raise SystemExit(main())
