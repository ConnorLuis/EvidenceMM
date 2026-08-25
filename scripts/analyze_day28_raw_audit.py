#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from evidencemm.day28_raw_audit import DEFAULT_RECORD_PATHS, analyze_raw_audit, validate_final_analysis, write_outputs

DEFAULT_RAW_ROOT = Path("/mnt/c/Users/Administrator/projects/embodied-agent-arm/outputs/episodes_root_cause_v2_final")

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    p.add_argument("--day22-plan", type=Path, default=Path("data/protocol/day22_root_cause_collection_plan.csv"))
    p.add_argument("--manifest", type=Path, default=Path("data/protocol/day28_registered_source_manifest.csv"))
    p.add_argument("--analysis-json", type=Path, default=Path("data/protocol/day28_raw_audit_analysis.json"))
    p.add_argument("--analysis-csv", type=Path, default=Path("data/protocol/day28_raw_audit_analysis.csv"))
    return p.parse_args()

def main():
    a = args()
    manifest, analysis = analyze_raw_audit(raw_root=a.raw_root, record_paths=DEFAULT_RECORD_PATHS, day22_plan=a.day22_plan)
    write_outputs(manifest=manifest, analysis=analysis, manifest_path=a.manifest, analysis_json_path=a.analysis_json, analysis_csv_path=a.analysis_csv)
    print("===== DAY28 RAW AUDIT ANALYSIS =====")
    print(f"registered={analysis['registered_collection']['registered_attempt_count']}/92 canonical={analysis['registered_collection']['registered_canonical_count']}/90 fresh_technical={analysis['fresh_registered_source_audit']['fresh_technical_pass_count']}/92 raw_dirs={analysis['raw_root_inventory']['directory_count']}/93")
    print("registered_experimental_exclusions=", analysis["registered_collection"]["registered_experimental_exclusion_count"])
    print("anonymous_technical_exclusions=", analysis["anonymous_technical_exclusion"]["count"])
    print("registered_raw_set_sha256=", analysis["source_binding"]["registered_raw_set_sha256"])
    print("anonymous_technical_exclusion_tree_sha256=", analysis["source_binding"]["anonymous_technical_exclusion_tree_sha256"])
    print("raw_root_content_set_sha256=", analysis["source_binding"]["raw_root_content_set_sha256"])
    print("raw_attempt_retention_completeness_asserted=", analysis["raw_attempt_retention_completeness_asserted"])
    print("status=", analysis["status"])
    print("manifest:", a.manifest); print("analysis:", a.analysis_json); print("summary :", a.analysis_csv)
    errors = validate_final_analysis(analysis)
    if errors:
        print("VALIDATION: NOT READY")
        for item in errors: print("FAIL:", item)
        return 1
    print("VALIDATION: PASS")
    return 0

if __name__ == "__main__": raise SystemExit(main())
