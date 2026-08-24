#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from evidencemm.day24_target_collection import load_csv
from evidencemm.day26_trajectory_collection import analyze_collection,selected_day24_clean_anchors

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--plan",type=Path,default=Path("data/protocol/day26_trajectory_collection_plan.csv"))
    p.add_argument("--records",type=Path,default=Path("data/protocol/day26_trajectory_collection_records.csv"))
    p.add_argument("--day24-plan",type=Path,default=Path("data/protocol/day24_target_collection_plan.csv"))
    p.add_argument("--day24-records",type=Path,default=Path("data/protocol/day24_target_collection_records.csv"))
    p.add_argument("--raw-root",type=Path,default=Path(
        "/mnt/c/Users/Administrator/projects/embodied-agent-arm/outputs/episodes_root_cause_v2_final"))
    p.add_argument("--output-json",type=Path,default=Path("data/protocol/day26_trajectory_collection_analysis.json"))
    p.add_argument("--output-csv",type=Path,default=Path("data/protocol/day26_trajectory_collection_analysis.csv"))
    return p.parse_args()

def main():
    a=args()
    anchors=selected_day24_clean_anchors(load_csv(a.day24_plan),load_csv(a.day24_records))
    analysis=analyze_collection(plan_rows=load_csv(a.plan),records=load_csv(a.records),
                                clean_anchors=anchors,raw_root=a.raw_root)
    a.output_json.write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    fields=("plan_row_id","pair_group_id","episode_id","task_success","technical_valid",
            "duration_seconds","max_tracking_error","recorder_script_version")
    with a.output_csv.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(analysis["canonical_details"])
    print("===== DAY26 TRAJECTORY COLLECTION ANALYSIS =====")
    print(f"trajectory_canonical={analysis['new_trajectory_canonical_count']}/20 "
          f"trajectory_failure={analysis['trajectory_failure_count']}/20 "
          f"clean_anchors={analysis['clean_anchor_count']}/15 "
          f"complete_groups={analysis['complete_pair_group_count']}/15")
    print(f"new_attempts={analysis['new_attempt_count']} "
          f"technical_exclusions={analysis['technical_exclusion_attempt_count']} "
          f"experimental_exclusions={analysis['experimental_exclusion_attempt_count']}")
    print("new_clean_collection_required=",analysis["new_clean_collection_required"])
    print("status=",analysis["status"])
    print("JSON:",a.output_json)
    print("CSV :",a.output_csv)
    return 0
if __name__=="__main__":
    raise SystemExit(main())
