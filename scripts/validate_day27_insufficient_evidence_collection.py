#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencemm.day24_target_collection import (
    DAY22_COLLECTION_PLAN_SHA256,
    file_sha256,
    load_csv,
)
from evidencemm.day27_insufficient_evidence_collection import (
    expected_day27_rows_from_day22,
    validate_day27_plan_shape,
    validate_final_analysis,
)


def args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--day22-plan",
        type=Path,
        default=Path("data/protocol/day22_root_cause_collection_plan.csv"),
    )
    p.add_argument(
        "--day27-plan",
        type=Path,
        default=Path(
            "data/protocol/day27_insufficient_evidence_collection_plan.csv"
        ),
    )
    p.add_argument(
        "--analysis",
        type=Path,
        default=Path(
            "data/protocol/day27_insufficient_evidence_collection_analysis.json"
        ),
    )
    return p.parse_args()


def main():
    a = args()
    errors = []
    actual_sha = file_sha256(a.day22_plan)
    frozen_ok = actual_sha == DAY22_COLLECTION_PLAN_SHA256
    if not frozen_ok:
        errors.append(
            "frozen Day22 plan SHA mismatch: "
            f"expected={DAY22_COLLECTION_PLAN_SHA256} actual={actual_sha}"
        )

    day22 = load_csv(a.day22_plan)
    actual = load_csv(a.day27_plan)
    try:
        validate_day27_plan_shape(actual)
    except Exception as exc:
        errors.append(f"Day27 plan shape: {exc}")

    expected = expected_day27_rows_from_day22(day22)
    plan_exact = actual == expected
    if not plan_exact:
        errors.append("Day27 plan is not the exact deterministic projection of Day22 s06")

    analysis = json.loads(a.analysis.read_text(encoding="utf-8"))
    errors.extend(validate_final_analysis(analysis))

    if errors:
        print("===== DAY27 INSUFFICIENT-EVIDENCE COLLECTION: NOT READY =====")
        for item in errors:
            print("FAIL:", item)
        print(
            "CURRENT: insufficient="
            f"{analysis.get('new_insufficient_candidate_canonical_count')}/15 "
            f"failures={analysis.get('insufficient_candidate_failure_count')}/15 "
            f"clean={analysis.get('clean_anchor_count')}/15 "
            f"controlled={analysis.get('controlled_canonical_count')}/60 "
            f"groups={analysis.get('complete_pair_group_count')}/15"
        )
        return 1

    print("===== DAY27 INSUFFICIENT-EVIDENCE COLLECTION: PASS =====")
    print("frozen_day22_plan_sha256: PASS")
    print("derived_s06_plan: 15/15 PASS")
    print("day24_clean_anchors: 15/15 PASS")
    print("preexisting_controlled_causes: 60/60 PASS")
    print("insufficient_candidate_failures: 15/15 PASS")
    print("eligible_target_episodes: 90/90 PASS")
    print("ambiguity_protocol: blinded_single_cause_challenge_v2 PASS")
    print("ambiguity_variants: target/gripper/trajectory = 5/5/5 PASS")
    print("answerability_prejudged_on_day27: false PASS")
    print("new_clean_collection_required: false PASS")
    print("preexisting_recollection_required: false PASS")
    print("future_split_materialized: false PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
