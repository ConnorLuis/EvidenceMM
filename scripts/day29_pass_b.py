#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CONTRACT_PATH = ROOT / "data/protocol/day29_pass_b_operational_contract.json"
PASS_A_RECORDS_PATH = ROOT / "data/annotations/day29_pass_a_records.jsonl"
PASS_A_RECEIPT_PATH = ROOT / "data/protocol/day29_pass_a_freeze_receipt.json"

DAY22_PROTOCOL_PATH = ROOT / "data/protocol/day22_root_cause_benchmark_v2_protocol.json"
DAY24_RECORDS_PATH = ROOT / "data/protocol/day24_target_collection_records.csv"
DAY25_RECORDS_PATH = ROOT / "data/protocol/day25_gripper_collection_records.csv"
DAY26_RECORDS_PATH = ROOT / "data/protocol/day26_trajectory_collection_records.csv"
DAY27_PLAN_PATH = ROOT / "data/protocol/day27_insufficient_evidence_collection_plan.csv"
DAY27_RECORDS_PATH = ROOT / "data/protocol/day27_insufficient_evidence_collection_records.csv"
DAY28_ANALYSIS_PATH = ROOT / "data/protocol/day28_raw_audit_analysis.json"
DAY29_REVIEW_SCHEMA_PATH = ROOT / "data/protocol/day29_review_record_schema.json"
DAY29_CONFIG_PATH = ROOT / "configs/day29_human_causal_review.yaml"

GT_RECORDS_PATH = ROOT / "data/annotations/day29_ground_truth_records.jsonl"
FREEZE_RECEIPT_PATH = ROOT / "data/protocol/day29_ground_truth_freeze_receipt.json"

PASS_A_FREEZE_COMMIT = "1ae4556d73c8dd409ee4e25d0025fce8a3064a1a"
PASS_A_RECORDS_SHA256 = "ebc8630e98ba409088e93e088230d60be6f4082c95a9f0cef45014049a3ddd7c"

REQUIRED_ANCESTORS = {
    "day24": "8c1035b6592488f6cf2a7cf4bbdddc86a62b6394",
    "day25": "2eb16ae1fb9418af0a7c712dc321b69fd3f0ed42",
    "day26": "ba7669b5503cab13524c4ab1e4a5ad68c503abe2",
    "day27": "eaa29a3ebc9f41fa26ffa6de3291c6a28d93a4cd",
    "day28": "48ee78f7ba07f6e053e1581edff3d271e964c581",
    "pass_a": PASS_A_FREEZE_COMMIT,
}

EXPECTED_FROZEN_BLOBS = {
    "data/protocol/day22_root_cause_benchmark_v2_protocol.json": "388ecb388375046b208e85eb9617e79961a5bf52",
    "data/protocol/day24_target_collection_records.csv": "81b2cfb2f2a6c0cd113cd46f32c31b4d32684b44",
    "data/protocol/day25_gripper_collection_records.csv": "ed2efc2b04859c5bf06f217880b319688a51419b",
    "data/protocol/day26_trajectory_collection_records.csv": "0685f1078fcbff9e2e3232e3e66f3879d9c807c1",
    "data/protocol/day27_insufficient_evidence_collection_plan.csv": "cbd196b887cb061d4c803cf10fe805760507e0e4",
    "data/protocol/day27_insufficient_evidence_collection_records.csv": "574ae1d2da744dd7707adf30d37634f6bb5bb494",
    "data/protocol/day28_raw_audit_analysis.json": "60a2760740d59fe9fb5e0434e47387de17cd286d",
    "data/protocol/day29_review_record_schema.json": "63ea6fa79179cd6abbcd1e9b135b5767e7d007b7",
    "configs/day29_human_causal_review.yaml": "e164e7775819bb5b77e078ccb517a5a90349651b",
    "data/annotations/day29_pass_a_records.jsonl": "526f15a0e70cb5e46109b05c27e234ec3163314a",
    "data/protocol/day29_pass_a_freeze_receipt.json": "98546da5901aea8f1aba8495ff6ab9fc3c04580c",
}

EXPECTED_COUNTS = {
    "day24_canonical": 35,
    "day25_canonical": 20,
    "day26_canonical": 20,
    "day27_registered_attempts": 17,
    "day27_canonical": 15,
    "canonical_total": 90,
    "pair_group_count": 15,
    "episodes_per_pair_group": 6,
}

EXPECTED_PHYSICAL_COUNTS = {
    "none_clean": 15,
    "target_offset_or_perception": 25,
    "gripper_close_timing": 25,
    "trajectory_execution_deviation": 25,
    "unknown": 0,
}

EXPECTED_DECISION_COUNTS = {
    "clean_success": 15,
    "target_offset_or_perception": 25,
    "gripper_close_timing": 25,
    "trajectory_execution_deviation": 25,
    "insufficient_evidence": 0,
}

EXPECTED_ANSWERABILITY_COUNTS = {
    "answerable": 75,
    "not_applicable_clean": 15,
    "insufficient_evidence": 0,
}

DAY27_VARIANT_MAP = {
    "blinded_single_cause_challenge_v2:target_mild_20mm_forward":
        "target_offset_or_perception",
    "blinded_single_cause_challenge_v2:gripper_late_30_40mm_upward_progress":
        "gripper_close_timing",
    "blinded_single_cause_challenge_v2:trajectory_mild_25mm_forward":
        "trajectory_execution_deviation",
}

GT_SCHEMA = "evidencemm_day29_ground_truth_record_v1"


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def require_ancestor(commit: str) -> None:
    rc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode
    if rc != 0:
        raise RuntimeError(
            f"required frozen commit is not an ancestor: {commit}"
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_bool(value: str, *, field: str, episode_id: str) -> bool:
    v = value.strip().lower()
    if v == "true":
        return True
    if v == "false":
        return False
    raise ValueError(
        f"{episode_id}: {field} must be true/false, got {value!r}"
    )


def verify_tooling_is_committed() -> None:
    required = [
        "scripts/day29_pass_b.py",
        "data/protocol/day29_pass_b_operational_contract.json",
        "docs/day29_pass_b_operator_guide.md",
    ]
    for rel in required:
        try:
            git_output("rev-parse", f"HEAD:{rel}")
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Pass B tooling must be committed before build/freeze: {rel}"
            ) from exc

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *required],
        cwd=ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(
            "Pass B tooling has uncommitted changes:\n" + dirty
        )


def verify_contract() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)

    if contract.get("schema_version") != \
            "evidencemm_day29_pass_b_operational_contract_v1":
        raise RuntimeError("Pass B contract schema mismatch")

    if contract.get("pass_a_freeze_commit") != PASS_A_FREEZE_COMMIT:
        raise RuntimeError("Pass B contract Pass A commit mismatch")

    if contract.get("pass_a_records_sha256") != PASS_A_RECORDS_SHA256:
        raise RuntimeError("Pass B contract Pass A records SHA mismatch")

    if contract.get("required_ancestor_commits") != REQUIRED_ANCESTORS:
        raise RuntimeError("Pass B contract ancestor set mismatch")

    if contract.get("frozen_git_blobs") != EXPECTED_FROZEN_BLOBS:
        raise RuntimeError("Pass B contract frozen blob set mismatch")

    return contract


def verify_frozen_environment() -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    if git_output("branch", "--show-current") != "master":
        raise RuntimeError("Day29 Pass B must run on master")

    contract = verify_contract()

    for commit in REQUIRED_ANCESTORS.values():
        require_ancestor(commit)

    for rel, expected_blob in EXPECTED_FROZEN_BLOBS.items():
        actual_blob = git_output("rev-parse", f"HEAD:{rel}")
        if actual_blob != expected_blob:
            raise RuntimeError(
                f"frozen blob mismatch: {rel}\n"
                f"expected={expected_blob}\nactual={actual_blob}"
            )

    if sha256_file(PASS_A_RECORDS_PATH) != PASS_A_RECORDS_SHA256:
        raise RuntimeError("Pass A records SHA256 changed after freeze")

    pass_a_receipt = read_json(PASS_A_RECEIPT_PATH)
    if pass_a_receipt.get("status") != \
            "blind_review_complete_admin_unrevealed":
        raise RuntimeError("unexpected Pass A receipt status")
    if pass_a_receipt.get("pass_a_records_sha256") != PASS_A_RECORDS_SHA256:
        raise RuntimeError("Pass A receipt record SHA mismatch")
    if pass_a_receipt.get("case_count") != 90:
        raise RuntimeError("Pass A receipt case count mismatch")
    if pass_a_receipt.get("human_review_completed") is not True:
        raise RuntimeError("Pass A human review not complete")
    if pass_a_receipt.get("admin_reveal_started") is not False:
        raise RuntimeError("Pass A receipt indicates prior admin reveal")
    if pass_a_receipt.get("ground_truth_frozen") is not False:
        raise RuntimeError("Pass A receipt unexpectedly has GT frozen")
    if pass_a_receipt.get("future_split_materialized") is not False:
        raise RuntimeError("Pass A receipt unexpectedly has future split")

    day28 = read_json(DAY28_ANALYSIS_PATH)
    checks = {
        "status": day28.get("status") == "complete",
        "registered_attempt_count":
            day28.get("registered_collection", {}).get("registered_attempt_count") == 92,
        "registered_canonical_count":
            day28.get("registered_collection", {}).get("registered_canonical_count") == 90,
        "registered_technical_exclusion_count":
            day28.get("registered_collection", {}).get("registered_technical_exclusion_count") == 0,
        "registered_experimental_exclusion_count":
            day28.get("registered_collection", {}).get("registered_experimental_exclusion_count") == 2,
        "fresh_technical_pass_count":
            day28.get("fresh_registered_source_audit", {}).get("fresh_technical_pass_count") == 92,
        "record_vs_raw_mismatch_count":
            day28.get("fresh_registered_source_audit", {}).get("record_vs_raw_mismatch_count") == 0,
        "eligible_target_episode_count":
            day28.get("eligible_target_episode_count") == 90,
        "ground_truth_materialized_on_day28":
            day28.get("ground_truth_materialized_on_day28") is False,
        "answerability_prejudged_on_day28":
            day28.get("answerability_prejudged_on_day28") is False,
        "future_split_materialized":
            day28.get("future_split_materialized") is False,
        "errors": day28.get("errors") == [],
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(
            "Day28 frozen audit checks failed: " + ", ".join(failed)
        )

    pass_a = read_jsonl(PASS_A_RECORDS_PATH)
    if len(pass_a) != 90:
        raise RuntimeError("Pass A record count must be 90")
    if len({r["episode_id"] for r in pass_a}) != 90:
        raise RuntimeError("Pass A episode IDs must be unique")
    if [r.get("review_position") for r in pass_a] != list(range(1, 91)):
        raise RuntimeError("Pass A review order positions invalid")

    answerability = Counter(
        r.get("evidence_answerability_gt") for r in pass_a
    )
    normalized_answerability = {
        key: answerability.get(key, 0)
        for key in EXPECTED_ANSWERABILITY_COUNTS
    }
    if normalized_answerability != EXPECTED_ANSWERABILITY_COUNTS:
        raise RuntimeError(
            f"unexpected Pass A answerability counts: "
            f"{normalized_answerability}"
        )

    return contract, pass_a


def selected(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if parse_bool(
            row["selected_canonical"],
            field="selected_canonical",
            episode_id=row["episode_id"],
        )
    ]


def derive_admin_labels() -> dict[str, dict[str, Any]]:
    day24_all = read_csv(DAY24_RECORDS_PATH)
    day25_all = read_csv(DAY25_RECORDS_PATH)
    day26_all = read_csv(DAY26_RECORDS_PATH)
    day27_all = read_csv(DAY27_RECORDS_PATH)
    day27_plan = read_csv(DAY27_PLAN_PATH)

    day24 = selected(day24_all)
    day25 = selected(day25_all)
    day26 = selected(day26_all)
    day27 = selected(day27_all)

    if len(day24) != EXPECTED_COUNTS["day24_canonical"]:
        raise RuntimeError(f"day24 canonical count={len(day24)}")
    if len(day25) != EXPECTED_COUNTS["day25_canonical"]:
        raise RuntimeError(f"day25 canonical count={len(day25)}")
    if len(day26) != EXPECTED_COUNTS["day26_canonical"]:
        raise RuntimeError(f"day26 canonical count={len(day26)}")
    if len(day27_all) != EXPECTED_COUNTS["day27_registered_attempts"]:
        raise RuntimeError(f"day27 attempt count={len(day27_all)}")
    if len(day27) != EXPECTED_COUNTS["day27_canonical"]:
        raise RuntimeError(f"day27 canonical count={len(day27)}")

    plan_by_row = {row["plan_row_id"]: row for row in day27_plan}
    if len(plan_by_row) != 15:
        raise RuntimeError("Day27 plan must contain 15 unique plan rows")

    labels: dict[str, dict[str, Any]] = {}

    def common(row: dict[str, str]) -> tuple[bool, bool, bool]:
        episode_id = row["episode_id"]
        technical = parse_bool(
            row["technical_valid"],
            field="technical_valid",
            episode_id=episode_id,
        )
        experimental = parse_bool(
            row["experimental_valid"],
            field="experimental_valid",
            episode_id=episode_id,
        )
        task_success = parse_bool(
            row["task_success"],
            field="task_success",
            episode_id=episode_id,
        )
        if not technical or not experimental:
            raise RuntimeError(
                f"{episode_id}: canonical episode must be technical+experimental valid"
            )
        return technical, experimental, task_success

    def insert(
        row: dict[str, str],
        *,
        cause: str,
        intervention_verified: bool,
        source_day: str,
        detail: str,
    ) -> None:
        episode_id = row["episode_id"]
        if episode_id in labels:
            raise RuntimeError(f"duplicate canonical episode: {episode_id}")

        technical, experimental, task_success = common(row)

        if cause == "none_clean":
            if not task_success:
                raise RuntimeError(
                    f"{episode_id}: clean GT requires task success"
                )
            if intervention_verified:
                raise RuntimeError(
                    f"{episode_id}: clean GT cannot verify an intervention"
                )
        else:
            if task_success:
                raise RuntimeError(
                    f"{episode_id}: controlled-cause GT requires task failure"
                )
            if not intervention_verified:
                raise RuntimeError(
                    f"{episode_id}: cause GT requires verified intervention"
                )

        labels[episode_id] = {
            "pair_group_id": row["pair_group_id"],
            "plan_row_id": row["plan_row_id"],
            "technical_valid": technical,
            "experimental_valid": experimental,
            "task_success": task_success,
            "intervention_verified": intervention_verified,
            "physical_cause_gt": cause,
            "source_day": source_day,
            "detail": detail,
        }

    # Day24: clean anchors + target offset interventions.
    for row in day24:
        eid = row["episode_id"]
        applied = parse_bool(
            row["intervention_applied"],
            field="intervention_applied",
            episode_id=eid,
        )
        single = parse_bool(
            row["single_primary_intervention"],
            field="single_primary_intervention",
            episode_id=eid,
        )
        observable = parse_bool(
            row["changed_factor_observable"],
            field="changed_factor_observable",
            episode_id=eid,
        )
        task_success = parse_bool(
            row["task_success"],
            field="task_success",
            episode_id=eid,
        )

        if task_success:
            if applied:
                raise RuntimeError(
                    f"{eid}: Day24 clean control unexpectedly has intervention"
                )
            insert(
                row,
                cause="none_clean",
                intervention_verified=False,
                source_day="day24",
                detail=(
                    "canonical clean control; no intervention applied by design; "
                    "technical/experimental validity and task success verified"
                ),
            )
        else:
            if not (applied and single and observable):
                raise RuntimeError(
                    f"{eid}: Day24 target intervention not fully verified"
                )
            if row.get("parameter_direction") != "follower_forward":
                raise RuntimeError(f"{eid}: unexpected Day24 target direction")
            if row.get("parameter_value") != "40":
                raise RuntimeError(f"{eid}: unexpected Day24 target magnitude")
            if row.get("parameter_unit") != "mm":
                raise RuntimeError(f"{eid}: unexpected Day24 target unit")
            insert(
                row,
                cause="target_offset_or_perception",
                intervention_verified=True,
                source_day="day24",
                detail=(
                    "canonical target-offset intervention; intervention applied, "
                    "single-primary, observable, 40 mm follower-forward; task failure verified"
                ),
            )

    # Day25: late gripper close.
    for row in day25:
        eid = row["episode_id"]
        flags = [
            parse_bool(
                row[name],
                field=name,
                episode_id=eid,
            )
            for name in (
                "intervention_applied",
                "single_primary_intervention",
                "changed_factor_observable",
                "phase_proxy_met",
            )
        ]
        if not all(flags):
            raise RuntimeError(
                f"{eid}: Day25 gripper intervention not fully verified"
            )
        if row.get("parameter_direction") != "late":
            raise RuntimeError(f"{eid}: unexpected Day25 timing direction")
        if row.get("parameter_min") != "30" or row.get("parameter_max") != "40":
            raise RuntimeError(f"{eid}: unexpected Day25 timing range")
        if row.get("parameter_unit") != "mm":
            raise RuntimeError(f"{eid}: unexpected Day25 unit")
        insert(
            row,
            cause="gripper_close_timing",
            intervention_verified=True,
            source_day="day25",
            detail=(
                "canonical late-gripper intervention; single-primary and observable; "
                "phase proxy met; 30-40 mm upward progress before close; task failure verified"
            ),
        )

    # Day26: bounded trajectory deviation.
    for row in day26:
        eid = row["episode_id"]
        flags = [
            parse_bool(
                row[name],
                field=name,
                episode_id=eid,
            )
            for name in (
                "intervention_applied",
                "single_primary_intervention",
                "changed_factor_observable",
                "deviation_proxy_met",
            )
        ]
        if not all(flags):
            raise RuntimeError(
                f"{eid}: Day26 trajectory intervention not fully verified"
            )
        if row.get("parameter_direction") != "follower_forward":
            raise RuntimeError(f"{eid}: unexpected Day26 direction")
        if row.get("parameter_min") != "40" or row.get("parameter_max") != "60":
            raise RuntimeError(f"{eid}: unexpected Day26 deviation range")
        if row.get("parameter_unit") != "mm":
            raise RuntimeError(f"{eid}: unexpected Day26 unit")
        insert(
            row,
            cause="trajectory_execution_deviation",
            intervention_verified=True,
            source_day="day26",
            detail=(
                "canonical bounded trajectory intervention; single-primary and observable; "
                "deviation proxy met; 40-60 mm follower-forward path deviation; task failure verified"
            ),
        )

    # Day27: admin-known single-cause challenge. Planned physical cause remains
    # unknown in the public plan field by design; the frozen ambiguity_protocol
    # variant + canonical collection record verifies the admin-known cause.
    for row in day27:
        eid = row["episode_id"]
        plan = plan_by_row.get(row["plan_row_id"])
        if plan is None:
            raise RuntimeError(f"{eid}: missing Day27 plan row")

        if plan["pair_group_id"] != row["pair_group_id"]:
            raise RuntimeError(f"{eid}: Day27 plan pair-group mismatch")
        if plan["planned_physical_cause"] != "unknown":
            raise RuntimeError(f"{eid}: Day27 planned cause must remain unknown")
        if plan["planned_intervention_type"] != "ambiguity_protocol":
            raise RuntimeError(f"{eid}: Day27 intervention type mismatch")
        if plan["expected_task_outcome"] != "failure":
            raise RuntimeError(f"{eid}: Day27 expected outcome mismatch")

        variant = plan["ambiguity_protocol"]
        cause = DAY27_VARIANT_MAP.get(variant)
        if cause is None:
            raise RuntimeError(
                f"{eid}: unknown Day27 challenge variant: {variant}"
            )

        required_true = (
            "ambiguity_protocol_followed",
            "deliberate_known_cause_intervention",
            "intentional_failure_injection",
            "scene_comparable",
        )
        for name in required_true:
            if not parse_bool(
                row[name],
                field=name,
                episode_id=eid,
            ):
                raise RuntimeError(
                    f"{eid}: Day27 {name} must be true"
                )

        if parse_bool(
            row["multiple_primary_interventions"],
            field="multiple_primary_interventions",
            episode_id=eid,
        ):
            raise RuntimeError(
                f"{eid}: Day27 multiple primary interventions forbidden"
            )

        insert(
            row,
            cause=cause,
            intervention_verified=True,
            source_day="day27",
            detail=(
                "canonical blinded single-cause challenge; admin-known variant "
                f"{variant}; protocol followed; deliberate single cause injected; "
                "scene comparable; task failure verified"
            ),
        )

    if len(labels) != EXPECTED_COUNTS["canonical_total"]:
        raise RuntimeError(f"canonical GT label count={len(labels)}")

    groups: dict[str, list[str]] = defaultdict(list)
    for episode_id, label in labels.items():
        groups[label["pair_group_id"]].append(episode_id)

    if len(groups) != EXPECTED_COUNTS["pair_group_count"]:
        raise RuntimeError(f"pair-group count={len(groups)}")

    bad_groups = {
        group: len(episodes)
        for group, episodes in groups.items()
        if len(episodes) != EXPECTED_COUNTS["episodes_per_pair_group"]
    }
    if bad_groups:
        raise RuntimeError(
            f"pair-group episode counts invalid: {bad_groups}"
        )

    return labels


def build_records(
    pass_a: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    pass_a_ids = {r["episode_id"] for r in pass_a}
    if pass_a_ids != set(labels):
        missing_admin = sorted(pass_a_ids - set(labels))
        extra_admin = sorted(set(labels) - pass_a_ids)
        raise RuntimeError(
            f"PassA/admin canonical set mismatch; "
            f"missing_admin={missing_admin}; extra_admin={extra_admin}"
        )

    records: list[dict[str, Any]] = []

    for blind in pass_a:
        episode_id = blind["episode_id"]
        admin = labels[episode_id]
        answerability = blind["evidence_answerability_gt"]
        task_success = admin["task_success"]
        physical = admin["physical_cause_gt"]

        if task_success:
            if answerability != "not_applicable_clean":
                raise RuntimeError(
                    f"{episode_id}: clean success requires not_applicable_clean"
                )
            if physical != "none_clean":
                raise RuntimeError(
                    f"{episode_id}: clean success requires none_clean"
                )
            decision = "clean_success"
        else:
            if physical not in {
                "target_offset_or_perception",
                "gripper_close_timing",
                "trajectory_execution_deviation",
                "unknown",
            }:
                raise RuntimeError(
                    f"{episode_id}: invalid failed physical cause {physical}"
                )

            if answerability == "answerable":
                if physical == "unknown":
                    raise RuntimeError(
                        f"{episode_id}: answerable case cannot have unknown GT"
                    )
                decision = physical
            elif answerability == "insufficient_evidence":
                decision = "insufficient_evidence"
            else:
                raise RuntimeError(
                    f"{episode_id}: failed case has invalid answerability "
                    f"{answerability}"
                )

        records.append(
            {
                "schema_version": GT_SCHEMA,
                "review_position": blind["review_position"],
                "episode_id": episode_id,
                "pair_group_id": admin["pair_group_id"],
                "technical_valid": admin["technical_valid"],
                "experimental_valid": admin["experimental_valid"],
                "task_success": task_success,
                "intervention_verified": admin["intervention_verified"],
                "physical_cause_gt": physical,
                "evidence_answerability_gt": answerability,
                "diagnostic_decision_gt": decision,
                "confidence": 1.0,
                "review_notes": (
                    "admin_causal_verification: "
                    + admin["detail"]
                    + "; physical_cause_gt derived only from frozen admin evidence; "
                    "Pass A blind_cause_hypothesis was not used"
                ),
            }
        )

    return records


def normalized_counter(
    values: list[str],
    keys: list[str],
) -> dict[str, int]:
    counts = Counter(values)
    return {key: counts.get(key, 0) for key in keys}


def validate_records(
    records: list[dict[str, Any]],
    pass_a: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []

    if len(records) != 90:
        errors.append(f"record count={len(records)}")

    if len({r.get("episode_id") for r in records}) != len(records):
        errors.append("episode IDs not unique")

    if [r.get("review_position") for r in records] != list(range(1, 91)):
        errors.append("review positions not exactly 1..90")

    expected_order = [r["episode_id"] for r in pass_a]
    actual_order = [r.get("episode_id") for r in records]
    if actual_order != expected_order:
        errors.append("GT order differs from frozen Pass A order")

    pass_a_by_id = {r["episode_id"]: r for r in pass_a}

    required_fields = {
        "schema_version",
        "review_position",
        "episode_id",
        "pair_group_id",
        "technical_valid",
        "experimental_valid",
        "task_success",
        "intervention_verified",
        "physical_cause_gt",
        "evidence_answerability_gt",
        "diagnostic_decision_gt",
        "confidence",
        "review_notes",
    }

    for record in records:
        eid = record.get("episode_id", "")
        if set(record) != required_fields:
            errors.append(f"{eid}: field set mismatch")
            continue

        if record["schema_version"] != GT_SCHEMA:
            errors.append(f"{eid}: schema mismatch")

        if not isinstance(record["pair_group_id"], str) or \
                not record["pair_group_id"].startswith("rcv2_g"):
            errors.append(f"{eid}: invalid pair_group_id")

        for name in (
            "technical_valid",
            "experimental_valid",
            "task_success",
            "intervention_verified",
        ):
            if not isinstance(record[name], bool):
                errors.append(f"{eid}: {name} must be bool")

        if record["technical_valid"] is not True:
            errors.append(f"{eid}: canonical technical_valid must be true")
        if record["experimental_valid"] is not True:
            errors.append(f"{eid}: canonical experimental_valid must be true")

        if record["physical_cause_gt"] not in {
            "target_offset_or_perception",
            "gripper_close_timing",
            "trajectory_execution_deviation",
            "unknown",
            "none_clean",
        }:
            errors.append(f"{eid}: invalid physical_cause_gt")

        if record["evidence_answerability_gt"] not in {
            "answerable",
            "insufficient_evidence",
            "not_applicable_clean",
        }:
            errors.append(f"{eid}: invalid evidence_answerability_gt")

        if record["diagnostic_decision_gt"] not in {
            "target_offset_or_perception",
            "gripper_close_timing",
            "trajectory_execution_deviation",
            "insufficient_evidence",
            "clean_success",
        }:
            errors.append(f"{eid}: invalid diagnostic_decision_gt")

        confidence = record["confidence"]
        if isinstance(confidence, bool) or \
                not isinstance(confidence, (int, float)) or \
                not 0.0 <= float(confidence) <= 1.0:
            errors.append(f"{eid}: invalid confidence")

        if not str(record["review_notes"]).strip():
            errors.append(f"{eid}: review_notes required")

        admin = labels.get(eid)
        blind = pass_a_by_id.get(eid)
        if admin is None or blind is None:
            errors.append(f"{eid}: missing source join")
            continue

        # Critical no-circularity validation: compare output only against
        # independently derived admin labels and frozen Pass A answerability.
        if record["pair_group_id"] != admin["pair_group_id"]:
            errors.append(f"{eid}: pair_group mismatch")
        for name in (
            "technical_valid",
            "experimental_valid",
            "task_success",
            "intervention_verified",
            "physical_cause_gt",
        ):
            if record[name] != admin[name]:
                errors.append(f"{eid}: {name} differs from admin derivation")

        if record["evidence_answerability_gt"] != \
                blind["evidence_answerability_gt"]:
            errors.append(f"{eid}: answerability changed after admin reveal")

        if record["task_success"]:
            expected_decision = "clean_success"
            if record["physical_cause_gt"] != "none_clean":
                errors.append(f"{eid}: clean cause must be none_clean")
            if record["intervention_verified"] is not False:
                errors.append(
                    f"{eid}: clean intervention_verified must be false/not-applicable"
                )
        elif record["evidence_answerability_gt"] == "insufficient_evidence":
            expected_decision = "insufficient_evidence"
        else:
            expected_decision = record["physical_cause_gt"]

        if record["diagnostic_decision_gt"] != expected_decision:
            errors.append(f"{eid}: diagnostic decision rule violated")

    physical_counts = normalized_counter(
        [r["physical_cause_gt"] for r in records],
        list(EXPECTED_PHYSICAL_COUNTS),
    )
    decision_counts = normalized_counter(
        [r["diagnostic_decision_gt"] for r in records],
        list(EXPECTED_DECISION_COUNTS),
    )
    answerability_counts = normalized_counter(
        [r["evidence_answerability_gt"] for r in records],
        list(EXPECTED_ANSWERABILITY_COUNTS),
    )

    if physical_counts != EXPECTED_PHYSICAL_COUNTS:
        errors.append(f"physical cause counts={physical_counts}")
    if decision_counts != EXPECTED_DECISION_COUNTS:
        errors.append(f"diagnostic decision counts={decision_counts}")
    if answerability_counts != EXPECTED_ANSWERABILITY_COUNTS:
        errors.append(f"answerability counts={answerability_counts}")

    task_success_counts = Counter(r["task_success"] for r in records)
    intervention_counts = Counter(
        r["intervention_verified"] for r in records
    )

    if task_success_counts != Counter({False: 75, True: 15}):
        errors.append(f"task success counts={dict(task_success_counts)}")
    if intervention_counts != Counter({True: 75, False: 15}):
        errors.append(
            f"intervention verified counts={dict(intervention_counts)}"
        )

    summary = {
        "case_count": len(records),
        "unique_episode_ids": len({r["episode_id"] for r in records}),
        "pair_group_count": len({r["pair_group_id"] for r in records}),
        "technical_valid_true": sum(r["technical_valid"] for r in records),
        "experimental_valid_true": sum(
            r["experimental_valid"] for r in records
        ),
        "task_success_counts": {
            "false": task_success_counts.get(False, 0),
            "true": task_success_counts.get(True, 0),
        },
        "intervention_verified_counts": {
            "false": intervention_counts.get(False, 0),
            "true": intervention_counts.get(True, 0),
        },
        "physical_cause_gt_counts": physical_counts,
        "diagnostic_decision_gt_counts": decision_counts,
        "evidence_answerability_gt_counts": answerability_counts,
        "errors": errors,
    }
    return summary


def load_and_validate() -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    contract, pass_a = verify_frozen_environment()
    labels = derive_admin_labels()

    if not GT_RECORDS_PATH.exists():
        raise RuntimeError(
            "Ground truth records do not exist; run build"
        )

    records = read_jsonl(GT_RECORDS_PATH)
    summary = validate_records(records, pass_a, labels)

    print("===== DAY29 PASS B VALIDATION =====")
    for key, value in summary.items():
        print(f"{key} =", value)

    if summary["errors"]:
        raise SystemExit(1)

    print("DAY29 PASS B VALIDATION: PASS")

    return contract, pass_a, labels, records, summary


def cmd_preflight() -> None:
    contract, pass_a = verify_frozen_environment()
    labels = derive_admin_labels()

    print("===== DAY29 PASS B PREFLIGHT =====")
    print("branch =", git_output("branch", "--show-current"))
    print("head =", git_output("rev-parse", "HEAD"))
    print("pass_a_freeze_commit =", PASS_A_FREEZE_COMMIT)
    print("pass_a_records_sha256 =", sha256_file(PASS_A_RECORDS_PATH))
    print("pass_a_case_count =", len(pass_a))
    print("admin_canonical_count =", len(labels))
    print(
        "pair_group_count =",
        len({v["pair_group_id"] for v in labels.values()}),
    )
    print("future_split_materialized =", False)
    print("ground_truth_output_exists =", GT_RECORDS_PATH.exists())
    print("freeze_receipt_exists =", FREEZE_RECEIPT_PATH.exists())

    if GT_RECORDS_PATH.exists() or FREEZE_RECEIPT_PATH.exists():
        raise RuntimeError(
            "Pass B outputs already exist; refuse preflight for a fresh build"
        )

    if contract["boundaries"]["future_split_materialized"] is not False:
        raise RuntimeError("contract unexpectedly materializes split")

    print("DAY29 PASS B PREFLIGHT: PASS")


def cmd_build() -> None:
    verify_tooling_is_committed()
    _, pass_a = verify_frozen_environment()

    if GT_RECORDS_PATH.exists():
        raise RuntimeError(
            "Ground truth records already exist; refuse overwrite"
        )
    if FREEZE_RECEIPT_PATH.exists():
        raise RuntimeError(
            "Freeze receipt already exists before build"
        )

    labels = derive_admin_labels()
    records = build_records(pass_a, labels)
    summary = validate_records(records, pass_a, labels)

    if summary["errors"]:
        raise RuntimeError(
            "candidate GT invalid: " + repr(summary["errors"])
        )

    write_jsonl(GT_RECORDS_PATH, records)

    reloaded = read_jsonl(GT_RECORDS_PATH)
    if reloaded != records:
        raise RuntimeError("GT records changed after reload")

    print("ground_truth_records =", len(records))
    print("ground_truth_sha256 =", sha256_file(GT_RECORDS_PATH))
    print(
        "physical_cause_gt_counts =",
        summary["physical_cause_gt_counts"],
    )
    print(
        "diagnostic_decision_gt_counts =",
        summary["diagnostic_decision_gt_counts"],
    )
    print("admin_reveal_started = true")
    print("future_split_materialized = false")
    print("DAY29 PASS B BUILD: PASS")


def cmd_validate() -> None:
    load_and_validate()


def cmd_freeze() -> None:
    verify_tooling_is_committed()
    contract, _, _, records, summary = load_and_validate()

    if FREEZE_RECEIPT_PATH.exists():
        raise RuntimeError(
            "Ground truth freeze receipt already exists"
        )

    source_paths = list(EXPECTED_FROZEN_BLOBS)
    source_sha256 = {
        rel: sha256_file(ROOT / rel)
        for rel in source_paths
    }

    receipt = {
        "schema_version":
            "evidencemm_day29_ground_truth_freeze_receipt_v1",
        "status": "ground_truth_frozen_day29_complete",
        "tooling_commit": git_output("rev-parse", "HEAD"),
        "pass_a_freeze_commit": PASS_A_FREEZE_COMMIT,
        "pass_a_records_sha256": PASS_A_RECORDS_SHA256,
        "pass_b_operational_contract_sha256":
            sha256_file(CONTRACT_PATH),
        "ground_truth_records_path":
            "data/annotations/day29_ground_truth_records.jsonl",
        "ground_truth_records_sha256":
            sha256_file(GT_RECORDS_PATH),
        "source_git_blobs": EXPECTED_FROZEN_BLOBS,
        "source_sha256": source_sha256,
        "case_count": len(records),
        "pair_group_count": summary["pair_group_count"],
        "technical_valid_true":
            summary["technical_valid_true"],
        "experimental_valid_true":
            summary["experimental_valid_true"],
        "task_success_counts":
            summary["task_success_counts"],
        "intervention_verified_counts":
            summary["intervention_verified_counts"],
        "physical_cause_gt_counts":
            summary["physical_cause_gt_counts"],
        "diagnostic_decision_gt_counts":
            summary["diagnostic_decision_gt_counts"],
        "evidence_answerability_gt_counts":
            summary["evidence_answerability_gt_counts"],
        "human_review_completed": True,
        "admin_reveal_started": True,
        "ground_truth_frozen": True,
        "future_split_materialized": False,
        "pass_a_rewritten_after_reveal": False,
        "ground_truth_derivation_used_blind_cause_hypothesis": False,
        "day30_split_materialized": False,
    }

    FREEZE_RECEIPT_PATH.write_text(
        json.dumps(
            receipt,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        "ground_truth_records_sha256 =",
        receipt["ground_truth_records_sha256"],
    )
    print(
        "freeze_receipt_sha256 =",
        sha256_file(FREEZE_RECEIPT_PATH),
    )
    print("admin_reveal_started = true")
    print("ground_truth_frozen = true")
    print("future_split_materialized = false")
    print("DAY29 GROUND TRUTH FREEZE RECEIPT: PASS")


def cmd_audit() -> None:
    contract, pass_a, labels, records, summary = load_and_validate()

    if not FREEZE_RECEIPT_PATH.exists():
        raise RuntimeError(
            "Ground truth freeze receipt does not exist"
        )

    receipt = read_json(FREEZE_RECEIPT_PATH)

    errors: list[str] = []

    if receipt.get("schema_version") != \
            "evidencemm_day29_ground_truth_freeze_receipt_v1":
        errors.append("receipt schema mismatch")
    if receipt.get("status") != "ground_truth_frozen_day29_complete":
        errors.append("receipt status mismatch")

    tooling_commit = receipt.get("tooling_commit")
    if not isinstance(tooling_commit, str) or not tooling_commit:
        errors.append("receipt tooling_commit missing")
    else:
        if tooling_commit == PASS_A_FREEZE_COMMIT:
            errors.append("tooling_commit must be later than Pass A freeze commit")
        else:
            rc = subprocess.run(
                ["git", "merge-base", "--is-ancestor", tooling_commit, "HEAD"],
                cwd=ROOT,
                check=False,
            ).returncode
            if rc != 0:
                errors.append("receipt tooling_commit is not an ancestor of HEAD")

            for rel in (
                "scripts/day29_pass_b.py",
                "data/protocol/day29_pass_b_operational_contract.json",
                "docs/day29_pass_b_operator_guide.md",
            ):
                try:
                    frozen_blob = git_output(
                        "rev-parse", f"{tooling_commit}:{rel}"
                    )
                    current_blob = git_output(
                        "rev-parse", f"HEAD:{rel}"
                    )
                    if frozen_blob != current_blob:
                        errors.append(
                            f"tooling file changed after tooling commit: {rel}"
                        )
                except subprocess.CalledProcessError:
                    errors.append(
                        f"tooling file missing from tooling commit: {rel}"
                    )
    if receipt.get("pass_a_freeze_commit") != PASS_A_FREEZE_COMMIT:
        errors.append("receipt Pass A commit mismatch")
    if receipt.get("pass_a_records_sha256") != PASS_A_RECORDS_SHA256:
        errors.append("receipt Pass A records SHA mismatch")
    if receipt.get("pass_b_operational_contract_sha256") != \
            sha256_file(CONTRACT_PATH):
        errors.append("receipt contract SHA mismatch")
    if receipt.get("ground_truth_records_sha256") != \
            sha256_file(GT_RECORDS_PATH):
        errors.append("receipt GT records SHA mismatch")
    if receipt.get("source_git_blobs") != EXPECTED_FROZEN_BLOBS:
        errors.append("receipt source blob map mismatch")

    expected_source_sha256 = {
        rel: sha256_file(ROOT / rel)
        for rel in EXPECTED_FROZEN_BLOBS
    }
    if receipt.get("source_sha256") != expected_source_sha256:
        errors.append("receipt source SHA256 map mismatch")

    expected_fields = {
        "case_count": 90,
        "pair_group_count": 15,
        "technical_valid_true": 90,
        "experimental_valid_true": 90,
        "task_success_counts": {"false": 75, "true": 15},
        "intervention_verified_counts": {"false": 15, "true": 75},
        "physical_cause_gt_counts":
            summary["physical_cause_gt_counts"],
        "diagnostic_decision_gt_counts":
            summary["diagnostic_decision_gt_counts"],
        "evidence_answerability_gt_counts":
            summary["evidence_answerability_gt_counts"],
        "human_review_completed": True,
        "admin_reveal_started": True,
        "ground_truth_frozen": True,
        "future_split_materialized": False,
        "pass_a_rewritten_after_reveal": False,
        "ground_truth_derivation_used_blind_cause_hypothesis": False,
        "day30_split_materialized": False,
    }
    for key, expected in expected_fields.items():
        if receipt.get(key) != expected:
            errors.append(
                f"receipt {key} mismatch: "
                f"expected={expected!r} actual={receipt.get(key)!r}"
            )

    # Re-check Pass A content hash after admin reveal.
    if sha256_file(PASS_A_RECORDS_PATH) != PASS_A_RECORDS_SHA256:
        errors.append("Pass A records changed after admin reveal")

    # Re-derive labels independently; do not compare GT against blind causes.
    rebuilt = build_records(pass_a, labels)
    if rebuilt != records:
        errors.append("frozen GT differs from independent admin re-derivation")

    print("===== DAY29 PASS B FREEZE AUDIT =====")
    print("case_count =", len(records))
    print("pair_group_count =", summary["pair_group_count"])
    print(
        "ground_truth_records_sha256 =",
        sha256_file(GT_RECORDS_PATH),
    )
    print(
        "freeze_receipt_sha256 =",
        sha256_file(FREEZE_RECEIPT_PATH),
    )
    print("pass_a_records_sha256 =", sha256_file(PASS_A_RECORDS_PATH))
    print("admin_reveal_started =", receipt.get("admin_reveal_started"))
    print("ground_truth_frozen =", receipt.get("ground_truth_frozen"))
    print(
        "future_split_materialized =",
        receipt.get("future_split_materialized"),
    )
    print(
        "ground_truth_derivation_used_blind_cause_hypothesis =",
        receipt.get(
            "ground_truth_derivation_used_blind_cause_hypothesis"
        ),
    )
    print("errors =", errors)

    if errors:
        raise SystemExit(1)

    print("DAY29 PASS B AUDIT: PASS")
    print("DAY29: CLOSED / FROZEN")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("build")
    sub.add_parser("validate")
    sub.add_parser("freeze")
    sub.add_parser("audit")
    args = parser.parse_args()

    if args.command == "preflight":
        cmd_preflight()
    elif args.command == "build":
        cmd_build()
    elif args.command == "validate":
        cmd_validate()
    elif args.command == "freeze":
        cmd_freeze()
    elif args.command == "audit":
        cmd_audit()


if __name__ == "__main__":
    main()
