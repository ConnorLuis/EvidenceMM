#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CFG = ROOT / "configs/day34_metrics_error_efficiency.json"
CONTRACT = ROOT / "data/protocol/day34_metrics_error_efficiency_operational_contract.json"

D31_PRED = ROOT / "data/eval/day31_development_baseline_predictions.jsonl"
D31_MET = ROOT / "data/eval/day31_development_baseline_metrics.json"
D31_CFG = ROOT / "configs/day31_root_cause_baseline.json"
D31_PROMPT = ROOT / "data/protocol/day31_baseline_prompt_contract.json"
D31_SCRIPT = ROOT / "scripts/day31_root_cause_baseline.py"

D32_SCORE = ROOT / "data/eval/day32_development_scoring_predictions.jsonl"
D32_CAL_PRED = ROOT / "data/eval/day32_development_calibrated_predictions.jsonl"
D32_MET = ROOT / "data/eval/day32_development_calibrated_metrics.json"
D32_CFG = ROOT / "configs/day32_development_calibration.json"
D32_FROZEN = ROOT / "data/protocol/day32_frozen_diagnostic_config.json"
D32_PROMPT = ROOT / "data/protocol/day32_scoring_prompt_contract.json"
D32_SCRIPT = ROOT / "scripts/day32_development_calibration.py"

D33_SCORE = ROOT / "data/eval/day33_heldout_scoring_predictions.jsonl"
D33_FINAL_PRED = ROOT / "data/eval/day33_heldout_final_predictions.jsonl"
D33_MET = ROOT / "data/eval/day33_heldout_final_metrics.json"
D33_RECEIPT = ROOT / "data/protocol/day33_heldout_eval_freeze_receipt.json"

SPLIT = ROOT / "data/splits/day30_episode_split.jsonl"
GT = ROOT / "data/annotations/day29_ground_truth_records.jsonl"
SRC = ROOT / "data/protocol/day28_registered_source_manifest.csv"
RAW_CFG = ROOT / "configs/day28_raw_audit.yaml"

FINAL_METRICS = ROOT / "data/eval/day34_final_metrics_report.json"
ERROR_ANALYSIS = ROOT / "data/eval/day34_error_analysis.json"
PER_CASE = ROOT / "data/eval/day34_per_case_analysis.jsonl"
E2E_PROFILE = ROOT / "data/eval/day34_development_e2e_profile.json"
EFFICIENCY = ROOT / "data/eval/day34_efficiency_report.json"
RECEIPT = ROOT / "data/protocol/day34_metrics_freeze_receipt.json"

WORK = ROOT / "reports/day34_e2e_profile_work"
PARTIAL_PROFILE = WORK / "partial_profile_rows.jsonl"
STARTUP_PROFILE = WORK / "startup_profile.json"

DAY33_FINAL_COMMIT = "a8a8b796eecdab6118c9ad637c41f7c2b987304d"

HASHES = {
    D31_PRED: "6c323f1432723e897306f20a8a0804c713b7f7b8c8d93a48b99492f9c394d768",
    D31_MET: "2e57c70b91eda0cc385be63a218d0a4802ca3a3800953749919330ec098437cd",
    D31_CFG: "eef3ed506ce434c9df2aafd236d4c848cb640bbbce7646fa6d143ac4798eb63f",
    D31_PROMPT: "faee60d40b710005a265ef7c657a2b19921c8b40c41ddda8c3d69d4916dbd79f",
    D32_SCORE: "a39c5af8029bb624b78798e1861db7db0ece86d75c72598688bf7af55181acef",
    D32_CAL_PRED: "bb7b1b0d33d879d0627a12a298f0434afda9813d334fafb9d835bcca718bf9ee",
    D32_MET: "95effed7203e612626b0a1d04cf79e7b24e33c0664265b1c9866f6d1a6687d46",
    D32_CFG: "f2cc794fe3bc718efbe6a639c51ade321447a1539f29373ad621ad47d88d8b61",
    D32_FROZEN: "15a11ebebf06edf10ba6cc8e015f1b04da4693a57800692025322ad69c63685d",
    D32_PROMPT: "f02290eb9fd0ad3de92363352fc921d13e9ce318de4d6db969149ec37fcd2cf7",
    D33_SCORE: "787dd204275c0f453d3700e95d165b607ad91946e38f866a7f0250a1bc8fde06",
    D33_FINAL_PRED: "c6046cc34de98cf0ab892236028a96d78a990145dfdbdd98fe68f90a53ea3289",
    D33_MET: "4165d639e9ff31bec09f76d133e45ba344d462cba04f0c72b1d97e1c65958369",
    D33_RECEIPT: "18663b49577187c305be349aa733d4697976142036b81de8457b9dfeb2f9c711",
    SPLIT: "0b37a499904dcf8568ac39a9641097f7d73c952a01a79f00cbcda2b3b7793312",
    GT: "e03ec1ab443e4fb4dab606e16fbae8439411d7c3acbcf5f078ed5a0660d389bf",
}

D31_SCRIPT_BLOB = "afd55294495b6cb552a3484662072fb8d46c3ef9"
D32_SCRIPT_BLOB = "c39df7aaa7d5dcb2daf9cd3f985d94e68aba7f6a"
SOURCE_MANIFEST_BLOB = "46a8a5655c17ca20f5aae88c1be05c18092a02c6"
RAW_CONFIG_BLOB = "eaef86f7aa514845a1160aa85d4d25cc4a79f279"

CAUSE = (
    "target_offset_or_perception",
    "gripper_close_timing",
    "trajectory_execution_deviation",
)
LABELS = CAUSE + ("insufficient_evidence", "clean_success")
SUBSTANTIVE = CAUSE + ("clean_success",)

TOOLING = (
    "configs/day34_metrics_error_efficiency.json",
    "data/protocol/day34_metrics_error_efficiency_operational_contract.json",
    "docs/day34_metrics_error_efficiency.md",
    "scripts/day34_metrics_error_efficiency.py",
    "tests/test_day34_metrics_error_efficiency.py",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_ancestor(commit: str) -> None:
    rc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode
    if rc != 0:
        raise RuntimeError(f"required ancestor missing: {commit}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        handle.flush()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tooling_commit() -> str:
    for rel in TOOLING:
        git("rev-parse", f"HEAD:{rel}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *TOOLING],
        cwd=ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Day34 tooling has uncommitted changes")
    return git("rev-parse", "HEAD")


def verify_frozen(*, require_raw: bool = False) -> dict[str, Any]:
    if git("branch", "--show-current") != "master":
        raise RuntimeError("Day34 must run on master")
    require_ancestor(DAY33_FINAL_COMMIT)

    if git("rev-parse", "HEAD:scripts/day31_root_cause_baseline.py") != D31_SCRIPT_BLOB:
        raise RuntimeError("Day31 script drift")
    if git("rev-parse", "HEAD:scripts/day32_development_calibration.py") != D32_SCRIPT_BLOB:
        raise RuntimeError("Day32 script drift")
    if git("rev-parse", "HEAD:data/protocol/day28_registered_source_manifest.csv") != SOURCE_MANIFEST_BLOB:
        raise RuntimeError("Day28 source manifest drift")
    if git("rev-parse", "HEAD:configs/day28_raw_audit.yaml") != RAW_CONFIG_BLOB:
        raise RuntimeError("Day28 raw config drift")

    for path, expected in HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen SHA mismatch: {path}\nexpected={expected}\nactual={actual}"
            )

    receipt = read_json(D33_RECEIPT)
    if receipt["status"] != "single_frozen_heldout_final_evaluation_day33_complete":
        raise RuntimeError("Day33 receipt status drift")
    if receipt["held_out_final_evaluation_count_consumed"] != 1:
        raise RuntimeError("Day33 single evaluation count drift")
    for key in (
        "prompt_changed_after_authorization",
        "calibration_changed_after_authorization",
        "evidence_selection_changed_after_authorization",
        "retrieval_changed_after_authorization",
        "result_conditioned_regeneration_used",
        "parse_failure_retry_used",
        "tuning_after_heldout",
    ):
        if receipt[key] is not False:
            raise RuntimeError(f"Day33 anti-tuning flag drift: {key}")

    split = read_jsonl(SPLIT)
    dev = [row for row in split if row["split"] == "development"]
    held = [row for row in split if row["split"] == "held_out"]
    if len(dev) != 60 or len(held) != 30:
        raise RuntimeError("Day30 split population drift")

    dev_ids = [row["episode_id"] for row in dev]
    held_ids = [row["episode_id"] for row in held]
    if set(dev_ids) & set(held_ids):
        raise RuntimeError("development/held-out overlap")

    env: dict[str, Any] = {
        "config": read_json(CFG),
        "dev_rows": dev,
        "held_rows": held,
        "dev_ids": dev_ids,
        "held_ids": held_ids,
        "episode_to_group": {
            row["episode_id"]: row["pair_group_id"] for row in split
        },
    }

    if require_raw:
        import yaml
        raw_cfg = yaml.safe_load(RAW_CFG.read_text(encoding="utf-8"))
        root = Path(raw_cfg["raw_source"]["compatibility_wsl_root"])
        if not root.is_dir():
            raise RuntimeError(f"raw root unavailable: {root}")
        env["raw_root"] = root

    return env


def load_gt_for_ids(ids: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    pattern = re.compile(r'"episode_id":"([^"]+)"')
    for line in GT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = pattern.search(line)
        if not match:
            raise RuntimeError("GT row missing episode_id")
        episode_id = match.group(1)
        if episode_id not in ids:
            continue
        row = json.loads(line)
        out[episode_id] = {
            "physical_cause_gt": row["physical_cause_gt"],
            "diagnostic_decision_gt": row["diagnostic_decision_gt"],
            "evidence_answerability_gt": row["evidence_answerability_gt"],
            "task_success": row["task_success"],
        }
    if set(out) != ids:
        raise RuntimeError(
            f"GT coverage mismatch: expected={len(ids)} actual={len(out)}"
        )
    return out


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if not 0 <= p <= 100:
        raise ValueError("percentile outside [0,100]")
    ordered = sorted(float(x) for x in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p / 100.0
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def stats(values: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(vals),
        "mean": sum(vals) / len(vals),
        "min": min(vals),
        "p50": percentile(vals, 50),
        "p95": percentile(vals, 95),
        "max": max(vals),
    }


def confusion_and_report(
    y_true: list[str],
    y_pred: list[str],
    labels: tuple[str, ...] = LABELS,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, Any]]]:
    matrix = {
        true: {pred: 0 for pred in labels}
        for true in labels
    }
    for true, pred in zip(y_true, y_pred):
        if true not in matrix or pred not in matrix[true]:
            raise RuntimeError(f"unknown label true={true!r} pred={pred!r}")
        matrix[true][pred] += 1

    report: dict[str, dict[str, Any]] = {}
    for label in labels:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[true][label] for true in labels)
        fp = predicted - tp
        fn = support - tp
        precision = None if tp + fp == 0 else tp / (tp + fp)
        recall = None if tp + fn == 0 else tp / (tp + fn)
        if precision is None or recall is None or precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        report[label] = {
            "support": support,
            "predicted_count": predicted,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return matrix, report


def select_profile_ids(
    development_ids: list[str],
    *,
    seed: str,
    count: int,
) -> tuple[list[str], list[dict[str, str]]]:
    ranked = [
        {
            "episode_id": episode_id,
            "sha256": hashlib.sha256(
                f"{seed}|{episode_id}".encode("utf-8")
            ).hexdigest(),
        }
        for episode_id in development_ids
    ]
    ranked.sort(key=lambda item: item["sha256"])
    return [item["episode_id"] for item in ranked[:count]], ranked


def parse_error_family(value: Any) -> str:
    if value is None:
        return "none"
    text = str(value)
    return text.split(":", 1)[0] if ":" in text else text


def error_category(
    *,
    gt_decision: str,
    pred_decision: str,
    parse_ok: bool,
) -> str:
    if gt_decision == pred_decision:
        return "correct"
    if not parse_ok:
        return "parse_failure_abstention"
    if gt_decision == "clean_success":
        if pred_decision in CAUSE:
            return "clean_false_positive_cause"
        if pred_decision == "insufficient_evidence":
            return "clean_false_abstention"
        return "clean_other_error"
    if pred_decision == "insufficient_evidence":
        return "false_abstention"
    if pred_decision == "clean_success":
        return "failure_predicted_clean"
    if pred_decision in CAUSE:
        return "cause_confusion"
    return "other_error"


def build_static_payloads(env: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    d32 = load_module(D32_SCRIPT, "day34_d32_static")

    d31_metrics = read_json(D31_MET)
    d32_metrics = read_json(D32_MET)
    d33_metrics = read_json(D33_MET)

    d32_scores = read_jsonl(D32_SCORE)
    d33_scores = read_jsonl(D33_SCORE)
    d33_final = read_jsonl(D33_FINAL_PRED)

    if len(d32_scores) != 60 or len(d33_scores) != 30 or len(d33_final) != 30:
        raise RuntimeError("frozen prediction count drift")

    held_gt = load_gt_for_ids(set(env["held_ids"]))
    score_by_id = {row["episode_id"]: row for row in d33_scores}
    final_by_id = {row["episode_id"]: row for row in d33_final}

    if set(score_by_id) != set(env["held_ids"]) or set(final_by_id) != set(env["held_ids"]):
        raise RuntimeError("Day33 held-out population drift")

    decisions = {
        episode_id: final_by_id[episode_id]["diagnostic_decision"]
        for episode_id in env["held_ids"]
    }
    recomputed = d32.metrics(env["held_ids"], held_gt, decisions)

    expected_primary = {
        "answerable_three_class_macro_f1":
            recomputed["answerable_three_class_macro_f1"],
        "failed_case_four_way_diagnostic_macro_f1":
            recomputed["failed_case_four_way_diagnostic_macro_f1"],
        "abstention_accuracy": recomputed["abstention_accuracy"],
        "false_answer_rate": recomputed["false_answer_rate"],
        "false_abstention_rate": recomputed["false_abstention_rate"],
        "clean_control_false_positive_cause_rate":
            recomputed["clean_control_false_positive_cause_rate"],
    }
    expected_secondary = {
        "substantive_four_class_macro_f1":
            recomputed["substantive_four_class_macro_f1"],
        "clean_control_accuracy": recomputed["clean_control_accuracy"],
        "heldout_decision_accuracy": recomputed["development_decision_accuracy"],
        "prediction_parse_rate":
            sum(bool(row["parse_ok"]) for row in d33_scores) / 30,
    }
    if d33_metrics["primary_metrics"] != expected_primary:
        raise RuntimeError("Day33 primary metrics do not recompute")
    if d33_metrics["secondary_metrics"] != expected_secondary:
        raise RuntimeError("Day33 secondary metrics do not recompute")

    dev_full = d32_metrics["full_development_metrics"]
    overlap = (
        "answerable_three_class_macro_f1",
        "failed_case_four_way_diagnostic_macro_f1",
        "substantive_four_class_macro_f1",
        "abstention_accuracy",
        "false_abstention_rate",
        "clean_control_false_positive_cause_rate",
        "clean_control_accuracy",
    )
    held_metric_lookup = {
        **d33_metrics["primary_metrics"],
        **d33_metrics["secondary_metrics"],
    }
    gap: dict[str, Any] = {}
    for key in overlap:
        dev_value = dev_full.get(key)
        held_value = held_metric_lookup.get(key)
        gap[key] = {
            "development": dev_value,
            "held_out": held_value,
            "heldout_minus_development":
                None if dev_value is None or held_value is None
                else held_value - dev_value,
        }

    metrics_payload = {
        "schema_version": "evidencemm_day34_final_metrics_report_v1",
        "status": "post_day33_final_metrics_recomputed",
        "day33_final_commit": DAY33_FINAL_COMMIT,
        "day31_development_baseline": {
            "primary_metrics": d31_metrics["primary_metrics"],
            "secondary_metrics": d31_metrics["secondary_metrics"],
            "prediction_parse_ok_count": d31_metrics["prediction_parse_ok_count"],
            "prediction_parse_failure_count":
                d31_metrics["prediction_parse_failure_count"],
        },
        "day32_development_calibrated": {
            "internal_fit_metrics": d32_metrics["internal_fit_metrics"],
            "internal_validation_metrics": d32_metrics["internal_validation_metrics"],
            "full_development_metrics": d32_metrics["full_development_metrics"],
            "score_parse_ok_count": d32_metrics["score_parse_ok_count"],
            "score_parse_failure_count": d32_metrics.get(
                "score_parse_failure_count",
                60 - int(d32_metrics["score_parse_ok_count"]),
            ),
        },
        "day33_single_frozen_heldout": {
            "primary_metrics": d33_metrics["primary_metrics"],
            "secondary_metrics": d33_metrics["secondary_metrics"],
            "gt_support": d33_metrics["gt_support"],
            "score_parse_ok_count": d33_metrics["score_parse_ok_count"],
            "score_parse_failure_count": d33_metrics["score_parse_failure_count"],
            "final_decision_counts": d33_metrics["final_decision_counts"],
            "held_out_final_evaluation_count_consumed":
                d33_metrics["held_out_final_evaluation_count_consumed"],
        },
        "development_to_heldout_generalization_gap": gap,
        "metric_family_coverage": {
            "diagnosis": {
                "status": "measured",
                "source": "Day33 single frozen held-out evaluation",
            },
            "efficiency": {
                "status": "measured",
                "source": (
                    "frozen Day31-Day33 generation logs plus Day34 "
                    "development-only E2E profile"
                ),
            },
            "temporal_localization": {
                "status": "not_emitted_by_root_cause_v2_predictor",
                "historical_separate_result":
                    "data/eval/day20_heldout_interval_eval.json",
                "merged_into_day33_final_metrics": False,
            },
            "grounding": {
                "status": "not_applicable_to_day31_day33_no_retrieval_baseline",
                "retrieval_used": False,
                "manual_corpus_used": False,
            },
        },
        "post_heldout_tuning_performed": False,
    }

    y_true = [
        held_gt[eid]["diagnostic_decision_gt"]
        for eid in env["held_ids"]
    ]
    y_pred = [
        final_by_id[eid]["diagnostic_decision"]
        for eid in env["held_ids"]
    ]
    matrix, class_report = confusion_and_report(y_true, y_pred)

    raw_top_counts = Counter(
        row["derived_top_substantive_decision"]
        for row in d33_scores
        if row["parse_ok"]
    )
    final_counts = Counter(y_pred)

    transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for eid in env["held_ids"]:
        score = score_by_id[eid]
        final = final_by_id[eid]
        raw_top = (
            score["derived_top_substantive_decision"]
            if score["parse_ok"]
            else "parse_failure"
        )
        transition_counts[raw_top][final["diagnostic_decision"]] += 1

    parse_failures = [
        score_by_id[eid]
        for eid in env["held_ids"]
        if not score_by_id[eid]["parse_ok"]
    ]
    parse_error_families = Counter(
        parse_error_family(row.get("parse_error"))
        for row in parse_failures
    )
    parse_failure_gt = Counter(
        held_gt[row["episode_id"]]["diagnostic_decision_gt"]
        for row in parse_failures
    )

    clean_ids = [
        eid for eid in env["held_ids"]
        if held_gt[eid]["task_success"] is True
    ]
    clean_prediction_counts = Counter(
        final_by_id[eid]["diagnostic_decision"]
        for eid in clean_ids
    )

    pair_group_report: dict[str, Any] = {}
    held_groups = sorted(
        {env["episode_to_group"][eid] for eid in env["held_ids"]}
    )
    for group in held_groups:
        ids = [
            eid for eid in env["held_ids"]
            if env["episode_to_group"][eid] == group
        ]
        correct = sum(
            held_gt[eid]["diagnostic_decision_gt"]
            == final_by_id[eid]["diagnostic_decision"]
            for eid in ids
        )
        parse_ok = sum(bool(score_by_id[eid]["parse_ok"]) for eid in ids)
        pair_group_report[group] = {
            "episode_count": len(ids),
            "accuracy": correct / len(ids),
            "parse_rate": parse_ok / len(ids),
            "gt_counts": dict(sorted(Counter(
                held_gt[eid]["diagnostic_decision_gt"] for eid in ids
            ).items())),
            "prediction_counts": dict(sorted(Counter(
                final_by_id[eid]["diagnostic_decision"] for eid in ids
            ).items())),
        }

    raw_gripper_ids = [
        eid for eid in env["held_ids"]
        if score_by_id[eid]["parse_ok"]
        and score_by_id[eid]["derived_top_substantive_decision"]
        == "gripper_close_timing"
    ]
    raw_gripper_transitions = Counter(
        final_by_id[eid]["diagnostic_decision"]
        for eid in raw_gripper_ids
    )

    categories = Counter()
    per_case_rows: list[dict[str, Any]] = []
    for eid in env["held_ids"]:
        gt_row = held_gt[eid]
        score = score_by_id[eid]
        final = final_by_id[eid]
        category = error_category(
            gt_decision=gt_row["diagnostic_decision_gt"],
            pred_decision=final["diagnostic_decision"],
            parse_ok=bool(score["parse_ok"]),
        )
        categories[category] += 1
        per_case_rows.append({
            "schema_version": "evidencemm_day34_per_case_analysis_v1",
            "episode_id": eid,
            "pair_group_id": env["episode_to_group"][eid],
            "gt_diagnostic_decision": gt_row["diagnostic_decision_gt"],
            "gt_physical_cause": gt_row["physical_cause_gt"],
            "task_success": gt_row["task_success"],
            "prediction": final["diagnostic_decision"],
            "correct":
                gt_row["diagnostic_decision_gt"]
                == final["diagnostic_decision"],
            "score_parse_ok": bool(score["parse_ok"]),
            "parse_error_family":
                parse_error_family(score.get("parse_error")),
            "raw_top_substantive_decision":
                score["derived_top_substantive_decision"],
            "calibrated_top_substantive_decision":
                final["top_substantive_decision"],
            "top_probability": final["top_probability"],
            "margin": final["margin"],
            "error_category": category,
            "generation_latency_sec": score["latency_sec"],
            "peak_gpu_memory_mb": score["peak_gpu_memory_mb"],
        })

    gt_gripper_support = class_report["gripper_close_timing"]["support"]
    error_payload = {
        "schema_version": "evidencemm_day34_error_analysis_v1",
        "status": "post_heldout_error_analysis_complete",
        "held_out_episode_count": 30,
        "confusion_matrix": matrix,
        "per_class_report": class_report,
        "error_category_counts": dict(sorted(categories.items())),
        "parse_failure_analysis": {
            "count": len(parse_failures),
            "rate": len(parse_failures) / 30,
            "error_family_counts": dict(sorted(parse_error_families.items())),
            "gt_label_counts": dict(sorted(parse_failure_gt.items())),
            "all_finalized_as_insufficient_evidence": all(
                final_by_id[row["episode_id"]]["diagnostic_decision"]
                == "insufficient_evidence"
                for row in parse_failures
            ),
        },
        "clean_control_analysis": {
            "gt_clean_count": len(clean_ids),
            "prediction_counts": dict(sorted(clean_prediction_counts.items())),
            "correct_clean_count":
                clean_prediction_counts.get("clean_success", 0),
            "cause_false_positive_count":
                sum(clean_prediction_counts.get(label, 0) for label in CAUSE),
        },
        "calibration_shift_analysis": {
            "frozen_log_biases":
                read_json(D32_FROZEN)["selected_log_biases"],
            "confidence_threshold":
                read_json(D32_FROZEN)["confidence_threshold"],
            "margin_threshold":
                read_json(D32_FROZEN)["margin_threshold"],
            "raw_top_counts_parse_ok": dict(sorted(raw_top_counts.items())),
            "final_decision_counts": dict(sorted(final_counts.items())),
            "raw_top_to_final_transition_counts": {
                raw: dict(sorted(counter.items()))
                for raw, counter in sorted(transition_counts.items())
            },
        },
        "gripper_class_analysis": {
            "gt_support": gt_gripper_support,
            "final_prediction_count":
                class_report["gripper_close_timing"]["predicted_count"],
            "final_recall": class_report["gripper_close_timing"]["recall"],
            "raw_top_gripper_count": len(raw_gripper_ids),
            "raw_top_gripper_final_transitions":
                dict(sorted(raw_gripper_transitions.items())),
        },
        "pair_group_report": pair_group_report,
        "development_vs_heldout_parse_rate": {
            "development_day32": sum(bool(r["parse_ok"]) for r in d32_scores) / 60,
            "heldout_day33": sum(bool(r["parse_ok"]) for r in d33_scores) / 30,
        },
        "analysis_only_no_tuning": True,
    }
    return metrics_payload, error_payload, per_case_rows


def source_manifest() -> dict[str, dict[str, str]]:
    with SRC.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["episode_id"]: row for row in rows}


def validate_profile_payload(
    payload: dict[str, Any],
    env: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    profile_cfg = env["config"]["efficiency_profile"]
    selected, ranked = select_profile_ids(
        env["dev_ids"],
        seed=profile_cfg["episode_selection_seed"],
        count=int(profile_cfg["episode_count"]),
    )
    if payload.get("selected_episode_ids") != selected:
        errors.append("profile selected episode IDs drift")
    if payload.get("profile_episode_count") != len(selected):
        errors.append("profile episode count drift")
    if payload.get("ground_truth_accessed") is not False:
        errors.append("profile claims GT access")
    rows = payload.get("episodes")
    if not isinstance(rows, list) or len(rows) != len(selected):
        errors.append("profile episode rows missing")
        return errors
    ids = [row.get("episode_id") for row in rows]
    if ids != selected:
        errors.append("profile row order/IDs drift")
    if not set(ids).issubset(set(env["dev_ids"])):
        errors.append("profile contains non-development episode")
    forbidden = {
        "physical_cause_gt",
        "diagnostic_decision_gt",
        "task_success",
        "evidence_answerability_gt",
        "pair_group_id",
    }
    for row in rows:
        if forbidden & set(row):
            errors.append(f"{row.get('episode_id')}: profile contains GT/admin field")
        for key in (
            "evidence_preparation_sec",
            "processor_preparation_sec",
            "generation_sec",
            "parse_and_calibration_sec",
            "episode_end_to_end_sec",
        ):
            value = row.get(key)
            if not isinstance(value, (int, float)) or value < 0:
                errors.append(f"{row.get('episode_id')}: invalid timing {key}")
    return errors


def build_efficiency_payload(
    env: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    errors = validate_profile_payload(profile, env)
    if errors:
        raise RuntimeError("invalid E2E profile: " + repr(errors))

    d31 = read_jsonl(D31_PRED)
    d32 = read_jsonl(D32_SCORE)
    d33 = read_jsonl(D33_SCORE)

    def logged(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "episode_count": len(rows),
            "generation_latency_sec": stats([
                float(row["latency_sec"]) for row in rows
            ]),
            "peak_gpu_memory_mb": stats([
                float(row["peak_gpu_memory_mb"])
                for row in rows
                if row.get("peak_gpu_memory_mb") is not None
            ]),
            "latency_semantics": "model_generation_only",
        }

    profile_rows = profile["episodes"]
    e2e_values = [
        float(row["episode_end_to_end_sec"])
        for row in profile_rows
    ]
    generation_values = [
        float(row["generation_sec"])
        for row in profile_rows
    ]

    source_frames = int(env["config"]["efficiency_reporting"]["source_frame_count"])
    evidence_frames = int(env["config"]["efficiency_reporting"]["evidence_frame_count"])
    density = evidence_frames / source_frames

    return {
        "schema_version": "evidencemm_day34_efficiency_report_v1",
        "status": "efficiency_summary_complete",
        "immutable_logged_measurements": {
            "day31_development_baseline": logged(d31),
            "day32_development_scoring": logged(d32),
            "day33_frozen_heldout_scoring": logged(d33),
        },
        "development_only_e2e_profile": {
            "selection_seed": profile["selection_seed"],
            "selected_episode_ids": profile["selected_episode_ids"],
            "profile_episode_count": profile["profile_episode_count"],
            "startup": profile["startup"],
            "stage_summaries": profile["stage_summaries"],
            "warm_episode_e2e_sec": stats(e2e_values),
            "generation_sec": stats(generation_values),
            "throughput_episodes_per_minute_based_on_warm_e2e_mean":
                None if not e2e_values
                else 60.0 / (sum(e2e_values) / len(e2e_values)),
            "cold_start_first_episode_estimate_sec":
                profile["startup"]["model_load_sec"]
                + profile["startup"]["processor_load_sec"]
                + e2e_values[0],
            "ground_truth_accessed": False,
            "heldout_inference_count": 0,
            "latency_semantics":
                "raw_episode_to_calibrated_decision_excluding_model_load",
        },
        "evidence_density": {
            "source_frame_count": source_frames,
            "selected_frame_count": evidence_frames,
            "density": density,
            "frame_reduction_ratio": 1.0 - density,
            "frame_count_compression_ratio": source_frames / evidence_frames,
        },
        "measurement_scope_notes": {
            "day31_day32_day33_latency":
                "generation-only measurements recorded during frozen runs",
            "day34_e2e_profile":
                "development-only performance profile; no held-out rerun",
            "model_load":
                "reported separately and excluded from warm per-episode E2E",
            "latency_is_hardware_specific": True,
        },
        "post_heldout_tuning_performed": False,
    }


def cmd_preflight() -> None:
    commit = tooling_commit()
    env = verify_frozen(require_raw=True)

    existing = [
        path for path in (
            FINAL_METRICS,
            ERROR_ANALYSIS,
            PER_CASE,
            E2E_PROFILE,
            EFFICIENCY,
            RECEIPT,
        )
        if path.exists()
    ]
    if existing:
        raise RuntimeError(f"Day34 output already exists: {existing}")

    profile_cfg = env["config"]["efficiency_profile"]
    selected, ranked = select_profile_ids(
        env["dev_ids"],
        seed=profile_cfg["episode_selection_seed"],
        count=int(profile_cfg["episode_count"]),
    )

    print("===== DAY34 PREFLIGHT =====")
    print("tooling_commit =", commit)
    print("day33_final_commit =", DAY33_FINAL_COMMIT)
    print("heldout_final_evaluation_count_consumed = 1")
    print("day34_heldout_inference_count = 0")
    print("post_heldout_tuning_allowed = false")
    print("profile_split = development")
    print("profile_episode_count =", len(selected))
    print("profile_selected_episode_ids =", selected)
    print("raw_root =", env["raw_root"])
    print("DAY34 PREFLIGHT: PASS")


def cmd_analyze() -> None:
    env = verify_frozen(require_raw=False)
    metrics_payload, error_payload, per_case_rows = build_static_payloads(env)

    for path, payload in (
        (FINAL_METRICS, metrics_payload),
        (ERROR_ANALYSIS, error_payload),
    ):
        if path.exists():
            if read_json(path) != payload:
                raise RuntimeError(f"existing Day34 artifact differs: {path}")
        else:
            write_json(path, payload)

    if PER_CASE.exists():
        if read_jsonl(PER_CASE) != per_case_rows:
            raise RuntimeError("existing Day34 per-case artifact differs")
    else:
        write_jsonl(PER_CASE, per_case_rows)

    print("===== DAY34 STATIC METRICS + ERROR ANALYSIS =====")
    held = metrics_payload["day33_single_frozen_heldout"]
    print("heldout_primary_metrics =", held["primary_metrics"])
    print("heldout_secondary_metrics =", held["secondary_metrics"])
    print("error_category_counts =", error_payload["error_category_counts"])
    print("gripper_class_analysis =", error_payload["gripper_class_analysis"])
    print("clean_control_analysis =", error_payload["clean_control_analysis"])
    print("parse_failure_analysis =", error_payload["parse_failure_analysis"])
    print("final_metrics_sha256 =", sha256(FINAL_METRICS))
    print("error_analysis_sha256 =", sha256(ERROR_ANALYSIS))
    print("per_case_sha256 =", sha256(PER_CASE))
    print("DAY34 STATIC ANALYSIS: PASS")


def cmd_profile() -> None:
    commit = tooling_commit()
    env = verify_frozen(require_raw=True)
    profile_cfg = env["config"]["efficiency_profile"]
    selected, ranked = select_profile_ids(
        env["dev_ids"],
        seed=profile_cfg["episode_selection_seed"],
        count=int(profile_cfg["episode_count"]),
    )

    if E2E_PROFILE.exists():
        payload = read_json(E2E_PROFILE)
        errors = validate_profile_payload(payload, env)
        if errors:
            raise RuntimeError("existing E2E profile invalid: " + repr(errors))
        print("DAY34 DEVELOPMENT E2E PROFILE already complete: PASS")
        return

    partial = read_jsonl(PARTIAL_PROFILE) if PARTIAL_PROFILE.exists() else []
    partial_ids = [row["episode_id"] for row in partial]
    if len(partial_ids) != len(set(partial_ids)):
        raise RuntimeError("duplicate Day34 profile partial rows")
    if partial_ids != selected[:len(partial_ids)]:
        raise RuntimeError("Day34 profile partial prefix drift")
    done = set(partial_ids)
    remaining = [eid for eid in selected if eid not in done]

    print("===== DAY34 DEVELOPMENT-ONLY E2E PROFILE =====")
    print("tooling_commit =", commit)
    print("selected_episode_ids =", selected)
    print("completed_before_run =", len(done))
    print("remaining =", len(remaining))
    print("heldout_inference_count = 0")
    print("ground_truth_accessed = false")

    if remaining:
        import torch
        import transformers
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from evidencemm.state_action_selection import load_state_action_samples

        d31 = load_module(D31_SCRIPT, "day34_d31_profile")
        d32 = load_module(D32_SCRIPT, "day34_d32_profile")
        d31_cfg = read_json(D31_CFG)
        d32_cfg = read_json(D32_CFG)
        frozen = read_json(D32_FROZEN)
        prompt = read_json(D32_PROMPT)
        day31_rows = {
            row["episode_id"]: row
            for row in read_jsonl(D31_PRED)
        }
        manifest = source_manifest()
        raw_root = env["raw_root"]

        model_name = frozen["model_name"]
        loading = frozen["model_loading"]

        model_start = time.perf_counter()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
            attn_implementation=loading["attn_implementation"],
            local_files_only=True,
        )
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_load_sec = time.perf_counter() - model_start

        processor_start = time.perf_counter()
        processor = AutoProcessor.from_pretrained(
            model_name,
            local_files_only=True,
        )
        processor_load_sec = time.perf_counter() - processor_start

        current_startup = {
            "model_name": model_name,
            "model_load_sec": model_load_sec,
            "processor_load_sec": processor_load_sec,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime_version": torch.version.cuda,
            "gpu_name":
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
            "model_resident_gpu_memory_mb":
                torch.cuda.memory_allocated() / 1024 / 1024
                if torch.cuda.is_available()
                else None,
        }
        if not STARTUP_PROFILE.exists():
            write_json(STARTUP_PROFILE, current_startup)
        startup = read_json(STARTUP_PROFILE)

        candidate = {
            "biases": frozen["selected_log_biases"],
            "confidence_threshold": frozen["confidence_threshold"],
            "margin_threshold": frozen["margin_threshold"],
        }

        for ordinal, episode_id in enumerate(remaining, 1):
            print(
                f"[{len(done)+ordinal:02d}/{len(selected):02d}] "
                f"episode={episode_id}",
                flush=True,
            )
            if episode_id not in day31_rows:
                raise RuntimeError(
                    f"profile development episode missing Day31 row: {episode_id}"
                )
            day31_row = day31_rows[episode_id]
            selected_frames = [
                int(x) for x in day31_row["selected_frame_indices"]
            ]

            episode_start = time.perf_counter()

            evidence_start = time.perf_counter()
            source = manifest[episode_id]
            episode_dir = raw_root / source["raw_episode_relpath"]
            samples_path = episode_dir / "samples.csv"
            samples_sha = sha256(samples_path)
            if samples_sha != source["samples_sha256"]:
                raise RuntimeError(f"{episode_id}: source samples SHA mismatch")
            if samples_sha != day31_row["samples_sha256"]:
                raise RuntimeError(f"{episode_id}: Day31 samples SHA mismatch")

            samples = load_state_action_samples(samples_path)
            state_text = d31.state_action_text(samples, selected_frames)
            sheets, raw_image_hash = d31.build_contact_sheets(
                episode_dir=episode_dir,
                selected=selected_frames,
                config=d31_cfg,
                output_dir=WORK / "inputs" / episode_id,
            )
            if raw_image_hash != day31_row["raw_selected_image_hashes_sha256"]:
                raise RuntimeError(f"{episode_id}: Day31 image fingerprint drift")
            evidence_fp = d31.evidence_fingerprint(
                episode_id=episode_id,
                selected=selected_frames,
                state_text=state_text,
                raw_image_hash_sha256=raw_image_hash,
                samples_sha256=samples_sha,
                prompt_sha256=HASHES[D31_PROMPT],
            )
            if evidence_fp != day31_row["evidence_input_sha256"]:
                raise RuntimeError(f"{episode_id}: Day31 evidence fingerprint drift")
            evidence_sec = time.perf_counter() - evidence_start

            processor_stage_start = time.perf_counter()
            messages = d32.messages(
                sheets,
                state_text,
                selected_frames,
                prompt,
                d32_cfg,
            )
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=int(d32_cfg["model"]["image_patch_size"]),
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            if videos is not None:
                videos, video_metadata = zip(*videos)
                videos = list(videos)
                video_metadata = list(video_metadata)
            else:
                video_metadata = None
            inputs = processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadata,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs,
            ).to(model.device)
            processor_sec = time.perf_counter() - processor_stage_start

            baseline_allocated_mb = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                baseline_allocated_mb = (
                    torch.cuda.memory_allocated() / 1024 / 1024
                )

            generation_start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=int(loading["max_new_tokens"]),
                    do_sample=False,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generation_sec = time.perf_counter() - generation_start

            parse_start = time.perf_counter()
            trimmed = [
                output[len(input_ids):]
                for input_ids, output in zip(inputs.input_ids, generated)
            ]
            response = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            parsed = d32.parse(response, selected_frames)
            calibrated = d32.decision(parsed, candidate)
            parse_sec = time.perf_counter() - parse_start

            end_to_end_sec = time.perf_counter() - episode_start
            peak_mb = (
                torch.cuda.max_memory_allocated() / 1024 / 1024
                if torch.cuda.is_available()
                else None
            )
            peak_incremental_mb = (
                None
                if peak_mb is None or baseline_allocated_mb is None
                else peak_mb - baseline_allocated_mb
            )

            row = {
                "schema_version": "evidencemm_day34_development_e2e_profile_row_v1",
                "episode_id": episode_id,
                "split": "development",
                "selected_frame_indices": selected_frames,
                "day31_evidence_input_sha256": evidence_fp,
                "score_prompt_sha256": HASHES[D32_PROMPT],
                "score_parse_ok": parsed["parse_ok"],
                "calibrated_decision": calibrated["diagnostic_decision"],
                "generated_token_count": int(trimmed[0].shape[-1]),
                "evidence_preparation_sec": evidence_sec,
                "processor_preparation_sec": processor_sec,
                "generation_sec": generation_sec,
                "parse_and_calibration_sec": parse_sec,
                "episode_end_to_end_sec": end_to_end_sec,
                "baseline_gpu_memory_mb": baseline_allocated_mb,
                "peak_gpu_memory_mb": peak_mb,
                "peak_incremental_gpu_memory_mb": peak_incremental_mb,
            }
            append_jsonl(PARTIAL_PROFILE, row)

            del inputs, generated, trimmed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rows = read_jsonl(PARTIAL_PROFILE)
    if [row["episode_id"] for row in rows] != selected:
        raise RuntimeError("Day34 profile did not complete exact selected population")
    if not STARTUP_PROFILE.exists():
        raise RuntimeError("Day34 startup profile missing")

    startup = read_json(STARTUP_PROFILE)
    payload = {
        "schema_version": "evidencemm_day34_development_e2e_profile_v1",
        "status": "development_only_e2e_profile_complete",
        "selection_seed": profile_cfg["episode_selection_seed"],
        "selection_rule": profile_cfg["episode_selection_rule"],
        "selected_episode_ids": selected,
        "ranked_development_episode_ids": ranked,
        "profile_episode_count": len(selected),
        "ground_truth_accessed": False,
        "heldout_inference_count": 0,
        "startup": startup,
        "episodes": rows,
        "stage_summaries": {
            "evidence_preparation_sec": stats([
                row["evidence_preparation_sec"] for row in rows
            ]),
            "processor_preparation_sec": stats([
                row["processor_preparation_sec"] for row in rows
            ]),
            "generation_sec": stats([
                row["generation_sec"] for row in rows
            ]),
            "parse_and_calibration_sec": stats([
                row["parse_and_calibration_sec"] for row in rows
            ]),
            "episode_end_to_end_sec": stats([
                row["episode_end_to_end_sec"] for row in rows
            ]),
            "peak_gpu_memory_mb": stats([
                row["peak_gpu_memory_mb"]
                for row in rows
                if row["peak_gpu_memory_mb"] is not None
            ]),
            "peak_incremental_gpu_memory_mb": stats([
                row["peak_incremental_gpu_memory_mb"]
                for row in rows
                if row["peak_incremental_gpu_memory_mb"] is not None
            ]),
        },
        "profile_is_accuracy_neutral": True,
        "post_heldout_tuning_performed": False,
    }
    errors = validate_profile_payload(payload, env)
    if errors:
        raise RuntimeError("Day34 completed profile invalid: " + repr(errors))
    write_json(E2E_PROFILE, payload)

    print("profile_sha256 =", sha256(E2E_PROFILE))
    print("warm_e2e_summary =", payload["stage_summaries"]["episode_end_to_end_sec"])
    print("generation_summary =", payload["stage_summaries"]["generation_sec"])
    print("peak_gpu_summary =", payload["stage_summaries"]["peak_gpu_memory_mb"])
    print("heldout_inference_count = 0")
    print("ground_truth_accessed = false")
    print("DAY34 DEVELOPMENT E2E PROFILE: PASS")


def cmd_finalize() -> None:
    env = verify_frozen(require_raw=False)
    for path in (FINAL_METRICS, ERROR_ANALYSIS, PER_CASE, E2E_PROFILE):
        if not path.exists():
            raise RuntimeError(f"Day34 prerequisite missing: {path}")

    profile = read_json(E2E_PROFILE)
    payload = build_efficiency_payload(env, profile)

    if EFFICIENCY.exists():
        if read_json(EFFICIENCY) != payload:
            raise RuntimeError("existing Day34 efficiency report differs")
    else:
        write_json(EFFICIENCY, payload)

    print("===== DAY34 EFFICIENCY REPORT =====")
    print(
        "day33_generation_latency =",
        payload["immutable_logged_measurements"]
        ["day33_frozen_heldout_scoring"]["generation_latency_sec"],
    )
    print(
        "day33_peak_gpu_memory_mb =",
        payload["immutable_logged_measurements"]
        ["day33_frozen_heldout_scoring"]["peak_gpu_memory_mb"],
    )
    print(
        "development_warm_e2e =",
        payload["development_only_e2e_profile"]["warm_episode_e2e_sec"],
    )
    print("evidence_density =", payload["evidence_density"])
    print("efficiency_report_sha256 =", sha256(EFFICIENCY))
    print("DAY34 EFFICIENCY FINALIZE: PASS")


def cmd_freeze() -> None:
    commit = tooling_commit()
    env = verify_frozen(require_raw=False)

    for path in (
        FINAL_METRICS,
        ERROR_ANALYSIS,
        PER_CASE,
        E2E_PROFILE,
        EFFICIENCY,
    ):
        if not path.exists():
            raise RuntimeError(f"Day34 artifact missing: {path}")

    if RECEIPT.exists():
        raise RuntimeError("Day34 freeze receipt already exists")

    tooling_blobs = {
        rel: git("rev-parse", f"{commit}:{rel}")
        for rel in TOOLING
    }
    profile = read_json(E2E_PROFILE)
    errors = validate_profile_payload(profile, env)
    if errors:
        raise RuntimeError("cannot freeze invalid profile: " + repr(errors))

    receipt = {
        "schema_version": "evidencemm_day34_metrics_freeze_receipt_v1",
        "status": "metrics_error_efficiency_frozen_day34_complete",
        "tooling_commit": commit,
        "day33_final_commit": DAY33_FINAL_COMMIT,
        "day33_final_metrics_sha256": HASHES[D33_MET],
        "day33_final_predictions_sha256": HASHES[D33_FINAL_PRED],
        "day33_score_predictions_sha256": HASHES[D33_SCORE],
        "day33_freeze_receipt_sha256": HASHES[D33_RECEIPT],
        "tooling_git_blobs": tooling_blobs,
        "final_metrics_report_sha256": sha256(FINAL_METRICS),
        "error_analysis_sha256": sha256(ERROR_ANALYSIS),
        "per_case_analysis_sha256": sha256(PER_CASE),
        "development_e2e_profile_sha256": sha256(E2E_PROFILE),
        "efficiency_report_sha256": sha256(EFFICIENCY),
        "heldout_final_evaluation_count_consumed": 1,
        "day34_heldout_inference_count": 0,
        "day34_heldout_regeneration_count": 0,
        "development_e2e_profile_episode_count":
            profile["profile_episode_count"],
        "development_e2e_profile_ground_truth_accessed": False,
        "prompt_tuning_after_day33": False,
        "calibration_tuning_after_day33": False,
        "retrieval_tuning_after_day33": False,
        "model_selection_after_day33": False,
        "result_conditioned_regeneration_after_day33": False,
        "day33_predictions_modified": False,
    }
    write_json(RECEIPT, receipt)

    print("final_metrics_report_sha256 =", receipt["final_metrics_report_sha256"])
    print("error_analysis_sha256 =", receipt["error_analysis_sha256"])
    print("per_case_analysis_sha256 =", receipt["per_case_analysis_sha256"])
    print("development_e2e_profile_sha256 =", receipt["development_e2e_profile_sha256"])
    print("efficiency_report_sha256 =", receipt["efficiency_report_sha256"])
    print("freeze_receipt_sha256 =", sha256(RECEIPT))
    print("day34_heldout_inference_count = 0")
    print("DAY34 FREEZE RECEIPT: PASS")


def cmd_audit() -> None:
    env = verify_frozen(require_raw=False)
    required = (
        FINAL_METRICS,
        ERROR_ANALYSIS,
        PER_CASE,
        E2E_PROFILE,
        EFFICIENCY,
        RECEIPT,
    )
    for path in required:
        if not path.exists():
            raise RuntimeError(f"Day34 frozen artifact missing: {path}")

    errors: list[str] = []
    expected_metrics, expected_error, expected_cases = build_static_payloads(env)
    if read_json(FINAL_METRICS) != expected_metrics:
        errors.append("final metrics report differs from recomputation")
    if read_json(ERROR_ANALYSIS) != expected_error:
        errors.append("error analysis differs from recomputation")
    if read_jsonl(PER_CASE) != expected_cases:
        errors.append("per-case analysis differs from recomputation")

    profile = read_json(E2E_PROFILE)
    errors.extend(validate_profile_payload(profile, env))
    expected_efficiency = build_efficiency_payload(env, profile)
    if read_json(EFFICIENCY) != expected_efficiency:
        errors.append("efficiency report differs from recomputation")

    receipt = read_json(RECEIPT)
    checks = {
        "status": "metrics_error_efficiency_frozen_day34_complete",
        "day33_final_commit": DAY33_FINAL_COMMIT,
        "day33_final_metrics_sha256": HASHES[D33_MET],
        "day33_final_predictions_sha256": HASHES[D33_FINAL_PRED],
        "day33_score_predictions_sha256": HASHES[D33_SCORE],
        "day33_freeze_receipt_sha256": HASHES[D33_RECEIPT],
        "final_metrics_report_sha256": sha256(FINAL_METRICS),
        "error_analysis_sha256": sha256(ERROR_ANALYSIS),
        "per_case_analysis_sha256": sha256(PER_CASE),
        "development_e2e_profile_sha256": sha256(E2E_PROFILE),
        "efficiency_report_sha256": sha256(EFFICIENCY),
        "heldout_final_evaluation_count_consumed": 1,
        "day34_heldout_inference_count": 0,
        "day34_heldout_regeneration_count": 0,
        "development_e2e_profile_episode_count": 5,
        "development_e2e_profile_ground_truth_accessed": False,
        "prompt_tuning_after_day33": False,
        "calibration_tuning_after_day33": False,
        "retrieval_tuning_after_day33": False,
        "model_selection_after_day33": False,
        "result_conditioned_regeneration_after_day33": False,
        "day33_predictions_modified": False,
    }
    for key, expected in checks.items():
        if receipt.get(key) != expected:
            errors.append(
                f"receipt {key} mismatch: expected={expected!r} "
                f"actual={receipt.get(key)!r}"
            )

    commit = receipt.get("tooling_commit")
    blobs = receipt.get("tooling_git_blobs")
    if not isinstance(commit, str) or not commit:
        errors.append("tooling commit missing")
    elif not isinstance(blobs, dict):
        errors.append("tooling blobs missing")
    else:
        try:
            require_ancestor(commit)
            for rel in TOOLING:
                frozen_blob = git("rev-parse", f"{commit}:{rel}")
                current_blob = git("rev-parse", f"HEAD:{rel}")
                if blobs.get(rel) != frozen_blob:
                    errors.append(f"receipt tooling blob mismatch: {rel}")
                if current_blob != frozen_blob:
                    errors.append(f"Day34 tooling drift: {rel}")
        except Exception as exc:
            errors.append(f"tooling verification failed: {exc}")

    print("===== DAY34 METRICS/ERROR/EFFICIENCY AUDIT =====")
    print("final_metrics_report_sha256 =", sha256(FINAL_METRICS))
    print("error_analysis_sha256 =", sha256(ERROR_ANALYSIS))
    print("per_case_analysis_sha256 =", sha256(PER_CASE))
    print("development_e2e_profile_sha256 =", sha256(E2E_PROFILE))
    print("efficiency_report_sha256 =", sha256(EFFICIENCY))
    print("freeze_receipt_sha256 =", sha256(RECEIPT))
    print("day34_heldout_inference_count = 0")
    print("heldout_final_evaluation_count_consumed = 1")
    print("errors =", errors)
    if errors:
        raise SystemExit(1)
    print("DAY34 METRICS/ERROR/EFFICIENCY AUDIT: PASS")
    print("DAY34: CLOSED / FROZEN")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in (
        "preflight",
        "analyze",
        "profile",
        "finalize",
        "freeze",
        "audit",
    ):
        sub.add_parser(name)
    args = parser.parse_args()
    {
        "preflight": cmd_preflight,
        "analyze": cmd_analyze,
        "profile": cmd_profile,
        "finalize": cmd_finalize,
        "freeze": cmd_freeze,
        "audit": cmd_audit,
    }[args.cmd]()


if __name__ == "__main__":
    main()
