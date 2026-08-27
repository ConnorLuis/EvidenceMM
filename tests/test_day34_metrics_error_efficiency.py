from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "day34",
    ROOT / "scripts/day34_metrics_error_efficiency.py",
)
assert spec is not None and spec.loader is not None
day34 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(day34)


def test_percentile_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert day34.percentile(values, 50) == 2.5
    assert day34.percentile(values, 0) == 1.0
    assert day34.percentile(values, 100) == 4.0


def test_stats_reports_expected_shape() -> None:
    result = day34.stats([1.0, 2.0, 3.0])
    assert result["count"] == 3
    assert result["mean"] == 2.0
    assert result["p50"] == 2.0
    assert result["min"] == 1.0
    assert result["max"] == 3.0


def test_confusion_and_report_synthetic() -> None:
    labels = ("a", "b")
    matrix, report = day34.confusion_and_report(
        ["a", "a", "b"],
        ["a", "b", "b"],
        labels=labels,
    )
    assert matrix == {
        "a": {"a": 1, "b": 1},
        "b": {"a": 0, "b": 1},
    }
    assert report["a"]["support"] == 2
    assert report["b"]["predicted_count"] == 2


def test_profile_selection_is_deterministic_label_blind() -> None:
    ids = ["e3", "e1", "e2", "e5", "e4"]
    selected1, ranked1 = day34.select_profile_ids(
        ids,
        seed="seed",
        count=3,
    )
    selected2, ranked2 = day34.select_profile_ids(
        list(reversed(ids)),
        seed="seed",
        count=3,
    )
    assert selected1 == selected2
    assert ranked1 == ranked2
    assert len(selected1) == 3


def test_error_category_contract() -> None:
    assert day34.error_category(
        gt_decision="target_offset_or_perception",
        pred_decision="target_offset_or_perception",
        parse_ok=True,
    ) == "correct"
    assert day34.error_category(
        gt_decision="gripper_close_timing",
        pred_decision="insufficient_evidence",
        parse_ok=False,
    ) == "parse_failure_abstention"
    assert day34.error_category(
        gt_decision="clean_success",
        pred_decision="trajectory_execution_deviation",
        parse_ok=True,
    ) == "clean_false_positive_cause"


def test_day33_frozen_metrics_are_exact() -> None:
    metrics = json.loads(day34.D33_MET.read_text(encoding="utf-8"))
    assert metrics["held_out_episode_count"] == 30
    assert metrics["held_out_final_evaluation_count_consumed"] == 1
    assert metrics["score_parse_ok_count"] == 25
    assert metrics["score_parse_failure_count"] == 5
    assert metrics["primary_metrics"]["answerable_three_class_macro_f1"] == (
        0.16722408026755853
    )
    assert metrics["secondary_metrics"]["prediction_parse_rate"] == (
        0.8333333333333334
    )


def test_day33_receipt_freezes_no_post_result_tuning() -> None:
    receipt = json.loads(day34.D33_RECEIPT.read_text(encoding="utf-8"))
    assert receipt["held_out_final_evaluation_count_consumed"] == 1
    assert receipt["prompt_changed_after_authorization"] is False
    assert receipt["calibration_changed_after_authorization"] is False
    assert receipt["retrieval_changed_after_authorization"] is False
    assert receipt["result_conditioned_regeneration_used"] is False
    assert receipt["tuning_after_heldout"] is False


def test_day34_config_profiles_development_only() -> None:
    cfg = json.loads(day34.CFG.read_text(encoding="utf-8"))
    profile = cfg["efficiency_profile"]
    assert profile["split"] == "development"
    assert profile["episode_count"] == 5
    assert profile["ground_truth_access_allowed"] is False
    analysis = cfg["analysis"]
    assert analysis["heldout_result_modification_allowed"] is False
    assert analysis["prompt_tuning_allowed"] is False
    assert analysis["calibration_tuning_allowed"] is False


def test_static_payload_recomputes_frozen_day33_metrics() -> None:
    env = day34.verify_frozen(require_raw=False)
    metrics, errors, cases = day34.build_static_payloads(env)
    held = metrics["day33_single_frozen_heldout"]
    assert len(cases) == 30
    assert held["score_parse_ok_count"] == 25
    assert held["primary_metrics"]["false_abstention_rate"] == 0.12
    assert errors["parse_failure_analysis"]["count"] == 5
    assert errors["analysis_only_no_tuning"] is True
