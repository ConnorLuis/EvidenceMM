from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/day31_root_cause_baseline.py"

spec = importlib.util.spec_from_file_location("day31_baseline", MODULE_PATH)
assert spec is not None and spec.loader is not None
day31 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(day31)


def test_uniform_frame_indices_are_deterministic() -> None:
    assert day31.uniform_frame_indices(900, 7) == [
        0, 150, 300, 450, 599, 749, 899
    ]


def test_parse_prediction_accepts_strict_valid_json() -> None:
    selected = [0, 100, 200, 300, 400, 500, 600, 700, 800, 850, 875, 899]
    response = json.dumps({
        "diagnostic_decision": "gripper_close_timing",
        "confidence": 0.8,
        "failure_start_frame": 500,
        "failure_end_frame": 700,
        "evidence_frame_indices": [500, 600, 700],
        "rationale": "closure is delayed relative to the approach",
    })
    parsed = day31.parse_prediction(response, selected=selected)
    assert parsed["parse_ok"] is True
    assert parsed["diagnostic_decision"] == "gripper_close_timing"
    assert parsed["evidence_frame_indices"] == [500, 600, 700]


def test_parse_prediction_fails_closed_on_out_of_bundle_frame() -> None:
    selected = [0, 100, 200, 300, 400, 500, 600, 700, 800, 850, 875, 899]
    response = json.dumps({
        "diagnostic_decision": "target_offset_or_perception",
        "confidence": 0.9,
        "failure_start_frame": 500,
        "failure_end_frame": 700,
        "evidence_frame_indices": [501],
        "rationale": "unsupported frame",
    })
    parsed = day31.parse_prediction(response, selected=selected)
    assert parsed["parse_ok"] is False
    assert parsed["diagnostic_decision"] == "insufficient_evidence"
    assert parsed["confidence"] == 0.0


def test_macro_f1_synthetic_perfect_three_class() -> None:
    y = [
        "target_offset_or_perception",
        "gripper_close_timing",
        "trajectory_execution_deviation",
    ]
    per_class = day31.per_class_f1(y, y, day31.CAUSE_LABELS)
    assert day31.macro_f1(per_class) == 1.0


def test_fixed_four_way_macro_has_zero_for_absent_class() -> None:
    y_true = [
        "target_offset_or_perception",
        "gripper_close_timing",
        "trajectory_execution_deviation",
    ]
    y_pred = list(y_true)
    per_class = day31.per_class_f1(
        y_true, y_pred, day31.FAILED_DECISION_LABELS
    )
    assert per_class["insufficient_evidence"]["support"] == 0
    assert per_class["insufficient_evidence"]["f1"] == 0.0
    assert day31.macro_f1(per_class) == 0.75


def test_frozen_day30_split_is_60_development_30_heldout() -> None:
    rows = day31.read_jsonl(day31.EPISODE_SPLIT_PATH)
    dev = [row for row in rows if row["split"] == "development"]
    held = [row for row in rows if row["split"] == "held_out"]
    assert len(dev) == 60
    assert len(held) == 30
    assert {r["episode_id"] for r in dev}.isdisjoint(
        {r["episode_id"] for r in held}
    )


def test_config_freezes_no_retrieval_and_no_heldout_inference() -> None:
    config = json.loads(day31.CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["baseline_mode"] == "direct_zero_shot_multimodal_no_retrieval"
    assert config["population"]["split"] == "development"
    assert config["population"]["held_out_inference_allowed"] is False
    assert config["boundaries"]["retrieval_used"] is False
    assert config["boundaries"]["calibration_performed"] is False
