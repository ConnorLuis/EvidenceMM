from __future__ import annotations
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "day33",
    ROOT / "scripts/day33_single_heldout_eval.py",
)
assert spec is not None and spec.loader is not None
day33 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(day33)


def test_day32_frozen_config_exact_parameters() -> None:
    frozen = json.loads(day33.D32_FROZEN.read_text(encoding="utf-8"))
    assert frozen["status"] == "frozen_for_single_day33_heldout_evaluation"
    assert frozen["selected_log_biases"] == {
        "target_offset_or_perception": 1.0,
        "gripper_close_timing": -0.5,
        "trajectory_execution_deviation": 1.0,
        "clean_success": 0.0,
    }
    assert frozen["confidence_threshold"] == 0.0
    assert frozen["margin_threshold"] == 0.0
    assert frozen["held_out_boundary"]["day33_final_evaluation_count_consumed"] == 0


def test_day30_split_is_exactly_60_dev_30_heldout_5_groups() -> None:
    rows = day33.read_jsonl(day33.SPLIT)
    dev = [r for r in rows if r["split"] == "development"]
    held = [r for r in rows if r["split"] == "held_out"]
    assert len(dev) == 60
    assert len(held) == 30
    assert len({r["pair_group_id"] for r in held}) == 5
    assert {r["episode_id"] for r in dev}.isdisjoint(
        {r["episode_id"] for r in held}
    )


def test_day33_config_forbids_result_conditioned_retry() -> None:
    cfg = json.loads(day33.CFG.read_text(encoding="utf-8"))
    policy = cfg["single_evaluation_policy"]
    assert policy["authorization_must_be_committed_and_pushed_before_first_heldout_inference"] is True
    assert policy["held_out_final_evaluation_count_consumed_on_authorization"] == 1
    assert policy["result_conditioned_regeneration_allowed"] is False
    assert policy["parse_failure_retry_allowed"] is False
    assert policy["operational_interruption_resume_allowed"] is True


def test_day33_config_freezes_evidence_interpretation() -> None:
    cfg = json.loads(day33.CFG.read_text(encoding="utf-8"))
    evidence = cfg["evidence_interpretation"]
    assert evidence["frame_count_per_episode"] == 12
    assert evidence["day31_contact_sheet_convention_reused"] is True
    assert evidence["day31_state_action_text_reused"] is True
    assert evidence["retrieval_used"] is False
    assert evidence["manual_corpus_used"] is False


def test_start_receipt_consumes_exactly_one_evaluation() -> None:
    env = {
        "held_ids": [str(i) for i in range(30)],
        "held_groups": [str(i) for i in range(5)],
    }
    receipt = day33.start_receipt_payload("abc", env)
    assert receipt["held_out_episode_count"] == 30
    assert receipt["held_out_pair_group_count"] == 5
    assert receipt["held_out_final_evaluation_count_consumed"] == 1
    assert receipt["held_out_gt_rows_parsed_before_authorization"] == 0


def test_score_validation_rejects_development_rows() -> None:
    env = {"held_ids": ["held"], "dev_ids": {"dev"}}
    row = {
        "episode_id": "dev",
        "split": "held_out",
        "score_prompt_sha256": day33.HASHES[day33.D32_PROMPT],
    }
    errors = day33.validate_score_rows([row], env, complete=False)
    assert errors


def test_frozen_prompt_hash_matches_day32_config() -> None:
    frozen = json.loads(day33.D32_FROZEN.read_text(encoding="utf-8"))
    assert day33.sha256(day33.D32_PROMPT) == frozen["scoring_prompt_sha256"]


def test_day33_tooling_does_not_define_new_prompt_file() -> None:
    assert not (ROOT / "data/protocol/day33_scoring_prompt_contract.json").exists()
