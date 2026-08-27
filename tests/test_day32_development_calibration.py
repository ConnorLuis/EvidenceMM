from __future__ import annotations
import importlib.util,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("d32",ROOT/"scripts/day32_development_calibration.py")
assert spec is not None and spec.loader is not None
d32=importlib.util.module_from_spec(spec);spec.loader.exec_module(d32)

def test_internal_group_split_deterministic_disjoint():
 g=[f"g{i}" for i in range(10)]
 a,b,r=d32.internal(g,"seed",6);a2,b2,r2=d32.internal(g,"seed",6)
 assert (a,b,r)==(a2,b2,r2)
 assert len(a)==6 and len(b)==4 and set(a).isdisjoint(b)

def test_parse_valid_score_json():
 selected=[0,100,200,300,400,500,600,700,800,850,875,899]
 x={"scores":{"target_offset_or_perception":70,"gripper_close_timing":15,"trajectory_execution_deviation":10,"clean_success":5},"top_substantive_decision":"target_offset_or_perception","confidence":.8,"evidence_frame_indices":[300,500],"rationale":"alignment mismatch"}
 p=d32.parse(json.dumps(x),selected)
 assert p["parse_ok"] is True
 assert p["derived_top_substantive_decision"]=="target_offset_or_perception"
 assert abs(sum(p["normalized_scores"].values())-1)<1e-12

def test_parse_rejects_insufficient_raw_score_label():
 selected=[0,100,200,300,400,500,600,700,800,850,875,899]
 x={"scores":{"target_offset_or_perception":25,"gripper_close_timing":25,"trajectory_execution_deviation":25,"clean_success":20,"insufficient_evidence":5},"top_substantive_decision":"target_offset_or_perception","confidence":.2,"evidence_frame_indices":[],"rationale":"uncertain"}
 assert d32.parse(json.dumps(x),selected)["parse_ok"] is False

def test_calibrated_probabilities_sum_to_one():
 p=d32.cprobs({"target_offset_or_perception":.4,"gripper_close_timing":.3,"trajectory_execution_deviation":.2,"clean_success":.1},{"target_offset_or_perception":.5,"gripper_close_timing":0,"trajectory_execution_deviation":-.5,"clean_success":0})
 assert abs(sum(p.values())-1)<1e-12
 assert p["target_offset_or_perception"]>p["gripper_close_timing"]

def test_margin_threshold_can_abstain():
 row={"parse_ok":True,"normalized_scores":{"target_offset_or_perception":.30,"gripper_close_timing":.29,"trajectory_execution_deviation":.21,"clean_success":.20}}
 cand={"biases":{k:0.0 for k in d32.SUB},"confidence_threshold":0.0,"margin_threshold":.05}
 assert d32.decision(row,cand)["diagnostic_decision"]=="insufficient_evidence"

def test_candidate_grid_count():
 cfg=json.loads(d32.CFG.read_text(encoding="utf-8"))
 assert len(d32.grid(cfg))==3750
 assert any(c["biases"]=={k:0.0 for k in d32.SUB} and c["confidence_threshold"]==0 and c["margin_threshold"]==0 for c in d32.grid(cfg))

def test_frozen_split_60_30_10_groups():
 rows=d32.rjl(d32.SPLIT);dev=[r for r in rows if r["split"]=="development"];held=[r for r in rows if r["split"]=="held_out"]
 assert len(dev)==60 and len(held)==30
 assert len({r["pair_group_id"] for r in dev})==10
 assert {r["episode_id"] for r in dev}.isdisjoint({r["episode_id"] for r in held})

def test_day31_predictions_exact_development_population():
 rows=d32.rjl(d32.SPLIT);dev={r["episode_id"] for r in rows if r["split"]=="development"}
 p=d32.rjl(d32.D31PRED)
 assert len(p)==60 and {r["episode_id"] for r in p}==dev

def test_config_forbids_heldout_and_validation_selection():
 cfg=json.loads(d32.CFG.read_text(encoding="utf-8"))
 assert cfg["population"]["held_out_inference_allowed"] is False
 assert cfg["population"]["held_out_ground_truth_use_allowed"] is False
 assert cfg["calibration_grid"]["selection_uses_internal_validation"] is False
 assert cfg["calibration_grid"]["refit_after_validation"] is False
 assert cfg["boundaries"]["held_out_labels_allowed"] is False
