#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CFG = ROOT / "configs/day33_single_heldout_eval.json"
CONTRACT = ROOT / "data/protocol/day33_heldout_eval_operational_contract.json"

D32_FROZEN = ROOT / "data/protocol/day32_frozen_diagnostic_config.json"
D32_PROMPT = ROOT / "data/protocol/day32_scoring_prompt_contract.json"
D32_SCRIPT = ROOT / "scripts/day32_development_calibration.py"
D31_SCRIPT = ROOT / "scripts/day31_root_cause_baseline.py"
D31_CFG = ROOT / "configs/day31_root_cause_baseline.json"

SPLIT = ROOT / "data/splits/day30_episode_split.jsonl"
PAIR_SPLIT = ROOT / "data/splits/day30_pair_group_split.json"
GT = ROOT / "data/annotations/day29_ground_truth_records.jsonl"
SRC = ROOT / "data/protocol/day28_registered_source_manifest.csv"
RAW_CFG = ROOT / "configs/day28_raw_audit.yaml"

START = ROOT / "data/protocol/day33_heldout_eval_start_receipt.json"
SCORES = ROOT / "data/eval/day33_heldout_scoring_predictions.jsonl"
FINAL_PRED = ROOT / "data/eval/day33_heldout_final_predictions.jsonl"
METRICS = ROOT / "data/eval/day33_heldout_final_metrics.json"
RECEIPT = ROOT / "data/protocol/day33_heldout_eval_freeze_receipt.json"

WORK = ROOT / "reports/day33_heldout_eval_work"
PARTIAL = WORK / "partial_scoring_predictions.jsonl"

DAY32_FINAL = "2b2b71c4489021f5637e8a4a5f6e6b3df36b0aa1"

HASHES = {
    D32_FROZEN: "15a11ebebf06edf10ba6cc8e015f1b04da4693a57800692025322ad69c63685d",
    D32_PROMPT: "f02290eb9fd0ad3de92363352fc921d13e9ce318de4d6db969149ec37fcd2cf7",
    D31_CFG: "eef3ed506ce434c9df2aafd236d4c848cb640bbbce7646fa6d143ac4798eb63f",
    SPLIT: "0b37a499904dcf8568ac39a9641097f7d73c952a01a79f00cbcda2b3b7793312",
    PAIR_SPLIT: "d43937c60279bbddc71ff078334dda40c900ff3dabe53cc06164773f3f77f5d2",
    GT: "e03ec1ab443e4fb4dab606e16fbae8439411d7c3acbcf5f078ed5a0660d389bf",
}
D32_SCRIPT_BLOB = "c39df7aaa7d5dcb2daf9cd3f985d94e68aba7f6a"
D31_SCRIPT_BLOB = "afd55294495b6cb552a3484662072fb8d46c3ef9"
D31_PROMPT_SHA = "faee60d40b710005a265ef7c657a2b19921c8b40c41ddda8c3d69d4916dbd79f"
SOURCE_MANIFEST_BLOB = "46a8a5655c17ca20f5aae88c1be05c18092a02c6"
RAW_CONFIG_BLOB = "eaef86f7aa514845a1160aa85d4d25cc4a79f279"

TOOLING = (
    "configs/day33_single_heldout_eval.json",
    "data/protocol/day33_heldout_eval_operational_contract.json",
    "docs/day33_single_frozen_heldout_eval.md",
    "scripts/day33_single_heldout_eval.py",
    "tests/test_day33_single_heldout_eval.py",
)

CAUSE = (
    "target_offset_or_perception",
    "gripper_close_timing",
    "trajectory_execution_deviation",
)
SUB = CAUSE + ("clean_success",)
ALL = CAUSE + ("insufficient_evidence", "clean_success")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


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
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_ancestor(commit: str) -> None:
    rc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode
    if rc != 0:
        raise RuntimeError(f"required ancestor missing: {commit}")


def verify_frozen() -> dict[str, Any]:
    if git("branch", "--show-current") != "master":
        raise RuntimeError("Day33 must run on master")
    require_ancestor(DAY32_FINAL)

    if git("rev-parse", "HEAD:scripts/day32_development_calibration.py") != D32_SCRIPT_BLOB:
        raise RuntimeError("Day32 script drift")
    if git("rev-parse", "HEAD:scripts/day31_root_cause_baseline.py") != D31_SCRIPT_BLOB:
        raise RuntimeError("Day31 script drift")
    if git("rev-parse", "HEAD:data/protocol/day28_registered_source_manifest.csv") != SOURCE_MANIFEST_BLOB:
        raise RuntimeError("Day28 source manifest drift")
    if git("rev-parse", "HEAD:configs/day28_raw_audit.yaml") != RAW_CONFIG_BLOB:
        raise RuntimeError("Day28 raw config drift")

    for path, expected in HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen SHA mismatch: {path}")

    cfg = read_json(CFG)
    frozen = read_json(D32_FROZEN)

    if frozen["status"] != "frozen_for_single_day33_heldout_evaluation":
        raise RuntimeError("Day32 config not frozen for Day33")
    if frozen["held_out_boundary"]["day33_final_evaluation_count_consumed"] != 0:
        raise RuntimeError("Day32 says held-out final evaluation already consumed")
    if frozen["scoring_prompt_sha256"] != HASHES[D32_PROMPT]:
        raise RuntimeError("Day32 scoring prompt SHA mismatch")
    if frozen["selected_log_biases"] != {
        "target_offset_or_perception": 1.0,
        "gripper_close_timing": -0.5,
        "trajectory_execution_deviation": 1.0,
        "clean_success": 0.0,
    }:
        raise RuntimeError("Day32 frozen biases drift")
    if frozen["confidence_threshold"] != 0.0 or frozen["margin_threshold"] != 0.0:
        raise RuntimeError("Day32 frozen thresholds drift")

    rows = read_jsonl(SPLIT)
    dev = [r for r in rows if r["split"] == "development"]
    held = [r for r in rows if r["split"] == "held_out"]
    if len(dev) != 60 or len(held) != 30:
        raise RuntimeError("Day30 split count drift")
    dev_ids = {r["episode_id"] for r in dev}
    held_ids = [r["episode_id"] for r in held]
    if dev_ids & set(held_ids):
        raise RuntimeError("split overlap")
    held_groups = sorted({r["pair_group_id"] for r in held})
    if len(held_groups) != 5:
        raise RuntimeError("held-out pair group count drift")

    return {
        "cfg": cfg,
        "frozen": frozen,
        "dev_ids": dev_ids,
        "held_ids": held_ids,
        "held_groups": held_groups,
    }


def tooling_commit() -> str:
    for rel in TOOLING:
        git("rev-parse", f"HEAD:{rel}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *TOOLING],
        cwd=ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Day33 tooling has uncommitted changes")
    return git("rev-parse", "HEAD")


def source_manifest() -> dict[str, dict[str, str]]:
    with SRC.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["episode_id"]: row for row in rows}


def raw_root() -> Path:
    import yaml
    payload = yaml.safe_load(RAW_CFG.read_text(encoding="utf-8"))
    root = Path(payload["raw_source"]["compatibility_wsl_root"])
    if not root.is_dir():
        raise RuntimeError(f"raw root unavailable: {root}")
    return root


def start_receipt_payload(tool_commit: str, env: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "evidencemm_day33_heldout_eval_start_receipt_v1",
        "status": "single_heldout_final_evaluation_authorized_day33",
        "tooling_commit": tool_commit,
        "day32_final_commit": DAY32_FINAL,
        "day32_frozen_inference_config_sha256": HASHES[D32_FROZEN],
        "day32_scoring_prompt_sha256": HASHES[D32_PROMPT],
        "held_out_episode_count": 30,
        "held_out_pair_group_count": 5,
        "held_out_final_evaluation_count_consumed": 1,
        "held_out_prediction_count_before_authorization": 0,
        "held_out_gt_rows_parsed_before_authorization": 0,
        "prompt_changes_allowed_after_authorization": False,
        "calibration_changes_allowed_after_authorization": False,
        "evidence_selection_changes_allowed_after_authorization": False,
        "retrieval_changes_allowed_after_authorization": False,
        "result_conditioned_regeneration_allowed": False,
        "parse_failure_retry_allowed": False,
        "operational_interruption_resume_allowed": True,
    }


def validate_score_rows(
    rows: list[dict[str, Any]], env: dict[str, Any], complete: bool
) -> list[str]:
    errors: list[str] = []
    ids = [r.get("episode_id") for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate held-out score IDs")
    if set(ids) - set(env["held_ids"]):
        errors.append("non-held-out score row")
    if set(ids) & env["dev_ids"]:
        errors.append("development score row in Day33")
    if complete and (len(rows) != 30 or set(ids) != set(env["held_ids"])):
        errors.append("incomplete held-out score population")

    forbidden = {
        "pair_group_id",
        "physical_cause_gt",
        "diagnostic_decision_gt",
        "evidence_answerability_gt",
        "task_success",
        "intervention_type",
        "human_review_notes",
    }
    for row in rows:
        if forbidden & set(row):
            errors.append(f"{row.get('episode_id')}: forbidden field")
        if row.get("schema_version") != "evidencemm_day33_heldout_scoring_prediction_v1":
            errors.append(f"{row.get('episode_id')}: bad score schema")
        if row.get("split") != "held_out":
            errors.append(f"{row.get('episode_id')}: bad split")
        if row.get("score_prompt_sha256") != HASHES[D32_PROMPT]:
            errors.append(f"{row.get('episode_id')}: prompt SHA drift")
        selected = row.get("selected_frame_indices")
        if not isinstance(selected, list) or len(selected) != 12:
            errors.append(f"{row.get('episode_id')}: selected frame count != 12")
        normalized = row.get("normalized_scores")
        if not isinstance(normalized, dict) or set(normalized) != set(SUB):
            errors.append(f"{row.get('episode_id')}: normalized score contract")
        elif abs(sum(float(v) for v in normalized.values()) - 1.0) > 1e-8:
            errors.append(f"{row.get('episode_id')}: normalized scores do not sum to one")
    return errors


def heldout_gt(env: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not SCORES.exists():
        raise RuntimeError("held-out score predictions missing")
    rows = read_jsonl(SCORES)
    if validate_score_rows(rows, env, True):
        raise RuntimeError("cannot open held-out GT before all 30 predictions validate")

    held = set(env["held_ids"])
    out: dict[str, dict[str, Any]] = {}
    pattern = re.compile(r'"episode_id":"([^"]+)"')
    for line in GT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = pattern.search(line)
        if not match:
            raise RuntimeError("GT row missing episode_id")
        episode_id = match.group(1)

        if episode_id in env["dev_ids"]:
            continue
        if episode_id not in held:
            raise RuntimeError(f"GT episode outside frozen population: {episode_id}")

        row = json.loads(line)
        out[episode_id] = {
            "physical_cause_gt": row["physical_cause_gt"],
            "diagnostic_decision_gt": row["diagnostic_decision_gt"],
            "evidence_answerability_gt": row["evidence_answerability_gt"],
            "task_success": row["task_success"],
        }

    if len(out) != 30:
        raise RuntimeError(f"held-out GT row count={len(out)}")
    return out


def cmd_preflight() -> None:
    tool_commit = tooling_commit()
    env = verify_frozen()

    for path in (START, SCORES, FINAL_PRED, METRICS, RECEIPT):
        if path.exists():
            raise RuntimeError(f"Day33 output already exists: {path}")

    root = raw_root()
    manifest = source_manifest()
    missing = sorted(set(env["held_ids"]) - set(manifest))
    if missing:
        raise RuntimeError(f"missing held-out source bindings: {missing}")

    print("===== DAY33 HELD-OUT PREFLIGHT =====")
    print("tooling_commit =", tool_commit)
    print("held_out_episode_count = 30")
    print("held_out_pair_group_count = 5")
    print("development_inference_allowed = false")
    print("held_out_gt_rows_parsed = 0")
    print("frozen_config_sha256 =", HASHES[D32_FROZEN])
    print("raw_root =", root)
    print("DAY33 HELD-OUT PREFLIGHT: PASS")


def cmd_authorize() -> None:
    tool_commit = tooling_commit()
    env = verify_frozen()
    for path in (SCORES, FINAL_PRED, METRICS, RECEIPT):
        if path.exists():
            raise RuntimeError("cannot authorize after held-out outputs exist")

    expected = start_receipt_payload(tool_commit, env)
    if START.exists():
        actual = read_json(START)
        if actual != expected:
            raise RuntimeError("existing Day33 authorization receipt differs")
    else:
        write_json(START, expected)

    print("authorization_receipt_sha256 =", sha256(START))
    print("held_out_final_evaluation_count_consumed = 1")
    print("held_out_gt_rows_parsed_before_authorization = 0")
    print("DAY33 HELD-OUT AUTHORIZATION: PASS")


def verify_authorization_committed() -> dict[str, Any]:
    env = verify_frozen()
    if not START.exists():
        raise RuntimeError("Day33 authorization receipt missing")
    git("rev-parse", "HEAD:data/protocol/day33_heldout_eval_start_receipt.json")
    expected = start_receipt_payload(
        read_json(START)["tooling_commit"],
        env,
    )
    if read_json(START) != expected:
        raise RuntimeError("Day33 authorization receipt drift")
    return env


def cmd_run() -> None:
    env = verify_authorization_committed()

    if SCORES.exists():
        errors = validate_score_rows(read_jsonl(SCORES), env, True)
        if errors:
            raise RuntimeError(repr(errors))
        print("DAY33 held-out scoring already complete: PASS")
        return

    partial = read_jsonl(PARTIAL) if PARTIAL.exists() else []
    errors = validate_score_rows(partial, env, False)
    if errors:
        raise RuntimeError(repr(errors))

    done = {r["episode_id"] for r in partial}
    remaining = [eid for eid in env["held_ids"] if eid not in done]

    print("===== DAY33 SINGLE HELD-OUT SCORING RUN =====")
    print("completed_before_run =", len(done))
    print("remaining =", len(remaining))
    print("evaluation_count_consumed = 1")
    print("result_conditioned_retry = false")

    if remaining:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        from evidencemm.state_action_selection import load_state_action_samples

        d31 = load_module(D31_SCRIPT, "day31_frozen")
        d32 = load_module(D32_SCRIPT, "day32_frozen")
        d31_cfg = read_json(D31_CFG)
        prompt = read_json(D32_PROMPT)
        frozen = env["frozen"]
        manifest = source_manifest()
        raw = raw_root()

        model_name = frozen["model_name"]
        loading = frozen["model_loading"]

        print("loading model:", model_name)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
            attn_implementation=loading["attn_implementation"],
            local_files_only=loading["local_files_only"],
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(
            model_name,
            local_files_only=True,
        )

        sel_cfg = d31_cfg["evidence_selection"]

        for ordinal, episode_id in enumerate(remaining, 1):
            print(f"[{len(done)+ordinal:02d}/30] episode={episode_id}", flush=True)

            source = manifest[episode_id]
            episode_dir = raw / source["raw_episode_relpath"]
            samples_path = episode_dir / "samples.csv"

            if sha256(samples_path) != source["samples_sha256"]:
                raise RuntimeError(f"{episode_id}: samples SHA mismatch")

            samples = load_state_action_samples(samples_path)
            selected = d31.select_baseline_frames(
                samples,
                uniform_count=int(sel_cfg["uniform_anchor_count"]),
                dynamic_count=int(sel_cfg["dynamic_frame_count"]),
                min_separation_frames=int(sel_cfg["dynamic_min_separation_frames"]),
            )
            if len(selected) != 12:
                raise RuntimeError(f"{episode_id}: selected frame count != 12")

            state_text = d31.state_action_text(samples, selected)
            sheets, raw_image_hash = d31.build_contact_sheets(
                episode_dir=episode_dir,
                selected=selected,
                config=d31_cfg,
                output_dir=WORK / "inputs" / episode_id,
            )

            evidence_fingerprint = d31.evidence_fingerprint(
                episode_id=episode_id,
                selected=selected,
                state_text=state_text,
                raw_image_hash_sha256=raw_image_hash,
                samples_sha256=sha256(samples_path),
                prompt_sha256=D31_PROMPT_SHA,
            )

            messages = d32.messages(
                sheets,
                state_text,
                selected,
                prompt,
                {
                    "model": {
                        "min_pixels": 50176,
                        "max_pixels": 786432,
                    }
                },
            )
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
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

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=int(loading["max_new_tokens"]),
                    do_sample=False,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - start

            trimmed = [
                output[len(input_ids):]
                for input_ids, output in zip(inputs.input_ids, generated)
            ]
            response = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            peak_mb = (
                torch.cuda.max_memory_allocated() / 1024 / 1024
                if torch.cuda.is_available()
                else None
            )

            parsed = d32.parse(response, selected)
            row = {
                "schema_version": "evidencemm_day33_heldout_scoring_prediction_v1",
                "episode_id": episode_id,
                "split": "held_out",
                "model_name": model_name,
                "score_prompt_sha256": HASHES[D32_PROMPT],
                "selected_frame_indices": selected,
                "day31_evidence_contract_fingerprint_sha256": evidence_fingerprint,
                "raw_selected_image_hashes_sha256": raw_image_hash,
                "samples_sha256": sha256(samples_path),
                "response_raw": response,
                **parsed,
                "latency_sec": latency,
                "peak_gpu_memory_mb": peak_mb,
            }
            append_jsonl(PARTIAL, row)

            del inputs, generated, trimmed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rows = read_jsonl(PARTIAL)
    errors = validate_score_rows(rows, env, True)
    if errors:
        raise RuntimeError(repr(errors))
    by_id = {r["episode_id"]: r for r in rows}
    write_jsonl(SCORES, [by_id[eid] for eid in env["held_ids"]])

    print("held_out_score_prediction_count = 30")
    print("parse_ok_count =", sum(bool(r["parse_ok"]) for r in rows))
    print("score_predictions_sha256 =", sha256(SCORES))
    print("DAY33 SINGLE HELD-OUT SCORING: PASS")


def cmd_validate() -> None:
    env = verify_authorization_committed()
    rows = read_jsonl(SCORES)
    errors = validate_score_rows(rows, env, True)
    print("===== DAY33 HELD-OUT SCORE VALIDATION =====")
    print("prediction_count =", len(rows))
    print("parse_ok_count =", sum(bool(r["parse_ok"]) for r in rows))
    print("held_out_only =", not bool(set(r["episode_id"] for r in rows) & env["dev_ids"]))
    print("errors =", errors)
    if errors:
        raise SystemExit(1)
    print("DAY33 HELD-OUT SCORE VALIDATION: PASS")


def cmd_evaluate() -> None:
    env = verify_authorization_committed()
    rows = read_jsonl(SCORES)
    errors = validate_score_rows(rows, env, True)
    if errors:
        raise RuntimeError(repr(errors))

    d32 = load_module(D32_SCRIPT, "day32_eval")
    frozen = env["frozen"]
    candidate = {
        "biases": frozen["selected_log_biases"],
        "confidence_threshold": frozen["confidence_threshold"],
        "margin_threshold": frozen["margin_threshold"],
    }

    final_rows = []
    for row in rows:
        calibrated = d32.decision(row, candidate)
        final_rows.append({
            "schema_version": "evidencemm_day33_heldout_final_prediction_v1",
            "episode_id": row["episode_id"],
            "split": "held_out",
            "score_parse_ok": row["parse_ok"],
            "selected_frame_indices": row["selected_frame_indices"],
            "raw_scores": row["raw_scores"],
            "normalized_scores": row["normalized_scores"],
            **calibrated,
            "evidence_frame_indices": row["evidence_frame_indices"],
            "rationale": row["rationale"],
        })

    gt = heldout_gt(env)
    decisions = {
        row["episode_id"]: row["diagnostic_decision"]
        for row in final_rows
    }
    m = d32.metrics(env["held_ids"], gt, decisions)

    payload = {
        "schema_version": "evidencemm_day33_heldout_final_metrics_v1",
        "status": "single_frozen_heldout_evaluation_complete",
        "held_out_episode_count": 30,
        "held_out_final_evaluation_count_consumed": 1,
        "frozen_inference_config_sha256": HASHES[D32_FROZEN],
        "score_predictions_sha256": sha256(SCORES),
        "score_parse_ok_count": sum(bool(r["parse_ok"]) for r in rows),
        "score_parse_failure_count": sum(not bool(r["parse_ok"]) for r in rows),
        "final_decision_counts": dict(sorted(Counter(decisions.values()).items())),
        "primary_metrics": {
            "answerable_three_class_macro_f1": m["answerable_three_class_macro_f1"],
            "failed_case_four_way_diagnostic_macro_f1": m["failed_case_four_way_diagnostic_macro_f1"],
            "abstention_accuracy": m["abstention_accuracy"],
            "false_answer_rate": m["false_answer_rate"],
            "false_abstention_rate": m["false_abstention_rate"],
            "clean_control_false_positive_cause_rate": m["clean_control_false_positive_cause_rate"],
        },
        "secondary_metrics": {
            "substantive_four_class_macro_f1": m["substantive_four_class_macro_f1"],
            "clean_control_accuracy": m["clean_control_accuracy"],
            "heldout_decision_accuracy": m["development_decision_accuracy"],
            "prediction_parse_rate": sum(bool(r["parse_ok"]) for r in rows) / 30,
        },
        "gt_support": m["gt_support"],
        "tuning_after_heldout": False,
    }

    if FINAL_PRED.exists():
        if read_jsonl(FINAL_PRED) != final_rows:
            raise RuntimeError("existing final predictions differ")
    else:
        write_jsonl(FINAL_PRED, final_rows)

    if METRICS.exists():
        if read_json(METRICS) != payload:
            raise RuntimeError("existing held-out metrics differ")
    else:
        write_json(METRICS, payload)

    print("===== DAY33 SINGLE FROZEN HELD-OUT METRICS =====")
    print("primary_metrics =", payload["primary_metrics"])
    print("secondary_metrics =", payload["secondary_metrics"])
    print("gt_support =", payload["gt_support"])
    print("final_decision_counts =", payload["final_decision_counts"])
    print("final_predictions_sha256 =", sha256(FINAL_PRED))
    print("metrics_sha256 =", sha256(METRICS))
    print("DAY33 SINGLE HELD-OUT EVALUATION: PASS")


def cmd_freeze() -> None:
    env = verify_authorization_committed()
    for path in (SCORES, FINAL_PRED, METRICS):
        if not path.exists():
            raise RuntimeError(f"missing Day33 artifact: {path}")

    authorization_commit = git("rev-parse", "HEAD")
    start_sha = sha256(START)
    metrics = read_json(METRICS)

    receipt = {
        "schema_version": "evidencemm_day33_heldout_eval_freeze_receipt_v1",
        "status": "single_frozen_heldout_final_evaluation_day33_complete",
        "authorization_commit": authorization_commit,
        "tooling_commit": read_json(START)["tooling_commit"],
        "authorization_receipt_sha256": start_sha,
        "day32_final_commit": DAY32_FINAL,
        "day32_frozen_inference_config_sha256": HASHES[D32_FROZEN],
        "day32_scoring_prompt_sha256": HASHES[D32_PROMPT],
        "held_out_episode_count": 30,
        "held_out_final_evaluation_count_consumed": 1,
        "score_predictions_sha256": sha256(SCORES),
        "final_predictions_sha256": sha256(FINAL_PRED),
        "final_metrics_sha256": sha256(METRICS),
        "score_parse_ok_count": metrics["score_parse_ok_count"],
        "score_parse_failure_count": metrics["score_parse_failure_count"],
        "primary_metrics": metrics["primary_metrics"],
        "secondary_metrics": metrics["secondary_metrics"],
        "gt_support": metrics["gt_support"],
        "prompt_changed_after_authorization": False,
        "calibration_changed_after_authorization": False,
        "evidence_selection_changed_after_authorization": False,
        "retrieval_changed_after_authorization": False,
        "result_conditioned_regeneration_used": False,
        "parse_failure_retry_used": False,
        "tuning_after_heldout": False,
    }

    if RECEIPT.exists():
        if read_json(RECEIPT) != receipt:
            raise RuntimeError("existing freeze receipt differs")
    else:
        write_json(RECEIPT, receipt)

    print("freeze_receipt_sha256 =", sha256(RECEIPT))
    print("held_out_final_evaluation_count_consumed = 1")
    print("DAY33 HELD-OUT FREEZE RECEIPT: PASS")


def cmd_audit() -> None:
    env = verify_authorization_committed()
    errors: list[str] = []

    score_rows = read_jsonl(SCORES)
    errors.extend(validate_score_rows(score_rows, env, True))

    d32 = load_module(D32_SCRIPT, "day32_audit")
    frozen = env["frozen"]
    candidate = {
        "biases": frozen["selected_log_biases"],
        "confidence_threshold": frozen["confidence_threshold"],
        "margin_threshold": frozen["margin_threshold"],
    }
    expected_final = []
    for row in score_rows:
        expected_final.append({
            "schema_version": "evidencemm_day33_heldout_final_prediction_v1",
            "episode_id": row["episode_id"],
            "split": "held_out",
            "score_parse_ok": row["parse_ok"],
            "selected_frame_indices": row["selected_frame_indices"],
            "raw_scores": row["raw_scores"],
            "normalized_scores": row["normalized_scores"],
            **d32.decision(row, candidate),
            "evidence_frame_indices": row["evidence_frame_indices"],
            "rationale": row["rationale"],
        })
    if read_jsonl(FINAL_PRED) != expected_final:
        errors.append("final predictions differ from frozen calibration recomputation")

    gt = heldout_gt(env)
    decisions = {r["episode_id"]: r["diagnostic_decision"] for r in expected_final}
    m = d32.metrics(env["held_ids"], gt, decisions)
    actual_metrics = read_json(METRICS)

    expected_primary = {
        "answerable_three_class_macro_f1": m["answerable_three_class_macro_f1"],
        "failed_case_four_way_diagnostic_macro_f1": m["failed_case_four_way_diagnostic_macro_f1"],
        "abstention_accuracy": m["abstention_accuracy"],
        "false_answer_rate": m["false_answer_rate"],
        "false_abstention_rate": m["false_abstention_rate"],
        "clean_control_false_positive_cause_rate": m["clean_control_false_positive_cause_rate"],
    }
    if actual_metrics["primary_metrics"] != expected_primary:
        errors.append("primary metrics drift")

    receipt = read_json(RECEIPT)
    frozen_tooling_commit = receipt.get("tooling_commit")
    if not isinstance(frozen_tooling_commit, str) or not frozen_tooling_commit:
        errors.append("tooling commit missing")
    else:
        try:
            require_ancestor(frozen_tooling_commit)
            for rel in TOOLING:
                frozen_blob = git("rev-parse", f"{frozen_tooling_commit}:{rel}")
                current_blob = git("rev-parse", f"HEAD:{rel}")
                if frozen_blob != current_blob:
                    errors.append(f"Day33 tooling changed after authorization: {rel}")
        except Exception as exc:
            errors.append(f"tooling commit invalid: {exc}")

    authorization_commit = receipt.get("authorization_commit")
    if not isinstance(authorization_commit, str) or not authorization_commit:
        errors.append("authorization commit missing")
    else:
        try:
            require_ancestor(authorization_commit)
            start_blob_at_auth = git("rev-parse", f"{authorization_commit}:data/protocol/day33_heldout_eval_start_receipt.json")
            start_blob_now = git("rev-parse", "HEAD:data/protocol/day33_heldout_eval_start_receipt.json")
            if start_blob_at_auth != start_blob_now:
                errors.append("authorization receipt changed after authorization")
        except Exception as exc:
            errors.append(f"authorization commit invalid: {exc}")
    checks = {
        "status": "single_frozen_heldout_final_evaluation_day33_complete",
        "authorization_receipt_sha256": sha256(START),
        "day32_frozen_inference_config_sha256": HASHES[D32_FROZEN],
        "held_out_episode_count": 30,
        "held_out_final_evaluation_count_consumed": 1,
        "score_predictions_sha256": sha256(SCORES),
        "final_predictions_sha256": sha256(FINAL_PRED),
        "final_metrics_sha256": sha256(METRICS),
        "prompt_changed_after_authorization": False,
        "calibration_changed_after_authorization": False,
        "evidence_selection_changed_after_authorization": False,
        "retrieval_changed_after_authorization": False,
        "result_conditioned_regeneration_used": False,
        "parse_failure_retry_used": False,
        "tuning_after_heldout": False,
    }
    for key, value in checks.items():
        if receipt.get(key) != value:
            errors.append(f"receipt {key} mismatch")

    print("===== DAY33 SINGLE HELD-OUT FREEZE AUDIT =====")
    print("held_out_score_prediction_count =", len(score_rows))
    print("score_parse_ok_count =", sum(bool(r["parse_ok"]) for r in score_rows))
    print("primary_metrics =", actual_metrics["primary_metrics"])
    print("secondary_metrics =", actual_metrics["secondary_metrics"])
    print("score_predictions_sha256 =", sha256(SCORES))
    print("final_predictions_sha256 =", sha256(FINAL_PRED))
    print("metrics_sha256 =", sha256(METRICS))
    print("freeze_receipt_sha256 =", sha256(RECEIPT))
    print("held_out_final_evaluation_count_consumed = 1")
    print("errors =", errors)
    if errors:
        raise SystemExit(1)
    print("DAY33 HELD-OUT AUDIT: PASS")
    print("DAY33: CLOSED / FROZEN")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "authorize", "run", "validate", "evaluate", "freeze", "audit"):
        sub.add_parser(name)
    args = parser.parse_args()
    {
        "preflight": cmd_preflight,
        "authorize": cmd_authorize,
        "run": cmd_run,
        "validate": cmd_validate,
        "evaluate": cmd_evaluate,
        "freeze": cmd_freeze,
        "audit": cmd_audit,
    }[args.cmd]()


if __name__ == "__main__":
    main()
