#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs/day31_root_cause_baseline.json"
PROMPT_PATH = ROOT / "data/protocol/day31_baseline_prompt_contract.json"
CONTRACT_PATH = ROOT / "data/protocol/day31_baseline_operational_contract.json"

DAY22_PROTOCOL_PATH = ROOT / "data/protocol/day22_root_cause_benchmark_v2_protocol.json"
DAY30_RECEIPT_PATH = ROOT / "data/protocol/day30_split_freeze_receipt.json"
EPISODE_SPLIT_PATH = ROOT / "data/splits/day30_episode_split.jsonl"
PAIR_SPLIT_PATH = ROOT / "data/splits/day30_pair_group_split.json"
GT_PATH = ROOT / "data/annotations/day29_ground_truth_records.jsonl"
SOURCE_MANIFEST_PATH = ROOT / "data/protocol/day28_registered_source_manifest.csv"
RAW_CONFIG_PATH = ROOT / "configs/day28_raw_audit.yaml"

PREDICTIONS_PATH = ROOT / "data/eval/day31_development_baseline_predictions.jsonl"
METRICS_PATH = ROOT / "data/eval/day31_development_baseline_metrics.json"
FREEZE_RECEIPT_PATH = ROOT / "data/protocol/day31_baseline_freeze_receipt.json"

WORK_ROOT = ROOT / "reports/day31_baseline_work"
PARTIAL_PREDICTIONS_PATH = WORK_ROOT / "partial_predictions.jsonl"

DAY30_FINAL_COMMIT = "21a68f1df2a6f0770b3db4b5ad99fa6f16d481b1"
DAY22_PROTOCOL_BLOB = "388ecb388375046b208e85eb9617e79961a5bf52"
DAY30_RECEIPT_SHA256 = "1523fd3fdfea33d2c5818ddee92c5fc161d73baa6acab5b56bf0c9c385f1465d"
EPISODE_SPLIT_SHA256 = "0b37a499904dcf8568ac39a9641097f7d73c952a01a79f00cbcda2b3b7793312"
PAIR_SPLIT_SHA256 = "d43937c60279bbddc71ff078334dda40c900ff3dabe53cc06164773f3f77f5d2"
GT_SHA256 = "e03ec1ab443e4fb4dab606e16fbae8439411d7c3acbcf5f078ed5a0660d389bf"
SOURCE_MANIFEST_BLOB = "46a8a5655c17ca20f5aae88c1be05c18092a02c6"
RAW_CONFIG_BLOB = "eaef86f7aa514845a1160aa85d4d25cc4a79f279"

PREDICTION_SCHEMA = "evidencemm_day31_baseline_prediction_v1"
METRICS_SCHEMA = "evidencemm_day31_development_baseline_metrics_v1"
RECEIPT_SCHEMA = "evidencemm_day31_baseline_freeze_receipt_v1"

CAUSE_LABELS = (
    "target_offset_or_perception",
    "gripper_close_timing",
    "trajectory_execution_deviation",
)
FAILED_DECISION_LABELS = CAUSE_LABELS + ("insufficient_evidence",)
ALL_DECISIONS = FAILED_DECISION_LABELS + ("clean_success",)

TOOLING_PATHS = (
    "configs/day31_root_cause_baseline.json",
    "data/protocol/day31_baseline_prompt_contract.json",
    "data/protocol/day31_baseline_operational_contract.json",
    "docs/day31_root_cause_baseline.md",
    "scripts/day31_root_cause_baseline.py",
    "tests/test_day31_root_cause_baseline.py",
)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_ancestor(commit: str) -> None:
    rc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode
    if rc != 0:
        raise RuntimeError(f"required commit is not an ancestor: {commit}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected YAML mapping: {path}")
    return payload


def verify_tooling_committed() -> str:
    for rel in TOOLING_PATHS:
        git_output("rev-parse", f"HEAD:{rel}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *TOOLING_PATHS],
        cwd=ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Day31 tooling has uncommitted changes:\n" + dirty)
    return git_output("rev-parse", "HEAD")


def load_split_rows() -> list[dict[str, Any]]:
    rows = read_jsonl(EPISODE_SPLIT_PATH)
    if len(rows) != 90:
        raise RuntimeError(f"Day30 episode split count={len(rows)}")
    if len({row["episode_id"] for row in rows}) != 90:
        raise RuntimeError("Day30 split episode IDs not unique")
    return rows


def split_ids() -> tuple[list[str], set[str]]:
    rows = load_split_rows()
    dev = [
        row["episode_id"]
        for row in rows
        if row["split"] == "development"
    ]
    held = {
        row["episode_id"]
        for row in rows
        if row["split"] == "held_out"
    }
    if len(dev) != 60 or len(held) != 30:
        raise RuntimeError(
            f"split count mismatch development={len(dev)} held_out={len(held)}"
        )
    if set(dev) & held:
        raise RuntimeError("development/held-out episode overlap")
    return dev, held


def verify_frozen_environment(*, require_raw: bool) -> dict[str, Any]:
    if git_output("branch", "--show-current") != "master":
        raise RuntimeError("Day31 must run on master")
    require_ancestor(DAY30_FINAL_COMMIT)

    if git_output(
        "rev-parse",
        "HEAD:data/protocol/day22_root_cause_benchmark_v2_protocol.json",
    ) != DAY22_PROTOCOL_BLOB:
        raise RuntimeError("Day22 protocol blob changed")

    if git_output(
        "rev-parse",
        "HEAD:data/protocol/day28_registered_source_manifest.csv",
    ) != SOURCE_MANIFEST_BLOB:
        raise RuntimeError("Day28 source manifest blob changed")

    if git_output(
        "rev-parse",
        "HEAD:configs/day28_raw_audit.yaml",
    ) != RAW_CONFIG_BLOB:
        raise RuntimeError("Day28 raw config blob changed")

    fixed_hashes = {
        DAY30_RECEIPT_PATH: DAY30_RECEIPT_SHA256,
        EPISODE_SPLIT_PATH: EPISODE_SPLIT_SHA256,
        PAIR_SPLIT_PATH: PAIR_SPLIT_SHA256,
    }
    for path, expected in fixed_hashes.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen Day30 SHA mismatch: {path}\n"
                f"expected={expected}\nactual={actual}"
            )

    # Anti-leakage boundary: Day30 receipt contains administrative held-out
    # aggregate label audits. Day31 verifies its exact frozen SHA256 above but
    # deliberately does not parse that receipt. Structural membership comes
    # only from the label-free Day30 episode split artifact.
    dev, held = split_ids()

    config = read_json(CONFIG_PATH)
    prompt = read_json(PROMPT_PATH)
    contract = read_json(CONTRACT_PATH)

    if config.get("baseline_mode") != "direct_zero_shot_multimodal_no_retrieval":
        raise RuntimeError("Day31 baseline mode mismatch")
    if config["population"]["split"] != "development":
        raise RuntimeError("Day31 config must be development-only")
    if config["population"]["expected_episode_count"] != 60:
        raise RuntimeError("Day31 expected development size must be 60")

    prompt_sha = sha256_file(PROMPT_PATH)
    if contract["baseline"]["prompt_contract_sha256"] != prompt_sha:
        raise RuntimeError("Day31 prompt contract SHA mismatch")

    raw_root = None
    if require_raw:
        raw_cfg = load_yaml(RAW_CONFIG_PATH)
        raw_root = Path(raw_cfg["raw_source"]["compatibility_wsl_root"])
        if not raw_root.is_dir():
            raise RuntimeError(f"raw root unavailable: {raw_root}")

    return {
        "development_ids": dev,
        "held_out_ids": held,
        "config": config,
        "prompt": prompt,
        "contract": contract,
        "raw_root": raw_root,
    }


def load_source_manifest() -> dict[str, dict[str, str]]:
    with SOURCE_MANIFEST_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        episode_id = row["episode_id"]
        if episode_id in result:
            raise RuntimeError(f"duplicate source manifest episode {episode_id}")
        result[episode_id] = row
    return result


def uniform_frame_indices(frame_count: int, count: int) -> list[int]:
    if frame_count <= 0 or count <= 0:
        raise ValueError("frame_count/count must be positive")
    if count == 1:
        return [0]
    if count >= frame_count:
        return list(range(frame_count))
    return sorted({
        int(round(i * (frame_count - 1) / (count - 1)))
        for i in range(count)
    })


def select_baseline_frames(
    samples: list[Any],
    *,
    uniform_count: int,
    dynamic_count: int,
    min_separation_frames: int,
) -> list[int]:
    from evidencemm.review_pack import build_state_scores

    anchors = uniform_frame_indices(len(samples), uniform_count)
    scores = build_state_scores(samples)
    ranked = sorted(
        scores,
        key=lambda item: (
            -float(item.fused_state_action_score),
            int(item.frame_index),
        ),
    )

    selected = list(anchors)
    for item in ranked:
        frame = int(item.frame_index)
        if frame in selected:
            continue
        if all(
            abs(frame - chosen) >= min_separation_frames
            for chosen in selected
        ):
            selected.append(frame)
            if len(selected) >= uniform_count + dynamic_count:
                break

    if len(selected) != uniform_count + dynamic_count:
        raise RuntimeError(
            f"could not select required frames: got {len(selected)}"
        )
    return sorted(selected)


def rms(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values))


def state_action_text(samples: list[Any], selected: list[int]) -> str:
    joints = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
    by_idx = {sample.frame_index: sample for sample in samples}
    lines = [
        "STATE/ACTION EVIDENCE",
        "Values are ordered [shoulder_pan, shoulder_lift, elbow_flex, "
        "wrist_flex, wrist_roll, gripper].",
        "tracking_rms is RMS(|action-observation|) across 6 joints.",
    ]
    for frame in selected:
        sample = by_idx[frame]
        obs = [round(float(getattr(sample.observation, j)), 3) for j in joints]
        act = [round(float(getattr(sample.action, j)), 3) for j in joints]
        err = [float(getattr(sample.tracking_error, j)) for j in joints]
        lines.append(
            f"frame={frame} t={sample.timestamp_sec:.3f}s "
            f"obs={obs} action={act} tracking_rms={rms(err):.3f}"
        )
    return "\n".join(lines)


def transform_image(image: Any, transform: str) -> Any:
    from PIL import ImageOps, Image
    image = ImageOps.exif_transpose(image)
    normalized = transform.lower().replace("-", "").replace("_", "").replace(" ", "")
    if normalized in ("", "none", "identity"):
        return image.copy()
    if "ccw" in normalized and "90" in normalized:
        return image.transpose(Image.Transpose.ROTATE_90)
    if "cw" in normalized and "ccw" not in normalized and "90" in normalized:
        return image.transpose(Image.Transpose.ROTATE_270)
    if "180" in normalized:
        return image.transpose(Image.Transpose.ROTATE_180)
    raise ValueError(f"unsupported transform: {transform}")


def fit_image(image: Any, max_width: int, max_height: int) -> Any:
    from PIL import Image
    ratio = min(max_width / image.width, max_height / image.height)
    size = (
        max(1, int(round(image.width * ratio))),
        max(1, int(round(image.height * ratio))),
    )
    return image.resize(size, Image.Resampling.LANCZOS)


def build_contact_sheets(
    *,
    episode_dir: Path,
    selected: list[int],
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[list[Path], str]:
    from PIL import Image, ImageDraw, ImageFont

    sheet_cfg = config["contact_sheets"]
    transforms = config["evidence_selection"]["camera_transforms"]
    frames_per_sheet = int(sheet_cfg["frames_per_sheet"])
    sheet_width = int(sheet_cfg["sheet_width"])
    row_height = int(sheet_cfg["row_height"])
    quality = int(sheet_cfg["jpeg_quality"])

    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    raw_hash_payload: list[dict[str, Any]] = []
    sheet_paths: list[Path] = []

    for sheet_index in range(0, len(selected), frames_per_sheet):
        frames = selected[sheet_index: sheet_index + frames_per_sheet]
        sheet = Image.new("RGB", (sheet_width, row_height * len(frames)), "white")
        draw = ImageDraw.Draw(sheet)

        for local_row, frame in enumerate(frames):
            y0 = local_row * row_height
            label_width = 105
            gap = 8
            available = sheet_width - label_width - 3 * gap
            front_w = int(available * 0.60)
            wrist_w = available - front_w

            front_path = episode_dir / "front" / f"{frame:06d}.jpg"
            wrist_path = episode_dir / "wrist" / f"{frame:06d}.jpg"
            if not front_path.is_file() or not wrist_path.is_file():
                raise RuntimeError(f"selected camera frame missing at {frame}")

            raw_hash_payload.append({
                "frame_index": frame,
                "front_sha256": sha256_file(front_path),
                "wrist_sha256": sha256_file(wrist_path),
            })

            with Image.open(front_path) as source:
                front = transform_image(source, transforms["front"])
                front = fit_image(front, front_w, row_height - 24)
            with Image.open(wrist_path) as source:
                wrist = transform_image(source, transforms["wrist"])
                wrist = fit_image(wrist, wrist_w, row_height - 24)

            draw.text((8, y0 + 8), f"frame {frame}", fill="black", font=font)
            draw.text((label_width + gap, y0 + 4), "front", fill="black", font=font)
            x_front = label_width + gap
            y_front = y0 + 20 + max(0, (row_height - 24 - front.height) // 2)
            sheet.paste(front, (x_front, y_front))

            x_wrist = label_width + 2 * gap + front_w
            draw.text((x_wrist, y0 + 4), "wrist", fill="black", font=font)
            y_wrist = y0 + 20 + max(0, (row_height - 24 - wrist.height) // 2)
            sheet.paste(wrist, (x_wrist, y_wrist))

            if local_row > 0:
                draw.line((0, y0, sheet_width, y0), fill="gray", width=1)

        path = output_dir / f"sheet_{sheet_index // frames_per_sheet + 1:02d}.jpg"
        sheet.save(path, "JPEG", quality=quality, optimize=True)
        sheet_paths.append(path)

    return sheet_paths, canonical_sha256(raw_hash_payload)


def evidence_fingerprint(
    *,
    episode_id: str,
    selected: list[int],
    state_text: str,
    raw_image_hash_sha256: str,
    samples_sha256: str,
    prompt_sha256: str,
) -> str:
    return canonical_sha256({
        "episode_id": episode_id,
        "selected_frame_indices": selected,
        "state_action_text": state_text,
        "raw_selected_image_hashes_sha256": raw_image_hash_sha256,
        "samples_sha256": samples_sha256,
        "prompt_sha256": prompt_sha256,
    })


def build_messages(
    sheet_paths: list[Path],
    state_text: str,
    prompt_contract: dict[str, Any],
    selected: list[int],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    model_cfg = config["model"]
    content: list[dict[str, Any]] = []
    for path in sheet_paths:
        content.append({
            "type": "image",
            "image": path.resolve().as_uri(),
            "min_pixels": int(model_cfg["min_pixels"]),
            "max_pixels": int(model_cfg["max_pixels"]),
        })

    taxonomy = prompt_contract["taxonomy_definitions"]
    taxonomy_text = "\n".join(
        f"- {label}: {definition}"
        for label, definition in taxonomy.items()
    )
    schema_text = json.dumps(
        prompt_contract["required_output_schema"],
        ensure_ascii=False,
        indent=2,
    )

    user_text = (
        prompt_contract["user_instruction_template"]
        + "\n\nFROZEN TAXONOMY\n"
        + taxonomy_text
        + "\n\nSUPPLIED FRAME INDICES\n"
        + json.dumps(selected)
        + "\n\n"
        + state_text
        + "\n\nOUTPUT JSON SCHEMA\n"
        + schema_text
    )
    content.append({"type": "text", "text": user_text})

    return [
        {"role": "system", "content": prompt_contract["system_prompt"]},
        {"role": "user", "content": content},
    ]


def strip_code_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def parse_prediction(
    response: str,
    *,
    selected: list[int],
) -> dict[str, Any]:
    selected_set = set(selected)
    try:
        candidate = strip_code_fence(response)
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", candidate, flags=re.S)
            if not match:
                raise
            payload = json.loads(match.group(0))

        if not isinstance(payload, dict):
            raise ValueError("response JSON is not an object")

        decision = payload.get("diagnostic_decision")
        if decision not in ALL_DECISIONS:
            raise ValueError(f"invalid diagnostic_decision={decision!r}")

        confidence = float(payload.get("confidence"))
        if not math.isfinite(confidence) or not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence outside [0,1]")

        start = payload.get("failure_start_frame")
        end = payload.get("failure_end_frame")
        for name, value in (("failure_start_frame", start), ("failure_end_frame", end)):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 899
            ):
                raise ValueError(f"{name} invalid")
        if start is not None and end is not None and end < start:
            raise ValueError("failure interval end before start")

        evidence = payload.get("evidence_frame_indices")
        if not isinstance(evidence, list):
            raise ValueError("evidence_frame_indices must be list")
        normalized_evidence: list[int] = []
        for value in evidence:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("non-integer evidence frame")
            if value not in selected_set:
                raise ValueError(f"evidence frame not supplied: {value}")
            if value not in normalized_evidence:
                normalized_evidence.append(value)

        rationale = payload.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("rationale missing")

        return {
            "parse_ok": True,
            "parse_error": None,
            "diagnostic_decision": decision,
            "confidence": confidence,
            "failure_start_frame": start,
            "failure_end_frame": end,
            "evidence_frame_indices": normalized_evidence,
            "rationale": rationale.strip(),
        }

    except Exception as exc:
        return {
            "parse_ok": False,
            "parse_error": f"{type(exc).__name__}: {exc}",
            "diagnostic_decision": "insufficient_evidence",
            "confidence": 0.0,
            "failure_start_frame": None,
            "failure_end_frame": None,
            "evidence_frame_indices": [],
            "rationale": "parse_failure_abstention",
        }


def load_partial_predictions() -> list[dict[str, Any]]:
    if not PARTIAL_PREDICTIONS_PATH.exists():
        return []
    rows = read_jsonl(PARTIAL_PREDICTIONS_PATH)
    ids = [row["episode_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate partial prediction episode IDs")
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        handle.flush()


def validate_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    development_ids: list[str],
    held_out_ids: set[str],
    require_complete: bool,
) -> list[str]:
    errors: list[str] = []
    ids = [row.get("episode_id") for row in rows]

    if len(ids) != len(set(ids)):
        errors.append("duplicate prediction episode IDs")

    unknown = sorted(set(ids) - set(development_ids))
    if unknown:
        errors.append(f"non-development predictions present: {unknown}")

    held = sorted(set(ids) & held_out_ids)
    if held:
        errors.append(f"held-out predictions present: {held}")

    if require_complete:
        if len(rows) != 60:
            errors.append(f"prediction count={len(rows)}, expected 60")
        missing = sorted(set(development_ids) - set(ids))
        if missing:
            errors.append(f"missing development predictions: {missing}")

    for row in rows:
        if row.get("schema_version") != PREDICTION_SCHEMA:
            errors.append(f"{row.get('episode_id')}: schema mismatch")
        if row.get("split") != "development":
            errors.append(f"{row.get('episode_id')}: split not development")
        if row.get("diagnostic_decision") not in ALL_DECISIONS:
            errors.append(f"{row.get('episode_id')}: invalid decision")
        selected = row.get("selected_frame_indices")
        if not isinstance(selected, list) or len(selected) != 12:
            errors.append(f"{row.get('episode_id')}: selected frame count != 12")
        evidence = row.get("evidence_frame_indices")
        if isinstance(selected, list) and isinstance(evidence, list):
            if not set(evidence).issubset(set(selected)):
                errors.append(f"{row.get('episode_id')}: evidence frame leakage")
    return errors


def load_development_gt(
    development_ids: list[str],
    held_out_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], int]:
    if sha256_file(GT_PATH) != GT_SHA256:
        raise RuntimeError("frozen GT SHA mismatch")

    dev_set = set(development_ids)
    result: dict[str, dict[str, Any]] = {}
    held_out_gt_rows_used = 0

    # Strict anti-leakage read: extract only episode_id from raw JSONL text
    # first. Held-out lines are skipped before json.loads(), so their GT label
    # fields are not parsed or copied into Day31 evaluation state.
    episode_pattern = re.compile(r'"episode_id":"([^"]+)"')
    for line in GT_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = episode_pattern.search(line)
        if not match:
            raise RuntimeError("GT row missing episode_id")
        episode_id = match.group(1)

        if episode_id in held_out_ids:
            continue
        if episode_id not in dev_set:
            raise RuntimeError(
                f"GT row outside frozen development/held-out population: {episode_id}"
            )

        row = json.loads(line)
        result[episode_id] = {
            "episode_id": episode_id,
            "physical_cause_gt": row["physical_cause_gt"],
            "diagnostic_decision_gt": row["diagnostic_decision_gt"],
            "evidence_answerability_gt": row["evidence_answerability_gt"],
            "task_success": row["task_success"],
        }

    if len(result) != 60:
        raise RuntimeError(f"development GT rows={len(result)}, expected 60")
    return result, held_out_gt_rows_used


def safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def per_class_f1(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        support = sum(t == label for t in y_true)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        if precision is None or recall is None or precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        result[label] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return result


def macro_f1(per_class: dict[str, dict[str, Any]]) -> float:
    if not per_class:
        return 0.0
    return sum(float(v["f1"]) for v in per_class.values()) / len(per_class)


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    development_ids: list[str],
    held_out_ids: set[str],
) -> dict[str, Any]:
    gt, heldout_used = load_development_gt(development_ids, held_out_ids)
    pred_by_id = {row["episode_id"]: row for row in predictions}

    ordered = development_ids
    answerable_ids = [
        eid for eid in ordered
        if gt[eid]["evidence_answerability_gt"] == "answerable"
        and gt[eid]["task_success"] is False
    ]
    failed_ids = [
        eid for eid in ordered
        if gt[eid]["task_success"] is False
    ]
    clean_ids = [
        eid for eid in ordered
        if gt[eid]["task_success"] is True
    ]
    insufficient_gt_ids = [
        eid for eid in failed_ids
        if gt[eid]["evidence_answerability_gt"] == "insufficient_evidence"
    ]

    y_true_three = [gt[eid]["physical_cause_gt"] for eid in answerable_ids]
    y_pred_three = [pred_by_id[eid]["diagnostic_decision"] for eid in answerable_ids]
    three_per_class = per_class_f1(y_true_three, y_pred_three, CAUSE_LABELS)

    y_true_four = [gt[eid]["diagnostic_decision_gt"] for eid in failed_ids]
    y_pred_four = [pred_by_id[eid]["diagnostic_decision"] for eid in failed_ids]
    four_per_class = per_class_f1(
        y_true_four, y_pred_four, FAILED_DECISION_LABELS
    )
    supported_labels = [
        label for label in FAILED_DECISION_LABELS
        if any(value == label for value in y_true_four)
    ]
    supported_per_class = {
        label: four_per_class[label] for label in supported_labels
    }

    abstention_correct = sum(
        (
            gt[eid]["evidence_answerability_gt"] == "insufficient_evidence"
            and pred_by_id[eid]["diagnostic_decision"] == "insufficient_evidence"
        )
        or (
            gt[eid]["evidence_answerability_gt"] == "answerable"
            and pred_by_id[eid]["diagnostic_decision"] != "insufficient_evidence"
        )
        for eid in failed_ids
    )

    false_answers = sum(
        pred_by_id[eid]["diagnostic_decision"] != "insufficient_evidence"
        for eid in insufficient_gt_ids
    )
    false_abstentions = sum(
        pred_by_id[eid]["diagnostic_decision"] == "insufficient_evidence"
        for eid in answerable_ids
    )
    clean_false_positive_causes = sum(
        pred_by_id[eid]["diagnostic_decision"] in CAUSE_LABELS
        for eid in clean_ids
    )
    clean_correct = sum(
        pred_by_id[eid]["diagnostic_decision"] == "clean_success"
        for eid in clean_ids
    )
    overall_correct = sum(
        pred_by_id[eid]["diagnostic_decision"]
        == gt[eid]["diagnostic_decision_gt"]
        for eid in ordered
    )

    parse_ok = sum(bool(row["parse_ok"]) for row in predictions)
    decision_counts = Counter(
        row["diagnostic_decision"] for row in predictions
    )

    four_support = {
        label: four_per_class[label]["support"]
        for label in FAILED_DECISION_LABELS
    }

    return {
        "schema_version": METRICS_SCHEMA,
        "status": "development_baseline_evaluated",
        "baseline_mode": "direct_zero_shot_multimodal_no_retrieval",
        "development_episode_count": 60,
        "held_out_prediction_count": 0,
        "held_out_gt_rows_used": heldout_used,
        "prediction_parse_ok_count": parse_ok,
        "prediction_parse_failure_count": 60 - parse_ok,
        "prediction_decision_counts": dict(sorted(decision_counts.items())),
        "development_gt_support": {
            "answerable_failure_count": len(answerable_ids),
            "failed_case_count": len(failed_ids),
            "clean_control_count": len(clean_ids),
            "insufficient_evidence_failure_count": len(insufficient_gt_ids),
        },
        "primary_metrics": {
            "answerable_three_class_macro_f1": macro_f1(three_per_class),
            "failed_case_four_way_diagnostic_macro_f1": macro_f1(four_per_class),
            "abstention_accuracy": abstention_correct / len(failed_ids),
            "false_answer_rate": safe_div(
                false_answers, len(insufficient_gt_ids)
            ),
            "false_abstention_rate": false_abstentions / len(answerable_ids),
            "clean_control_false_positive_cause_rate":
                clean_false_positive_causes / len(clean_ids),
        },
        "secondary_metrics": {
            "failed_case_supported_label_macro_f1":
                macro_f1(supported_per_class),
            "clean_control_accuracy": clean_correct / len(clean_ids),
            "development_decision_accuracy": overall_correct / len(ordered),
            "prediction_parse_rate": parse_ok / len(predictions),
        },
        "per_class_three_way": three_per_class,
        "per_class_four_way": four_per_class,
        "four_way_class_support": four_support,
        "metric_notes": {
            "failed_case_four_way_diagnostic_macro_f1": (
                "Fixed four-label macro-F1 with zero-division=0. "
                "The frozen development GT has zero insufficient_evidence support, "
                "so this literal four-way metric has a structural ceiling below 1.0."
            ),
            "false_answer_rate": (
                "null when frozen development GT contains no insufficient-evidence "
                "failures; no held-out support is consulted."
            ),
        },
        "boundaries": {
            "development_only": True,
            "held_out_labels_aggregated": False,
            "calibration_performed": False,
            "prompt_tuning_after_results": False,
            "model_selection_using_development_results": False,
            "retrieval_used": False,
            "manual_corpus_used": False,
        },
    }


def cmd_preflight() -> None:
    verify_tooling_committed()
    env = verify_frozen_environment(require_raw=True)
    dev = env["development_ids"]
    held = env["held_out_ids"]
    manifest = load_source_manifest()

    if set(dev) & held:
        raise RuntimeError("development/held-out overlap")

    missing_manifest = sorted(set(dev) - set(manifest))
    if missing_manifest:
        raise RuntimeError(f"development source bindings missing: {missing_manifest}")

    if PREDICTIONS_PATH.exists() or METRICS_PATH.exists() or FREEZE_RECEIPT_PATH.exists():
        raise RuntimeError("Day31 frozen output already exists")

    print("===== DAY31 BASELINE PREFLIGHT =====")
    print("branch =", git_output("branch", "--show-current"))
    print("head =", git_output("rev-parse", "HEAD"))
    print("baseline_mode = direct_zero_shot_multimodal_no_retrieval")
    print("model_name =", env["config"]["model"]["model_name"])
    print("development_episode_count =", len(dev))
    print("held_out_episode_count =", len(held))
    print("inference_reads_ground_truth = false")
    print("held_out_inference_allowed = false")
    print("retrieval_used = false")
    print("manual_corpus_used = false")
    print("calibration_started = false")
    print("DAY31 BASELINE PREFLIGHT: PASS")


def cmd_run() -> None:
    tooling_commit = verify_tooling_committed()
    env = verify_frozen_environment(require_raw=True)
    dev_ids = env["development_ids"]
    held_ids = env["held_out_ids"]
    config = env["config"]
    prompt_contract = env["prompt"]
    raw_root: Path = env["raw_root"]
    source_manifest = load_source_manifest()

    if PREDICTIONS_PATH.exists():
        rows = read_jsonl(PREDICTIONS_PATH)
        errors = validate_prediction_rows(
            rows,
            development_ids=dev_ids,
            held_out_ids=held_ids,
            require_complete=True,
        )
        if errors:
            raise RuntimeError("existing final predictions invalid: " + repr(errors))
        print("DAY31 predictions already complete; run is idempotent: PASS")
        return

    partial = load_partial_predictions()
    partial_errors = validate_prediction_rows(
        partial,
        development_ids=dev_ids,
        held_out_ids=held_ids,
        require_complete=False,
    )
    if partial_errors:
        raise RuntimeError("partial predictions invalid: " + repr(partial_errors))
    completed = {row["episode_id"] for row in partial}
    remaining = [eid for eid in dev_ids if eid not in completed]

    print("===== DAY31 BASELINE RUN =====")
    print("tooling_commit =", tooling_commit)
    print("completed_before_run =", len(completed))
    print("remaining =", len(remaining))
    print("held_out_inference_count = 0")

    if remaining:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from evidencemm.state_action_selection import load_state_action_samples

        model_cfg = config["model"]
        model_name = model_cfg["model_name"]

        print("loading model:", model_name)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
            attn_implementation=model_cfg["attn_implementation"],
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(model_name)

        prompt_sha = sha256_file(PROMPT_PATH)
        selection_cfg = config["evidence_selection"]

        for ordinal, episode_id in enumerate(remaining, start=1):
            print(
                f"[{len(completed)+ordinal:02d}/60] episode={episode_id}",
                flush=True,
            )

            manifest_row = source_manifest[episode_id]
            episode_dir = raw_root / manifest_row["raw_episode_relpath"]
            samples_path = episode_dir / "samples.csv"
            if not samples_path.is_file():
                raise RuntimeError(f"samples missing: {episode_id}")

            actual_samples_sha = sha256_file(samples_path)
            expected_samples_sha = manifest_row["samples_sha256"]
            if actual_samples_sha != expected_samples_sha:
                raise RuntimeError(f"samples SHA mismatch: {episode_id}")

            samples = load_state_action_samples(samples_path)
            if len(samples) != int(selection_cfg["frame_count"]):
                raise RuntimeError(
                    f"{episode_id}: sample count={len(samples)}"
                )

            selected = select_baseline_frames(
                samples,
                uniform_count=int(selection_cfg["uniform_anchor_count"]),
                dynamic_count=int(selection_cfg["dynamic_frame_count"]),
                min_separation_frames=int(
                    selection_cfg["dynamic_min_separation_frames"]
                ),
            )

            state_text = state_action_text(samples, selected)
            episode_work = WORK_ROOT / "inputs" / episode_id
            sheet_paths, raw_image_hashes_sha = build_contact_sheets(
                episode_dir=episode_dir,
                selected=selected,
                config=config,
                output_dir=episode_work,
            )

            fingerprint = evidence_fingerprint(
                episode_id=episode_id,
                selected=selected,
                state_text=state_text,
                raw_image_hash_sha256=raw_image_hashes_sha,
                samples_sha256=actual_samples_sha,
                prompt_sha256=prompt_sha,
            )

            messages = build_messages(
                sheet_paths,
                state_text,
                prompt_contract,
                selected,
                config,
            )
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=int(model_cfg["image_patch_size"]),
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
            )
            inputs = inputs.to(model.device)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=int(model_cfg["max_new_tokens"]),
                    do_sample=False,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - start

            trimmed = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            peak_mb = None
            if torch.cuda.is_available():
                peak_mb = (
                    torch.cuda.max_memory_allocated()
                    / 1024
                    / 1024
                )

            parsed = parse_prediction(response, selected=selected)

            row = {
                "schema_version": PREDICTION_SCHEMA,
                "episode_id": episode_id,
                "split": "development",
                "model_name": model_name,
                "baseline_mode": "direct_zero_shot_multimodal_no_retrieval",
                "prompt_contract_sha256": prompt_sha,
                "selected_frame_indices": selected,
                "evidence_input_sha256": fingerprint,
                "raw_selected_image_hashes_sha256": raw_image_hashes_sha,
                "samples_sha256": actual_samples_sha,
                "response_raw": response,
                **parsed,
                "latency_sec": latency,
                "peak_gpu_memory_mb": peak_mb,
            }
            append_jsonl(PARTIAL_PREDICTIONS_PATH, row)

            del inputs, generated_ids, trimmed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rows = load_partial_predictions()
    errors = validate_prediction_rows(
        rows,
        development_ids=dev_ids,
        held_out_ids=held_ids,
        require_complete=True,
    )
    if errors:
        raise RuntimeError("completed predictions invalid: " + repr(errors))

    by_id = {row["episode_id"]: row for row in rows}
    ordered = [by_id[eid] for eid in dev_ids]
    write_jsonl(PREDICTIONS_PATH, ordered)

    print("prediction_count =", len(ordered))
    print("held_out_prediction_count = 0")
    print("predictions_sha256 =", sha256_file(PREDICTIONS_PATH))
    print("DAY31 BASELINE RUN: PASS")


def cmd_validate_predictions() -> None:
    env = verify_frozen_environment(require_raw=False)
    if not PREDICTIONS_PATH.exists():
        raise RuntimeError("final predictions missing")
    rows = read_jsonl(PREDICTIONS_PATH)
    errors = validate_prediction_rows(
        rows,
        development_ids=env["development_ids"],
        held_out_ids=env["held_out_ids"],
        require_complete=True,
    )
    print("===== DAY31 PREDICTION VALIDATION =====")
    print("prediction_count =", len(rows))
    print("unique_episode_ids =", len({r["episode_id"] for r in rows}))
    print("held_out_prediction_count =",
          sum(r["episode_id"] in env["held_out_ids"] for r in rows))
    print("parse_ok_count =", sum(bool(r["parse_ok"]) for r in rows))
    print("decision_counts =",
          dict(sorted(Counter(r["diagnostic_decision"] for r in rows).items())))
    print("errors =", errors)
    if errors:
        raise SystemExit(1)
    print("DAY31 PREDICTION VALIDATION: PASS")


def cmd_evaluate() -> None:
    env = verify_frozen_environment(require_raw=False)
    if not PREDICTIONS_PATH.exists():
        raise RuntimeError("predictions missing")
    rows = read_jsonl(PREDICTIONS_PATH)
    errors = validate_prediction_rows(
        rows,
        development_ids=env["development_ids"],
        held_out_ids=env["held_out_ids"],
        require_complete=True,
    )
    if errors:
        raise RuntimeError("prediction validation failed: " + repr(errors))

    metrics = evaluate_predictions(
        rows,
        env["development_ids"],
        env["held_out_ids"],
    )

    if METRICS_PATH.exists():
        existing = read_json(METRICS_PATH)
        if existing != metrics:
            raise RuntimeError("existing Day31 metrics differ from recomputation")
    else:
        write_json(METRICS_PATH, metrics)

    print("===== DAY31 DEVELOPMENT BASELINE METRICS =====")
    for key, value in metrics["primary_metrics"].items():
        print(f"{key} = {value}")
    for key, value in metrics["secondary_metrics"].items():
        print(f"{key} = {value}")
    print("held_out_gt_rows_used =", metrics["held_out_gt_rows_used"])
    print("metrics_sha256 =", sha256_file(METRICS_PATH))
    print("DAY31 BASELINE EVALUATION: PASS")


def cmd_freeze() -> None:
    tooling_commit = verify_tooling_committed()
    env = verify_frozen_environment(require_raw=False)
    cmd_validate_predictions()

    if not METRICS_PATH.exists():
        raise RuntimeError("metrics missing; run evaluate first")

    predictions = read_jsonl(PREDICTIONS_PATH)
    expected_metrics = evaluate_predictions(
        predictions,
        env["development_ids"],
        env["held_out_ids"],
    )
    metrics = read_json(METRICS_PATH)
    if metrics != expected_metrics:
        raise RuntimeError("metrics do not match deterministic recomputation")

    if FREEZE_RECEIPT_PATH.exists():
        raise RuntimeError("Day31 freeze receipt already exists")

    tooling_git_blobs = {
        rel: git_output("rev-parse", f"{tooling_commit}:{rel}")
        for rel in TOOLING_PATHS
    }

    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "development_root_cause_baseline_frozen_day31_complete",
        "tooling_commit": tooling_commit,
        "day30_split_freeze_commit": DAY30_FINAL_COMMIT,
        "day30_split_freeze_receipt_sha256": DAY30_RECEIPT_SHA256,
        "day30_episode_split_sha256": EPISODE_SPLIT_SHA256,
        "day30_pair_group_split_sha256": PAIR_SPLIT_SHA256,
        "day29_ground_truth_records_sha256": GT_SHA256,
        "config_sha256": sha256_file(CONFIG_PATH),
        "prompt_contract_sha256": sha256_file(PROMPT_PATH),
        "operational_contract_sha256": sha256_file(CONTRACT_PATH),
        "tooling_git_blobs": tooling_git_blobs,
        "predictions_path":
            "data/eval/day31_development_baseline_predictions.jsonl",
        "predictions_sha256": sha256_file(PREDICTIONS_PATH),
        "metrics_path":
            "data/eval/day31_development_baseline_metrics.json",
        "metrics_sha256": sha256_file(METRICS_PATH),
        "baseline_mode": "direct_zero_shot_multimodal_no_retrieval",
        "model_name": env["config"]["model"]["model_name"],
        "development_prediction_count": 60,
        "held_out_prediction_count": 0,
        "held_out_gt_rows_used": metrics["held_out_gt_rows_used"],
        "inference_read_ground_truth": False,
        "selection_used_ground_truth": False,
        "retrieval_used": False,
        "manual_corpus_used": False,
        "model_training_performed": False,
        "calibration_performed": False,
        "prompt_tuning_after_results": False,
        "model_selection_using_development_results": False,
        "held_out_prompt_tuning": False,
        "held_out_retrieval_tuning": False,
        "held_out_model_selection": False,
        "held_out_evaluation_started": False,
        "held_out_final_evaluation_count_consumed": 0,
        "primary_metrics": metrics["primary_metrics"],
        "secondary_metrics": metrics["secondary_metrics"],
    }
    write_json(FREEZE_RECEIPT_PATH, receipt)

    print("predictions_sha256 =", receipt["predictions_sha256"])
    print("metrics_sha256 =", receipt["metrics_sha256"])
    print("freeze_receipt_sha256 =", sha256_file(FREEZE_RECEIPT_PATH))
    print("held_out_prediction_count = 0")
    print("held_out_gt_rows_used =", receipt["held_out_gt_rows_used"])
    print("calibration_performed = false")
    print("DAY31 BASELINE FREEZE RECEIPT: PASS")


def cmd_audit() -> None:
    env = verify_frozen_environment(require_raw=False)
    if not PREDICTIONS_PATH.exists() or not METRICS_PATH.exists() or not FREEZE_RECEIPT_PATH.exists():
        raise RuntimeError("Day31 frozen artifacts missing")

    predictions = read_jsonl(PREDICTIONS_PATH)
    prediction_errors = validate_prediction_rows(
        predictions,
        development_ids=env["development_ids"],
        held_out_ids=env["held_out_ids"],
        require_complete=True,
    )
    metrics = read_json(METRICS_PATH)
    expected_metrics = evaluate_predictions(
        predictions,
        env["development_ids"],
        env["held_out_ids"],
    )
    receipt = read_json(FREEZE_RECEIPT_PATH)

    errors = list(prediction_errors)
    if metrics != expected_metrics:
        errors.append("metrics differ from fresh development-only recomputation")

    expected_receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "development_root_cause_baseline_frozen_day31_complete",
        "day30_split_freeze_commit": DAY30_FINAL_COMMIT,
        "day30_split_freeze_receipt_sha256": DAY30_RECEIPT_SHA256,
        "day30_episode_split_sha256": EPISODE_SPLIT_SHA256,
        "day30_pair_group_split_sha256": PAIR_SPLIT_SHA256,
        "day29_ground_truth_records_sha256": GT_SHA256,
        "config_sha256": sha256_file(CONFIG_PATH),
        "prompt_contract_sha256": sha256_file(PROMPT_PATH),
        "operational_contract_sha256": sha256_file(CONTRACT_PATH),
        "predictions_sha256": sha256_file(PREDICTIONS_PATH),
        "metrics_sha256": sha256_file(METRICS_PATH),
        "baseline_mode": "direct_zero_shot_multimodal_no_retrieval",
        "model_name": env["config"]["model"]["model_name"],
        "development_prediction_count": 60,
        "held_out_prediction_count": 0,
        "held_out_gt_rows_used": 0,
        "inference_read_ground_truth": False,
        "selection_used_ground_truth": False,
        "retrieval_used": False,
        "manual_corpus_used": False,
        "model_training_performed": False,
        "calibration_performed": False,
        "prompt_tuning_after_results": False,
        "model_selection_using_development_results": False,
        "held_out_prompt_tuning": False,
        "held_out_retrieval_tuning": False,
        "held_out_model_selection": False,
        "held_out_evaluation_started": False,
        "held_out_final_evaluation_count_consumed": 0,
        "primary_metrics": metrics["primary_metrics"],
        "secondary_metrics": metrics["secondary_metrics"],
    }
    for key, expected in expected_receipt.items():
        if receipt.get(key) != expected:
            errors.append(
                f"receipt {key} mismatch: expected={expected!r}, "
                f"actual={receipt.get(key)!r}"
            )

    tooling_commit = receipt.get("tooling_commit")
    if not isinstance(tooling_commit, str) or not tooling_commit:
        errors.append("receipt tooling_commit missing")
    else:
        try:
            require_ancestor(tooling_commit)
        except RuntimeError as exc:
            errors.append(str(exc))
        blobs = receipt.get("tooling_git_blobs")
        if not isinstance(blobs, dict):
            errors.append("receipt tooling_git_blobs missing")
        else:
            for rel in TOOLING_PATHS:
                try:
                    frozen_blob = git_output("rev-parse", f"{tooling_commit}:{rel}")
                    current_blob = git_output("rev-parse", f"HEAD:{rel}")
                except subprocess.CalledProcessError:
                    errors.append(f"tooling path missing from Git: {rel}")
                    continue
                if blobs.get(rel) != frozen_blob:
                    errors.append(f"receipt tooling blob mismatch: {rel}")
                if current_blob != frozen_blob:
                    errors.append(f"tooling changed after freeze: {rel}")

    print("===== DAY31 BASELINE FREEZE AUDIT =====")
    print("development_prediction_count =", len(predictions))
    print("held_out_prediction_count =",
          sum(p["episode_id"] in env["held_out_ids"] for p in predictions))
    print("held_out_gt_rows_used =", metrics["held_out_gt_rows_used"])
    print("predictions_sha256 =", sha256_file(PREDICTIONS_PATH))
    print("metrics_sha256 =", sha256_file(METRICS_PATH))
    print("freeze_receipt_sha256 =", sha256_file(FREEZE_RECEIPT_PATH))
    print("primary_metrics =", metrics["primary_metrics"])
    print("secondary_metrics =", metrics["secondary_metrics"])
    print("errors =", errors)
    if errors:
        raise SystemExit(1)
    print("DAY31 BASELINE AUDIT: PASS")
    print("DAY31: CLOSED / FROZEN")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("run")
    sub.add_parser("validate-predictions")
    sub.add_parser("evaluate")
    sub.add_parser("freeze")
    sub.add_parser("audit")
    args = parser.parse_args()

    if args.command == "preflight":
        cmd_preflight()
    elif args.command == "run":
        cmd_run()
    elif args.command == "validate-predictions":
        cmd_validate_predictions()
    elif args.command == "evaluate":
        cmd_evaluate()
    elif args.command == "freeze":
        cmd_freeze()
    elif args.command == "audit":
        cmd_audit()


if __name__ == "__main__":
    main()
