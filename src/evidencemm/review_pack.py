from __future__ import annotations

import csv
import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageChops, ImageOps

from evidencemm.data_binding import sha256_file
from evidencemm.robot_failure_dataset import (
    AnomalyReviewCase,
    SourceAuditRecord,
    load_anomaly_review_cases,
    load_source_audit,
)
from evidencemm.state_action_selection import (
    JOINT_ORDER,
    StateActionFrameScore,
    StateActionSample,
    load_state_action_samples,
    score_state_action_sample,
    validate_source_semantics,
)
from evidencemm.temporal_evidence import (
    EpisodeManifest,
    FrameRecord,
    load_frame_records,
)


REVIEW_TEMPLATE_SCHEMA = "evidencemm_day16_human_review_v2"
LEGACY_REVIEW_TEMPLATE_SCHEMA = "evidencemm_day16_human_review_v1"
REVIEW_PACK_SCHEMA = "evidencemm_day16_review_pack_v1"


@dataclass(frozen=True)
class SelectionConfig:
    uniform_count: int = 7
    top_state_action: int = 5
    top_tracking_gap: int = 3
    top_gripper_change: int = 4
    top_visual_motion_per_camera: int = 4
    min_separation_frames: int = 12
    visual_stride: int = 3
    visual_width: int = 160
    visual_height: int = 120
    max_selected_frames: int = 24
    thumbnail_width: int = 360

    def validate(self) -> None:
        for name in (
            "uniform_count",
            "top_state_action",
            "top_tracking_gap",
            "top_gripper_change",
            "top_visual_motion_per_camera",
            "min_separation_frames",
            "visual_stride",
            "visual_width",
            "visual_height",
            "max_selected_frames",
            "thumbnail_width",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_selected_frames < self.uniform_count:
            raise ValueError(
                "max_selected_frames must be >= uniform_count"
            )


@dataclass
class CandidateAccumulator:
    frame_index: int
    timestamp_sec: float
    reasons: set[str] = field(default_factory=set)
    reason_priorities: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def add_reason(
        self,
        reason: str,
        *,
        priority: float,
        metrics: dict[str, float] | None = None,
    ) -> None:
        self.reasons.add(reason)
        previous = self.reason_priorities.get(reason)
        if previous is None or priority > previous:
            self.reason_priorities[reason] = float(priority)
        if metrics:
            for key, value in metrics.items():
                self.metrics[key] = float(value)

    @property
    def best_priority(self) -> float:
        if not self.reason_priorities:
            return 0.0
        return max(self.reason_priorities.values())


def uniform_frame_indices(
    frame_count: int,
    count: int,
) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if count == 1:
        return [0]
    if count >= frame_count:
        return list(range(frame_count))

    values = {
        int(round(i * (frame_count - 1) / (count - 1)))
        for i in range(count)
    }
    return sorted(values)


def select_peaks_with_nms(
    scored_frames: Iterable[tuple[int, float]],
    *,
    top_k: int,
    min_separation_frames: int,
) -> list[tuple[int, float]]:
    if top_k <= 0:
        return []
    if min_separation_frames < 0:
        raise ValueError(
            "min_separation_frames must be non-negative"
        )

    ranked = sorted(
        (
            (int(frame_index), float(score))
            for frame_index, score in scored_frames
            if math.isfinite(float(score))
        ),
        key=lambda item: (-item[1], item[0]),
    )

    selected: list[tuple[int, float]] = []
    for frame_index, score in ranked:
        if all(
            abs(frame_index - chosen_index)
            >= min_separation_frames
            for chosen_index, _ in selected
        ):
            selected.append((frame_index, score))
            if len(selected) >= top_k:
                break
    return selected


def _camera_transform(
    image: Image.Image,
    transform: str,
) -> Image.Image:
    normalized = (
        transform.lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )
    if not normalized or normalized in {"none", "identity"}:
        return image.copy()

    if "ccw" in normalized and "90" in normalized:
        return image.transpose(Image.Transpose.ROTATE_90)
    if (
        "cw" in normalized
        and "ccw" not in normalized
        and "90" in normalized
    ):
        return image.transpose(Image.Transpose.ROTATE_270)
    if "180" in normalized:
        return image.transpose(Image.Transpose.ROTATE_180)

    return image.copy()


def _load_motion_image(
    path: Path,
    *,
    transform: str,
    width: int,
    height: int,
) -> Image.Image:
    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source)
        transformed = _camera_transform(
            source,
            transform,
        )
        gray = ImageOps.grayscale(transformed)
        return gray.resize(
            (width, height),
            Image.Resampling.BILINEAR,
        )


def visual_motion_scores(
    *,
    episode_dir: Path,
    records_by_frame: dict[int, FrameRecord],
    transform: str,
    frame_count: int,
    stride: int,
    width: int,
    height: int,
) -> list[tuple[int, float]]:
    if stride <= 0:
        raise ValueError("stride must be positive")

    indices = list(range(0, frame_count, stride))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)

    if len(indices) < 2:
        return []

    scores: list[tuple[int, float]] = []
    previous_index = indices[0]
    previous_record = records_by_frame[previous_index]
    previous = _load_motion_image(
        episode_dir / previous_record.image_relpath,
        transform=transform,
        width=width,
        height=height,
    )

    for frame_index in indices[1:]:
        record = records_by_frame[frame_index]
        current = _load_motion_image(
            episode_dir / record.image_relpath,
            transform=transform,
            width=width,
            height=height,
        )
        diff = ImageChops.difference(
            current,
            previous,
        )
        histogram = diff.histogram()
        weighted_sum = sum(
            value * count
            for value, count in enumerate(histogram)
        )
        pixel_count = width * height
        score = (
            weighted_sum / pixel_count / 255.0
            if pixel_count
            else 0.0
        )
        scores.append(
            (frame_index, float(score))
        )
        previous = current
        previous_index = frame_index

    return scores


def build_state_scores(
    samples: list[StateActionSample],
) -> list[StateActionFrameScore]:
    scores = []
    for index, sample in enumerate(samples):
        previous = (
            samples[index - 1]
            if index > 0
            else None
        )
        scores.append(
            score_state_action_sample(
                current=sample,
                previous=previous,
            )
        )
    return scores


def _gripper_action_delta(
    samples: list[StateActionSample],
) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    previous = None
    for sample in samples:
        if previous is None:
            score = 0.0
        else:
            score = abs(
                float(sample.action.gripper)
                - float(previous.action.gripper)
            )
        values.append(
            (sample.frame_index, score)
        )
        previous = sample
    return values


def select_review_candidates(
    *,
    samples: list[StateActionSample],
    state_scores: list[StateActionFrameScore],
    front_motion: list[tuple[int, float]],
    wrist_motion: list[tuple[int, float]],
    config: SelectionConfig,
) -> list[CandidateAccumulator]:
    config.validate()
    if not samples:
        raise ValueError("samples must be non-empty")
    if len(samples) != len(state_scores):
        raise ValueError(
            "samples/state_scores length mismatch"
        )

    by_frame: dict[int, CandidateAccumulator] = {}

    def get(frame_index: int) -> CandidateAccumulator:
        if frame_index < 0 or frame_index >= len(samples):
            raise ValueError(
                f"candidate frame outside episode: {frame_index}"
            )
        item = by_frame.get(frame_index)
        if item is None:
            item = CandidateAccumulator(
                frame_index=frame_index,
                timestamp_sec=(
                    samples[frame_index].timestamp_sec
                ),
            )
            by_frame[frame_index] = item
        return item

    uniform = uniform_frame_indices(
        len(samples),
        config.uniform_count,
    )
    for frame_index in uniform:
        get(frame_index).add_reason(
            "uniform_anchor",
            priority=1000.0,
        )

    def add_ranked(
        reason: str,
        scored: Iterable[tuple[int, float]],
        *,
        top_k: int,
        base_priority: float,
        metric_name: str,
    ) -> None:
        selected = select_peaks_with_nms(
            scored,
            top_k=top_k,
            min_separation_frames=(
                config.min_separation_frames
            ),
        )
        for rank, (frame_index, score) in enumerate(
            selected,
            start=1,
        ):
            get(frame_index).add_reason(
                reason,
                priority=(
                    base_priority
                    - rank
                ),
                metrics={
                    metric_name: score
                },
            )

    add_ranked(
        "state_action_change",
        (
            (
                score.frame_index,
                score.fused_state_action_score,
            )
            for score in state_scores
        ),
        top_k=config.top_state_action,
        base_priority=900.0,
        metric_name="fused_state_action_score",
    )

    add_ranked(
        "tracking_gap",
        (
            (
                score.frame_index,
                score.tracking_gap_rms,
            )
            for score in state_scores
        ),
        top_k=config.top_tracking_gap,
        base_priority=700.0,
        metric_name="tracking_gap_rms",
    )

    add_ranked(
        "gripper_action_change",
        _gripper_action_delta(samples),
        top_k=config.top_gripper_change,
        base_priority=850.0,
        metric_name="gripper_action_delta",
    )

    add_ranked(
        "front_visual_motion",
        front_motion,
        top_k=config.top_visual_motion_per_camera,
        base_priority=800.0,
        metric_name="front_visual_motion",
    )

    add_ranked(
        "wrist_visual_motion",
        wrist_motion,
        top_k=config.top_visual_motion_per_camera,
        base_priority=800.0,
        metric_name="wrist_visual_motion",
    )

    mandatory = [
        item
        for item in by_frame.values()
        if "uniform_anchor" in item.reasons
    ]
    optional = [
        item
        for item in by_frame.values()
        if "uniform_anchor" not in item.reasons
    ]

    optional.sort(
        key=lambda item: (
            -item.best_priority,
            item.frame_index,
        )
    )

    remaining = max(
        0,
        config.max_selected_frames
        - len(mandatory),
    )
    selected = mandatory + optional[:remaining]

    return sorted(
        selected,
        key=lambda item: item.frame_index,
    )


def _round_dict(
    values: dict[str, float],
    digits: int = 6,
) -> dict[str, float]:
    return {
        key: round(float(value), digits)
        for key, value in values.items()
    }


def _joint_dict(vector) -> dict[str, float]:
    return {
        joint: round(
            float(getattr(vector, joint)),
            6,
        )
        for joint in JOINT_ORDER
    }


def _make_thumbnail(
    *,
    source_path: Path,
    output_path: Path,
    transform: str,
    target_width: int,
) -> None:
    with Image.open(source_path) as source:
        source = ImageOps.exif_transpose(source)
        image = _camera_transform(
            source,
            transform,
        )
        ratio = (
            target_width / image.width
        )
        target_height = max(
            1,
            int(round(image.height * ratio)),
        )
        image = image.resize(
            (target_width, target_height),
            Image.Resampling.LANCZOS,
        )
        if image.mode != "RGB":
            image = image.convert("RGB")
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        image.save(
            output_path,
            "JPEG",
            quality=88,
            optimize=True,
        )


def _camera_transform_map(
    metadata: dict,
) -> dict[str, str]:
    raw = (
        metadata.get("settings", {})
        .get("camera_transforms", {})
    )
    return {
        "front": str(raw.get("front", "none")),
        "wrist": str(raw.get("wrist", "none")),
    }


def _records_by_camera(
    records: list[FrameRecord],
    frame_count: int,
) -> dict[str, dict[int, FrameRecord]]:
    result = {
        "front": {},
        "wrist": {},
    }
    for record in records:
        result[record.camera][
            record.frame_index
        ] = record

    expected = set(range(frame_count))
    for camera in ("front", "wrist"):
        if set(result[camera]) != expected:
            raise ValueError(
                f"{camera} frame records do not cover episode"
            )
    return result


def _review_template(
    review: AnomalyReviewCase,
) -> dict:
    return {
        "schema_version": REVIEW_TEMPLATE_SCHEMA,
        "review_id": review.review_id,
        "episode_id": review.episode_id,
        "task_success": review.task_success,
        "operation_anomaly": True,
        "original_failure_reason": (
            review.original_failure_reason
        ),
        "events": [
            {
                "event_id": event.event_id,
                "observed_failure_mode": (
                    event.observed_failure_mode.value
                ),
                "failure_interval": None,
                "causal_diagnosis": None,
                "supporting_robot_refs": [],
                "counterevidence_robot_refs": [],
                "confidence": None,
                "event_status": "draft",
            }
            for event in review.events
        ],
        "reviewer": None,
        "notes": None,
        "instructions": {
            "failure_interval": (
                "Choose the smallest defensible start/end frame "
                "for each event after inspecting both cameras and state/action."
            ),
            "causal_diagnosis": (
                "Do not infer cause from original_failure_reason or "
                "observed_failure_mode. Use insufficient_evidence when "
                "the evidence does not support a causal claim."
            ),
            "supporting_robot_refs": (
                "Use canonical robot EvidenceRef objects for frames that "
                "support the event-level diagnosis."
            ),
            "counterevidence_robot_refs": (
                "Use canonical robot EvidenceRef objects that weaken "
                "alternative hypotheses."
            ),
        },
    }


def _legacy_template_is_unedited(
    payload: dict,
) -> bool:
    return (
        payload.get("schema_version")
        == LEGACY_REVIEW_TEMPLATE_SCHEMA
        and payload.get("review_status") == "draft"
        and payload.get("failure_interval") is None
        and payload.get("causal_diagnosis") is None
        and payload.get("confidence") is None
        and payload.get("supporting_frames", []) == []
        and payload.get("counterevidence_frames", []) == []
        and payload.get("reviewer") is None
    )


def _write_or_migrate_review_template(
    path: Path,
    review: AnomalyReviewCase,
) -> str:
    desired = _review_template(review)

    if not path.exists():
        path.write_text(
            json.dumps(
                desired,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return "created"

    existing = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    schema = existing.get(
        "schema_version"
    )

    if schema == REVIEW_TEMPLATE_SCHEMA:
        return "preserved"

    if _legacy_template_is_unedited(
        existing
    ):
        path.write_text(
            json.dumps(
                desired,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return "migrated_v1_to_v2"

    raise ValueError(
        f"refusing to overwrite edited or unknown review template: {path}"
    )


def _write_selected_csv(
    path: Path,
    rows: list[dict],
) -> None:
    fieldnames = [
        "frame_index",
        "timestamp_sec",
        "reasons",
        "fused_state_action_score",
        "tracking_gap_rms",
        "gripper_action_delta",
        "front_visual_motion",
        "wrist_visual_motion",
        *[
            f"observation_{joint}"
            for joint in JOINT_ORDER
        ],
        *[
            f"action_{joint}"
            for joint in JOINT_ORDER
        ],
        *[
            f"tracking_error_{joint}"
            for joint in JOINT_ORDER
        ],
    ]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        for row in rows:
            flat = {
                "frame_index": row["frame_index"],
                "timestamp_sec": row["timestamp_sec"],
                "reasons": "|".join(
                    row["reasons"]
                ),
                "fused_state_action_score": (
                    row["state_action"][
                        "fused_state_action_score"
                    ]
                ),
                "tracking_gap_rms": (
                    row["state_action"][
                        "tracking_gap_rms"
                    ]
                ),
                "gripper_action_delta": (
                    row["metrics"].get(
                        "gripper_action_delta",
                        "",
                    )
                ),
                "front_visual_motion": (
                    row["metrics"].get(
                        "front_visual_motion",
                        "",
                    )
                ),
                "wrist_visual_motion": (
                    row["metrics"].get(
                        "wrist_visual_motion",
                        "",
                    )
                ),
            }
            for joint in JOINT_ORDER:
                flat[
                    f"observation_{joint}"
                ] = row["observation"][joint]
                flat[
                    f"action_{joint}"
                ] = row["action"][joint]
                flat[
                    f"tracking_error_{joint}"
                ] = row[
                    "tracking_error"
                ][joint]
            writer.writerow(flat)


def _render_episode_html(
    *,
    review: AnomalyReviewCase,
    manifest: EpisodeManifest,
    rows: list[dict],
    output_path: Path,
    source_status: str,
    source_overall_pass: bool,
    source_failed_checks: list[str],
) -> None:
    cards = []
    for row in rows:
        joint_rows = []
        for joint in JOINT_ORDER:
            joint_rows.append(
                "<tr>"
                f"<td>{html.escape(joint)}</td>"
                f"<td>{row['observation'][joint]:.4f}</td>"
                f"<td>{row['action'][joint]:.4f}</td>"
                f"<td>{row['tracking_error'][joint]:.4f}</td>"
                "</tr>"
            )

        metrics = " · ".join(
            f"{html.escape(key)}={value:.5f}"
            for key, value in sorted(
                row["metrics"].items()
            )
        ) or "none"

        cards.append(
            f"""
<section class="frame-card" id="f{row['frame_index']}">
  <h3>Frame {row['frame_index']} · {row['timestamp_sec']:.3f}s</h3>
  <p><b>Selection:</b> {html.escape(', '.join(row['reasons']))}</p>
  <p><b>Metrics:</b> {html.escape(metrics)}</p>
  <div class="camera-grid">
    <figure>
      <img src="{html.escape(row['front_thumb'])}" loading="lazy">
      <figcaption>front · source age {row['front_source_age_ms']:.2f} ms</figcaption>
    </figure>
    <figure>
      <img src="{html.escape(row['wrist_thumb'])}" loading="lazy">
      <figcaption>wrist · source age {row['wrist_source_age_ms']:.2f} ms</figcaption>
    </figure>
  </div>
  <table>
    <thead><tr><th>joint</th><th>observation</th><th>action</th><th>|tracking error|</th></tr></thead>
    <tbody>{''.join(joint_rows)}</tbody>
  </table>
  <p>
    fused state/action change = {row['state_action']['fused_state_action_score']:.5f} ·
    tracking gap RMS = {row['state_action']['tracking_gap_rms']:.5f}
  </p>
</section>
"""
        )

    event_rows = "".join(
        "<li>"
        f"<b>{html.escape(event.event_id)}</b>: "
        f"{html.escape(event.observed_failure_mode.value)} "
        "(draft causal diagnosis)"
        "</li>"
        for event in review.events
    )
    failed_checks = (
        ", ".join(source_failed_checks)
        if source_failed_checks
        else "none"
    )

    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Day16 Review · {html.escape(review.episode_id)}</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1500px; margin: 24px auto; padding: 0 18px; color: #222; }}
header {{ position: sticky; top: 0; background: white; padding: 10px 0; border-bottom: 1px solid #ddd; z-index: 10; }}
.warning {{ padding: 12px; border: 1px solid #aaa; background: #f7f7f7; }}
.camera-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
figure {{ margin: 0; }}
img {{ width: 100%; height: auto; border: 1px solid #ddd; }}
.frame-card {{ margin: 30px 0; padding: 16px; border: 1px solid #ccc; border-radius: 8px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
th, td {{ border: 1px solid #ddd; padding: 6px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
code {{ background: #f4f4f4; padding: 2px 4px; }}
@media (max-width: 800px) {{ .camera-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
  <b>{html.escape(review.episode_id)}</b> ·
  <a href="../index.html">back to index</a> ·
  <a href="selected_frames.csv">CSV</a> ·
  <a href="review_template.json">review template</a>
</header>

<h1>Human Review Pack</h1>
<div class="warning">
  <b>Event-level contract:</b> this episode contains {len(review.events)} anomaly event(s).
  Do not merge separate events into one interval and do not infer causal diagnosis from the source label.
</div>

<ul>
<li><b>task:</b> {html.escape(manifest.task)}</li>
<li><b>task_success:</b> {review.task_success}</li>
<li><b>operation_anomaly:</b> true</li>
<li><b>original_failure_reason:</b> {html.escape(review.original_failure_reason)}</li>
<li><b>source status:</b> {html.escape(source_status)}</li>
<li><b>source overall_pass:</b> {source_overall_pass}</li>
<li><b>source failed checks:</b> {html.escape(failed_checks)}</li>
<li><b>diagnostic episode SHA256:</b> <code>{manifest.episode_sha256}</code></li>
<li><b>selected frame count:</b> {len(rows)}</li>
</ul>

<h2>Events to verify</h2>
<ul>
{event_rows}
</ul>

<p>
Use the chronological cards below to identify a separate smallest defensible interval for each event.
If these candidates are insufficient, inspect nearby raw frames around the closest candidate rather than scanning the full episode.
</p>

{''.join(cards)}
</body>
</html>
""",
        encoding="utf-8",
        newline="\n",
    )


def _render_index_html(
    *,
    packs: list[dict],
    output_path: Path,
) -> None:
    rows = []
    for pack in packs:
        rows.append(
            "<tr>"
            f"<td><a href=\"{html.escape(pack['episode_id'])}/review.html\">"
            f"{html.escape(pack['episode_id'])}</a></td>"
            f"<td>{html.escape(pack['original_failure_reason'])}</td>"
            f"<td>{html.escape(', '.join(pack['observed_failure_modes']))}</td>"
            f"<td>{pack['event_count']}</td>"
            f"<td>{pack['selected_frame_count']}</td>"
            f"<td>{html.escape(pack['source_status'])}</td>"
            "</tr>"
        )

    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EvidenceMM Day16 Review Packs</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1400px; margin: 28px auto; padding: 0 18px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
th {{ text-align: left; }}
.notice {{ padding: 12px; border: 1px solid #aaa; background: #f7f7f7; margin: 14px 0; }}
</style>
</head>
<body>
<h1>EvidenceMM Day16 · 8-Episode Human Review Pack</h1>
<div class="notice">
The pack preserves 9 anomaly events across 8 episodes.
The 24-frame selection algorithm is unchanged and remains a review aid, not a causal diagnosis model.
</div>
<table>
<thead>
<tr>
<th>episode</th>
<th>original failure reason</th>
<th>observed modes</th>
<th>events</th>
<th>selected frames</th>
<th>source status</th>
</tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body>
</html>
""",
        encoding="utf-8",
        newline="\n",
    )


def generate_episode_review_pack(
    *,
    project_root: Path,
    review: AnomalyReviewCase,
    source_audit: SourceAuditRecord,
    output_root: Path,
    config: SelectionConfig,
    binding_provenance: dict | None = None,
) -> dict:
    episode_dir = Path(
        source_audit.raw_episode_dir
    ).resolve()
    metadata_path = episode_dir / "metadata.json"
    samples_path = episode_dir / "samples.csv"

    manifest_path = (
        project_root
        / review.diagnostic_manifest_path
    ).resolve()
    frames_path = (
        project_root
        / review.diagnostic_frames_path
    ).resolve()

    manifest = EpisodeManifest.model_validate_json(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if manifest.episode_id != review.episode_id:
        raise ValueError(
            "diagnostic manifest episode_id mismatch"
        )
    if sha256_file(metadata_path) != manifest.metadata_sha256:
        raise ValueError(
            "metadata.json SHA256 differs from diagnostic manifest"
        )
    if sha256_file(samples_path) != manifest.samples_csv_sha256:
        raise ValueError(
            "samples.csv SHA256 differs from diagnostic manifest"
        )

    validate_source_semantics(metadata_path)
    samples = load_state_action_samples(
        samples_path,
        verify_tracking_error=True,
    )
    if len(samples) != manifest.frame_count:
        raise ValueError(
            "state/action sample count differs from diagnostic manifest"
        )

    records = load_frame_records(
        frames_path
    )
    by_camera = _records_by_camera(
        records,
        manifest.frame_count,
    )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )
    transforms = _camera_transform_map(
        metadata
    )

    state_scores = build_state_scores(
        samples
    )
    front_motion = visual_motion_scores(
        episode_dir=episode_dir,
        records_by_frame=by_camera["front"],
        transform=transforms["front"],
        frame_count=manifest.frame_count,
        stride=config.visual_stride,
        width=config.visual_width,
        height=config.visual_height,
    )
    wrist_motion = visual_motion_scores(
        episode_dir=episode_dir,
        records_by_frame=by_camera["wrist"],
        transform=transforms["wrist"],
        frame_count=manifest.frame_count,
        stride=config.visual_stride,
        width=config.visual_width,
        height=config.visual_height,
    )

    candidates = select_review_candidates(
        samples=samples,
        state_scores=state_scores,
        front_motion=front_motion,
        wrist_motion=wrist_motion,
        config=config,
    )

    score_by_frame = {
        score.frame_index: score
        for score in state_scores
    }

    episode_output = (
        output_root / review.episode_id
    )
    thumbs_dir = (
        episode_output / "thumbs"
    )
    episode_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict] = []
    for candidate in candidates:
        frame_index = candidate.frame_index
        sample = samples[frame_index]
        score = score_by_frame[frame_index]
        front = by_camera["front"][
            frame_index
        ]
        wrist = by_camera["wrist"][
            frame_index
        ]

        front_name = (
            f"f{frame_index:06d}_front.jpg"
        )
        wrist_name = (
            f"f{frame_index:06d}_wrist.jpg"
        )

        _make_thumbnail(
            source_path=(
                episode_dir
                / front.image_relpath
            ),
            output_path=(
                thumbs_dir
                / front_name
            ),
            transform=transforms["front"],
            target_width=config.thumbnail_width,
        )
        _make_thumbnail(
            source_path=(
                episode_dir
                / wrist.image_relpath
            ),
            output_path=(
                thumbs_dir
                / wrist_name
            ),
            transform=transforms["wrist"],
            target_width=config.thumbnail_width,
        )

        metrics = dict(
            candidate.metrics
        )
        metrics.setdefault(
            "fused_state_action_score",
            score.fused_state_action_score,
        )
        metrics.setdefault(
            "tracking_gap_rms",
            score.tracking_gap_rms,
        )

        rows.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": round(
                    sample.timestamp_sec,
                    6,
                ),
                "reasons": sorted(
                    candidate.reasons
                ),
                "metrics": _round_dict(
                    metrics
                ),
                "state_action": {
                    "state_change_rms": round(
                        score.state_change_rms,
                        6,
                    ),
                    "action_change_rms": round(
                        score.action_change_rms,
                        6,
                    ),
                    "fused_state_action_score": round(
                        score.fused_state_action_score,
                        6,
                    ),
                    "tracking_gap_rms": round(
                        score.tracking_gap_rms,
                        6,
                    ),
                },
                "observation": _joint_dict(
                    sample.observation
                ),
                "action": _joint_dict(
                    sample.action
                ),
                "tracking_error": _joint_dict(
                    sample.tracking_error
                ),
                "state_delta": _joint_dict(
                    score.state_delta
                ),
                "action_delta": _joint_dict(
                    score.action_delta
                ),
                "front_thumb": (
                    f"thumbs/{front_name}"
                ),
                "wrist_thumb": (
                    f"thumbs/{wrist_name}"
                ),
                "front_source_age_ms": round(
                    front.source_age_ms,
                    3,
                ),
                "wrist_source_age_ms": round(
                    wrist.source_age_ms,
                    3,
                ),
                "front_source_frame": (
                    front.image_relpath
                ),
                "wrist_source_frame": (
                    wrist.image_relpath
                ),
            }
        )

    selected_json = (
        episode_output
        / "selected_frames.json"
    )
    selected_json.write_text(
        json.dumps(
            {
                "schema_version": REVIEW_PACK_SCHEMA,
                "episode_id": review.episode_id,
                "selected_frame_count": len(rows),
                "selection_config": {
                    key: getattr(config, key)
                    for key in config.__dataclass_fields__
                },
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    _write_selected_csv(
        episode_output
        / "selected_frames.csv",
        rows,
    )
    template_action = (
        _write_or_migrate_review_template(
            episode_output
            / "review_template.json",
            review,
        )
    )

    provenance = (
        binding_provenance or {}
    )
    source_status = str(
        provenance.get(
            "source_status",
            metadata.get(
                "status",
                "",
            ),
        )
    )
    source_overall_pass = bool(
        provenance.get(
            "source_overall_pass",
            metadata.get(
                "overall_pass",
                False,
            ),
        )
    )
    source_failed_checks = list(
        provenance.get(
            "source_failed_checks",
            [
                key
                for key, value
                in metadata.get(
                    "checks",
                    {}
                ).items()
                if value is not True
            ],
        )
    )

    _render_episode_html(
        review=review,
        manifest=manifest,
        rows=rows,
        output_path=(
            episode_output
            / "review.html"
        ),
        source_status=source_status,
        source_overall_pass=(
            source_overall_pass
        ),
        source_failed_checks=(
            source_failed_checks
        ),
    )

    return {
        "episode_id": review.episode_id,
        "original_failure_reason": (
            review.original_failure_reason
        ),
        "observed_failure_modes": [
            event.observed_failure_mode.value
            for event in review.events
        ],
        "event_count": len(
            review.events
        ),
        "selected_frame_count": len(rows),
        "source_status": source_status,
        "source_overall_pass": (
            source_overall_pass
        ),
        "source_failed_checks": (
            source_failed_checks
        ),
        "episode_sha256": (
            manifest.episode_sha256
        ),
        "review_template_action": (
            template_action
        ),
        "review_html": (
            f"{review.episode_id}/review.html"
        ),
        "review_template": (
            f"{review.episode_id}/review_template.json"
        ),
    }


def generate_review_packs(
    *,
    project_root: Path,
    audit_path: Path,
    review_cases_path: Path,
    output_root: Path,
    config: SelectionConfig,
    binding_report_path: Path | None = None,
) -> dict:
    audits = load_source_audit(
        audit_path
    )
    reviews = load_anomaly_review_cases(
        review_cases_path
    )

    audit_by_id = {
        item.episode_id: item
        for item in audits
    }
    if len(reviews) != 8:
        raise ValueError(
            f"expected 8 anomaly review cases, got {len(reviews)}"
        )
    if len({
        item.episode_id
        for item in reviews
    }) != 8:
        raise ValueError(
            "review case episode IDs must be unique"
        )

    total_events = sum(
        len(review.events)
        for review in reviews
    )
    if total_events != 9:
        raise ValueError(
            f"expected 9 anomaly events, got {total_events}"
        )

    binding_by_id: dict[str, dict] = {}
    if (
        binding_report_path is not None
        and binding_report_path.is_file()
    ):
        payload = json.loads(
            binding_report_path.read_text(
                encoding="utf-8"
            )
        )
        binding_by_id = {
            item["episode_id"]: item
            for item in payload.get(
                "results",
                []
            )
        }

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    packs = []
    for review in reviews:
        audit = audit_by_id.get(
            review.episode_id
        )
        if audit is None:
            raise ValueError(
                f"missing source audit for {review.episode_id}"
            )
        packs.append(
            generate_episode_review_pack(
                project_root=project_root,
                review=review,
                source_audit=audit,
                output_root=output_root,
                config=config,
                binding_provenance=(
                    binding_by_id.get(
                        review.episode_id
                    )
                ),
            )
        )

    _render_index_html(
        packs=packs,
        output_path=(
            output_root
            / "index.html"
        ),
    )

    manifest = {
        "schema_version": (
            "evidencemm_day16_review_pack_manifest_v2"
        ),
        "pack_count": len(packs),
        "event_count": total_events,
        "episode_ids": [
            pack["episode_id"]
            for pack in packs
        ],
        "multi_event_episode_ids": [
            review.episode_id
            for review in reviews
            if len(review.events) > 1
        ],
        "all_causal_diagnoses_unset_at_generation": all(
            event.causal_diagnosis
            is None
            for review in reviews
            for event in review.events
        ),
        "packs": packs,
        "non_claims": [
            "selected frames are review candidates, not verified failure boundaries",
            "visual motion and state/action peaks are diagnostic aids, not causal labels",
            "the generator does not assign event-level causal_diagnosis",
        ],
    }

    (
        output_root
        / "review_pack_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def validate_review_pack_output(
    output_root: Path,
    *,
    expected_episode_ids: list[str],
) -> dict:
    manifest_path = (
        output_root
        / "review_pack_manifest.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            manifest_path
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    errors: list[str] = []
    actual_ids = manifest.get(
        "episode_ids",
        [],
    )
    if actual_ids != expected_episode_ids:
        errors.append(
            "review pack episode IDs differ from expected order"
        )
    if int(
        manifest.get(
            "event_count",
            0,
        )
    ) != 9:
        errors.append(
            "review pack event_count must be 9"
        )
    if manifest.get(
        "multi_event_episode_ids",
        [],
    ) != ["20260815_111613"]:
        errors.append(
            "review pack multi-event episode must be 20260815_111613"
        )

    for episode_id in expected_episode_ids:
        episode_dir = (
            output_root
            / episode_id
        )
        required = [
            episode_dir / "review.html",
            episode_dir / "selected_frames.json",
            episode_dir / "selected_frames.csv",
            episode_dir / "review_template.json",
        ]
        for path in required:
            if not path.is_file():
                errors.append(
                    f"missing {path}"
                )

        selected_path = (
            episode_dir
            / "selected_frames.json"
        )
        if selected_path.is_file():
            payload = json.loads(
                selected_path.read_text(
                    encoding="utf-8"
                )
            )
            rows = payload.get(
                "rows",
                [],
            )
            if not rows:
                errors.append(
                    f"{episode_id} has no selected frames"
                )
            frames = [
                int(row["frame_index"])
                for row in rows
            ]
            if frames != sorted(
                set(frames)
            ):
                errors.append(
                    f"{episode_id} selected frames are not unique/chronological"
                )
            for row in rows:
                for key in (
                    "front_thumb",
                    "wrist_thumb",
                ):
                    thumb = (
                        episode_dir
                        / row[key]
                    )
                    if not thumb.is_file():
                        errors.append(
                            f"missing thumbnail {thumb}"
                        )

        review_path = (
            episode_dir
            / "review_template.json"
        )
        if review_path.is_file():
            review = json.loads(
                review_path.read_text(
                    encoding="utf-8"
                )
            )
            if (
                review.get(
                    "schema_version"
                )
                != REVIEW_TEMPLATE_SCHEMA
            ):
                errors.append(
                    f"{episode_id} review template schema mismatch"
                )
            events = review.get(
                "events",
                [],
            )
            expected_event_count = (
                2
                if episode_id
                == "20260815_111613"
                else 1
            )
            if len(events) != expected_event_count:
                errors.append(
                    f"{episode_id} review template event count mismatch"
                )
            for event in events:
                if (
                    event.get(
                        "event_status"
                    )
                    != "draft"
                ):
                    errors.append(
                        f"{episode_id} event is not draft"
                    )
                if (
                    event.get(
                        "causal_diagnosis"
                    )
                    is not None
                ):
                    errors.append(
                        f"{episode_id} causal diagnosis was prefilled"
                    )

    return {
        "mode": "day16_review_pack_validation",
        "expected_pack_count": len(
            expected_episode_ids
        ),
        "actual_pack_count": int(
            manifest.get(
                "pack_count",
                0,
            )
        ),
        "expected_event_count": 9,
        "actual_event_count": int(
            manifest.get(
                "event_count",
                0,
            )
        ),
        "multi_event_episode_ids": (
            manifest.get(
                "multi_event_episode_ids",
                [],
            )
        ),
        "errors": errors,
        "valid": (
            not errors
            and int(
                manifest.get(
                    "pack_count",
                    0,
                )
            )
            == len(
                expected_episode_ids
            )
        ),
    }
