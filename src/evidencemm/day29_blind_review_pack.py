from __future__ import annotations

import csv
import hashlib
import html
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageOps

from evidencemm.data_binding import sha256_file
from evidencemm.review_pack import (
    SelectionConfig,
    build_state_scores,
    select_review_candidates,
    visual_motion_scores,
)
from evidencemm.state_action_selection import (
    JOINT_ORDER,
    load_state_action_samples,
)


PACK_SCHEMA = "evidencemm_day29_blind_review_pack_v1"
CASE_SCHEMA = "evidencemm_day29_blind_review_case_v1"

BLANK_RECORD_KEYS = {
    "episode_id",
    "observed_symptom",
    "failure_interval",
    "supporting_robot_refs",
    "counterevidence_robot_refs",
    "supporting_manual_refs",
    "counterevidence_manual_refs",
    "evidence_answerability_gt",
    "explicit_uncertainty_reason",
    "blind_confidence",
    "blind_review_notes",
}


@dataclass(frozen=True)
class RawFrameRef:
    image_relpath: str


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        payload = json.loads(line)

        if not isinstance(payload, dict):
            raise ValueError(
                f"{path}:{line_number}: expected object"
            )

        rows.append(payload)

    return rows


def _is_blank_record(record: dict[str, Any]) -> bool:
    if set(record) != BLANK_RECORD_KEYS:
        return False

    if not record.get("episode_id"):
        return False

    scalar_blank = (
        "observed_symptom",
        "failure_interval",
        "evidence_answerability_gt",
        "explicit_uncertainty_reason",
        "blind_confidence",
        "blind_review_notes",
    )

    if any(
        record[key] is not None
        for key in scalar_blank
    ):
        return False

    list_blank = (
        "supporting_robot_refs",
        "counterevidence_robot_refs",
        "supporting_manual_refs",
        "counterevidence_manual_refs",
    )

    return all(
        record[key] == []
        for key in list_blank
    )


def _population_sha(path: Path) -> str:
    return sha256_file(path)


def _load_source_manifest(
    path: Path,
) -> dict[str, dict[str, str]]:
    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        rows = list(csv.DictReader(handle))

    result: dict[str, dict[str, str]] = {}

    for row in rows:
        episode_id = row.get("episode_id", "")

        if not episode_id:
            raise ValueError(
                "source manifest row lacks episode_id"
            )

        if episode_id in result:
            raise ValueError(
                f"duplicate source manifest episode: "
                f"{episode_id}"
            )

        result[episode_id] = row

    return result


def _resolve_project_path(
    project_root: Path,
    value: str,
) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return (project_root / path).resolve()


def _raw_root(
    project_root: Path,
    config: dict[str, Any],
) -> Path:
    raw_config_path = _resolve_project_path(
        project_root,
        config["inputs"]["raw_audit_config"],
    )

    raw_config = load_yaml(raw_config_path)

    root = Path(
        raw_config[
            "raw_source"
        ][
            "compatibility_wsl_root"
        ]
    )

    if not root.is_dir():
        raise FileNotFoundError(
            f"raw root unavailable: {root}"
        )

    return root


def _selection_config(
    config: dict[str, Any],
) -> SelectionConfig:
    selection = SelectionConfig(
        **config["selection"]
    )
    selection.validate()
    return selection


def _ordered_episode_ids(
    episode_ids: list[str],
    seed: str,
) -> list[str]:
    def key(episode_id: str) -> str:
        payload = (
            seed + "\0" + episode_id
        ).encode("utf-8")

        return hashlib.sha256(
            payload
        ).hexdigest()

    return sorted(
        episode_ids,
        key=key,
    )


def _ordered_ids_sha256(
    ordered_ids: list[str],
) -> str:
    payload = (
        "\n".join(ordered_ids) + "\n"
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


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

    if normalized in {
        "",
        "none",
        "identity",
    }:
        return image.copy()

    if "ccw" in normalized and "90" in normalized:
        return image.transpose(
            Image.Transpose.ROTATE_90
        )

    if (
        "cw" in normalized
        and "ccw" not in normalized
        and "90" in normalized
    ):
        return image.transpose(
            Image.Transpose.ROTATE_270
        )

    if "180" in normalized:
        return image.transpose(
            Image.Transpose.ROTATE_180
        )

    raise ValueError(
        f"unsupported camera transform: "
        f"{transform}"
    )


def _make_thumbnail(
    source: Path,
    destination: Path,
    transform: str,
    width: int,
) -> None:
    with Image.open(source) as raw:
        raw = ImageOps.exif_transpose(raw)

        image = _camera_transform(
            raw,
            transform,
        )

        ratio = width / image.width

        height = max(
            1,
            int(
                round(
                    image.height * ratio
                )
            ),
        )

        image = image.resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )

        if image.mode != "RGB":
            image = image.convert("RGB")

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image.save(
            destination,
            "JPEG",
            quality=88,
            optimize=True,
        )


def _joint_dict(vector: Any) -> dict[str, float]:
    return {
        joint: round(
            float(
                getattr(
                    vector,
                    joint,
                )
            ),
            6,
        )
        for joint in JOINT_ORDER
    }


def _frame_refs(
    camera: str,
    frame_count: int,
) -> dict[int, RawFrameRef]:
    return {
        index: RawFrameRef(
            image_relpath=(
                f"{camera}/"
                f"{index:06d}.jpg"
            )
        )
        for index in range(frame_count)
    }


def _manual_info(
    project_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = _resolve_project_path(
        project_root,
        config[
            "inputs"
        ][
            "approved_manual_manifest"
        ],
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    manual_path = _resolve_project_path(
        project_root,
        manifest["local_path"],
    )

    if not manual_path.is_file():
        raise FileNotFoundError(
            manual_path
        )

    actual_sha = sha256_file(
        manual_path
    )

    if actual_sha != manifest["sha256"]:
        raise ValueError(
            "approved manual SHA256 mismatch"
        )

    page_dir = _resolve_project_path(
        project_root,
        config[
            "inputs"
        ][
            "approved_manual_page_dir"
        ],
    )

    page_glob = config[
        "inputs"
    ][
        "approved_manual_page_glob"
    ]

    pages = sorted(
        page_dir.glob(page_glob)
    )

    if len(pages) != int(
        manifest["page_count"]
    ):
        raise ValueError(
            "approved manual rendered page "
            "count mismatch"
        )

    return {
        "source_id": manifest[
            "source_id"
        ],
        "sha256": actual_sha,
        "page_count": int(
            manifest["page_count"]
        ),
        "pages": pages,
    }


def preflight(
    *,
    project_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    config = load_yaml(
        config_path
    )

    population_path = _resolve_project_path(
        project_root,
        config[
            "inputs"
        ][
            "population_records"
        ],
    )

    source_manifest_path = _resolve_project_path(
        project_root,
        config[
            "inputs"
        ][
            "source_manifest"
        ],
    )

    expected = config["expected"]

    records = load_jsonl(
        population_path
    )

    if len(records) != int(
        expected["canonical_episode_count"]
    ):
        raise ValueError(
            "unexpected canonical population size"
        )

    if not all(
        _is_blank_record(record)
        for record in records
    ):
        raise ValueError(
            "population records are no longer "
            "in frozen blank state"
        )

    stored_sha = config[
        "provenance"
    ][
        "frozen_blank_records_sha256"
    ]

    actual_sha = _population_sha(
        population_path
    )

    if actual_sha != stored_sha:
        raise ValueError(
            "frozen blank review population "
            "SHA256 mismatch"
        )

    ids = [
        record["episode_id"]
        for record in records
    ]

    if len(set(ids)) != len(ids):
        raise ValueError(
            "duplicate review population episode IDs"
        )

    manifest = _load_source_manifest(
        source_manifest_path
    )

    root = _raw_root(
        project_root,
        config,
    )

    frame_count = int(
        expected["frame_count_per_episode"]
    )

    errors: list[str] = []

    for episode_id in ids:
        row = manifest.get(
            episode_id
        )

        if row is None:
            errors.append(
                f"{episode_id}: missing source binding"
            )
            continue

        episode_dir = (
            root
            / row["raw_episode_relpath"]
        )

        metadata = (
            episode_dir
            / "metadata.json"
        )

        samples = (
            episode_dir
            / "samples.csv"
        )

        if not metadata.is_file():
            errors.append(
                f"{episode_id}: metadata missing"
            )
            continue

        if not samples.is_file():
            errors.append(
                f"{episode_id}: samples missing"
            )
            continue

        if (
            sha256_file(metadata)
            != row["metadata_sha256"]
        ):
            errors.append(
                f"{episode_id}: metadata SHA mismatch"
            )

        if (
            sha256_file(samples)
            != row["samples_sha256"]
        ):
            errors.append(
                f"{episode_id}: samples SHA mismatch"
            )

        for camera in (
            "front",
            "wrist",
        ):
            camera_dir = (
                episode_dir
                / camera
            )

            files = list(
                camera_dir.glob("*.jpg")
            )

            if len(files) != frame_count:
                errors.append(
                    f"{episode_id}/{camera}: "
                    f"{len(files)} images"
                )

    if errors:
        raise ValueError(
            "preflight source errors:\n"
            + "\n".join(errors)
        )

    manual = _manual_info(
        project_root,
        config,
    )

    ordered_ids = _ordered_episode_ids(
        ids,
        config[
            "review_order"
        ][
            "seed"
        ],
    )

    return {
        "episode_count": len(ids),
        "raw_root": str(root),
        "population_sha256": actual_sha,
        "manual_source_id": manual[
            "source_id"
        ],
        "manual_sha256": manual[
            "sha256"
        ],
        "manual_page_count": manual[
            "page_count"
        ],
        "review_order_sha256": (
            _ordered_ids_sha256(
                ordered_ids
            )
        ),
    }


def _write_selected_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fieldnames = [
        "frame_index",
        "timestamp_sec",
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
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            flat: dict[str, Any] = {
                "frame_index": row[
                    "frame_index"
                ],
                "timestamp_sec": row[
                    "timestamp_sec"
                ],
            }

            for joint in JOINT_ORDER:
                flat[
                    f"observation_{joint}"
                ] = row[
                    "observation"
                ][joint]

                flat[
                    f"action_{joint}"
                ] = row[
                    "action"
                ][joint]

                flat[
                    f"tracking_error_{joint}"
                ] = row[
                    "tracking_error"
                ][joint]

            writer.writerow(flat)



def _full_state_rows(
    samples: list[Any],
) -> list[dict[str, Any]]:
    return [
        {
            "frame_index": sample.frame_index,
            "timestamp_sec": round(
                sample.timestamp_sec,
                6,
            ),
            "observation": _joint_dict(
                sample.observation
            ),
            "action": _joint_dict(
                sample.action
            ),
            "tracking_error": _joint_dict(
                sample.tracking_error
            ),
        }
        for sample in samples
    ]


def _ensure_camera_symlink(
    *,
    source: Path,
    destination: Path,
) -> None:
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(
            f"refusing to overwrite camera link: "
            f"{destination}"
        )

    if not source.is_dir():
        raise FileNotFoundError(
            f"camera source missing: {source}"
        )

    destination.symlink_to(
        source.resolve(),
        target_is_directory=True,
    )


def _render_frame_explorer(
    *,
    episode_id: str,
    rows: list[dict[str, Any]],
    front_transform: str,
    wrist_transform: str,
    output_path: Path,
) -> None:
    embedded = json.dumps(
        rows,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Day29 Full Frame Explorer · {html.escape(episode_id)}</title>
<style>
body {{
  font-family: Arial, sans-serif;
  max-width: 1500px;
  margin: 24px auto;
  padding: 0 18px;
  color: #222;
}}
.notice {{
  border: 1px solid #aaa;
  padding: 12px;
  background: #f7f7f7;
  margin-bottom: 18px;
}}
.controls {{
  position: sticky;
  top: 0;
  background: white;
  padding: 12px 0;
  border-bottom: 1px solid #ddd;
  z-index: 5;
}}
.controls input[type="range"] {{
  width: min(900px, 70vw);
}}
.cams {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-top: 18px;
}}
figure {{
  margin: 0;
}}
canvas {{
  display: block;
  width: 100%;
  height: auto;
  border: 1px solid #ddd;
  background: #111;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  margin-top: 18px;
}}
th, td {{
  border: 1px solid #ddd;
  padding: 6px;
  text-align: right;
}}
th:first-child, td:first-child {{
  text-align: left;
}}
@media (max-width: 900px) {{
  .cams {{
    grid-template-columns: 1fr;
  }}
}}
</style>
</head>
<body>

<p>
<a href="review.html">selected-frame review</a> ·
<a href="../index.html">review index</a> ·
<a href="../manual/index.html">approved manual</a>
</p>

<h1>Full 900-Frame Blind Explorer</h1>

<div class="notice">
This explorer exposes the complete model-visible robot sequence:
front image, wrist image, observation, action, and tracking_error.
Candidate-selection reasons and administrative metadata are not shown.
</div>

<div class="controls">
<button id="prev" type="button">Previous</button>
<button id="next" type="button">Next</button>

<input
  id="slider"
  type="range"
  min="0"
  max="899"
  value="0"
  step="1"
>

<label>
Frame
<input
  id="frameNumber"
  type="number"
  min="0"
  max="899"
  value="0"
  step="1"
>
</label>

<span id="timeLabel"></span>
</div>

<div class="cams">
<figure>
<canvas id="frontCanvas"></canvas>
<figcaption>front</figcaption>
</figure>

<figure>
<canvas id="wristCanvas"></canvas>
<figcaption>wrist</figcaption>
</figure>
</div>

<table>
<thead>
<tr>
<th>joint</th>
<th>observation</th>
<th>action</th>
<th>tracking_error</th>
</tr>
</thead>
<tbody id="jointRows"></tbody>
</table>

<script>
const frames = {embedded};
const frontTransform = {json.dumps(front_transform)};
const wristTransform = {json.dumps(wrist_transform)};

const slider = document.getElementById("slider");
const frameNumber = document.getElementById("frameNumber");
const timeLabel = document.getElementById("timeLabel");
const jointRows = document.getElementById("jointRows");

function padFrame(index) {{
  return String(index).padStart(6, "0");
}}

function drawImage(canvas, src, transform) {{
  const image = new Image();

  image.onload = () => {{
    const ctx = canvas.getContext("2d");

    if (transform === "ccw90") {{
      canvas.width = image.height;
      canvas.height = image.width;

      ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
      );

      ctx.save();
      ctx.translate(0, image.width);
      ctx.rotate(-Math.PI / 2);
      ctx.drawImage(image, 0, 0);
      ctx.restore();
      return;
    }}

    if (transform === "cw90") {{
      canvas.width = image.height;
      canvas.height = image.width;

      ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
      );

      ctx.save();
      ctx.translate(image.height, 0);
      ctx.rotate(Math.PI / 2);
      ctx.drawImage(image, 0, 0);
      ctx.restore();
      return;
    }}

    if (transform === "180") {{
      canvas.width = image.width;
      canvas.height = image.height;

      ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
      );

      ctx.save();
      ctx.translate(
        image.width,
        image.height
      );
      ctx.rotate(Math.PI);
      ctx.drawImage(image, 0, 0);
      ctx.restore();
      return;
    }}

    canvas.width = image.width;
    canvas.height = image.height;

    ctx.clearRect(
      0,
      0,
      canvas.width,
      canvas.height
    );

    ctx.drawImage(image, 0, 0);
  }};

  image.src = src;
}}

function render(index) {{
  index = Math.max(
    0,
    Math.min(
      frames.length - 1,
      Number(index)
    )
  );

  const frame = frames[index];
  const name = padFrame(index);

  slider.value = String(index);
  frameNumber.value = String(index);

  timeLabel.textContent =
    " · " +
    Number(frame.timestamp_sec).toFixed(3) +
    " s";

  drawImage(
    document.getElementById("frontCanvas"),
    "front_frames/" + name + ".jpg",
    frontTransform
  );

  drawImage(
    document.getElementById("wristCanvas"),
    "wrist_frames/" + name + ".jpg",
    wristTransform
  );

  const joints = Object.keys(
    frame.observation
  );

  jointRows.innerHTML = joints.map(
    joint =>
      "<tr>" +
      "<td>" + joint + "</td>" +
      "<td>" +
      Number(
        frame.observation[joint]
      ).toFixed(4) +
      "</td>" +
      "<td>" +
      Number(
        frame.action[joint]
      ).toFixed(4) +
      "</td>" +
      "<td>" +
      Number(
        frame.tracking_error[joint]
      ).toFixed(4) +
      "</td>" +
      "</tr>"
  ).join("");
}}

slider.addEventListener(
  "input",
  () => render(slider.value)
);

frameNumber.addEventListener(
  "change",
  () => render(frameNumber.value)
);

document.getElementById("prev").addEventListener(
  "click",
  () => render(Number(slider.value) - 1)
);

document.getElementById("next").addEventListener(
  "click",
  () => render(Number(slider.value) + 1)
);

document.addEventListener(
  "keydown",
  event => {{
    if (event.key === "ArrowLeft") {{
      render(Number(slider.value) - 1);
    }}

    if (event.key === "ArrowRight") {{
      render(Number(slider.value) + 1);
    }}
  }}
);

render(0);
</script>

</body>
</html>
""",
        encoding="utf-8",
        newline="\n",
    )


def _render_case_html(
    *,
    episode_id: str,
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    cards: list[str] = []

    for row in rows:
        joint_rows = "".join(
            "<tr>"
            f"<td>{html.escape(joint)}</td>"
            f"<td>{row['observation'][joint]:.4f}</td>"
            f"<td>{row['action'][joint]:.4f}</td>"
            f"<td>{row['tracking_error'][joint]:.4f}</td>"
            "</tr>"
            for joint in JOINT_ORDER
        )

        cards.append(
            f"""
<section class="card">
<h3>Frame {row['frame_index']} · {row['timestamp_sec']:.3f}s</h3>
<div class="cams">
<figure>
<img src="{html.escape(row['front_thumb'])}" loading="lazy">
<figcaption>front</figcaption>
</figure>
<figure>
<img src="{html.escape(row['wrist_thumb'])}" loading="lazy">
<figcaption>wrist</figcaption>
</figure>
</div>
<table>
<thead>
<tr>
<th>joint</th>
<th>observation</th>
<th>action</th>
<th>tracking_error</th>
</tr>
</thead>
<tbody>
{joint_rows}
</tbody>
</table>
</section>
"""
        )

    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Day29 Blind Review · {html.escape(episode_id)}</title>
<style>
body {{
  font-family: Arial, sans-serif;
  max-width: 1450px;
  margin: 24px auto;
  padding: 0 18px;
  color: #222;
}}
.notice {{
  border: 1px solid #aaa;
  padding: 12px;
  background: #f7f7f7;
}}
.cams {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}}
img {{
  width: 100%;
  height: auto;
}}
.card {{
  border: 1px solid #ccc;
  margin: 28px 0;
  padding: 16px;
}}
table {{
  border-collapse: collapse;
  width: 100%;
}}
th, td {{
  border: 1px solid #ddd;
  padding: 6px;
  text-align: right;
}}
th:first-child, td:first-child {{
  text-align: left;
}}
</style>
</head>
<body>
<p>
<a href="../index.html">review index</a> ·
<a href="frame_explorer.html">full 900-frame explorer</a> ·
<a href="../manual/index.html">approved manual</a> ·
<a href="selected_frames.csv">selected CSV</a>
</p>

<h1>Blind Evidence Review</h1>
<p><b>episode:</b> {html.escape(episode_id)}</p>

<div class="notice">
Judge only from front/wrist images,
observation, action, tracking_error,
and the approved manual corpus.
Do not use collection or intervention metadata.
</div>

{''.join(cards)}
</body>
</html>
""",
        encoding="utf-8",
        newline="\n",
    )


def _render_index(
    *,
    cases: list[dict[str, Any]],
    output_path: Path,
) -> None:
    rows = "".join(
        "<tr>"
        f"<td>{case['review_position']}</td>"
        f"<td><a href=\"{html.escape(case['episode_id'])}/review.html\">"
        f"{html.escape(case['episode_id'])}</a></td>"
        f"<td>{case['selected_frame_count']}</td>"
        "</tr>"
        for case in cases
    )

    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EvidenceMM Day29 Blind Review</title>
<style>
body {{
  font-family: Arial, sans-serif;
  max-width: 1100px;
  margin: 28px auto;
  padding: 0 18px;
}}
table {{
  border-collapse: collapse;
  width: 100%;
}}
th, td {{
  border: 1px solid #ddd;
  padding: 8px;
  text-align: left;
}}
.notice {{
  border: 1px solid #aaa;
  padding: 12px;
  background: #f7f7f7;
}}
</style>
</head>
<body>
<h1>Day29 Blind Evidence Review</h1>
<div class="notice">
Review order is deterministically shuffled.
Only model-visible evidence and the approved
manual corpus may be used.
</div>
<p><a href="manual/index.html">Approved manual corpus</a></p>
<table>
<thead>
<tr>
<th>order</th>
<th>episode</th>
<th>selected frames</th>
</tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
</body>
</html>
""",
        encoding="utf-8",
        newline="\n",
    )


def _build_manual_pack(
    *,
    manual: dict[str, Any],
    output_root: Path,
) -> None:
    manual_root = (
        output_root
        / "manual"
    )

    manual_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    page_entries: list[dict[str, Any]] = []
    html_pages: list[str] = []

    for page_number, source in enumerate(
        manual["pages"],
        start=1,
    ):
        name = (
            f"page_{page_number:04d}.png"
        )

        destination = (
            manual_root
            / name
        )

        shutil.copy2(
            source,
            destination,
        )

        page_entries.append(
            {
                "page_number": page_number,
                "image": name,
            }
        )

        html_pages.append(
            f"""
<section>
<h2>Page {page_number}</h2>
<img src="{name}" style="max-width:100%;height:auto">
</section>
"""
        )

    manifest = {
        "schema_version": (
            "evidencemm_day29_approved_manual_pack_v1"
        ),
        "source_id": manual[
            "source_id"
        ],
        "source_sha256": manual[
            "sha256"
        ],
        "page_count": manual[
            "page_count"
        ],
        "pages": page_entries,
    }

    (
        manual_root
        / "manifest.json"
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

    (
        manual_root
        / "index.html"
    ).write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Approved Manual Corpus</title>
</head>
<body>
<p><a href="../index.html">review index</a></p>
<h1>Approved Manual Corpus</h1>
<p>source_id: {html.escape(manual['source_id'])}</p>
{''.join(html_pages)}
</body>
</html>
""",
        encoding="utf-8",
        newline="\n",
    )


def build_pack(
    *,
    project_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    preflight_result = preflight(
        project_root=project_root,
        config_path=config_path,
    )

    config = load_yaml(
        config_path
    )

    output_root = _resolve_project_path(
        project_root,
        config["output"]["root"],
    )

    if output_root.exists():
        raise RuntimeError(
            "refusing to overwrite existing "
            f"blind review pack: {output_root}"
        )

    population_path = _resolve_project_path(
        project_root,
        config[
            "inputs"
        ][
            "population_records"
        ],
    )

    source_manifest_path = _resolve_project_path(
        project_root,
        config[
            "inputs"
        ][
            "source_manifest"
        ],
    )

    records = load_jsonl(
        population_path
    )

    manifest = _load_source_manifest(
        source_manifest_path
    )

    root = _raw_root(
        project_root,
        config,
    )

    manual = _manual_info(
        project_root,
        config,
    )

    selection = _selection_config(
        config
    )

    frame_count = int(
        config[
            "expected"
        ][
            "frame_count_per_episode"
        ]
    )

    episode_ids = [
        row["episode_id"]
        for row in records
    ]

    ordered_ids = _ordered_episode_ids(
        episode_ids,
        config[
            "review_order"
        ][
            "seed"
        ],
    )

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    _build_manual_pack(
        manual=manual,
        output_root=output_root,
    )

    cases: list[dict[str, Any]] = []

    for position, episode_id in enumerate(
        ordered_ids,
        start=1,
    ):
        source = manifest[
            episode_id
        ]

        episode_dir = (
            root
            / source[
                "raw_episode_relpath"
            ]
        )

        samples_path = (
            episode_dir
            / "samples.csv"
        )

        samples = load_state_action_samples(
            samples_path,
            verify_tracking_error=True,
        )

        if len(samples) != frame_count:
            raise ValueError(
                f"{episode_id}: unexpected "
                "sample count"
            )

        front_transform = str(
            config[
                "camera_transforms"
            ][
                "front"
            ]
        )

        wrist_transform = str(
            config[
                "camera_transforms"
            ][
                "wrist"
            ]
        )

        front_refs = _frame_refs(
            "front",
            frame_count,
        )

        wrist_refs = _frame_refs(
            "wrist",
            frame_count,
        )

        state_scores = build_state_scores(
            samples
        )

        front_motion = visual_motion_scores(
            episode_dir=episode_dir,
            records_by_frame=front_refs,
            transform=front_transform,
            frame_count=frame_count,
            stride=selection.visual_stride,
            width=selection.visual_width,
            height=selection.visual_height,
        )

        wrist_motion = visual_motion_scores(
            episode_dir=episode_dir,
            records_by_frame=wrist_refs,
            transform=wrist_transform,
            frame_count=frame_count,
            stride=selection.visual_stride,
            width=selection.visual_width,
            height=selection.visual_height,
        )

        candidates = select_review_candidates(
            samples=samples,
            state_scores=state_scores,
            front_motion=front_motion,
            wrist_motion=wrist_motion,
            config=selection,
        )

        episode_output = (
            output_root
            / episode_id
        )

        thumbs = (
            episode_output
            / "thumbs"
        )

        episode_output.mkdir(
            parents=True,
            exist_ok=False,
        )

        full_rows = _full_state_rows(
            samples
        )

        if len(full_rows) != frame_count:
            raise ValueError(
                f"{episode_id}: unexpected "
                "full-state row count"
            )

        _ensure_camera_symlink(
            source=episode_dir / "front",
            destination=(
                episode_output
                / "front_frames"
            ),
        )

        _ensure_camera_symlink(
            source=episode_dir / "wrist",
            destination=(
                episode_output
                / "wrist_frames"
            ),
        )

        full_state_path = (
            episode_output
            / "full_state_action.json"
        )

        full_state_path.write_text(
            json.dumps(
                {
                    "schema_version": (
                        "evidencemm_day29_full_state_action_v1"
                    ),
                    "episode_id": episode_id,
                    "frame_count": frame_count,
                    "rows": full_rows,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

        _render_frame_explorer(
            episode_id=episode_id,
            rows=full_rows,
            front_transform=front_transform,
            wrist_transform=wrist_transform,
            output_path=(
                episode_output
                / "frame_explorer.html"
            ),
        )

        rows: list[dict[str, Any]] = []

        for candidate in candidates:
            frame_index = (
                candidate.frame_index
            )

            sample = samples[
                frame_index
            ]

            front_name = (
                f"f{frame_index:06d}_front.jpg"
            )

            wrist_name = (
                f"f{frame_index:06d}_wrist.jpg"
            )

            _make_thumbnail(
                episode_dir
                / "front"
                / f"{frame_index:06d}.jpg",
                thumbs / front_name,
                front_transform,
                selection.thumbnail_width,
            )

            _make_thumbnail(
                episode_dir
                / "wrist"
                / f"{frame_index:06d}.jpg",
                thumbs / wrist_name,
                wrist_transform,
                selection.thumbnail_width,
            )

            rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_sec": round(
                        sample.timestamp_sec,
                        6,
                    ),
                    "observation": _joint_dict(
                        sample.observation
                    ),
                    "action": _joint_dict(
                        sample.action
                    ),
                    "tracking_error": _joint_dict(
                        sample.tracking_error
                    ),
                    "front_thumb": (
                        f"thumbs/{front_name}"
                    ),
                    "wrist_thumb": (
                        f"thumbs/{wrist_name}"
                    ),
                    "front_source_frame": (
                        f"front/{frame_index:06d}.jpg"
                    ),
                    "wrist_source_frame": (
                        f"wrist/{frame_index:06d}.jpg"
                    ),
                }
            )

        selected_payload = {
            "schema_version": CASE_SCHEMA,
            "episode_id": episode_id,
            "selected_frame_count": len(
                rows
            ),
            "rows": rows,
        }

        selected_path = (
            episode_output
            / "selected_frames.json"
        )

        selected_path.write_text(
            json.dumps(
                selected_payload,
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

        review_context = {
            "schema_version": (
                "evidencemm_day29_blind_review_context_v1"
            ),
            "episode_id": episode_id,
            "frame_count": frame_count,
            "selected_frame_count": len(
                rows
            ),
            "full_frame_count": frame_count,
            "full_frame_explorer": (
                "frame_explorer.html"
            ),
            "full_state_action": (
                "full_state_action.json"
            ),
            "camera_streams": {
                "front": "front_frames",
                "wrist": "wrist_frames",
            },
            "allowed_evidence": [
                "front_images",
                "wrist_images",
                "observation",
                "action",
                "tracking_error",
                "approved_manual_corpus",
            ],
            "approved_manual_source_id": (
                manual["source_id"]
            ),
            "manual_pack": (
                "../manual/index.html"
            ),
        }

        (
            episode_output
            / "review_context.json"
        ).write_text(
            json.dumps(
                review_context,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

        _render_case_html(
            episode_id=episode_id,
            rows=rows,
            output_path=(
                episode_output
                / "review.html"
            ),
        )

        cases.append(
            {
                "review_position": position,
                "episode_id": episode_id,
                "selected_frame_count": len(
                    rows
                ),
                "selected_frames_sha256": (
                    sha256_file(
                        selected_path
                    )
                ),
                "full_frame_count": frame_count,
                "full_state_action_sha256": (
                    sha256_file(
                        full_state_path
                    )
                ),
            }
        )

    _render_index(
        cases=cases,
        output_path=(
            output_root
            / "index.html"
        ),
    )

    pack_manifest = {
        "schema_version": PACK_SCHEMA,
        "frozen_day29_population_commit": (
            config[
                "provenance"
            ][
                "frozen_day29_population_commit"
            ]
        ),
        "frozen_blank_records_sha256": (
            config[
                "provenance"
            ][
                "frozen_blank_records_sha256"
            ]
        ),
        "case_count": len(cases),
        "review_order_method": (
            config[
                "review_order"
            ][
                "method"
            ]
        ),
        "review_order_seed": (
            config[
                "review_order"
            ][
                "seed"
            ]
        ),
        "review_order_sha256": (
            _ordered_ids_sha256(
                ordered_ids
            )
        ),
        "approved_manual_source_id": (
            manual["source_id"]
        ),
        "approved_manual_sha256": (
            manual["sha256"]
        ),
        "cases": cases,
        "human_review_started": False,
        "ground_truth_frozen": False,
        "future_split_materialized": False,
    }

    (
        output_root
        / "manifest.json"
    ).write_text(
        json.dumps(
            pack_manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return {
        **preflight_result,
        "output_root": str(
            output_root
        ),
        "case_count": len(cases),
        "selected_frame_total": sum(
            case[
                "selected_frame_count"
            ]
            for case in cases
        ),
    }
