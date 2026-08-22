import json

import pytest
from PIL import Image

from evidencemm.review_pack import (
    CandidateAccumulator,
    LEGACY_REVIEW_TEMPLATE_SCHEMA,
    REVIEW_TEMPLATE_SCHEMA,
    SelectionConfig,
    _camera_transform,
    _review_template,
    _write_or_migrate_review_template,
    select_peaks_with_nms,
    uniform_frame_indices,
    visual_motion_scores,
)
from evidencemm.robot_failure_dataset import (
    AnomalyEvent,
    AnomalyReviewCase,
    ObservedFailureMode,
)


def review_case():
    return AnomalyReviewCase(
        review_id="r1",
        episode_id="ep1",
        task_success=True,
        original_failure_reason="第一次抓取掉落，放入后碰到方块",
        events=[
            AnomalyEvent(
                event_id="ep1_event_01",
                observed_failure_mode=ObservedFailureMode.GRASP_DROP,
            ),
            AnomalyEvent(
                event_id="ep1_event_02",
                observed_failure_mode=ObservedFailureMode.POST_PLACE_COLLISION,
            ),
        ],
        diagnostic_manifest_path="m.json",
        diagnostic_frames_path="f.jsonl",
    )


def test_uniform_frame_indices_include_episode_ends():
    assert uniform_frame_indices(10, 3) == [0, 4, 9]


def test_peak_selection_uses_temporal_nms_and_lower_frame_tie_break():
    selected = select_peaks_with_nms(
        [
            (10, 5.0),
            (11, 5.0),
            (30, 4.0),
            (50, 3.0),
        ],
        top_k=3,
        min_separation_frames=10,
    )
    assert selected == [
        (10, 5.0),
        (30, 4.0),
        (50, 3.0),
    ]


def test_candidate_accumulator_merges_reasons():
    item = CandidateAccumulator(
        frame_index=10,
        timestamp_sec=1.0,
    )
    item.add_reason(
        "state_action_change",
        priority=900.0,
        metrics={"x": 1.0},
    )
    item.add_reason(
        "front_visual_motion",
        priority=800.0,
        metrics={"y": 2.0},
    )
    assert item.reasons == {
        "state_action_change",
        "front_visual_motion",
    }
    assert item.best_priority == 900.0
    assert item.metrics == {
        "x": 1.0,
        "y": 2.0,
    }


def test_review_template_is_event_level_and_never_infers_cause():
    payload = _review_template(
        review_case()
    )
    assert payload["schema_version"] == REVIEW_TEMPLATE_SCHEMA
    assert len(payload["events"]) == 2
    assert payload["events"][0]["observed_failure_mode"] == "grasp_drop"
    assert payload["events"][1]["observed_failure_mode"] == "post_place_collision"
    assert all(
        event["causal_diagnosis"] is None
        for event in payload["events"]
    )
    assert all(
        event["failure_interval"] is None
        for event in payload["events"]
    )
    assert all(
        event["event_status"] == "draft"
        for event in payload["events"]
    )


def test_unedited_v1_template_is_migrated_to_event_v2(tmp_path):
    path = tmp_path / "review_template.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": LEGACY_REVIEW_TEMPLATE_SCHEMA,
                "review_id": "r1",
                "episode_id": "ep1",
                "task_success": True,
                "operation_anomaly": True,
                "original_failure_reason": "x",
                "observed_failure_modes": ["grasp_drop"],
                "failure_interval": None,
                "causal_diagnosis": None,
                "supporting_frames": [],
                "counterevidence_frames": [],
                "confidence": None,
                "reviewer": None,
                "review_status": "draft",
                "notes": None,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    action = _write_or_migrate_review_template(
        path,
        review_case(),
    )
    assert action == "migrated_v1_to_v2"
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == REVIEW_TEMPLATE_SCHEMA
    assert len(payload["events"]) == 2


def test_edited_v1_template_is_not_overwritten(tmp_path):
    path = tmp_path / "review_template.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": LEGACY_REVIEW_TEMPLATE_SCHEMA,
                "review_status": "draft",
                "failure_interval": {
                    "start_frame": 10,
                    "end_frame": 20,
                },
                "causal_diagnosis": None,
                "supporting_frames": [],
                "counterevidence_frames": [],
                "confidence": None,
                "reviewer": None,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="refusing to overwrite",
    ):
        _write_or_migrate_review_template(
            path,
            review_case(),
        )


def test_camera_transform_ccw90():
    image = Image.new(
        "RGB",
        (10, 20),
    )
    rotated = _camera_transform(
        image,
        "ccw90",
    )
    assert rotated.size == (
        20,
        10,
    )


def test_visual_motion_scores_detect_change(tmp_path):
    episode = tmp_path / "ep"
    front = episode / "front"
    front.mkdir(parents=True)

    for index, value in [
        (0, 0),
        (1, 0),
        (2, 255),
    ]:
        Image.new(
            "L",
            (20, 20),
            color=value,
        ).save(
            front / f"{index:06d}.jpg"
        )

    class Record:
        def __init__(self, frame_index):
            self.frame_index = frame_index
            self.image_relpath = (
                f"front/{frame_index:06d}.jpg"
            )

    records = {
        index: Record(index)
        for index in range(3)
    }

    scores = visual_motion_scores(
        episode_dir=episode,
        records_by_frame=records,
        transform="none",
        frame_count=3,
        stride=1,
        width=20,
        height=20,
    )

    assert scores[0][0] == 1
    assert scores[0][1] < 0.01
    assert scores[1][0] == 2
    assert scores[1][1] > 0.95


def test_selection_config_rejects_too_small_cap():
    config = SelectionConfig(
        uniform_count=8,
        max_selected_frames=7,
    )

    with pytest.raises(
        ValueError,
        match="max_selected_frames",
    ):
        config.validate()
