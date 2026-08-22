import json

import pytest

from evidencemm.robot_benchmark_split import (
    ANOMALY_CATEGORY,
    CLEAN_CATEGORY,
    EligibleEpisode,
    GoldEventMetadata,
    SplitAssignment,
    assign_stratified_split,
    build_split_artifact,
    split_counts,
    split_rank_digest,
    validate_split_artifact_no_gold_boundaries,
)


def _episodes():
    return [
        EligibleEpisode(
            episode_id=f"clean_{index:02d}",
            audit_category=CLEAN_CATEGORY,
            task_success=True,
            operation_anomaly=False,
        )
        for index in range(6)
    ] + [
        EligibleEpisode(
            episode_id=f"anom_{index:02d}",
            audit_category=ANOMALY_CATEGORY,
            task_success=True,
            operation_anomaly=True,
        )
        for index in range(4)
    ]


def test_split_rank_digest_is_stable():
    first = split_rank_digest(
        seed="seed",
        audit_category=CLEAN_CATEGORY,
        episode_id="episode",
    )
    second = split_rank_digest(
        seed="seed",
        audit_category=CLEAN_CATEGORY,
        episode_id="episode",
    )
    assert first == second
    assert len(first) == 64


def test_assign_stratified_split_is_deterministic():
    kwargs = {
        "seed": "fixed",
        "held_out_counts": {
            CLEAN_CATEGORY: 2,
            ANOMALY_CATEGORY: 1,
        },
    }
    first = assign_stratified_split(
        _episodes(),
        **kwargs,
    )
    second = assign_stratified_split(
        list(reversed(_episodes())),
        **kwargs,
    )
    assert first == second


def test_assign_stratified_split_preserves_category_counts():
    assignments = assign_stratified_split(
        _episodes(),
        seed="fixed",
        held_out_counts={
            CLEAN_CATEGORY: 2,
            ANOMALY_CATEGORY: 1,
        },
    )
    assert split_counts(assignments) == {
        "eligible_episode_count": 10,
        "development_episode_count": 7,
        "held_out_episode_count": 3,
        "development_clean_count": 4,
        "development_anomaly_count": 3,
        "held_out_clean_count": 2,
        "held_out_anomaly_count": 1,
    }


def test_split_rejects_duplicate_episode_ids():
    episodes = _episodes()
    episodes.append(episodes[0])
    with pytest.raises(
        ValueError,
        match="duplicate episode_id",
    ):
        assign_stratified_split(
            episodes,
            seed="fixed",
            held_out_counts={
                CLEAN_CATEGORY: 2,
                ANOMALY_CATEGORY: 1,
            },
        )


def test_split_rejects_holding_out_entire_category():
    with pytest.raises(
        ValueError,
        match="held-out count must be smaller",
    ):
        assign_stratified_split(
            [
                EligibleEpisode(
                    episode_id="clean",
                    audit_category=CLEAN_CATEGORY,
                    task_success=True,
                    operation_anomaly=False,
                ),
                EligibleEpisode(
                    episode_id="anom",
                    audit_category=ANOMALY_CATEGORY,
                    task_success=True,
                    operation_anomaly=True,
                ),
            ],
            seed="fixed",
            held_out_counts={
                CLEAN_CATEGORY: 1,
                ANOMALY_CATEGORY: 0,
            },
        )


def test_build_artifact_contains_no_failure_boundaries():
    episodes = [
        EligibleEpisode(
            episode_id="clean",
            audit_category=CLEAN_CATEGORY,
            task_success=True,
            operation_anomaly=False,
        ),
        EligibleEpisode(
            episode_id="anom",
            audit_category=ANOMALY_CATEGORY,
            task_success=True,
            operation_anomaly=True,
        ),
    ]
    assignments = [
        SplitAssignment(
            episode_id="clean",
            audit_category=CLEAN_CATEGORY,
            split="development",
            rank_digest="a" * 64,
        ),
        SplitAssignment(
            episode_id="anom",
            audit_category=ANOMALY_CATEGORY,
            split="held_out",
            rank_digest="b" * 64,
        ),
    ]
    events = [
        GoldEventMetadata(
            event_id="anom_event_01",
            episode_id="anom",
            review_disposition="verified",
        )
    ]

    artifact = build_split_artifact(
        eligible_episodes=episodes,
        assignments=assignments,
        gold_events=events,
        source_audit_path="audit.jsonl",
        source_audit_sha256="1" * 64,
        human_gt_path="gt.jsonl",
        human_gt_sha256="2" * 64,
        seed="fixed",
        held_out_counts={
            CLEAN_CATEGORY: 0,
            ANOMALY_CATEGORY: 1,
        },
        frozen_after_day17_commit="3" * 40,
    )

    validate_split_artifact_no_gold_boundaries(
        artifact
    )
    def collect_keys(value):
        keys = set()
        if isinstance(value, dict):
            for key, child in value.items():
                keys.add(key)
                keys.update(collect_keys(child))
        elif isinstance(value, list):
            for child in value:
                keys.update(collect_keys(child))
        return keys

    keys = collect_keys(artifact)
    for forbidden in (
        "failure_interval",
        "start_frame",
        "end_frame",
        "start_sec",
        "end_sec",
        "supporting_robot_refs",
    ):
        assert forbidden not in keys


def test_build_artifact_rejects_gold_episode_outside_anomaly_universe():
    episodes = [
        EligibleEpisode(
            episode_id="clean",
            audit_category=CLEAN_CATEGORY,
            task_success=True,
            operation_anomaly=False,
        ),
        EligibleEpisode(
            episode_id="anom",
            audit_category=ANOMALY_CATEGORY,
            task_success=True,
            operation_anomaly=True,
        ),
    ]
    assignments = [
        SplitAssignment(
            episode_id="clean",
            audit_category=CLEAN_CATEGORY,
            split="development",
            rank_digest="a" * 64,
        ),
        SplitAssignment(
            episode_id="anom",
            audit_category=ANOMALY_CATEGORY,
            split="held_out",
            rank_digest="b" * 64,
        ),
    ]
    bad_events = [
        GoldEventMetadata(
            event_id="wrong_event",
            episode_id="clean",
            review_disposition="verified",
        )
    ]

    with pytest.raises(
        ValueError,
        match="human-GT episode universe",
    ):
        build_split_artifact(
            eligible_episodes=episodes,
            assignments=assignments,
            gold_events=bad_events,
            source_audit_path="audit.jsonl",
            source_audit_sha256="1" * 64,
            human_gt_path="gt.jsonl",
            human_gt_sha256="2" * 64,
            seed="fixed",
            held_out_counts={
                CLEAN_CATEGORY: 0,
                ANOMALY_CATEGORY: 1,
            },
            frozen_after_day17_commit="3" * 40,
        )
