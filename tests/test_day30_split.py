from __future__ import annotations

import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/day30_split.py"

spec = importlib.util.spec_from_file_location("day30_split", MODULE_PATH)
assert spec is not None and spec.loader is not None
day30 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(day30)


def test_frozen_hash_ranking_matches_precommitted_order() -> None:
    groups = [f"rcv2_g{i:02d}" for i in range(1, 16)]
    ranked = day30.rank_groups(
        groups,
        "evidencemm-root-cause-v2-split-v3",
    )
    assert ranked == day30.EXPECTED_RANKED_GROUPS


def test_day22_future_split_contract_is_exact() -> None:
    protocol = json.loads(
        day30.DAY22_PROTOCOL_PATH.read_text(encoding="utf-8")
    )
    assert protocol["future_split"] == day30.EXPECTED_FUTURE_SPLIT


def test_frozen_ground_truth_population_is_15_groups_of_6() -> None:
    rows = day30.read_jsonl(day30.GT_RECORDS_PATH)
    assert len(rows) == 90
    assert len({row["episode_id"] for row in rows}) == 90

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["pair_group_id"]].append(row)

    assert len(groups) == 15
    assert {len(group_rows) for group_rows in groups.values()} == {6}


def test_deterministic_split_is_10_groups_60_vs_5_groups_30() -> None:
    protocol = json.loads(
        day30.DAY22_PROTOCOL_PATH.read_text(encoding="utf-8")
    )
    gt = day30.read_jsonl(day30.GT_RECORDS_PATH)
    pair_manifest, episode_rows, _ = day30.derive_artifacts(
        protocol["future_split"],
        gt,
    )

    assert pair_manifest["development_pair_groups"] == [
        "rcv2_g07",
        "rcv2_g11",
        "rcv2_g15",
        "rcv2_g05",
        "rcv2_g01",
        "rcv2_g10",
        "rcv2_g14",
        "rcv2_g13",
        "rcv2_g06",
        "rcv2_g04",
    ]
    assert pair_manifest["held_out_pair_groups"] == [
        "rcv2_g08",
        "rcv2_g09",
        "rcv2_g12",
        "rcv2_g02",
        "rcv2_g03",
    ]

    assert Counter(row["split"] for row in episode_rows) == {
        "development": 60,
        "held_out": 30,
    }


def test_no_pair_group_crosses_split_and_split_rows_leak_no_gt() -> None:
    protocol = json.loads(
        day30.DAY22_PROTOCOL_PATH.read_text(encoding="utf-8")
    )
    gt = day30.read_jsonl(day30.GT_RECORDS_PATH)
    pair_manifest, episode_rows, _ = day30.derive_artifacts(
        protocol["future_split"],
        gt,
    )

    split_by_group: dict[str, set[str]] = defaultdict(set)
    for row in episode_rows:
        split_by_group[row["pair_group_id"]].add(row["split"])

    assert all(len(values) == 1 for values in split_by_group.values())

    forbidden = {
        "physical_cause_gt",
        "diagnostic_decision_gt",
        "evidence_answerability_gt",
        "task_success",
        "intervention_verified",
        "review_notes",
    }
    assert all(not (forbidden & set(row)) for row in episode_rows)
    assert all(
        not (forbidden & set(row))
        for row in pair_manifest["ranked_pair_groups"]
    )
