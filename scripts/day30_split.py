#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CONTRACT_PATH = ROOT / "data/protocol/day30_split_operational_contract.json"
DAY22_PROTOCOL_PATH = ROOT / "data/protocol/day22_root_cause_benchmark_v2_protocol.json"
GT_RECORDS_PATH = ROOT / "data/annotations/day29_ground_truth_records.jsonl"
GT_RECEIPT_PATH = ROOT / "data/protocol/day29_ground_truth_freeze_receipt.json"

PAIR_GROUP_SPLIT_PATH = ROOT / "data/splits/day30_pair_group_split.json"
EPISODE_SPLIT_PATH = ROOT / "data/splits/day30_episode_split.jsonl"
FREEZE_RECEIPT_PATH = ROOT / "data/protocol/day30_split_freeze_receipt.json"

DAY29_GT_FREEZE_COMMIT = "98dfa730ae87193b907a818d75e50020daf5e567"
DAY22_PROTOCOL_BLOB = "388ecb388375046b208e85eb9617e79961a5bf52"
DAY22_PROTOCOL_SHA256 = "a1adc5473e32f1e62523f230e4ecda2945ef805ad266663cd027927cef3c75be"
GT_RECORDS_SHA256 = "e03ec1ab443e4fb4dab606e16fbae8439411d7c3acbcf5f078ed5a0660d389bf"
GT_RECEIPT_SHA256 = "2c31413b8be2e3e755ca595b9ee410230df677da798ebb078f4b3537b5d0a680"

SPLIT_SCHEMA = "evidencemm_day30_pair_group_split_v1"
EPISODE_SPLIT_SCHEMA = "evidencemm_day30_episode_split_record_v1"
RECEIPT_SCHEMA = "evidencemm_day30_split_freeze_receipt_v1"

TOOLING_PATHS = (
    "scripts/day30_split.py",
    "data/protocol/day30_split_operational_contract.json",
    "docs/day30_split_operator_guide.md",
    "tests/test_day30_split.py",
)

EXPECTED_FUTURE_SPLIT = {
    "materialize_membership_on_day22": False,
    "materialize_target_day": 30,
    "unit": "pair_group_id",
    "seed": "evidencemm-root-cause-v2-split-v3",
    "ranking_rule": "sha256(seed|pair_group_id)_ascending",
    "development_pair_group_count": 10,
    "held_out_pair_group_count": 5,
    "expected_development_episode_count": 60,
    "expected_held_out_episode_count": 30,
    "pair_group_cross_split_allowed": False,
    "held_out_model_selection_allowed": False,
    "held_out_prompt_tuning_allowed": False,
    "held_out_retrieval_tuning_allowed": False,
}

EXPECTED_RANKED_GROUPS = [
    ("rcv2_g07", "02d7261b78b30ff384de33423002fe767e1b10abaaa7b78eba529c994b02020c"),
    ("rcv2_g11", "2884ac16a06fe4ef29e8743eceaaa0e0fd4c820e5e8cf18f5cad05d836ab2a55"),
    ("rcv2_g15", "31002567c45aa5f43f674bc32f815a9ec6e58e76484638ebe453f54ea2be9e12"),
    ("rcv2_g05", "33fcedf2369e9c35fdc06ab06704bc82c32ae3630c92fc31a7d6a28f8c04d5bd"),
    ("rcv2_g01", "486f844e1fc1e4486ee6228b72293a2c40078e73e6b818ee339855b98c00bc90"),
    ("rcv2_g10", "4d7b09675d6be144a2d2e41a5d57f0a6de340fb57be7644cfc98d227397142df"),
    ("rcv2_g14", "5032e9c452bb582571fe9fd02b5e179db19975527abfbe86109c24e7c87f551e"),
    ("rcv2_g13", "50d4059dc42c451f7875a09bce9162a118abc6c729d5c704dbd91d6ba38e9406"),
    ("rcv2_g06", "52c7de2d870f81cec826d5a1b673c6b9fa95fd1cbe758e07412ffb0ac32cc793"),
    ("rcv2_g04", "6de06c8a010ca604ee6812982d129593a31d2000db11f3dc96864f1e2b6ce305"),
    ("rcv2_g08", "72ff3a0251e1f3f3a8089b26676449b5fc2d9105a89370e4acd01390a4261c1e"),
    ("rcv2_g09", "75fd43af61986e03e4766c5ffd667b527e03913c9b2dbb85357053c8914eccc9"),
    ("rcv2_g12", "9ac3fc1d481de1b47e6e6d7619d4028897a0749f6f8f1c01f7a4f5344c38c015"),
    ("rcv2_g02", "b1187bb1f9cbf1f0784546a918e31dcdfa845bd72b29086968462b3149cde395"),
    ("rcv2_g03", "f33196350e1287b56beda3953cbc75f7c03eb9a86a09ff1d7191624ba8375ec3"),
]


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True
    ).strip()


def require_ancestor(commit: str) -> None:
    rc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode
    if rc != 0:
        raise RuntimeError(
            f"required frozen commit is not an ancestor: {commit}"
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def rank_hash(seed: str, pair_group_id: str) -> str:
    return hashlib.sha256(
        f"{seed}|{pair_group_id}".encode("utf-8")
    ).hexdigest()


def rank_groups(
    pair_group_ids: list[str] | set[str],
    seed: str,
) -> list[tuple[str, str]]:
    ranked = [
        (pair_group_id, rank_hash(seed, pair_group_id))
        for pair_group_id in pair_group_ids
    ]
    ranked.sort(key=lambda item: item[1])
    return ranked


def verify_tooling_committed() -> str:
    for rel in TOOLING_PATHS:
        try:
            git_output("rev-parse", f"HEAD:{rel}")
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Day30 tooling must be committed before materialization: {rel}"
            ) from exc

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *TOOLING_PATHS],
        cwd=ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(
            "Day30 tooling has uncommitted changes:\n" + dirty
        )
    return git_output("rev-parse", "HEAD")


def verify_contract() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    if contract.get("schema_version") != \
            "evidencemm_day30_groupwise_split_operational_contract_v1":
        raise RuntimeError("Day30 operational contract schema mismatch")

    if contract.get("day29_ground_truth_freeze_commit") != DAY29_GT_FREEZE_COMMIT:
        raise RuntimeError("Day30 contract Day29 GT commit mismatch")

    if contract.get("split_rule") != EXPECTED_FUTURE_SPLIT:
        raise RuntimeError("Day30 contract split rule mismatch")

    expected_ranked = [
        {
            "rank": idx,
            "pair_group_id": group_id,
            "ranking_sha256": digest,
            "split": "development" if idx <= 10 else "held_out",
        }
        for idx, (group_id, digest) in enumerate(EXPECTED_RANKED_GROUPS, 1)
    ]
    if contract.get("expected_ranked_pair_groups") != expected_ranked:
        raise RuntimeError("Day30 contract expected ranking mismatch")

    return contract


def verify_frozen_environment() -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    if git_output("branch", "--show-current") != "master":
        raise RuntimeError("Day30 split must run on master")

    require_ancestor(DAY29_GT_FREEZE_COMMIT)
    contract = verify_contract()

    actual_day22_blob = git_output(
        "rev-parse",
        "HEAD:data/protocol/day22_root_cause_benchmark_v2_protocol.json",
    )
    if actual_day22_blob != DAY22_PROTOCOL_BLOB:
        raise RuntimeError(
            "Day22 protocol Git blob changed after freeze"
        )

    source_hashes = {
        DAY22_PROTOCOL_PATH: DAY22_PROTOCOL_SHA256,
        GT_RECORDS_PATH: GT_RECORDS_SHA256,
        GT_RECEIPT_PATH: GT_RECEIPT_SHA256,
    }
    for path, expected in source_hashes.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen source SHA256 mismatch: {path}\n"
                f"expected={expected}\nactual={actual}"
            )

    protocol = read_json(DAY22_PROTOCOL_PATH)
    future_split = protocol.get("future_split")
    if future_split != EXPECTED_FUTURE_SPLIT:
        raise RuntimeError(
            f"Day22 future_split contract changed: {future_split!r}"
        )

    gt_receipt = read_json(GT_RECEIPT_PATH)
    receipt_checks = {
        "status":
            gt_receipt.get("status") == "ground_truth_frozen_day29_complete",
        "ground_truth_records_sha256":
            gt_receipt.get("ground_truth_records_sha256") == GT_RECORDS_SHA256,
        "case_count": gt_receipt.get("case_count") == 90,
        "pair_group_count": gt_receipt.get("pair_group_count") == 15,
        "human_review_completed":
            gt_receipt.get("human_review_completed") is True,
        "admin_reveal_started":
            gt_receipt.get("admin_reveal_started") is True,
        "ground_truth_frozen":
            gt_receipt.get("ground_truth_frozen") is True,
        "future_split_materialized":
            gt_receipt.get("future_split_materialized") is False,
        "day30_split_materialized":
            gt_receipt.get("day30_split_materialized") is False,
        "ground_truth_derivation_used_blind_cause_hypothesis":
            gt_receipt.get(
                "ground_truth_derivation_used_blind_cause_hypothesis"
            ) is False,
    }
    failed = [key for key, ok in receipt_checks.items() if not ok]
    if failed:
        raise RuntimeError(
            "Day29 GT freeze receipt failed checks: " + ", ".join(failed)
        )

    gt = read_jsonl(GT_RECORDS_PATH)
    if len(gt) != 90:
        raise RuntimeError(f"GT record count={len(gt)}, expected 90")
    if len({row["episode_id"] for row in gt}) != 90:
        raise RuntimeError("GT episode IDs must be unique")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gt:
        if row.get("technical_valid") is not True:
            raise RuntimeError(
                f"{row['episode_id']}: technical_valid must be true"
            )
        if row.get("experimental_valid") is not True:
            raise RuntimeError(
                f"{row['episode_id']}: experimental_valid must be true"
            )
        groups[row["pair_group_id"]].append(row)

    if len(groups) != 15:
        raise RuntimeError(f"GT pair-group count={len(groups)}")

    bad = {
        group_id: len(rows)
        for group_id, rows in groups.items()
        if len(rows) != 6
    }
    if bad:
        raise RuntimeError(f"pair-group episode counts invalid: {bad}")

    ranked = rank_groups(
        list(groups),
        future_split["seed"],
    )
    if ranked != EXPECTED_RANKED_GROUPS:
        raise RuntimeError(
            f"deterministic ranking mismatch: {ranked!r}"
        )

    return contract, future_split, gt


def derive_artifacts(
    future_split: dict[str, Any],
    gt: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
]:
    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gt:
        group_rows[row["pair_group_id"]].append(row)

    ranked = rank_groups(
        list(group_rows),
        future_split["seed"],
    )

    dev_n = future_split["development_pair_group_count"]

    ranked_entries: list[dict[str, Any]] = []
    group_to_assignment: dict[str, dict[str, Any]] = {}

    for rank, (pair_group_id, digest) in enumerate(ranked, 1):
        split = "development" if rank <= dev_n else "held_out"
        entry = {
            "rank": rank,
            "pair_group_id": pair_group_id,
            "ranking_sha256": digest,
            "split": split,
            "episode_count": len(group_rows[pair_group_id]),
        }
        ranked_entries.append(entry)
        group_to_assignment[pair_group_id] = entry

    development_groups = [
        item["pair_group_id"]
        for item in ranked_entries
        if item["split"] == "development"
    ]
    held_out_groups = [
        item["pair_group_id"]
        for item in ranked_entries
        if item["split"] == "held_out"
    ]

    pair_manifest = {
        "schema_version": SPLIT_SCHEMA,
        "status": "deterministic_groupwise_split_materialized",
        "source_ground_truth_records_sha256": GT_RECORDS_SHA256,
        "unit": future_split["unit"],
        "seed": future_split["seed"],
        "ranking_rule": future_split["ranking_rule"],
        "development_pair_group_count":
            future_split["development_pair_group_count"],
        "held_out_pair_group_count":
            future_split["held_out_pair_group_count"],
        "expected_development_episode_count":
            future_split["expected_development_episode_count"],
        "expected_held_out_episode_count":
            future_split["expected_held_out_episode_count"],
        "pair_group_cross_split_allowed": False,
        "ranked_pair_groups": ranked_entries,
        "development_pair_groups": development_groups,
        "held_out_pair_groups": held_out_groups,
        "ranking_used_ground_truth_labels": False,
        "ranking_inputs": ["seed", "pair_group_id"],
        "contains_ground_truth_labels": False,
    }

    episode_rows: list[dict[str, Any]] = []
    for row in gt:
        assignment = group_to_assignment[row["pair_group_id"]]
        episode_rows.append(
            {
                "schema_version": EPISODE_SPLIT_SCHEMA,
                "review_position": row["review_position"],
                "episode_id": row["episode_id"],
                "pair_group_id": row["pair_group_id"],
                "pair_group_rank": assignment["rank"],
                "pair_group_ranking_sha256":
                    assignment["ranking_sha256"],
                "split": assignment["split"],
            }
        )

    label_summary: dict[str, Any] = {}
    gt_by_episode = {
        row["episode_id"]: row
        for row in gt
    }
    for split in ("development", "held_out"):
        split_rows = [
            row for row in episode_rows
            if row["split"] == split
        ]
        gt_rows = [
            gt_by_episode[row["episode_id"]]
            for row in split_rows
        ]

        physical = Counter(
            row["physical_cause_gt"]
            for row in gt_rows
        )
        diagnostic = Counter(
            row["diagnostic_decision_gt"]
            for row in gt_rows
        )
        answerability = Counter(
            row["evidence_answerability_gt"]
            for row in gt_rows
        )
        task_success = Counter(
            row["task_success"]
            for row in gt_rows
        )

        label_summary[split] = {
            "episode_count": len(gt_rows),
            "pair_group_count": len({
                row["pair_group_id"]
                for row in gt_rows
            }),
            "physical_cause_gt_counts": dict(sorted(physical.items())),
            "diagnostic_decision_gt_counts": dict(sorted(diagnostic.items())),
            "evidence_answerability_gt_counts":
                dict(sorted(answerability.items())),
            "task_success_counts": {
                "false": task_success.get(False, 0),
                "true": task_success.get(True, 0),
            },
        }

    return pair_manifest, episode_rows, label_summary


def validate_artifacts(
    pair_manifest: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    future_split: dict[str, Any],
    gt: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []

    expected_pair, expected_episode, label_summary = derive_artifacts(
        future_split, gt
    )

    if pair_manifest != expected_pair:
        errors.append("pair-group split differs from deterministic derivation")

    if episode_rows != expected_episode:
        errors.append("episode split differs from deterministic derivation")

    if len(episode_rows) != 90:
        errors.append(f"episode split count={len(episode_rows)}")

    if len({row["episode_id"] for row in episode_rows}) != 90:
        errors.append("episode split episode IDs not unique")

    if [row["review_position"] for row in episode_rows] != list(range(1, 91)):
        errors.append("episode split review positions not exactly 1..90")

    dev_rows = [
        row for row in episode_rows
        if row["split"] == "development"
    ]
    held_rows = [
        row for row in episode_rows
        if row["split"] == "held_out"
    ]

    if len(dev_rows) != future_split["expected_development_episode_count"]:
        errors.append(f"development episode count={len(dev_rows)}")

    if len(held_rows) != future_split["expected_held_out_episode_count"]:
        errors.append(f"held-out episode count={len(held_rows)}")

    dev_groups = {row["pair_group_id"] for row in dev_rows}
    held_groups = {row["pair_group_id"] for row in held_rows}

    if len(dev_groups) != future_split["development_pair_group_count"]:
        errors.append(f"development group count={len(dev_groups)}")

    if len(held_groups) != future_split["held_out_pair_group_count"]:
        errors.append(f"held-out group count={len(held_groups)}")

    overlap = sorted(dev_groups & held_groups)
    if overlap:
        errors.append(f"pair groups cross split: {overlap}")

    all_groups = dev_groups | held_groups
    gt_groups = {row["pair_group_id"] for row in gt}
    if all_groups != gt_groups:
        errors.append("split pair-group population differs from GT")

    # Enforce that split artifacts themselves contain no GT label fields.
    forbidden = {
        "physical_cause_gt",
        "diagnostic_decision_gt",
        "evidence_answerability_gt",
        "task_success",
        "intervention_verified",
        "review_notes",
    }
    for row in episode_rows:
        leaked = forbidden & set(row)
        if leaked:
            errors.append(
                f"{row['episode_id']}: split row leaks GT fields {sorted(leaked)}"
            )

    for entry in pair_manifest.get("ranked_pair_groups", []):
        leaked = forbidden & set(entry)
        if leaked:
            errors.append(
                f"{entry.get('pair_group_id')}: group split leaks GT fields "
                f"{sorted(leaked)}"
            )

    return errors, label_summary


def load_materialized() -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
]:
    contract, future_split, gt = verify_frozen_environment()

    if not PAIR_GROUP_SPLIT_PATH.exists():
        raise RuntimeError(
            "pair-group split missing; run materialize"
        )
    if not EPISODE_SPLIT_PATH.exists():
        raise RuntimeError(
            "episode split missing; run materialize"
        )

    pair_manifest = read_json(PAIR_GROUP_SPLIT_PATH)
    episode_rows = read_jsonl(EPISODE_SPLIT_PATH)

    errors, label_summary = validate_artifacts(
        pair_manifest,
        episode_rows,
        future_split,
        gt,
    )

    print("===== DAY30 SPLIT VALIDATION =====")
    print("case_count =", len(episode_rows))
    print(
        "development_pair_groups =",
        pair_manifest.get("development_pair_groups"),
    )
    print(
        "held_out_pair_groups =",
        pair_manifest.get("held_out_pair_groups"),
    )
    print(
        "development_episode_count =",
        sum(row["split"] == "development" for row in episode_rows),
    )
    print(
        "held_out_episode_count =",
        sum(row["split"] == "held_out" for row in episode_rows),
    )
    print("label_distribution_audit =", label_summary)
    print("errors =", errors)

    if errors:
        raise SystemExit(1)

    print("DAY30 SPLIT VALIDATION: PASS")

    return (
        contract,
        future_split,
        gt,
        pair_manifest,
        episode_rows,
        label_summary,
    )


def cmd_preflight() -> None:
    verify_tooling_committed()
    contract, future_split, gt = verify_frozen_environment()

    outputs = [
        PAIR_GROUP_SPLIT_PATH,
        EPISODE_SPLIT_PATH,
        FREEZE_RECEIPT_PATH,
    ]
    existing = [
        str(path.relative_to(ROOT))
        for path in outputs
        if path.exists()
    ]
    if existing:
        raise RuntimeError(
            f"Day30 outputs already exist: {existing}"
        )

    pair_manifest, episode_rows, label_summary = derive_artifacts(
        future_split, gt
    )

    print("===== DAY30 SPLIT PREFLIGHT =====")
    print("branch =", git_output("branch", "--show-current"))
    print("head =", git_output("rev-parse", "HEAD"))
    print("day29_gt_freeze_commit =", DAY29_GT_FREEZE_COMMIT)
    print("gt_records_sha256 =", sha256_file(GT_RECORDS_PATH))
    print("case_count =", len(gt))
    print("pair_group_count =", len({
        row["pair_group_id"] for row in gt
    }))
    print(
        "development_pair_groups =",
        pair_manifest["development_pair_groups"],
    )
    print(
        "held_out_pair_groups =",
        pair_manifest["held_out_pair_groups"],
    )
    print(
        "development_episode_count =",
        sum(row["split"] == "development" for row in episode_rows),
    )
    print(
        "held_out_episode_count =",
        sum(row["split"] == "held_out" for row in episode_rows),
    )
    print("post_assignment_label_audit =", label_summary)
    print("ranking_used_ground_truth_labels = false")
    print("future_split_materialized = false")
    print("DAY30 SPLIT PREFLIGHT: PASS")


def cmd_materialize() -> None:
    verify_tooling_committed()
    _, future_split, gt = verify_frozen_environment()

    for path in (
        PAIR_GROUP_SPLIT_PATH,
        EPISODE_SPLIT_PATH,
        FREEZE_RECEIPT_PATH,
    ):
        if path.exists():
            raise RuntimeError(
                f"refusing overwrite: {path.relative_to(ROOT)}"
            )

    pair_manifest, episode_rows, label_summary = derive_artifacts(
        future_split, gt
    )

    errors, _ = validate_artifacts(
        pair_manifest,
        episode_rows,
        future_split,
        gt,
    )
    if errors:
        raise RuntimeError(
            "candidate split invalid: " + repr(errors)
        )

    write_json(PAIR_GROUP_SPLIT_PATH, pair_manifest)
    write_jsonl(EPISODE_SPLIT_PATH, episode_rows)

    if read_json(PAIR_GROUP_SPLIT_PATH) != pair_manifest:
        raise RuntimeError("pair-group split changed after reload")
    if read_jsonl(EPISODE_SPLIT_PATH) != episode_rows:
        raise RuntimeError("episode split changed after reload")

    print(
        "pair_group_split_sha256 =",
        sha256_file(PAIR_GROUP_SPLIT_PATH),
    )
    print(
        "episode_split_sha256 =",
        sha256_file(EPISODE_SPLIT_PATH),
    )
    print(
        "development_pair_groups =",
        pair_manifest["development_pair_groups"],
    )
    print(
        "held_out_pair_groups =",
        pair_manifest["held_out_pair_groups"],
    )
    print("post_assignment_label_audit =", label_summary)
    print("ranking_used_ground_truth_labels = false")
    print("DAY30 SPLIT MATERIALIZATION: PASS")


def cmd_validate() -> None:
    load_materialized()


def cmd_freeze() -> None:
    tooling_commit = verify_tooling_committed()
    (
        contract,
        future_split,
        gt,
        pair_manifest,
        episode_rows,
        label_summary,
    ) = load_materialized()

    if FREEZE_RECEIPT_PATH.exists():
        raise RuntimeError(
            "Day30 split freeze receipt already exists"
        )

    tooling_git_blobs = {
        rel: git_output("rev-parse", f"{tooling_commit}:{rel}")
        for rel in TOOLING_PATHS
    }

    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "groupwise_split_frozen_day30_complete",
        "tooling_commit": tooling_commit,
        "day29_ground_truth_freeze_commit": DAY29_GT_FREEZE_COMMIT,
        "day22_protocol_sha256": DAY22_PROTOCOL_SHA256,
        "ground_truth_records_sha256": GT_RECORDS_SHA256,
        "ground_truth_freeze_receipt_sha256": GT_RECEIPT_SHA256,
        "operational_contract_sha256": sha256_file(CONTRACT_PATH),
        "tooling_git_blobs": tooling_git_blobs,
        "pair_group_split_path":
            "data/splits/day30_pair_group_split.json",
        "pair_group_split_sha256":
            sha256_file(PAIR_GROUP_SPLIT_PATH),
        "episode_split_path":
            "data/splits/day30_episode_split.jsonl",
        "episode_split_sha256":
            sha256_file(EPISODE_SPLIT_PATH),
        "unit": future_split["unit"],
        "seed": future_split["seed"],
        "ranking_rule": future_split["ranking_rule"],
        "development_pair_groups":
            pair_manifest["development_pair_groups"],
        "held_out_pair_groups":
            pair_manifest["held_out_pair_groups"],
        "development_pair_group_count":
            future_split["development_pair_group_count"],
        "held_out_pair_group_count":
            future_split["held_out_pair_group_count"],
        "development_episode_count":
            sum(row["split"] == "development" for row in episode_rows),
        "held_out_episode_count":
            sum(row["split"] == "held_out" for row in episode_rows),
        "pair_group_cross_split_count": 0,
        "post_assignment_label_distribution_audit": label_summary,
        "ranking_used_ground_truth_labels": False,
        "split_artifacts_contain_ground_truth_labels": False,
        "future_split_materialized": True,
        "held_out_model_selection_allowed": False,
        "held_out_prompt_tuning_allowed": False,
        "held_out_retrieval_tuning_allowed": False,
        "held_out_calibration_allowed": False,
        "held_out_final_evaluation_target_day": 33,
        "held_out_final_evaluation_started": False,
        "held_out_final_evaluation_count_consumed": 0,
        "model_training_started_on_day30": False,
        "calibration_started_on_day30": False,
    }

    write_json(FREEZE_RECEIPT_PATH, receipt)

    print(
        "pair_group_split_sha256 =",
        receipt["pair_group_split_sha256"],
    )
    print(
        "episode_split_sha256 =",
        receipt["episode_split_sha256"],
    )
    print(
        "freeze_receipt_sha256 =",
        sha256_file(FREEZE_RECEIPT_PATH),
    )
    print("future_split_materialized = true")
    print("held_out_final_evaluation_started = false")
    print("DAY30 SPLIT FREEZE RECEIPT: PASS")


def cmd_audit() -> None:
    (
        contract,
        future_split,
        gt,
        pair_manifest,
        episode_rows,
        label_summary,
    ) = load_materialized()

    if not FREEZE_RECEIPT_PATH.exists():
        raise RuntimeError(
            "Day30 split freeze receipt missing"
        )

    receipt = read_json(FREEZE_RECEIPT_PATH)
    errors: list[str] = []

    expected_scalars = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "groupwise_split_frozen_day30_complete",
        "day29_ground_truth_freeze_commit": DAY29_GT_FREEZE_COMMIT,
        "day22_protocol_sha256": DAY22_PROTOCOL_SHA256,
        "ground_truth_records_sha256": GT_RECORDS_SHA256,
        "ground_truth_freeze_receipt_sha256": GT_RECEIPT_SHA256,
        "operational_contract_sha256": sha256_file(CONTRACT_PATH),
        "pair_group_split_sha256": sha256_file(PAIR_GROUP_SPLIT_PATH),
        "episode_split_sha256": sha256_file(EPISODE_SPLIT_PATH),
        "unit": future_split["unit"],
        "seed": future_split["seed"],
        "ranking_rule": future_split["ranking_rule"],
        "development_pair_groups":
            pair_manifest["development_pair_groups"],
        "held_out_pair_groups":
            pair_manifest["held_out_pair_groups"],
        "development_pair_group_count": 10,
        "held_out_pair_group_count": 5,
        "development_episode_count": 60,
        "held_out_episode_count": 30,
        "pair_group_cross_split_count": 0,
        "post_assignment_label_distribution_audit": label_summary,
        "ranking_used_ground_truth_labels": False,
        "split_artifacts_contain_ground_truth_labels": False,
        "future_split_materialized": True,
        "held_out_model_selection_allowed": False,
        "held_out_prompt_tuning_allowed": False,
        "held_out_retrieval_tuning_allowed": False,
        "held_out_calibration_allowed": False,
        "held_out_final_evaluation_target_day": 33,
        "held_out_final_evaluation_started": False,
        "held_out_final_evaluation_count_consumed": 0,
        "model_training_started_on_day30": False,
        "calibration_started_on_day30": False,
    }

    for key, expected in expected_scalars.items():
        actual = receipt.get(key)
        if actual != expected:
            errors.append(
                f"receipt {key} mismatch: "
                f"expected={expected!r}, actual={actual!r}"
            )

    tooling_commit = receipt.get("tooling_commit")
    if not isinstance(tooling_commit, str) or not tooling_commit:
        errors.append("tooling_commit missing")
    else:
        try:
            require_ancestor(tooling_commit)
        except RuntimeError as exc:
            errors.append(str(exc))

        expected_blobs = receipt.get("tooling_git_blobs")
        if not isinstance(expected_blobs, dict):
            errors.append("tooling_git_blobs missing")
        else:
            for rel in TOOLING_PATHS:
                try:
                    frozen_blob = git_output(
                        "rev-parse", f"{tooling_commit}:{rel}"
                    )
                    current_blob = git_output(
                        "rev-parse", f"HEAD:{rel}"
                    )
                except subprocess.CalledProcessError:
                    errors.append(
                        f"tooling path missing from Git: {rel}"
                    )
                    continue
                if expected_blobs.get(rel) != frozen_blob:
                    errors.append(
                        f"receipt tooling blob mismatch: {rel}"
                    )
                if current_blob != frozen_blob:
                    errors.append(
                        f"tooling file changed after freeze: {rel}"
                    )

    # Recompute from frozen sources. This is the core deterministic audit.
    expected_pair, expected_episode, expected_summary = derive_artifacts(
        future_split, gt
    )
    if pair_manifest != expected_pair:
        errors.append(
            "pair-group split differs from fresh deterministic recomputation"
        )
    if episode_rows != expected_episode:
        errors.append(
            "episode split differs from fresh deterministic recomputation"
        )
    if label_summary != expected_summary:
        errors.append("label summary differs from fresh recomputation")

    print("===== DAY30 SPLIT FREEZE AUDIT =====")
    print("development_pair_groups =", pair_manifest["development_pair_groups"])
    print("held_out_pair_groups =", pair_manifest["held_out_pair_groups"])
    print("development_episode_count = 60")
    print("held_out_episode_count = 30")
    print("pair_group_cross_split_count = 0")
    print("ranking_used_ground_truth_labels = False")
    print("post_assignment_label_distribution_audit =", label_summary)
    print(
        "pair_group_split_sha256 =",
        sha256_file(PAIR_GROUP_SPLIT_PATH),
    )
    print(
        "episode_split_sha256 =",
        sha256_file(EPISODE_SPLIT_PATH),
    )
    print(
        "freeze_receipt_sha256 =",
        sha256_file(FREEZE_RECEIPT_PATH),
    )
    print("errors =", errors)

    if errors:
        raise SystemExit(1)

    print("DAY30 SPLIT AUDIT: PASS")
    print("DAY30: CLOSED / FROZEN")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("materialize")
    sub.add_parser("validate")
    sub.add_parser("freeze")
    sub.add_parser("audit")
    args = parser.parse_args()

    if args.command == "preflight":
        cmd_preflight()
    elif args.command == "materialize":
        cmd_materialize()
    elif args.command == "validate":
        cmd_validate()
    elif args.command == "freeze":
        cmd_freeze()
    elif args.command == "audit":
        cmd_audit()


if __name__ == "__main__":
    main()
