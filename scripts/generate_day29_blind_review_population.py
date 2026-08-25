#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

FROZEN_DAY29_PROTOCOL_COMMIT = (
    "c230657d27c0837991a88b2df87f692734217e0c"
)

RECORD_PATHS = [
    REPO_ROOT / "data/protocol/day24_target_collection_records.csv",
    REPO_ROOT / "data/protocol/day25_gripper_collection_records.csv",
    REPO_ROOT / "data/protocol/day26_trajectory_collection_records.csv",
    REPO_ROOT / "data/protocol/day27_insufficient_evidence_collection_records.csv",
]

DAY28_MANIFEST = (
    REPO_ROOT / "data/protocol/day28_registered_source_manifest.csv"
)

OUTPUT_RECORDS = (
    REPO_ROOT / "data/annotations/day29_blind_review_records.jsonl"
)

OUTPUT_POPULATION = (
    REPO_ROOT / "data/protocol/day29_blind_review_population.json"
)


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() == "true"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_canonical_ids() -> list[str]:
    all_rows: list[dict[str, str]] = []

    for path in RECORD_PATHS:
        with path.open(newline="", encoding="utf-8-sig") as f:
            all_rows.extend(csv.DictReader(f))

    episode_ids = [r["episode_id"] for r in all_rows]

    if len(episode_ids) != 92:
        raise RuntimeError(
            f"expected 92 registered collection records, got {len(episode_ids)}"
        )

    if len(set(episode_ids)) != 92:
        raise RuntimeError("registered collection episode IDs are not unique")

    canonical = sorted(
        r["episode_id"]
        for r in all_rows
        if truthy(r.get("selected_canonical"))
    )

    if len(canonical) != 90:
        raise RuntimeError(
            f"expected 90 canonical episodes, got {len(canonical)}"
        )

    if len(set(canonical)) != 90:
        raise RuntimeError("canonical episode IDs are not unique")

    return canonical


def load_registered_manifest_ids() -> set[str]:
    with DAY28_MANIFEST.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        rows = list(csv.DictReader(f))

    ids = {r["episode_id"] for r in rows}

    if len(rows) != 92:
        raise RuntimeError(
            f"expected 92 Day28 manifest rows, got {len(rows)}"
        )

    if len(ids) != 92:
        raise RuntimeError(
            "Day28 registered source manifest episode IDs are not unique"
        )

    return ids


def blank_record(episode_id: str) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "observed_symptom": None,
        "failure_interval": None,
        "supporting_robot_refs": [],
        "counterevidence_robot_refs": [],
        "supporting_manual_refs": [],
        "counterevidence_manual_refs": [],
        "evidence_answerability_gt": None,
        "explicit_uncertainty_reason": None,
        "blind_confidence": None,
        "blind_review_notes": None,
    }


def main() -> None:
    if OUTPUT_RECORDS.exists():
        raise RuntimeError(
            f"refusing to overwrite existing blind review records: "
            f"{OUTPUT_RECORDS.relative_to(REPO_ROOT)}"
        )

    if OUTPUT_POPULATION.exists():
        raise RuntimeError(
            f"refusing to overwrite existing population manifest: "
            f"{OUTPUT_POPULATION.relative_to(REPO_ROOT)}"
        )

    canonical = load_canonical_ids()
    registered = load_registered_manifest_ids()

    canonical_set = set(canonical)

    missing = canonical_set - registered
    noncanonical = registered - canonical_set

    if missing:
        raise RuntimeError(
            f"canonical episodes missing from Day28 manifest: {len(missing)}"
        )

    if len(noncanonical) != 2:
        raise RuntimeError(
            f"expected 2 registered noncanonical episodes, got {len(noncanonical)}"
        )

    records = [blank_record(episode_id) for episode_id in canonical]

    records_text = "".join(
        json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for record in records
    )

    ordered_id_payload = (
        "\n".join(canonical) + "\n"
    ).encode("utf-8")

    records_bytes = records_text.encode("utf-8")

    population = {
        "schema_version": (
            "evidencemm_day29_blind_review_population_v1"
        ),
        "frozen_day29_protocol_commit": (
            FROZEN_DAY29_PROTOCOL_COMMIT
        ),
        "source_day28_manifest": (
            "data/protocol/day28_registered_source_manifest.csv"
        ),
        "source_day28_manifest_sha256": (
            sha256_file(DAY28_MANIFEST)
        ),
        "registered_episode_count": 92,
        "canonical_episode_count": 90,
        "registered_noncanonical_excluded_count": 2,
        "ordered_episode_ids_sha256": (
            sha256_bytes(ordered_id_payload)
        ),
        "records_path": (
            "data/annotations/day29_blind_review_records.jsonl"
        ),
        "records_sha256": sha256_bytes(records_bytes),
        "records_are_blank": True,
        "human_review_started": False,
        "ground_truth_frozen": False,
        "future_split_materialized": False,
    }

    OUTPUT_RECORDS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_POPULATION.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_RECORDS.write_text(
        records_text,
        encoding="utf-8",
    )

    OUTPUT_POPULATION.write_text(
        json.dumps(
            population,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("===== DAY29 BLIND REVIEW POPULATION =====")
    print("registered =", len(registered))
    print("canonical =", len(canonical))
    print("registered_noncanonical_excluded =", len(noncanonical))
    print(
        "ordered_episode_ids_sha256 =",
        population["ordered_episode_ids_sha256"],
    )
    print(
        "records_sha256 =",
        population["records_sha256"],
    )
    print("records_are_blank = True")
    print("human_review_started = False")
    print("ground_truth_frozen = False")
    print("future_split_materialized = False")
    print("VALIDATION: PASS")


if __name__ == "__main__":
    main()
