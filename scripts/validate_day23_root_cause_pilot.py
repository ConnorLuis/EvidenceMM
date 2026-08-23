\
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from evidencemm.root_cause_pilot import (
    CONFIG_SCHEMA,
    FREEZE_SCHEMA,
    FREEZE_STATUS,
    build_analysis,
    canonical_json_bytes,
    load_plan_csv,
    load_records_csv,
    validate_pilot_and_select,
)


FROZEN_DAY22_FILES = (
    "data/protocol/day22_root_cause_benchmark_v2_protocol.json",
    "data/protocol/day22_root_cause_collection_plan.csv",
    "docs/day22_root_cause_benchmark_v2_protocol.md",
)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_day22_unchanged(root: Path, commit: str) -> None:
    for path in FROZEN_DAY22_FILES:
        result = subprocess.run(
            ["git", "diff", "--quiet", commit, "--", path],
            cwd=root,
        )
        if result.returncode != 0:
            raise ValueError(f"frozen Day22 file changed after protocol freeze: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/day23_root_cause_pilot.yaml"),
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    args = parser.parse_args()

    root = Path.cwd().resolve()
    config = yaml.safe_load(
        _resolve(root, args.config).read_text(encoding="utf-8")
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unexpected Day23 config schema")

    day22_commit = str(config["provenance"]["frozen_after_day22_commit"])
    _assert_day22_unchanged(root, day22_commit)

    day22_plan = _resolve(
        root, config["provenance"]["day22_collection_plan_path"]
    )
    if _sha256(day22_plan) != str(
        config["provenance"]["expected_day22_collection_plan_sha256"]
    ):
        raise ValueError("Day22 collection plan SHA256 drifted")

    plan = load_plan_csv(_resolve(root, config["pilot_plan"]["path"]))
    records = load_records_csv(
        _resolve(root, config["pilot_plan"]["records_path"])
    )
    dataset_root = (
        args.dataset_root.resolve()
        if args.dataset_root is not None
        else Path(config["acquisition"]["wsl_dataset_root"]).resolve()
    )

    rebuilt_analysis = build_analysis(
        records=records,
        plan=plan,
        dataset_root=dataset_root,
        config=config,
    )
    analysis_path = _resolve(root, config["analysis"]["output_path"])
    tracked_analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if canonical_json_bytes(rebuilt_analysis) != canonical_json_bytes(tracked_analysis):
        raise ValueError("Day23 analysis differs from deterministic rebuild")

    rebuilt_frozen = validate_pilot_and_select(
        records=records,
        plan=plan,
        analysis=rebuilt_analysis,
        config=config,
    )
    frozen_path = _resolve(
        root, config["parameter_selection"]["freeze_output"]
    )
    tracked_frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if tracked_frozen.get("schema_version") != FREEZE_SCHEMA:
        raise ValueError("unexpected Day23 frozen parameter schema")
    if tracked_frozen.get("freeze_status") != FREEZE_STATUS:
        raise ValueError("unexpected Day23 freeze status")
    if canonical_json_bytes(rebuilt_frozen) != canonical_json_bytes(tracked_frozen):
        raise ValueError("Day23 frozen parameters differ from deterministic rebuild")

    print(
        json.dumps(
            {
                "valid": True,
                "freeze_status": tracked_frozen["freeze_status"],
                "pilot_episode_count": len(records),
                "pilot_final_benchmark_eligible": False,
                "clean_control_success_count": tracked_frozen[
                    "clean_control_success_count"
                ],
                "eligible_failure_count_by_cause": tracked_frozen[
                    "eligible_failure_count_by_cause"
                ],
                "selected_parameters": tracked_frozen["selected_parameters"],
                "future_split_membership_materialized": False,
                "held_out_data_used": False,
                "day24_may_start": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
