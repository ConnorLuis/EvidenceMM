\
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.root_cause_pilot import (
    CONFIG_SCHEMA,
    build_analysis,
    load_plan_csv,
    load_records_csv,
)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


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

    plan = load_plan_csv(_resolve(root, config["pilot_plan"]["path"]))
    records = load_records_csv(
        _resolve(root, config["pilot_plan"]["records_path"])
    )
    dataset_root = (
        args.dataset_root.resolve()
        if args.dataset_root is not None
        else Path(config["acquisition"]["wsl_dataset_root"]).resolve()
    )

    analysis = build_analysis(
        records=records,
        plan=plan,
        dataset_root=dataset_root,
        config=config,
    )

    output = _resolve(root, config["analysis"]["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "schema_version": analysis["schema_version"],
                "row_count": len(analysis["row_results"]),
                "dataset_root": analysis["dataset_root"],
                "gripper_shifts": {
                    row_id: row.get("gripper_shift_frames_vs_pair_clean")
                    for row_id, row in analysis["row_results"].items()
                    if row_id.endswith("_gripper")
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
