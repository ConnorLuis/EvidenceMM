\
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.root_cause_pilot import (
    CONFIG_SCHEMA,
    load_plan_csv,
    load_records_csv,
    validate_pilot_and_select,
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
    analysis = json.loads(
        _resolve(root, config["analysis"]["output_path"]).read_text(
            encoding="utf-8"
        )
    )

    frozen = validate_pilot_and_select(
        records=records,
        plan=plan,
        analysis=analysis,
        config=config,
    )
    output = _resolve(root, config["parameter_selection"]["freeze_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(frozen, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
