from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from evidencemm.diagnostic_episode_binding import (
    bind_diagnostic_episode,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-dir",
        required=True,
    )
    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "day16_robot_failure_data.yaml"
        ),
    )
    args = parser.parse_args()

    config_path = Path(
        args.config
    )
    if not config_path.is_absolute():
        config_path = (
            ROOT / config_path
        )
    config = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    episode_dir = Path(
        args.episode_dir
    ).expanduser().resolve()
    episode_id = episode_dir.name

    binding = config[
        "diagnostic_binding"
    ]

    manifest_path = (
        ROOT
        / binding["manifest_root"]
        / f"{episode_id}.json"
    )
    frames_path = (
        ROOT
        / binding["processed_root"]
        / episode_id
        / "frames.jsonl"
    )

    result = bind_diagnostic_episode(
        episode_dir=episode_dir,
        manifest_path=manifest_path,
        frames_path=frames_path,
        timestamp_source=(
            binding[
                "timestamp_source"
            ]
        ),
    )

    print(
        f"episode_id={result.episode_id}"
    )
    print(
        f"frame_count={result.frame_count}"
    )
    print(
        f"episode_sha256={result.episode_sha256}"
    )
    print(
        "source_status="
        f"{result.source_status}"
    )
    print(
        "source_valid_for_training="
        f"{result.source_valid_for_training}"
    )
    print(
        "source_overall_pass="
        f"{result.source_overall_pass}"
    )
    print(
        "source_failed_checks="
        f"{list(result.source_failed_checks)}"
    )
    print(
        "diagnostic_binding=PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
