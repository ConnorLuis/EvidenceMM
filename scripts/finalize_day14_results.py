from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/day14_failure_diagnosis.json"
DOC = ROOT / "docs/day14_failure_diagnosis.md"

START = "<!-- DAY14_OBSERVED_START -->"
END = "<!-- DAY14_OBSERVED_END -->"


def main() -> int:
    report = json.loads(
        REPORT.read_text(
            encoding="utf-8"
        )
    )
    text = DOC.read_text(
        encoding="utf-8"
    )

    lines = [
        "## Observed Day 14 result",
        "",
        "```text",
        (
            "scenario_count = "
            f'{report["scenario_count"]}'
        ),
        (
            "all_expected_diagnoses_match = "
            f'{report["all_expected_diagnoses_match"]}'
        ),
        (
            "reference_document_pages = "
            f'{report["real_reference_document_pages"]}'
        ),
        (
            "reference_robot_frames = "
            f'{report["real_reference_robot_frames"]}'
        ),
        "```",
        "",
        "Scenario results:",
        "",
        "```text",
    ]

    for row in report["scenarios"]:
        lines.append(
            f'{row["scenario_id"]}: '
            f'expected={row["expected_codes"]} '
            f'actual={row["actual_codes"]} '
            f'match={row["match"]}'
        )

    lines.extend(
        [
            "```",
            "",
            (
                "This is deterministic system-pipeline "
                "failure diagnosis over fault-injected variants "
                "of a real EvidenceMM evidence bundle."
            ),
            (
                "It is not robot-operation outcome or "
                "failed-grasp cause diagnosis."
            ),
        ]
    )
    body = "\n".join(lines)

    if START not in text or END not in text:
        raise ValueError(
            "Day 14 observed markers missing"
        )

    before, remainder = text.split(
        START,
        1,
    )
    _, after = remainder.split(
        END,
        1,
    )

    DOC.write_text(
        before
        + START
        + "\n"
        + body
        + "\n"
        + END
        + after,
        encoding="utf-8",
        newline="\n",
    )

    print(
        "Day 14 observed failure-diagnosis result "
        "written into docs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
