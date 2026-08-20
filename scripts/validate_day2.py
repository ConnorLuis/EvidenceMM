from __future__ import annotations

import json
from pathlib import Path

from evidencemm.schemas import EvalCase, SourceManifest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "manifests" / "sources"
CANDIDATES = ROOT / "data" / "eval" / "day2_candidates.jsonl"
VERIFIED = ROOT / "data" / "eval" / "day2_verified_cases.jsonl"


def load_jsonl(path: Path) -> list[EvalCase]:
    if not path.exists():
        return []
    return [
        EvalCase.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    manifests: list[SourceManifest] = []
    if SOURCE_DIR.exists():
        for path in sorted(SOURCE_DIR.glob("*.json")):
            manifests.append(
                SourceManifest.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            )

    candidates = load_jsonl(CANDIDATES)
    verified = load_jsonl(VERIFIED)

    if len(candidates) != 4:
        raise SystemExit(
            f"expected 4 Day2 candidates, got {len(candidates)}"
        )

    if len(verified) != 4:
        raise SystemExit(
            f"expected 4 Day2 verified cases, got {len(verified)}"
        )

    if not all(
        case.annotation_status == "draft"
        for case in candidates
    ):
        raise SystemExit("Day2 candidates must remain draft")

    if not all(
        case.annotation_status == "verified"
        for case in verified
    ):
        raise SystemExit(
            "Day2 verified file contains non-verified case"
        )

    manifest_ids = {item.source_id for item in manifests}

    unresolved_inputs = sorted({
        source_id
        for case in verified
        for source_id in case.input_ids
        if source_id not in manifest_ids
    })

    unresolved_evidence = sorted({
        evidence.source_id
        for case in verified
        for evidence in case.expected_evidence
        if evidence.source_id not in manifest_ids
    })

    if unresolved_inputs:
        raise SystemExit(
            "verified cases reference unbound input sources: "
            + ", ".join(unresolved_inputs)
        )

    if unresolved_evidence:
        raise SystemExit(
            "verified evidence references unbound sources: "
            + ", ".join(unresolved_evidence)
        )

    print(f"bound_sources={len(manifests)}")
    print(f"day2_candidates={len(candidates)}")
    print(f"day2_verified={len(verified)}")
    print(
        "unresolved_verified_input_sources="
        f"{len(unresolved_inputs)}"
    )
    print(
        "unresolved_verified_evidence_sources="
        f"{len(unresolved_evidence)}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
