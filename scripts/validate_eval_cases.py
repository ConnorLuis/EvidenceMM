from __future__ import annotations

import json
from pathlib import Path

from evidencemm.schemas import EvalCase


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data" / "eval" / "day1_question_bank.jsonl"


def main() -> int:
    cases: list[EvalCase] = []
    for line_no, raw in enumerate(CASES_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            cases.append(EvalCase.model_validate(json.loads(raw)))
        except Exception as exc:
            raise SystemExit(f"invalid case at line {line_no}: {exc}") from exc

    ids = [case.case_id for case in cases]
    if len(cases) != 20:
        raise SystemExit(f"expected 20 cases, got {len(cases)}")
    if len(ids) != len(set(ids)):
        raise SystemExit("case_id values must be unique")

    verified = sum(case.annotation_status == "verified" for case in cases)
    print(f"valid_cases={len(cases)}")
    print(f"verified_cases={verified}")
    print(f"draft_cases={len(cases) - verified}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
