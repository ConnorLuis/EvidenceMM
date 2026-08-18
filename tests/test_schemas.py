from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evidencemm.schemas import EvalCase, NormalizedBBox


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data" / "eval" / "day1_question_bank.jsonl"


def load_cases() -> list[EvalCase]:
    return [
        EvalCase.model_validate(json.loads(line))
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_day1_question_bank_has_20_unique_draft_cases():
    cases = load_cases()
    assert len(cases) == 20
    assert len({case.case_id for case in cases}) == 20
    assert all(case.annotation_status == "draft" for case in cases)


def test_bbox_requires_valid_normalized_order():
    box = NormalizedBBox(x1=0.1, y1=0.2, x2=0.8, y2=0.9)
    assert box.x2 == 0.8

    with pytest.raises(ValidationError):
        NormalizedBBox(x1=0.8, y1=0.2, x2=0.1, y2=0.9)
