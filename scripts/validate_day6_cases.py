from __future__ import annotations

import json
from pathlib import Path

import yaml

from evidencemm.grounding import load_day6_cases
from evidencemm.text_retrieval import load_corpus


ROOT = Path(__file__).resolve().parents[1]


def compact(text: str) -> str:
    return "".join(text.lower().split())


def main() -> int:
    config = yaml.safe_load(
        (
            ROOT
            / "configs/grounded_generation.yaml"
        ).read_text(encoding="utf-8")
    )

    cases = load_day6_cases(
        ROOT / config["eval_cases"]
    )
    corpus = load_corpus(
        ROOT / config["text_corpus"]
    )

    if len(cases) != 3:
        raise SystemExit(
            f"expected 3 Day 6 cases, got {len(cases)}"
        )

    answerable = [
        case
        for case in cases
        if case.expected_answerable
    ]
    abstention = [
        case
        for case in cases
        if not case.expected_answerable
    ]

    if len(answerable) != 2:
        raise SystemExit(
            "expected exactly 2 answerable cases"
        )
    if len(abstention) != 1:
        raise SystemExit(
            "expected exactly 1 abstention case"
        )

    corpus_text = compact(
        "\n".join(page.text for page in corpus)
    )

    absence_checks = []
    for case in abstention:
        matches = [
            term
            for term in case.absence_terms
            if compact(term) in corpus_text
        ]
        if matches:
            raise SystemExit(
                f"{case.case_id} absence-term collision: "
                f"{matches}"
            )
        absence_checks.append(
            {
                "case_id": case.case_id,
                "absence_terms": case.absence_terms,
                "matches": matches,
            }
        )

    payload = {
        "day6_cases": len(cases),
        "answerable_cases": len(answerable),
        "abstention_cases": len(abstention),
        "corpus_pages": len(corpus),
        "absence_checks": absence_checks,
    }

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
