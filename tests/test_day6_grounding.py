from __future__ import annotations

from evidencemm.grounding import (
    EvidencePage,
    GroundedAnswer,
    build_messages,
    parse_grounded_answer,
    required_fact_coverage,
    validate_citation_policy,
)


def evidence(page_number: int) -> EvidencePage:
    return EvidencePage(
        source_id="manual",
        page_number=page_number,
        image_path="/tmp/page.png",
        text="evidence text",
        retrieval_rank=1,
        rrf_score=0.03,
    )


def test_parse_grounded_answer_from_json_fence():
    raw = (
        "```json\n"
        "{\"answer\":\"6V 和 7.4V\","
        "\"abstain\":false,"
        "\"citations\":[{\"source_id\":\"manual\","
        "\"page_number\":3}]}\n"
        "```"
    )
    parsed = parse_grounded_answer(raw)

    assert parsed.abstain is False
    assert parsed.citations[0].page_number == 3


def test_answer_citation_must_be_supplied():
    answer = GroundedAnswer.model_validate(
        {
            "answer": "supported",
            "abstain": False,
            "citations": [
                {
                    "source_id": "manual",
                    "page_number": 9,
                }
            ],
        }
    )

    valid, errors = validate_citation_policy(
        answer,
        [evidence(3)],
    )

    assert not valid
    assert errors


def test_answerable_response_requires_citation():
    answer = GroundedAnswer.model_validate(
        {
            "answer": "supported",
            "abstain": False,
            "citations": [],
        }
    )

    valid, _ = validate_citation_policy(
        answer,
        [evidence(3)],
    )
    assert not valid


def test_abstention_requires_empty_citations():
    answer = GroundedAnswer.model_validate(
        {
            "answer": "证据不足",
            "abstain": True,
            "citations": [
                {
                    "source_id": "manual",
                    "page_number": 3,
                }
            ],
        }
    )

    valid, _ = validate_citation_policy(
        answer,
        [evidence(3)],
    )
    assert not valid


def test_required_fact_coverage_handles_spacing():
    coverage = required_fact_coverage(
        "典型工作电压为 6 V 和 7.4 V。",
        [
            ["6V", "6 V"],
            ["7.4V", "7.4 V"],
        ],
    )

    assert coverage == 1.0


def test_build_messages_contains_only_supplied_pages():
    messages = build_messages(
        question="question",
        evidence=[
            evidence(3),
            evidence(4),
        ],
    )

    payload = repr(messages)
    assert "page_number=3" in payload
    assert "page_number=4" in payload
    assert "page_number=5" not in payload
