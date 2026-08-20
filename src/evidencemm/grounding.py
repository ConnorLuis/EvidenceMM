from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CitationRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    abstain: bool
    citations: list[CitationRef]


@dataclass(frozen=True)
class Day6Case:
    case_id: str
    question: str
    expected_answerable: bool
    source_id: str
    gold_pages: list[int]
    required_fact_groups: list[list[str]]
    absence_terms: list[str]
    tags: list[str]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Day6Case":
        return cls(**value)


@dataclass(frozen=True)
class EvidencePage:
    source_id: str
    page_number: int
    image_path: str
    text: str
    retrieval_rank: int
    rrf_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_day6_cases(path) -> list[Day6Case]:
    return [
        Day6Case.from_dict(json.loads(line))
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def extract_json_object(text: str) -> str:
    stripped = text.strip()

    if stripped.startswith("```"):
        stripped = re.sub(
            r"^```(?:json)?\s*",
            "",
            stripped,
            count=1,
            flags=re.IGNORECASE,
        )
        stripped = re.sub(
            r"\s*```$",
            "",
            stripped,
            count=1,
        )

    start = stripped.find("{")
    end = stripped.rfind("}")

    if start < 0 or end < start:
        raise ValueError(
            "model output does not contain a JSON object"
        )

    return stripped[start : end + 1]


def parse_grounded_answer(text: str) -> GroundedAnswer:
    payload = json.loads(
        extract_json_object(text)
    )
    return GroundedAnswer.model_validate(payload)


def citation_keys(
    citations: list[CitationRef],
) -> set[tuple[str, int]]:
    return {
        (
            citation.source_id,
            citation.page_number,
        )
        for citation in citations
    }


def validate_citation_policy(
    answer: GroundedAnswer,
    evidence: list[EvidencePage],
) -> tuple[bool, list[str]]:
    errors: list[str] = []

    allowed = {
        (
            page.source_id,
            page.page_number,
        )
        for page in evidence
    }
    cited = citation_keys(answer.citations)

    unsupported = cited - allowed
    if unsupported:
        errors.append(
            "citation_outside_supplied_evidence="
            + repr(sorted(unsupported))
        )

    if answer.abstain:
        if answer.citations:
            errors.append(
                "abstention_must_not_emit_citations"
            )
    elif not answer.citations:
        errors.append(
            "answerable_response_requires_citation"
        )

    return not errors, errors


def compact_text(text: str) -> str:
    normalized = unicodedata.normalize(
        "NFKC",
        text,
    ).lower()
    return re.sub(r"\s+", "", normalized)


def fact_group_covered(
    answer: str,
    aliases: list[str],
) -> bool:
    compact_answer = compact_text(answer)
    return any(
        compact_text(alias) in compact_answer
        for alias in aliases
    )


def required_fact_coverage(
    answer: str,
    groups: list[list[str]],
) -> float:
    if not groups:
        return 1.0

    matched = sum(
        fact_group_covered(
            answer,
            aliases,
        )
        for aliases in groups
    )
    return matched / len(groups)


def build_messages(
    *,
    question: str,
    evidence: list[EvidencePage],
):
    system_text = (
        "你是 EvidenceMM 的证据约束回答器。"
        "只能使用用户消息中给出的检索证据。"
        "禁止使用常识补全、外部知识或猜测。"
        "如果证据不能直接支持答案，必须拒答。"
        "最终只能输出一个 JSON 对象，不要 Markdown。"
    )

    schema_text = (
        '输出格式严格为：'
        '{"answer":"...","abstain":false,'
        '"citations":[{"source_id":"...","page_number":1}]}。'
        "如果证据不足，设置 abstain=true，"
        'answer 简洁说明“提供的证据不足以回答该问题”，'
        "citations 必须为空数组。"
        "如果能够回答，abstain=false，且 citations 至少包含"
        "一个实际支持答案的已提供页面。"
    )

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"问题：{question}\n\n"
                f"{schema_text}\n\n"
                "下面是唯一允许使用的证据："
            ),
        }
    ]

    for index, page in enumerate(
        evidence,
        start=1,
    ):
        content.append(
            {
                "type": "text",
                "text": (
                    f"[EVIDENCE {index}] "
                    f"source_id={page.source_id} "
                    f"page_number={page.page_number} "
                    f"retrieval_rank={page.retrieval_rank}"
                ),
            }
        )
        content.append(
            {
                "type": "image",
                "image": page.image_path,
            }
        )
        content.append(
            {
                "type": "text",
                "text": (
                    f"[EVIDENCE {index} EXTRACTED TEXT]\n"
                    f"{page.text}"
                ),
            }
        )

    content.append(
        {
            "type": "text",
            "text": (
                "现在只根据以上证据回答。"
                "不要引用未提供的页面。"
                "只输出 JSON。"
            ),
        }
    )

    return [
        {
            "role": "system",
            "content": system_text,
        },
        {
            "role": "user",
            "content": content,
        },
    ]
