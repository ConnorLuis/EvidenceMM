from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from evidencemm.grounding import extract_json_object
from evidencemm.schemas import EvidenceRef
from evidencemm.unified_evidence import (
    DocumentPagePayload,
    RobotSamplePayload,
    UnifiedEvidenceBundle,
    UnifiedEvidenceItem,
    UnifiedEvidenceKind,
    UnifiedGroundedAnswer,
    evidence_ref_key,
)
from evidencemm.unified_grounding import (
    build_unified_messages,
)


class CompactGroundedAnswer(BaseModel):
    """Day 12 generation surface with compact citation identifiers."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    abstain: bool
    citation_ids: list[str] = Field(default_factory=list)


def find_document_page_item(
    bundle: UnifiedEvidenceBundle,
    *,
    page_number: int,
) -> UnifiedEvidenceItem | None:
    for item in bundle.items:
        if item.kind != UnifiedEvidenceKind.DOCUMENT_PAGE:
            continue
        payload = item.payload
        if not isinstance(payload, DocumentPagePayload):
            raise TypeError(
                "document evidence payload type mismatch"
            )
        if payload.page_number == page_number:
            return item
    return None


def robot_items(
    bundle: UnifiedEvidenceBundle,
) -> list[UnifiedEvidenceItem]:
    items = [
        item
        for item in bundle.items
        if item.kind == UnifiedEvidenceKind.ROBOT_SAMPLE
    ]
    for item in items:
        if not isinstance(item.payload, RobotSamplePayload):
            raise TypeError(
                "robot evidence payload type mismatch"
            )
    return items


def required_generation_refs(
    *,
    bundle: UnifiedEvidenceBundle,
    document_page: int,
) -> list[EvidenceRef]:
    document_item = find_document_page_item(
        bundle,
        page_number=document_page,
    )
    if document_item is None:
        raise ValueError(
            f"required document page {document_page} "
            "was not retrieved into the bundle"
        )

    robots = robot_items(bundle)
    if len(robots) != 2:
        raise ValueError(
            "Day 12 grounded smoke requires exactly "
            "two selected robot samples"
        )

    refs = [
        *document_item.refs,
        *[
            ref
            for item in robots
            for ref in item.refs
        ],
    ]

    if len(refs) != 5:
        raise ValueError(
            "Day 12 grounded smoke requires one PDF ref "
            "and four robot camera refs"
        )
    return refs


def dynamic_robot_fact_groups(
    bundle: UnifiedEvidenceBundle,
) -> list[list[str]]:
    groups: list[list[str]] = []
    robots = robot_items(bundle)

    if len(robots) != 2:
        raise ValueError(
            "Day 12 grounded smoke requires exactly "
            "two selected robot samples"
        )

    for item in robots:
        payload = item.payload
        assert isinstance(payload, RobotSamplePayload)

        frame_index = payload.frame_index
        timestamp = payload.timestamp_sec

        groups.append(
            [
                f"frame {frame_index}",
                f"frame_index {frame_index}",
                f"frame_index={frame_index}",
                f"frame_index 是 {frame_index}",
                f"frame_index是{frame_index}",
                f"frame_index 为 {frame_index}",
                f"frame_index为{frame_index}",
                f"帧{frame_index}",
                f"帧 {frame_index}",
            ]
        )
        groups.append(
            [
                f"{timestamp:.7f}",
                f"{timestamp:.4f}",
                f"{timestamp:.3f}",
            ]
        )

    return groups


def count_visual_inputs(
    bundle: UnifiedEvidenceBundle,
) -> int:
    total = 0
    for item in bundle.items:
        if item.kind == UnifiedEvidenceKind.DOCUMENT_PAGE:
            payload = item.payload
            if not isinstance(payload, DocumentPagePayload):
                raise TypeError(
                    "document evidence payload type mismatch"
                )
            if payload.page_image_path:
                total += 1
        elif item.kind == UnifiedEvidenceKind.ROBOT_SAMPLE:
            payload = item.payload
            if not isinstance(payload, RobotSamplePayload):
                raise TypeError(
                    "robot evidence payload type mismatch"
                )
            total += len(payload.cameras)
    return total


def build_citation_alias_map(
    bundle: UnifiedEvidenceBundle,
) -> dict[str, EvidenceRef]:
    aliases: dict[str, EvidenceRef] = {}

    for item in bundle.items:
        if item.kind == UnifiedEvidenceKind.DOCUMENT_PAGE:
            payload = item.payload
            if not isinstance(payload, DocumentPagePayload):
                raise TypeError(
                    "document evidence payload type mismatch"
                )
            if len(item.refs) != 1:
                raise ValueError(
                    "document item requires exactly one citation ref"
                )
            alias = f"DOC_P{payload.page_number}"
            if alias in aliases:
                raise ValueError(
                    f"duplicate citation alias: {alias}"
                )
            aliases[alias] = item.refs[0]

        elif item.kind == UnifiedEvidenceKind.ROBOT_SAMPLE:
            payload = item.payload
            if not isinstance(payload, RobotSamplePayload):
                raise TypeError(
                    "robot evidence payload type mismatch"
                )
            for ref in item.refs:
                if ref.camera is None:
                    raise ValueError(
                        "robot frame citation requires camera"
                    )
                alias = (
                    f"ROBOT_F{payload.frame_index}_"
                    f"{ref.camera.upper()}"
                )
                if alias in aliases:
                    raise ValueError(
                        f"duplicate citation alias: {alias}"
                    )
                aliases[alias] = ref

    return aliases


def citation_alias_rows(
    aliases: dict[str, EvidenceRef],
) -> list[dict]:
    return [
        {
            "citation_id": alias,
            "evidence_ref": ref.model_dump(
                mode="json",
                exclude_none=True,
            ),
        }
        for alias, ref in aliases.items()
    ]


def build_compact_citation_messages(
    *,
    bundle: UnifiedEvidenceBundle,
    project_root: str,
    episode_dir: str,
) -> tuple[list[dict], dict[str, EvidenceRef]]:
    """Reuse Day 11 evidence rendering but simplify model citation output.

    Day 11 evidence text and images remain unchanged. Only the output-facing
    citation representation is adapted from full EvidenceRef JSON objects to
    deterministic compact IDs, which are resolved back to EvidenceRef after
    generation.
    """

    messages = build_unified_messages(
        bundle=bundle,
        project_root=project_root,
        episode_dir=episode_dir,
    )
    aliases = build_citation_alias_map(bundle)

    user_content = messages[1]["content"]
    if not isinstance(user_content, list) or len(user_content) < 2:
        raise ValueError(
            "unexpected Day 11 unified message structure"
        )

    user_content[0] = {
        "type": "text",
        "text": (
            f"问题：{bundle.question}\n\n"
            "输出严格为一个 JSON 对象："
            '{"answer":"...","abstain":false,'
            '"citation_ids":["..."]}。'
            "不要输出 citations 字段。"
            "citation_ids 中只能复制下面 ALLOWED_CITATION_IDS "
            "里的 citation_id；不要复制或改写完整 EvidenceRef。"
            "正文只陈述证据直接支持的事实，不需要在正文中抄写 "
            "EvidenceRef。"
            "如果证据不足：abstain=true，citation_ids=[]。"
            "如果能够回答：abstain=false，并为实际使用的证据选择"
            "对应 citation_id。\n\n"
            "ALLOWED_CITATION_IDS:\n"
            + json.dumps(
                citation_alias_rows(aliases),
                ensure_ascii=False,
                indent=2,
            )
        ),
    }

    user_content[-1] = {
        "type": "text",
        "text": (
            "回答时只陈述上述证据直接给出的事实。"
            "本题不是故障诊断，不要判断机器人是否成功、失败，"
            "也不要推断文档内容与机器人行为之间的因果关系。"
            "citation_ids 只填写 ALLOWED_CITATION_IDS 中的短 ID；"
            "不要自行生成 source_id/page/frame/camera JSON。"
            "只输出 JSON。"
        ),
    }

    return messages, aliases


def parse_compact_grounded_answer(
    text: str,
) -> CompactGroundedAnswer:
    payload = json.loads(
        extract_json_object(text)
    )
    return CompactGroundedAnswer.model_validate(payload)


def resolve_compact_grounded_answer(
    compact: CompactGroundedAnswer,
    aliases: dict[str, EvidenceRef],
) -> UnifiedGroundedAnswer:
    if len(set(compact.citation_ids)) != len(
        compact.citation_ids
    ):
        raise ValueError(
            "duplicate compact citation ids"
        )

    if compact.abstain:
        if compact.citation_ids:
            raise ValueError(
                "abstaining compact answer must not cite evidence"
            )
        return UnifiedGroundedAnswer(
            answer=compact.answer,
            abstain=True,
            citations=[],
        )

    if not compact.citation_ids:
        raise ValueError(
            "non-abstaining compact answer requires citation ids"
        )

    unknown = [
        citation_id
        for citation_id in compact.citation_ids
        if citation_id not in aliases
    ]
    if unknown:
        raise ValueError(
            "unsupported compact citation ids: "
            + repr(unknown)
        )

    return UnifiedGroundedAnswer(
        answer=compact.answer,
        abstain=False,
        citations=[
            aliases[citation_id]
            for citation_id in compact.citation_ids
        ],
    )


def required_citation_aliases(
    *,
    aliases: dict[str, EvidenceRef],
    required_refs: list[EvidenceRef],
) -> list[str]:
    reverse = {
        evidence_ref_key(ref): alias
        for alias, ref in aliases.items()
    }

    missing = [
        evidence_ref_key(ref)
        for ref in required_refs
        if evidence_ref_key(ref) not in reverse
    ]
    if missing:
        raise ValueError(
            "required refs missing from citation alias map: "
            + repr(missing)
        )

    return [
        reverse[evidence_ref_key(ref)]
        for ref in required_refs
    ]
