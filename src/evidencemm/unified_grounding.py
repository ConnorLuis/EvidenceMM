from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evidencemm.grounding import extract_json_object
from evidencemm.schemas import EvidenceRef
from evidencemm.unified_evidence import (
    DocumentPagePayload,
    RobotSamplePayload,
    UnifiedEvidenceBundle,
    UnifiedEvidenceKind,
    UnifiedGroundedAnswer,
    evidence_ref_key,
)


def citation_json(ref: EvidenceRef) -> dict[str, Any]:
    return ref.model_dump(
        mode="json",
        exclude_none=True,
    )


def allowed_citations_json(
    bundle: UnifiedEvidenceBundle,
) -> list[dict[str, Any]]:
    return [
        citation_json(ref)
        for item in bundle.items
        for ref in item.refs
    ]


def parse_unified_grounded_answer(
    text: str,
) -> UnifiedGroundedAnswer:
    payload = json.loads(
        extract_json_object(text)
    )
    return UnifiedGroundedAnswer.model_validate(payload)


def validate_required_citation_coverage(
    answer: UnifiedGroundedAnswer,
    required_refs: list[EvidenceRef],
) -> tuple[bool, list[str]]:
    required = {
        evidence_ref_key(ref)
        for ref in required_refs
    }
    cited = {
        evidence_ref_key(ref)
        for ref in answer.citations
    }

    missing = required - cited
    if not missing:
        return True, []

    return (
        False,
        [
            "missing_required_citations="
            + repr(sorted(missing))
        ],
    )


def build_unified_messages(
    *,
    bundle: UnifiedEvidenceBundle,
    project_root: str,
    episode_dir: str,
) -> list[dict[str, Any]]:
    system_text = (
        "你是 EvidenceMM 的跨域证据约束回答器。"
        "只能使用用户消息中明确提供的文档证据、机器人证据和元数据。"
        "禁止使用外部知识、常识补全、故障推断或猜测。"
        "如果证据不足以回答，必须拒答。"
        "最终只能输出一个 JSON 对象，不要 Markdown。"
    )

    allowed = allowed_citations_json(bundle)

    schema_text = (
        '输出严格为：'
        '{"answer":"...","abstain":false,"citations":[...]}。'
        "citations 中的每一项必须逐字段复制自下面的 "
        "ALLOWED_CITATIONS，不得创造新的 source_id、page_number、"
        "frame_index、camera 或 timestamp。"
        "如果证据不足：abstain=true，citations=[]。"
        "如果能够回答：abstain=false，并引用实际支持各项事实的证据。"
    )

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"问题：{bundle.question}\n\n"
                f"{schema_text}\n\n"
                "ALLOWED_CITATIONS:\n"
                + json.dumps(
                    allowed,
                    ensure_ascii=False,
                    indent=2,
                )
            ),
        }
    ]

    for item in bundle.items:
        if item.kind == UnifiedEvidenceKind.DOCUMENT_PAGE:
            payload = item.payload
            if not isinstance(payload, DocumentPagePayload):
                raise TypeError(
                    "document item payload type mismatch"
                )

            content.append(
                {
                    "type": "text",
                    "text": (
                        f"[DOCUMENT EVIDENCE] evidence_id={item.evidence_id}\n"
                        f"source_id={item.provenance.source_id}\n"
                        f"page_number={payload.page_number}\n"
                        f"text_sha256={payload.text_sha256}\n"
                        f"extracted_text:\n{payload.text_excerpt}"
                    ),
                }
            )

            if payload.page_image_path:
                image_path = (
                    Path(project_root)
                    / payload.page_image_path
                )
                content.append(
                    {
                        "type": "image",
                        "image": str(image_path),
                    }
                )

        elif item.kind == UnifiedEvidenceKind.ROBOT_SAMPLE:
            payload = item.payload
            if not isinstance(payload, RobotSamplePayload):
                raise TypeError(
                    "robot item payload type mismatch"
                )

            snapshot = payload.state_action
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"[ROBOT EVIDENCE] evidence_id={item.evidence_id}\n"
                        f"episode_id={payload.episode_id}\n"
                        f"frame_index={payload.frame_index}\n"
                        f"timestamp_sec={payload.timestamp_sec:.7f}\n"
                        "cameras=front,wrist\n"
                        "observation_6d="
                        + json.dumps(
                            snapshot.observation.model_dump(
                                mode="json"
                            ),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\naction_6d="
                        + json.dumps(
                            snapshot.action.model_dump(
                                mode="json"
                            ),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\ntracking_error_6d="
                        + json.dumps(
                            snapshot.tracking_error.model_dump(
                                mode="json"
                            ),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    ),
                }
            )

            for camera in payload.cameras:
                image_path = (
                    Path(episode_dir)
                    / camera.image_relpath
                )
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"[ROBOT CAMERA] camera={camera.camera} "
                            f"frame_index={camera.frame_index} "
                            f"timestamp_sec={camera.timestamp_sec:.7f}"
                        ),
                    }
                )
                content.append(
                    {
                        "type": "image",
                        "image": str(image_path),
                    }
                )

    content.append(
        {
            "type": "text",
            "text": (
                "回答时只陈述上述证据直接给出的事实。"
                "本题不是故障诊断，不要判断机器人是否成功、失败，"
                "也不要推断文档内容与机器人行为之间的因果关系。"
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
