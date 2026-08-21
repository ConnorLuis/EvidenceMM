from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import yaml

from evidencemm.canonical_pipeline import (
    build_canonical_retrieval,
    validate_document_mode,
)
from evidencemm.failure_diagnosis import (
    diagnose_bundle,
    diagnose_generation,
    diagnose_pipeline,
    diagnose_retrieval_pages,
)
from evidencemm.grounded_generation import (
    generate_grounded,
    load_generator,
)
from evidencemm.grounding import (
    required_fact_coverage,
)
from evidencemm.retrieval_grounded_generation import (
    build_compact_citation_messages,
    count_visual_inputs,
    dynamic_robot_fact_groups,
    find_document_page_item,
    parse_compact_grounded_answer,
    required_citation_aliases,
    required_generation_refs,
    resolve_compact_grounded_answer,
    robot_items,
)
from evidencemm.unified_evidence import (
    UnifiedEvidenceKind,
    validate_unified_citation_policy,
)
from evidencemm.unified_grounding import (
    validate_required_citation_coverage,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-dir",
        required=True,
    )
    parser.add_argument(
        "--document-mode",
        choices=["bm25", "hybrid"],
        default=None,
    )
    args = parser.parse_args()

    day15_config = yaml.safe_load(
        (
            ROOT
            / "configs/day15_canonical_e2e.yaml"
        ).read_text(encoding="utf-8")
    )
    generation_config = yaml.safe_load(
        (
            ROOT
            / day15_config["generation_config"]
        ).read_text(encoding="utf-8")
    )

    document_mode = validate_document_mode(
        args.document_mode
        or day15_config[
            "default_document_mode"
        ]
    )
    query = str(
        generation_config["question"]
    ).strip()

    composition, retrieval_trace = (
        build_canonical_retrieval(
            project_root=ROOT,
            episode_dir=args.episode_dir,
            query=query,
            document_mode=document_mode,
            generation_config_path=(
                day15_config[
                    "generation_config"
                ]
            ),
            hybrid_config_path=(
                day15_config[
                    "hybrid_config"
                ]
            ),
            robot_config_path=(
                day15_config[
                    "robot_config"
                ]
            ),
        )
    )
    bundle = composition.bundle

    expected_document_page = int(
        generation_config[
            "required_document_page"
        ]
    )
    required_document_item = (
        find_document_page_item(
            bundle,
            page_number=(
                expected_document_page
            ),
        )
    )
    expected_document_page_retrieved = (
        required_document_item is not None
    )

    selected_document_pages = [
        candidate.item.payload.page_number
        for candidate
        in composition.selected_candidates
        if (
            candidate.item.kind
            == UnifiedEvidenceKind.DOCUMENT_PAGE
        )
    ]

    if not expected_document_page_retrieved:
        payload = {
            "mode": "day15_canonical_e2e",
            "document_mode": document_mode,
            "generation_called": False,
            "expected_document_page": (
                expected_document_page
            ),
            "selected_document_pages": (
                selected_document_pages
            ),
            "retrieval_trace": (
                retrieval_trace
            ),
            "failure_reason": (
                "required document page absent "
                "from selected canonical bundle"
            ),
        }
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    required_refs = required_generation_refs(
        bundle=bundle,
        document_page=(
            expected_document_page
        ),
    )
    all_required_fact_groups = [
        *generation_config[
            "required_fact_groups"
        ],
        *dynamic_robot_fact_groups(
            bundle
        ),
    ]

    messages, citation_alias_map = (
        build_compact_citation_messages(
            bundle=bundle,
            project_root=str(ROOT),
            episode_dir=str(
                Path(args.episode_dir)
            ),
        )
    )
    required_aliases = (
        required_citation_aliases(
            aliases=citation_alias_map,
            required_refs=required_refs,
        )
    )

    model, processor, model_load_sec = (
        load_generator(
            model_name=generation_config[
                "generator_model"
            ],
            min_visual_tokens_per_image=int(
                generation_config[
                    "generator_min_visual_tokens_per_image"
                ]
            ),
            max_visual_tokens_per_image=int(
                generation_config[
                    "generator_max_visual_tokens_per_image"
                ]
            ),
            attn_implementation=(
                generation_config[
                    "attn_implementation"
                ]
            ),
        )
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    raw_output, generation_sec = (
        generate_grounded(
            model=model,
            processor=processor,
            messages=messages,
            max_new_tokens=int(
                generation_config[
                    "max_new_tokens"
                ]
            ),
        )
    )

    peak_allocated_gpu_mb = (
        torch.cuda.max_memory_allocated()
        / 1024
        / 1024
    )

    compact_answer = (
        parse_compact_grounded_answer(
            raw_output
        )
    )
    answer = (
        resolve_compact_grounded_answer(
            compact_answer,
            citation_alias_map,
        )
    )

    citation_valid, citation_errors = (
        validate_unified_citation_policy(
            answer,
            bundle,
        )
    )
    (
        required_citation_valid,
        required_citation_errors,
    ) = validate_required_citation_coverage(
        answer,
        required_refs,
    )

    fact_coverage = required_fact_coverage(
        answer.answer,
        all_required_fact_groups,
    )
    fact_coverage_valid = (
        abs(fact_coverage - 1.0)
        < 1e-12
    )
    answerable_valid = (
        answer.abstain is False
    )

    retrieval_findings = (
        diagnose_retrieval_pages(
            ranked_pages=(
                selected_document_pages
            ),
            gold_pages=[
                expected_document_page
            ],
            top_k=len(
                selected_document_pages
            ),
        )
    )
    evidence_findings = diagnose_bundle(
        bundle=bundle,
        required_refs=required_refs,
        require_cross_domain=True,
    )
    generation_findings = (
        diagnose_generation(
            answer=answer,
            bundle=bundle,
            required_refs=required_refs,
            required_fact_groups=(
                all_required_fact_groups
            ),
            expected_answerable=True,
        )
    )
    diagnosis = diagnose_pipeline(
        retrieval_findings=(
            retrieval_findings
        ),
        evidence_findings=(
            evidence_findings
        ),
        generation_findings=(
            generation_findings
        ),
    )

    robots = robot_items(bundle)
    selected_robot_samples = [
        {
            "frame_index": (
                item.payload.frame_index
            ),
            "timestamp_sec": (
                item.payload.timestamp_sec
            ),
            "cameras": [
                camera.camera
                for camera
                in item.payload.cameras
            ],
        }
        for item in robots
    ]

    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

    after_unload_allocated_gpu_mb = (
        torch.cuda.memory_allocated()
        / 1024
        / 1024
    )

    payload = {
        "mode": "day15_canonical_e2e",
        "scope": (
            "canonical_document_retrieval_plus_"
            "robot_retrieval_to_grounded_generation"
        ),
        "document_mode": document_mode,
        "document_retriever": (
            retrieval_trace[
                "document_retriever_name"
            ]
        ),
        "robot_retriever": (
            retrieval_trace[
                "robot_retriever_name"
            ]
        ),
        "retrieval_called": True,
        "generation_called": True,
        "gold_read_by_retriever": False,
        "robot_event_gold_used": False,
        "cross_domain_score_fusion": False,
        "query_rewrite": False,
        "agent_used": False,
        "bundle_valid": True,
        "bundle_item_count": len(
            bundle.items
        ),
        "selected_document_pages": (
            selected_document_pages
        ),
        "selected_robot_samples": (
            selected_robot_samples
        ),
        "expected_document_page": (
            expected_document_page
        ),
        "expected_document_page_retrieved": (
            expected_document_page_retrieved
        ),
        "visual_inputs": (
            count_visual_inputs(
                bundle
            )
        ),
        "trace_available": (
            retrieval_trace[
                "document_ranking_trace"
            ]
            is not None
            if document_mode == "hybrid"
            else True
        ),
        "retrieval_trace": (
            retrieval_trace
        ),
        "allowed_citation_ids": list(
            citation_alias_map
        ),
        "required_citation_ids": (
            required_aliases
        ),
        "model_citation_ids": (
            compact_answer.citation_ids
        ),
        "structured_output_valid": True,
        "citation_resolution_valid": True,
        "answerable_valid": (
            answerable_valid
        ),
        "citation_policy_valid": (
            citation_valid
        ),
        "citation_policy_errors": (
            citation_errors
        ),
        "required_citation_coverage_valid": (
            required_citation_valid
        ),
        "required_citation_coverage_errors": (
            required_citation_errors
        ),
        "required_fact_coverage": (
            fact_coverage
        ),
        "required_fact_coverage_valid": (
            fact_coverage_valid
        ),
        "pipeline_diagnosis_healthy": (
            diagnosis.healthy
        ),
        "pipeline_diagnosis_codes": (
            diagnosis.codes
        ),
        "compact_answer": (
            compact_answer.model_dump(
                mode="json"
            )
        ),
        "answer": answer.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "performance": {
            "generator_model_load_sec": (
                model_load_sec
            ),
            "generation_sec": (
                generation_sec
            ),
            "peak_allocated_gpu_mb": (
                peak_allocated_gpu_mb
            ),
            "after_unload_allocated_gpu_mb": (
                after_unload_allocated_gpu_mb
            ),
        },
        "non_claims": [
            "single canonical end-to-end smoke case only",
            "ColQwen visual retrieval remains a validated non-canonical branch",
            "no benchmark-scale retrieval or generation claim",
            "no natural-language robot event retrieval claim",
            "no robot failure root-cause diagnosis",
            "no Agent or MCP",
        ],
    }

    report_file = (
        ROOT
        / day15_config[
            "report_file"
        ]
    )
    report_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_file.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"report={report_file}")

    if not (
        answerable_valid
        and citation_valid
        and required_citation_valid
        and fact_coverage_valid
        and diagnosis.healthy
        and expected_document_page_retrieved
        and len(bundle.items) == 5
        and count_visual_inputs(
            bundle
        ) == 7
    ):
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
