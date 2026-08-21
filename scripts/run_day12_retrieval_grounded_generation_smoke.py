from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import yaml

from evidencemm.document_candidate_retrieval import (
    DocumentBM25CandidateRetriever,
)
from evidencemm.grounded_generation import (
    generate_grounded,
    load_generator,
)
from evidencemm.grounding import required_fact_coverage
from evidencemm.retrieval import validate_retrieved_bundle
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
from evidencemm.retrieval_ranking import (
    DAY12_BASELINE_BUDGET,
    compose_fixed_quota,
)
from evidencemm.robot_candidate_retrieval import (
    RobotSignalCandidateRetriever,
)
from evidencemm.unified_evidence import (
    UnifiedEvidenceKind,
    validate_cross_domain_bundle,
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
    args = parser.parse_args()

    config = yaml.safe_load(
        (
            ROOT
            / "configs/day12_retrieval_grounded_generation.yaml"
        ).read_text(encoding="utf-8")
    )
    robot_config = yaml.safe_load(
        (
            ROOT
            / "configs/robot_sequence_evidence.yaml"
        ).read_text(encoding="utf-8")
    )

    query = str(config["question"]).strip()
    candidate_pool_k = int(
        config["candidate_pool_k"]
    )
    if candidate_pool_k != 5:
        raise ValueError(
            "Day 12 candidate_pool_k must remain 5"
        )

    document_retriever = DocumentBM25CandidateRetriever(
        project_root=ROOT,
        source_manifest_path=config["document_manifest"],
        visual_manifest_path=config["document_visual_manifest"],
    )

    episode_id = str(config["episode_id"])
    episode_manifest_path = (
        ROOT
        / robot_config["manifest_root"]
        / f"{episode_id}.json"
    )
    frame_records_path = (
        ROOT
        / robot_config["processed_root"]
        / episode_id
        / "frames.jsonl"
    )

    robot_retriever = RobotSignalCandidateRetriever(
        project_root=ROOT,
        episode_manifest_path=episode_manifest_path,
        episode_dir=args.episode_dir,
        frame_records_path=frame_records_path,
    )

    document_candidates = document_retriever.search(
        query,
        top_k=candidate_pool_k,
    )
    robot_candidates = robot_retriever.search(
        query,
        top_k=candidate_pool_k,
    )

    composition = compose_fixed_quota(
        query=query,
        document_candidates=document_candidates,
        robot_candidates=robot_candidates,
        budget=DAY12_BASELINE_BUDGET,
        bundle_id=(
            f"day12_retrieval_grounded_{episode_id}"
        ),
    )
    bundle = composition.bundle

    validate_retrieved_bundle(
        query=query,
        top_k=DAY12_BASELINE_BUDGET.total_top_k,
        bundle=bundle,
    )

    cross_valid, cross_errors = (
        validate_cross_domain_bundle(bundle)
    )
    if not cross_valid:
        raise ValueError(
            "invalid retrieved cross-domain bundle: "
            + repr(cross_errors)
        )

    expected_document_page = int(
        config["required_document_page"]
    )
    required_document_item = find_document_page_item(
        bundle,
        page_number=expected_document_page,
    )
    expected_document_page_retrieved = (
        required_document_item is not None
    )

    if not expected_document_page_retrieved:
        payload = {
            "mode": (
                "day12_retrieval_grounded_generation_smoke"
            ),
            "scope": (
                "real_retrieval_to_grounded_generation_"
                "no_failure_diagnosis"
            ),
            "retrieval_called": True,
            "generation_called": False,
            "gold_read_by_retriever": False,
            "expected_document_page": (
                expected_document_page
            ),
            "expected_document_page_retrieved": False,
            "selected_document_pages": [
                candidate.item.payload.page_number
                for candidate
                in composition.selected_candidates
                if (
                    candidate.item.kind
                    == UnifiedEvidenceKind.DOCUMENT_PAGE
                )
            ],
            "failure_reason": (
                "required document evidence not present "
                "in selected retrieval bundle"
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
        document_page=expected_document_page,
    )

    dynamic_groups = dynamic_robot_fact_groups(
        bundle
    )
    all_required_fact_groups = [
        *config["required_fact_groups"],
        *dynamic_groups,
    ]

    episode_dir = Path(args.episode_dir)

    messages, citation_alias_map = (
        build_compact_citation_messages(
            bundle=bundle,
            project_root=str(ROOT),
            episode_dir=str(episode_dir),
        )
    )
    required_aliases = required_citation_aliases(
        aliases=citation_alias_map,
        required_refs=required_refs,
    )

    model, processor, model_load_sec = load_generator(
        model_name=config["generator_model"],
        min_visual_tokens_per_image=int(
            config[
                "generator_min_visual_tokens_per_image"
            ]
        ),
        max_visual_tokens_per_image=int(
            config[
                "generator_max_visual_tokens_per_image"
            ]
        ),
        attn_implementation=config[
            "attn_implementation"
        ],
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    raw_output, generation_sec = generate_grounded(
        model=model,
        processor=processor,
        messages=messages,
        max_new_tokens=int(config["max_new_tokens"]),
    )

    peak_allocated_gpu_mb = (
        torch.cuda.max_memory_allocated()
        / 1024
        / 1024
    )

    compact_answer = parse_compact_grounded_answer(
        raw_output
    )
    answer = resolve_compact_grounded_answer(
        compact_answer,
        citation_alias_map,
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
        abs(fact_coverage - 1.0) < 1e-12
    )
    answerable_valid = answer.abstain is False

    robots = robot_items(bundle)

    selected_document_pages = [
        candidate.item.payload.page_number
        for candidate in composition.selected_candidates
        if (
            candidate.item.kind
            == UnifiedEvidenceKind.DOCUMENT_PAGE
        )
    ]
    selected_robot_samples = [
        {
            "frame_index": item.payload.frame_index,
            "timestamp_sec": item.payload.timestamp_sec,
            "cameras": [
                camera.camera
                for camera in item.payload.cameras
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
        "mode": (
            "day12_retrieval_grounded_generation_smoke"
        ),
        "scope": (
            "real_retrieval_fixed_quota_to_"
            "grounded_generation_no_failure_diagnosis"
        ),
        "query": query,
        "model_name": config["generator_model"],
        "retrieval_called": True,
        "generation_called": True,
        "gold_read_by_retriever": False,
        "robot_event_gold_used": False,
        "failure_diagnosis_attempted": False,
        "agent_used": False,
        "cross_domain_score_fusion": False,
        "query_rewrite": False,
        "compact_citation_adapter_used": True,
        "candidate_pool_k_per_domain": (
            candidate_pool_k
        ),
        "budget": {
            "total_top_k": (
                DAY12_BASELINE_BUDGET.total_top_k
            ),
            "document_quota": (
                DAY12_BASELINE_BUDGET.document_quota
            ),
            "robot_quota": (
                DAY12_BASELINE_BUDGET.robot_quota
            ),
        },
        "bundle_schema_version": (
            bundle.schema_version
        ),
        "bundle_valid": True,
        "bundle_item_count": len(bundle.items),
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
        "visual_inputs": count_visual_inputs(
            bundle
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
        "answerable_valid": answerable_valid,
        "citation_policy_valid": citation_valid,
        "citation_policy_errors": citation_errors,
        "required_citation_coverage_valid": (
            required_citation_valid
        ),
        "required_citation_coverage_errors": (
            required_citation_errors
        ),
        "required_fact_group_count": len(
            all_required_fact_groups
        ),
        "required_fact_coverage": (
            fact_coverage
        ),
        "required_fact_coverage_valid": (
            fact_coverage_valid
        ),
        "compact_answer": compact_answer.model_dump(
            mode="json"
        ),
        "answer": answer.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "raw_model_output": raw_output,
        "performance": {
            "model_load_sec": model_load_sec,
            "generation_sec": generation_sec,
            "peak_allocated_gpu_mb": (
                peak_allocated_gpu_mb
            ),
            "after_unload_allocated_gpu_mb": (
                after_unload_allocated_gpu_mb
            ),
        },
        "non_claims": [
            "single end-to-end smoke case only",
            "no benchmark-scale generation claim",
            "no natural-language robot event retrieval claim",
            "no temporal diversity optimization",
            "no failure diagnosis",
            "no causal relation between manual and robot evidence",
            "no Agent or MCP",
        ],
    }

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )

    if not (
        answerable_valid
        and citation_valid
        and required_citation_valid
        and fact_coverage_valid
        and expected_document_page_retrieved
        and len(bundle.items) == 5
        and count_visual_inputs(bundle) == 7
    ):
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
