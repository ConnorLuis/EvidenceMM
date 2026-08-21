from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import yaml

from evidencemm.data_binding import sha256_file
from evidencemm.grounded_generation import (
    generate_grounded,
    load_generator,
)
from evidencemm.grounding import required_fact_coverage
from evidencemm.pdf_corpus import (
    load_source_manifest,
    standardize_pdf_pages,
)
from evidencemm.schemas import EvidenceRef, SourceType
from evidencemm.state_action_selection import (
    load_state_action_samples,
    validate_source_semantics,
)
from evidencemm.temporal_evidence import (
    EpisodeManifest,
    load_frame_records,
)
from evidencemm.unified_evidence import (
    DocumentPagePayload,
    EvidenceProvenance,
    RobotCameraAsset,
    RobotSamplePayload,
    RobotStateActionSnapshot,
    UnifiedEvidenceBundle,
    UnifiedEvidenceItem,
    UnifiedEvidenceKind,
    validate_cross_domain_bundle,
    validate_unified_citation_policy,
)
from evidencemm.unified_grounding import (
    build_unified_messages,
    parse_unified_grounded_answer,
    validate_required_citation_coverage,
)
from evidencemm.visual_corpus import load_visual_manifest


ROOT = Path(__file__).resolve().parents[1]


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-id",
        required=True,
    )
    parser.add_argument(
        "--episode-dir",
        required=True,
    )
    args = parser.parse_args()

    config = yaml.safe_load(
        (
            ROOT
            / "configs/unified_grounded_generation.yaml"
        ).read_text(encoding="utf-8")
    )
    robot_config = yaml.safe_load(
        (
            ROOT
            / "configs/robot_sequence_evidence.yaml"
        ).read_text(encoding="utf-8")
    )

    if args.episode_id != config["episode_id"]:
        raise ValueError(
            "episode-id differs from frozen Day 11 Gate B fixture"
        )

    page_number = int(config["page_number"])
    frame_index = int(config["frame_index"])

    pdf_manifest_path = ROOT / config["pdf_manifest"]
    pdf_manifest = load_source_manifest(
        pdf_manifest_path
    )
    if pdf_manifest.source_type != SourceType.PDF:
        raise ValueError("fixture manifest must be PDF")

    pages = standardize_pdf_pages(
        pdf_manifest,
        project_root=ROOT,
    )
    page = {
        item.page_number: item
        for item in pages
    }[page_number]

    visual_pages = load_visual_manifest(
        ROOT / config["pdf_visual_manifest"]
    )
    visual_page = {
        item.page_number: item
        for item in visual_pages
    }[page_number]
    if visual_page.source_id != pdf_manifest.source_id:
        raise ValueError(
            "visual page source_id differs from PDF manifest"
        )

    page_image_path = ROOT / visual_page.image_path
    if not page_image_path.is_file():
        raise FileNotFoundError(page_image_path)
    if (
        sha256_file(page_image_path)
        != visual_page.image_sha256
    ):
        raise ValueError(
            "rendered PDF page image SHA256 mismatch"
        )

    episode_manifest_path = (
        ROOT
        / robot_config["manifest_root"]
        / f"{args.episode_id}.json"
    )
    episode_manifest = (
        EpisodeManifest.model_validate_json(
            episode_manifest_path.read_text(
                encoding="utf-8"
            )
        )
    )

    episode_dir = Path(args.episode_dir)
    metadata_path = episode_dir / "metadata.json"
    samples_path = episode_dir / "samples.csv"

    if (
        sha256_file(metadata_path)
        != episode_manifest.metadata_sha256
    ):
        raise ValueError(
            "metadata.json SHA256 does not match episode manifest"
        )
    if (
        sha256_file(samples_path)
        != episode_manifest.samples_csv_sha256
    ):
        raise ValueError(
            "samples.csv SHA256 does not match episode manifest"
        )

    validate_source_semantics(metadata_path)

    processed_dir = (
        ROOT
        / robot_config["processed_root"]
        / args.episode_id
    )
    records = load_frame_records(
        processed_dir / "frames.jsonl"
    )
    pair = [
        record
        for record in records
        if record.frame_index == frame_index
    ]
    pair.sort(
        key=lambda record: (
            0 if record.camera == "front" else 1
        )
    )
    if [item.camera for item in pair] != [
        "front",
        "wrist",
    ]:
        raise ValueError(
            "frozen robot sample requires front/wrist pair"
        )

    for record in pair:
        image_path = episode_dir / record.image_relpath
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if sha256_file(image_path) != record.image_sha256:
            raise ValueError(
                f"{record.camera} frame SHA256 mismatch"
            )

    samples = load_state_action_samples(
        samples_path,
        verify_tracking_error=True,
    )
    sample = samples[frame_index]

    timestamp_sec = pair[0].timestamp_sec
    if abs(sample.timestamp_sec - timestamp_sec) > 1e-12:
        raise ValueError(
            "frame and state/action timestamps differ"
        )

    document_ref = EvidenceRef(
        source_id=pdf_manifest.source_id,
        source_type=SourceType.PDF,
        page_number=page_number,
    )
    robot_refs = [
        EvidenceRef(
            source_id=episode_manifest.episode_id,
            source_type=SourceType.ROBOT_SEQUENCE,
            time_start_sec=timestamp_sec,
            time_end_sec=timestamp_sec,
            frame_index=frame_index,
            camera=record.camera,
        )
        for record in pair
    ]

    bundle = UnifiedEvidenceBundle(
        bundle_id=(
            "day11_grounded_"
            + episode_manifest.episode_id
        ),
        question=config["question"],
        items=[
            UnifiedEvidenceItem(
                evidence_id=(
                    f"doc:{pdf_manifest.source_id}:p{page_number}"
                ),
                kind=UnifiedEvidenceKind.DOCUMENT_PAGE,
                refs=[document_ref],
                provenance=EvidenceProvenance(
                    source_id=pdf_manifest.source_id,
                    source_type=SourceType.PDF,
                    manifest_path=project_relative(
                        pdf_manifest_path
                    ),
                    canonical_sha256=pdf_manifest.sha256,
                ),
                payload=DocumentPagePayload(
                    page_number=page_number,
                    text_sha256=page.text_sha256,
                    char_count=page.char_count,
                    text_excerpt=page.text,
                    page_image_path=visual_page.image_path,
                ),
            ),
            UnifiedEvidenceItem(
                evidence_id=(
                    f"robot:{episode_manifest.episode_id}:"
                    f"f{frame_index}"
                ),
                kind=UnifiedEvidenceKind.ROBOT_SAMPLE,
                refs=robot_refs,
                provenance=EvidenceProvenance(
                    source_id=episode_manifest.episode_id,
                    source_type=SourceType.ROBOT_SEQUENCE,
                    manifest_path=project_relative(
                        episode_manifest_path
                    ),
                    canonical_sha256=(
                        episode_manifest.episode_sha256
                    ),
                    supporting_sha256={
                        "metadata.json": (
                            episode_manifest.metadata_sha256
                        ),
                        "samples.csv": (
                            episode_manifest.samples_csv_sha256
                        ),
                    },
                ),
                payload=RobotSamplePayload(
                    episode_id=episode_manifest.episode_id,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    cameras=[
                        RobotCameraAsset(
                            camera=record.camera,
                            frame_index=record.frame_index,
                            timestamp_sec=record.timestamp_sec,
                            image_relpath=record.image_relpath,
                            image_sha256=record.image_sha256,
                            source_timestamp_ns=(
                                record.source_timestamp_ns
                            ),
                            source_age_ms=record.source_age_ms,
                            width_px=record.width_px,
                            height_px=record.height_px,
                        )
                        for record in pair
                    ],
                    state_action=RobotStateActionSnapshot(
                        frame_index=sample.frame_index,
                        timestamp_sec=sample.timestamp_sec,
                        observation=sample.observation,
                        action=sample.action,
                        tracking_error=sample.tracking_error,
                    ),
                ),
            ),
        ],
    )

    cross_valid, cross_errors = (
        validate_cross_domain_bundle(bundle)
    )
    if not cross_valid:
        raise ValueError(
            "invalid cross-domain bundle: "
            + repr(cross_errors)
        )

    messages = build_unified_messages(
        bundle=bundle,
        project_root=str(ROOT),
        episode_dir=str(episode_dir),
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

    answer = parse_unified_grounded_answer(
        raw_output
    )
    citation_valid, citation_errors = (
        validate_unified_citation_policy(
            answer,
            bundle,
        )
    )
    required_citation_valid, required_citation_errors = (
        validate_required_citation_coverage(
            answer,
            [
                document_ref,
                *robot_refs,
            ],
        )
    )

    fact_coverage = required_fact_coverage(
        answer.answer,
        config["required_fact_groups"],
    )

    answerable_valid = answer.abstain is False

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
            "day11_unified_grounded_generation_smoke"
        ),
        "scope": (
            "fixed_cross_domain_evidence_no_retrieval_"
            "no_failure_diagnosis"
        ),
        "episode_id": args.episode_id,
        "page_number": page_number,
        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec,
        "model_name": config["generator_model"],
        "retrieval_called": False,
        "failure_diagnosis_attempted": False,
        "agent_used": False,
        "bundle_schema_version": bundle.schema_version,
        "bundle_valid": True,
        "structured_output_valid": True,
        "answerable_valid": answerable_valid,
        "citation_policy_valid": citation_valid,
        "citation_policy_errors": citation_errors,
        "required_citation_coverage_valid": (
            required_citation_valid
        ),
        "required_citation_coverage_errors": (
            required_citation_errors
        ),
        "required_fact_coverage": fact_coverage,
        "required_fact_coverage_valid": (
            abs(fact_coverage - 1.0) < 1e-12
        ),
        "answer": answer.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "raw_model_output": raw_output,
        "evidence_summary": {
            "document_items": 1,
            "robot_items": 1,
            "visual_inputs": 3,
            "document_page_image_sha256": (
                visual_page.image_sha256
            ),
            "front_image_sha256": pair[0].image_sha256,
            "wrist_image_sha256": pair[1].image_sha256,
            "observation_dimensions": len(
                sample.observation.ordered_values()
            ),
            "action_dimensions": len(
                sample.action.ordered_values()
            ),
        },
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
            "no unified retrieval",
            "no failure diagnosis",
            "no causal relation between manual and robot sample",
            "no Agent or MCP",
            "no production budget selection",
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
        and abs(fact_coverage - 1.0) < 1e-12
    ):
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
