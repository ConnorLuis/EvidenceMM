from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.data_binding import sha256_file
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
    UnifiedGroundedAnswer,
    validate_cross_domain_bundle,
    validate_unified_citation_policy,
)


ROOT = Path(__file__).resolve().parents[1]


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument(
        "--pdf-manifest",
        default=(
            "data/manifests/sources/"
            "sts3215_datasheet.json"
        ),
    )
    parser.add_argument("--page-number", type=int, default=1)
    parser.add_argument("--frame-index", type=int, default=15)
    args = parser.parse_args()

    config = yaml.safe_load(
        (
            ROOT
            / "configs/robot_sequence_evidence.yaml"
        ).read_text(encoding="utf-8")
    )

    pdf_manifest_path = Path(args.pdf_manifest)
    if not pdf_manifest_path.is_absolute():
        pdf_manifest_path = ROOT / pdf_manifest_path

    pdf_manifest = load_source_manifest(pdf_manifest_path)
    if pdf_manifest.source_type != SourceType.PDF:
        raise ValueError(
            "cross-domain smoke PDF manifest must be a PDF source"
        )

    pages = standardize_pdf_pages(
        pdf_manifest,
        project_root=ROOT,
    )
    page_by_number = {
        page.page_number: page
        for page in pages
    }
    if args.page_number not in page_by_number:
        raise ValueError(
            f"page-number must be in [1, {len(pages)}]"
        )
    page = page_by_number[args.page_number]

    episode_manifest_path = (
        ROOT
        / config["manifest_root"]
        / f"{args.episode_id}.json"
    )
    episode_manifest = EpisodeManifest.model_validate_json(
        episode_manifest_path.read_text(encoding="utf-8")
    )

    episode_dir = Path(args.episode_dir)
    metadata_path = episode_dir / "metadata.json"
    samples_path = episode_dir / "samples.csv"

    if sha256_file(metadata_path) != episode_manifest.metadata_sha256:
        raise ValueError(
            "metadata.json SHA256 does not match episode manifest"
        )
    if sha256_file(samples_path) != episode_manifest.samples_csv_sha256:
        raise ValueError(
            "samples.csv SHA256 does not match episode manifest"
        )

    validate_source_semantics(metadata_path)

    processed_dir = (
        ROOT
        / config["processed_root"]
        / args.episode_id
    )
    records = load_frame_records(
        processed_dir / "frames.jsonl"
    )
    pair = [
        record
        for record in records
        if record.frame_index == args.frame_index
    ]
    pair.sort(
        key=lambda record: (
            0 if record.camera == "front" else 1
        )
    )
    if [record.camera for record in pair] != ["front", "wrist"]:
        raise ValueError(
            "requested frame does not have a front/wrist pair"
        )

    for record in pair:
        image_path = episode_dir / record.image_relpath
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if sha256_file(image_path) != record.image_sha256:
            raise ValueError(
                f"{record.camera} frame SHA256 mismatch"
            )

    state_action_samples = load_state_action_samples(
        samples_path,
        verify_tracking_error=True,
    )
    if not 0 <= args.frame_index < len(state_action_samples):
        raise ValueError("frame-index outside state/action samples")
    sample = state_action_samples[args.frame_index]

    timestamp_sec = pair[0].timestamp_sec
    if abs(sample.timestamp_sec - timestamp_sec) > 1e-12:
        raise ValueError(
            "frame-record and state/action timestamps differ"
        )

    document_ref = EvidenceRef(
        source_id=pdf_manifest.source_id,
        source_type=SourceType.PDF,
        page_number=page.page_number,
    )

    robot_refs = [
        EvidenceRef(
            source_id=episode_manifest.episode_id,
            source_type=SourceType.ROBOT_SEQUENCE,
            time_start_sec=timestamp_sec,
            time_end_sec=timestamp_sec,
            frame_index=args.frame_index,
            camera=record.camera,
        )
        for record in pair
    ]

    document_item = UnifiedEvidenceItem(
        evidence_id=(
            f"doc:{pdf_manifest.source_id}:p{page.page_number}"
        ),
        kind=UnifiedEvidenceKind.DOCUMENT_PAGE,
        refs=[document_ref],
        provenance=EvidenceProvenance(
            source_id=pdf_manifest.source_id,
            source_type=SourceType.PDF,
            manifest_path=project_relative(pdf_manifest_path),
            canonical_sha256=pdf_manifest.sha256,
        ),
        payload=DocumentPagePayload(
            page_number=page.page_number,
            text_sha256=page.text_sha256,
            char_count=page.char_count,
            text_excerpt=page.text[:240],
            page_image_path=None,
        ),
    )

    robot_item = UnifiedEvidenceItem(
        evidence_id=(
            f"robot:{episode_manifest.episode_id}:"
            f"f{args.frame_index}"
        ),
        kind=UnifiedEvidenceKind.ROBOT_SAMPLE,
        refs=robot_refs,
        provenance=EvidenceProvenance(
            source_id=episode_manifest.episode_id,
            source_type=SourceType.ROBOT_SEQUENCE,
            manifest_path=project_relative(
                episode_manifest_path
            ),
            canonical_sha256=episode_manifest.episode_sha256,
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
            frame_index=args.frame_index,
            timestamp_sec=timestamp_sec,
            cameras=[
                RobotCameraAsset(
                    camera=record.camera,
                    frame_index=record.frame_index,
                    timestamp_sec=record.timestamp_sec,
                    image_relpath=record.image_relpath,
                    image_sha256=record.image_sha256,
                    source_timestamp_ns=record.source_timestamp_ns,
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
    )

    bundle = UnifiedEvidenceBundle(
        bundle_id=(
            f"day11_cross_domain_"
            f"{episode_manifest.episode_id}"
        ),
        question=(
            "Contract smoke: bind one real manual page and one "
            "real robot sample into one traceable evidence bundle."
        ),
        items=[document_item, robot_item],
    )

    cross_domain_valid, cross_domain_errors = (
        validate_cross_domain_bundle(bundle)
    )

    smoke_answer = UnifiedGroundedAnswer(
        answer=(
            "The supplied evidence bundle contains one traceable "
            "manual page and one traceable synchronized robot sample."
        ),
        abstain=False,
        citations=[
            document_ref,
            *robot_refs,
        ],
    )
    citation_valid, citation_errors = (
        validate_unified_citation_policy(
            smoke_answer,
            bundle,
        )
    )

    payload = {
        "mode": "day11_unified_evidence_contract_smoke",
        "scope": "contract_only_no_model_generation",
        "gold_read": False,
        "model_called": False,
        "failure_diagnosis_attempted": False,
        "bundle_schema_version": bundle.schema_version,
        "bundle_id": bundle.bundle_id,
        "cross_domain_valid": cross_domain_valid,
        "cross_domain_errors": cross_domain_errors,
        "citation_policy_valid": citation_valid,
        "citation_policy_errors": citation_errors,
        "document_evidence": {
            "evidence_id": document_item.evidence_id,
            "source_id": pdf_manifest.source_id,
            "source_sha256": pdf_manifest.sha256,
            "page_number": page.page_number,
            "text_sha256": page.text_sha256,
            "char_count": page.char_count,
            "citation": document_ref.model_dump(
                mode="json",
                exclude_none=True,
            ),
        },
        "robot_evidence": {
            "evidence_id": robot_item.evidence_id,
            "episode_id": episode_manifest.episode_id,
            "episode_sha256": episode_manifest.episode_sha256,
            "samples_csv_sha256": (
                episode_manifest.samples_csv_sha256
            ),
            "frame_index": args.frame_index,
            "timestamp_sec": timestamp_sec,
            "camera_count": len(pair),
            "cameras": [
                {
                    "camera": record.camera,
                    "image_relpath": record.image_relpath,
                    "image_sha256": record.image_sha256,
                }
                for record in pair
            ],
            "state_dimensions": len(
                sample.observation.ordered_values()
            ),
            "action_dimensions": len(
                sample.action.ordered_values()
            ),
            "citations": [
                ref.model_dump(
                    mode="json",
                    exclude_none=True,
                )
                for ref in robot_refs
            ],
        },
        "unified_item_count": len(bundle.items),
        "unified_citation_count": len(
            smoke_answer.citations
        ),
        "next_connection_not_claimed": (
            "unified retrieval/generation is not connected by "
            "this contract smoke"
        ),
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
