from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from evidencemm.data_binding import sha256_file
from evidencemm.robot_failure_dataset import (
    AuditCategory,
    SourceAuditRecord,
)
from evidencemm.temporal_evidence import (
    CameraSpec,
    EpisodeManifest,
    build_frame_records,
    canonical_episode_hash,
    load_sample_rows,
    load_source_metadata,
    save_frame_records,
)


class DiagnosticBindingError(ValueError):
    """Raised when raw episode bytes cannot be bound for diagnosis."""


@dataclass(frozen=True)
class DiagnosticBindingResult:
    episode_id: str
    manifest_path: str
    frames_path: str
    frame_count: int
    episode_sha256: str
    source_status: str
    source_overall_pass: bool
    source_valid_for_training: bool | None
    source_failed_checks: tuple[str, ...]


def diagnostic_binding_allowed(
    audit: SourceAuditRecord,
) -> bool:
    return (
        audit.diagnostic_eligible
        and audit.audit_category
        != AuditCategory.TECHNICAL_EXCLUSION
        and audit.technical_valid
    )


NON_DIAGNOSTIC_FAILED_CHECKS = frozenset(
    {
        "cleanup_home",
        "cleanup_park",
    }
)


def validate_diagnostic_recording_acceptance(
    metadata: dict,
) -> tuple[str, ...]:
    """
    Validate recording integrity for EvidenceMM diagnosis.

    Training/demo acceptance and diagnostic acceptance
    intentionally have different semantics.

    cleanup_home / cleanup_park occur outside the
    recorded task evidence window, so they are preserved
    as provenance but do not invalidate diagnostic evidence.
    """

    status = str(
        metadata.get(
            "status",
            "",
        )
    )

    completed = metadata.get(
        "completed"
    )

    # Backward-compatible fallback for older metadata.
    if completed is None:
        completed = (
            status == "completed"
        )

    if completed is not True:
        raise DiagnosticBindingError(
            "diagnostic source recording "
            "must be completed"
        )

    if metadata.get(
        "aborted",
        False,
    ) is True:
        raise DiagnosticBindingError(
            "diagnostic source recording "
            "must not be aborted"
        )

    error = metadata.get("error")
    if error not in (
        None,
        "",
    ):
        raise DiagnosticBindingError(
            "diagnostic source recording "
            f"contains error: {error!r}"
        )

    checks = metadata.get(
        "checks",
        {},
    )

    if not isinstance(
        checks,
        dict,
    ):
        raise DiagnosticBindingError(
            "metadata.checks must be an object"
        )

    if checks:
        if (
            checks.get(
                "normal_recording_completion"
            )
            is not True
        ):
            raise DiagnosticBindingError(
                "normal_recording_completion "
                "check must pass"
            )

        failed_checks = tuple(
            sorted(
                key
                for key, value
                in checks.items()
                if value is not True
            )
        )

        blocking_checks = tuple(
            key
            for key in failed_checks
            if (
                key
                not in NON_DIAGNOSTIC_FAILED_CHECKS
            )
        )

        if blocking_checks:
            raise DiagnosticBindingError(
                "diagnostic-critical source "
                "checks failed: "
                + ", ".join(
                    blocking_checks
                )
            )

        return failed_checks

    return ()


def _required_metadata_count(
    metadata: dict,
) -> int:
    try:
        return int(
            metadata["settings"][
                "expected_sample_count"
            ]
        )
    except Exception as exc:
        raise DiagnosticBindingError(
            "metadata missing settings.expected_sample_count"
        ) from exc


def bind_diagnostic_episode(
    *,
    episode_dir: str | Path,
    manifest_path: str | Path,
    frames_path: str | Path,
    timestamp_source: str,
) -> DiagnosticBindingResult:
    source_dir = Path(
        episode_dir
    ).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)

    metadata_path = source_dir / "metadata.json"
    samples_path = source_dir / "samples.csv"

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not samples_path.is_file():
        raise FileNotFoundError(samples_path)

    metadata = load_source_metadata(
        metadata_path
    )
    rows = load_sample_rows(
        samples_path
    )

    source_failed_checks = (
        validate_diagnostic_recording_acceptance(
            metadata
        )
    )

    expected_count = _required_metadata_count(
        metadata
    )
    if len(rows) != expected_count:
        raise DiagnosticBindingError(
            f"sample count mismatch: "
            f"{len(rows)} != {expected_count}"
        )

    results = metadata.get(
        "results",
        {}
    )
    for camera in ("front", "wrist"):
        key = f"{camera}_image_count"
        if key not in results:
            raise DiagnosticBindingError(
                f"metadata missing results.{key}"
            )
        if int(results[key]) != expected_count:
            raise DiagnosticBindingError(
                f"{key} mismatch"
            )

    episode_id = source_dir.name

    records = build_frame_records(
        episode_dir=source_dir,
        episode_id=episode_id,
        rows=rows,
    )

    front = [
        item
        for item in records
        if item.camera == "front"
    ]
    wrist = [
        item
        for item in records
        if item.camera == "wrist"
    ]

    front_dims = {
        (item.width_px, item.height_px)
        for item in front
    }
    wrist_dims = {
        (item.width_px, item.height_px)
        for item in wrist
    }
    if len(front_dims) != 1:
        raise DiagnosticBindingError(
            f"front dimension drift: {front_dims}"
        )
    if len(wrist_dims) != 1:
        raise DiagnosticBindingError(
            f"wrist dimension drift: {wrist_dims}"
        )

    metadata_sha = sha256_file(
        metadata_path
    )
    samples_sha = sha256_file(
        samples_path
    )
    episode_sha = canonical_episode_hash(
        metadata_sha256=metadata_sha,
        samples_csv_sha256=samples_sha,
        records=records,
    )

    front_width, front_height = next(
        iter(front_dims)
    )
    wrist_width, wrist_height = next(
        iter(wrist_dims)
    )

    camera_transforms = (
        metadata.get("settings", {})
        .get("camera_transforms", {})
    )

    manifest = EpisodeManifest(
        episode_id=episode_id,
        source_schema_version=str(
            metadata.get(
                "schema_version",
                "unknown",
            )
        ),
        source_script_version=str(
            metadata.get(
                "script_version",
                "unknown",
            )
        ),
        task=str(
            metadata.get(
                "task",
                "unknown",
            )
        ),
        frame_count=expected_count,
        nominal_hz=float(
            metadata["settings"]["hz"]
        ),
        actual_record_span_seconds=float(
            metadata["timing"][
                "actual_record_span_seconds"
            ]
        ),
        timestamp_source=timestamp_source,
        metadata_sha256=metadata_sha,
        samples_csv_sha256=samples_sha,
        episode_sha256=episode_sha,
        cameras=[
            CameraSpec(
                camera="front",
                frame_count=expected_count,
                width_px=front_width,
                height_px=front_height,
                transform=str(
                    camera_transforms.get(
                        "front",
                        "none",
                    )
                ),
            ),
            CameraSpec(
                camera="wrist",
                frame_count=expected_count,
                width_px=wrist_width,
                height_px=wrist_height,
                transform=str(
                    camera_transforms.get(
                        "wrist",
                        "none",
                    )
                ),
            ),
        ],
        # This field preserves source metadata truth. It is NOT used as an
        # acceptance gate for diagnostic binding.
        source_checks_overall_pass=bool(
            metadata.get(
                "overall_pass",
                False,
            )
        ),
    )

    out_manifest = Path(
        manifest_path
    )
    out_frames = Path(
        frames_path
    )

    out_manifest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    save_frame_records(
        records,
        out_frames,
    )
    out_manifest.write_text(
        json.dumps(
            manifest.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return DiagnosticBindingResult(
        episode_id=episode_id,
        manifest_path=str(
            out_manifest
        ),
        frames_path=str(
            out_frames
        ),
        frame_count=expected_count,
        episode_sha256=episode_sha,
        source_status=str(
            metadata.get(
                "status",
                "",
            )
        ),
        source_overall_pass=bool(
            metadata.get(
                "overall_pass",
                False,
            )
        ),
        source_valid_for_training=(
            bool(
                metadata[
                    "valid_for_training"
                ]
            )
            if (
                "valid_for_training"
                in metadata
            )
            else None
        ),
        source_failed_checks=(
            source_failed_checks
        ),
    )
