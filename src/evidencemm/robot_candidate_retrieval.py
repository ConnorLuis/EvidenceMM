from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from evidencemm.data_binding import sha256_file
from evidencemm.retrieval import (
    normalize_query,
    validate_top_k,
)
from evidencemm.retrieval_ranking import (
    RankedEvidenceCandidate,
    RetrievalDomain,
)
from evidencemm.schemas import EvidenceRef, SourceType
from evidencemm.state_action_selection import (
    JOINT_ORDER,
    JointVector,
    StateActionSample,
    load_state_action_samples,
    rms,
    score_state_action_sample,
    validate_source_semantics,
)
from evidencemm.temporal_evidence import (
    EpisodeManifest,
    FrameRecord,
    load_frame_records,
)
from evidencemm.text_retrieval import tokenize_mixed
from evidencemm.unified_evidence import (
    EvidenceProvenance,
    RobotCameraAsset,
    RobotSamplePayload,
    RobotStateActionSnapshot,
    UnifiedEvidenceItem,
    UnifiedEvidenceKind,
)


ROBOT_RETRIEVER_NAME = "robot_signal_change_v1"
ROBOT_SIGNAL_NAMES = (
    "observation",
    "action",
    "tracking_error",
)


@dataclass(frozen=True)
class RobotSignalQueryProfile:
    joints: tuple[str, ...]
    signals: tuple[str, ...]
    explicit_joint_terms: bool
    explicit_signal_terms: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RobotSignalRetrievalHit:
    rank: int
    frame_index: int
    timestamp_sec: float
    raw_score: float
    signal_scores: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def parse_robot_signal_query(
    query: str,
) -> RobotSignalQueryProfile:
    """Parse only canonical joint/signal identifiers.

    This intentionally performs no synonym injection, translation, or semantic
    event mapping. When the query does not name a canonical joint, all six
    joints are used. When it does not name a canonical signal, observation and
    action change are used.
    """

    tokens = set(
        tokenize_mixed(
            normalize_query(query)
        )
    )

    explicit_joints = tuple(
        joint
        for joint in JOINT_ORDER
        if joint in tokens
    )
    explicit_signals = tuple(
        signal
        for signal in ROBOT_SIGNAL_NAMES
        if signal in tokens
    )

    return RobotSignalQueryProfile(
        joints=(
            explicit_joints
            if explicit_joints
            else tuple(JOINT_ORDER)
        ),
        signals=(
            explicit_signals
            if explicit_signals
            else ("observation", "action")
        ),
        explicit_joint_terms=bool(explicit_joints),
        explicit_signal_terms=bool(explicit_signals),
    )


def _selected_rms(
    vector: JointVector,
    joints: tuple[str, ...],
) -> float:
    return rms(
        [
            float(getattr(vector, joint))
            for joint in joints
        ]
    )


def score_robot_signal_sample(
    *,
    current: StateActionSample,
    previous: StateActionSample | None,
    profile: RobotSignalQueryProfile,
) -> tuple[float, dict[str, float]]:
    frame_score = score_state_action_sample(
        current=current,
        previous=previous,
    )

    scores: dict[str, float] = {}
    for signal in profile.signals:
        if signal == "observation":
            scores[signal] = _selected_rms(
                frame_score.state_delta,
                profile.joints,
            )
        elif signal == "action":
            scores[signal] = _selected_rms(
                frame_score.action_delta,
                profile.joints,
            )
        elif signal == "tracking_error":
            scores[signal] = _selected_rms(
                frame_score.tracking_error,
                profile.joints,
            )
        else:
            raise ValueError(
                f"unsupported robot signal: {signal}"
            )

    return max(scores.values()), scores


def rank_robot_signal_samples(
    *,
    samples: list[StateActionSample],
    query: str,
    top_k: int,
) -> tuple[
    RobotSignalQueryProfile,
    list[RobotSignalRetrievalHit],
]:
    validated_top_k = validate_top_k(top_k)
    profile = parse_robot_signal_query(query)

    scored = []
    for index, current in enumerate(samples):
        previous = (
            samples[index - 1]
            if index > 0
            else None
        )
        raw_score, signal_scores = (
            score_robot_signal_sample(
                current=current,
                previous=previous,
                profile=profile,
            )
        )
        scored.append(
            (
                current.frame_index,
                current.timestamp_sec,
                raw_score,
                signal_scores,
            )
        )

    scored.sort(
        key=lambda row: (
            -row[2],
            row[0],
        )
    )

    return (
        profile,
        [
            RobotSignalRetrievalHit(
                rank=rank,
                frame_index=row[0],
                timestamp_sec=row[1],
                raw_score=row[2],
                signal_scores=row[3],
            )
            for rank, row in enumerate(
                scored[:validated_top_k],
                start=1,
            )
        ],
    )


def _stored_path(
    path: Path,
    *,
    project_root: Path,
) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(
            project_root.resolve()
        ).as_posix()
    except ValueError:
        return str(resolved)


class RobotSignalCandidateRetriever:
    """Query-conditioned state/action baseline over one real robot episode."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        episode_manifest_path: str | Path,
        episode_dir: str | Path,
        frame_records_path: str | Path,
    ) -> None:
        self.project_root = Path(project_root).resolve()

        manifest_path = Path(episode_manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = (
                self.project_root / manifest_path
            )
        self.episode_manifest_path = (
            manifest_path.resolve()
        )
        self.manifest = (
            EpisodeManifest.model_validate_json(
                self.episode_manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        )

        self.episode_dir = Path(episode_dir).resolve()
        metadata_path = self.episode_dir / "metadata.json"
        samples_path = self.episode_dir / "samples.csv"

        if (
            sha256_file(metadata_path)
            != self.manifest.metadata_sha256
        ):
            raise ValueError(
                "metadata.json SHA256 differs from episode manifest"
            )
        if (
            sha256_file(samples_path)
            != self.manifest.samples_csv_sha256
        ):
            raise ValueError(
                "samples.csv SHA256 differs from episode manifest"
            )

        validate_source_semantics(metadata_path)
        self.samples = load_state_action_samples(
            samples_path,
            verify_tracking_error=True,
        )
        if len(self.samples) != self.manifest.frame_count:
            raise ValueError(
                "state/action sample count differs from episode manifest"
            )

        records_path = Path(frame_records_path)
        if not records_path.is_absolute():
            records_path = (
                self.project_root / records_path
            )
        self.frame_records_path = records_path.resolve()
        records = load_frame_records(
            self.frame_records_path
        )

        by_frame: dict[int, list[FrameRecord]] = {}
        for record in records:
            if record.episode_id != self.manifest.episode_id:
                raise ValueError(
                    "frame record episode_id differs from manifest"
                )
            by_frame.setdefault(
                record.frame_index,
                [],
            ).append(record)

        if set(by_frame) != set(
            range(self.manifest.frame_count)
        ):
            raise ValueError(
                "frame records do not cover episode frames exactly"
            )

        self.pairs: dict[
            int,
            tuple[FrameRecord, FrameRecord],
        ] = {}
        for frame_index, pair in by_frame.items():
            pair.sort(
                key=lambda record: (
                    0
                    if record.camera == "front"
                    else 1
                )
            )
            if [
                record.camera
                for record in pair
            ] != ["front", "wrist"]:
                raise ValueError(
                    "each robot frame requires front/wrist records"
                )
            if abs(
                pair[0].timestamp_sec
                - pair[1].timestamp_sec
            ) > 1e-12:
                raise ValueError(
                    "front/wrist canonical timestamps differ"
                )
            self.pairs[frame_index] = (
                pair[0],
                pair[1],
            )

    def query_profile(
        self,
        query: str,
    ) -> RobotSignalQueryProfile:
        return parse_robot_signal_query(query)

    def search(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[RankedEvidenceCandidate]:
        _, hits = rank_robot_signal_samples(
            samples=self.samples,
            query=query,
            top_k=top_k,
        )

        candidates = []
        for hit in hits:
            sample = self.samples[
                hit.frame_index
            ]
            front, wrist = self.pairs[
                hit.frame_index
            ]
            pair = [front, wrist]

            if abs(
                sample.timestamp_sec
                - hit.timestamp_sec
            ) > 1e-12:
                raise ValueError(
                    "robot hit timestamp differs from state/action sample"
                )

            for record in pair:
                image_path = (
                    self.episode_dir
                    / record.image_relpath
                )
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                if (
                    sha256_file(image_path)
                    != record.image_sha256
                ):
                    raise ValueError(
                        f"{record.camera} image SHA256 mismatch"
                    )

            refs = [
                EvidenceRef(
                    source_id=self.manifest.episode_id,
                    source_type=SourceType.ROBOT_SEQUENCE,
                    time_start_sec=hit.timestamp_sec,
                    time_end_sec=hit.timestamp_sec,
                    frame_index=hit.frame_index,
                    camera=record.camera,
                )
                for record in pair
            ]

            item = UnifiedEvidenceItem(
                evidence_id=(
                    f"robot:{self.manifest.episode_id}:"
                    f"f{hit.frame_index}"
                ),
                kind=UnifiedEvidenceKind.ROBOT_SAMPLE,
                refs=refs,
                provenance=EvidenceProvenance(
                    source_id=self.manifest.episode_id,
                    source_type=SourceType.ROBOT_SEQUENCE,
                    manifest_path=_stored_path(
                        self.episode_manifest_path,
                        project_root=self.project_root,
                    ),
                    canonical_sha256=(
                        self.manifest.episode_sha256
                    ),
                    supporting_sha256={
                        "metadata.json": (
                            self.manifest.metadata_sha256
                        ),
                        "samples.csv": (
                            self.manifest.samples_csv_sha256
                        ),
                    },
                ),
                payload=RobotSamplePayload(
                    episode_id=self.manifest.episode_id,
                    frame_index=hit.frame_index,
                    timestamp_sec=hit.timestamp_sec,
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
            )

            candidates.append(
                RankedEvidenceCandidate(
                    domain=RetrievalDomain.ROBOT,
                    retriever_name=ROBOT_RETRIEVER_NAME,
                    rank=hit.rank,
                    raw_score=hit.raw_score,
                    item=item,
                )
            )

        return candidates
