# EvidenceMM Day 7.5 - Public Contract and Schema Alignment

## Purpose

Day 7.5 is a calibration pass before Day 8. It does not add a new model,
retriever, temporal selector, evaluation gold label, or benchmark result.

The purpose is to make the public repository contract match the actual Day 7
implementation.

## Problems corrected

### README drift

The previous root README still described the repository as a Day 1 direct-VLM
baseline and listed ColQwen/retrieval as non-goals even though Days 3-7 had
already implemented retrieval, grounding, and robot temporal evidence.

The README is updated to distinguish:

- completed document grounded-QA baseline;
- completed but separate robot temporal-evidence baseline;
- not-yet-connected cross-domain diagnostic pipeline;
- smoke-scale evaluation limitations.

### Robot source terminology

The project contract previously used `robot videos` as if video were the only
robot source representation.

The canonical Day 7 source is instead a sample-synchronized image sequence:

```text
metadata.json
samples.csv
front/*.jpg
wrist/*.jpg
```

The public contract now uses `robot-operation sequences`, with synchronized
image sequences and native video treated as possible adapters. A derived MP4
from Day 7 frames remains display-only.

### Schema drift

The generic `schemas.py` previously contained a legacy `RobotEpisodeManifest`
that required:

```text
wrist_video
front_video
state_file
```

That shape conflicts with the canonical Day 7 source.

The legacy class is removed. The canonical multi-file robot episode identity
remains `evidencemm.temporal_evidence.EpisodeManifest`.

`EvidenceRef` gains `SourceType.ROBOT_SEQUENCE` so frame/interval evidence can
participate in the generic locator contract without forcing the episode source
back into a video-shaped schema.

A robot-sequence frame reference requires:

- `source_id`;
- `frame_index`;
- `camera`.

A robot-sequence interval reference may use a time range without forcing a
single camera.

## Integration boundary after calibration

Completed and connected:

```text
PDF
-> BM25 / ColQwen2.5
-> RRF
-> Qwen3-VL
-> citation validation / abstention
```

Completed but separate:

```text
robot sequence
-> EpisodeManifest / FrameRecord
-> TemporalSlice
-> uniform midpoint
-> temporal gold / coverage
```

Not yet completed:

```text
robot temporal evidence
+ document evidence
-> cross-domain diagnostic generation
```

The flagship failed-grasp diagnosis therefore remains a target rather than a
current feature claim.

## Day 8 entry gate

Day 8 may start only after:

- full pytest passes;
- schema compatibility test passes;
- README/task definition no longer claim video-only canonical robot input;
- repository description is updated from `robot-operation videos` to
  `robot-operation sequences`;
- working tree is clean after commit/push.

Day 8 remains a visual-motion-aware selector comparison against the frozen
Day 7 baseline. It must not introduce `q_t/action` yet.
