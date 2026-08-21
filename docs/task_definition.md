# EvidenceMM Task Definition

## 1. Problem

EvidenceMM answers questions over multimodal operational evidence rather than
chatting with a single uploaded image.

The final system should retrieve and verify evidence from:

1. PDF manuals, diagrams, and tables;
2. still images;
3. robot-operation sequences;
4. robot state/action streams.

A robot-operation sequence may be:

- a sample-synchronized multi-camera image sequence, such as the Day 7
  `front/wrist` SO-ARM101 source; or
- a native video source handled by a future video adapter.

A valid answer should be traceable to evidence such as page number, timestamp,
frame index, camera view, or image region. If the available evidence is
insufficient, the system should abstain instead of fabricating a diagnosis.

## 2. Flagship scenario

Given a failed robot-grasp episode, locate the failure interval and distinguish
hypotheses such as:

- target offset / perception error;
- gripper-close timing error;
- trajectory execution deviation;
- insufficient evidence.

The final target is to combine robot-operation evidence, robot state/action
changes, and manual evidence in one traceable diagnostic answer.

This cross-domain diagnostic pipeline is **not yet completed**. At Day 7 the
document grounded-QA branch and robot temporal-evidence branch exist as
separate baselines.

## 3. Evidence contract

Evidence is represented with a generic `EvidenceRef`.

- PDF: `source_id + page_number`, optionally `bbox`.
- image: `source_id`, optionally `bbox`.
- native video: `source_id + time_start_sec/time_end_sec`, optionally
  `frame_index` and `camera`.
- robot sequence: `source_id + frame_index + camera`, or a temporal interval.
- robot state/action: `source_id + time_start_sec/time_end_sec`, with
  diagnostic notes stored outside the raw evidence locator.

`bbox` uses normalized coordinates in [0, 1] with origin at the top-left:

```text
(x1, y1) --------
   |             |
   |             |
   -------- (x2, y2)
```

Page numbers are 1-based. Times are seconds from the beginning of the bound
source timeline.

### Robot sequence identity

The generic locator contract does not force a multi-file robot episode into a
single video-shaped manifest.

The Day 7 canonical robot episode schema is
`evidencemm.temporal_evidence.EpisodeManifest`.

Its canonical source is:

```text
metadata.json
samples.csv
front/*.jpg
wrist/*.jpg
```

The aggregate episode identity is computed deterministically from source-file
hashes and ordered frame hashes. `samples.csv:elapsed_ns` is the canonical
relative sample timeline. Camera source timestamp and source age remain
camera-specific.

MP4 generated from this image sequence is a future derived display artifact,
not canonical evidence.

## 4. Current integration state

### Connected document baseline

```text
bound PDF
-> page text + page image representation
-> BM25 + ColQwen2.5
-> RRF
-> Top-k evidence
-> Qwen3-VL
-> structured answer
-> citation validation
-> answer / abstain
```

### Separate robot temporal baseline

```text
sample-synchronized front/wrist sequence
-> EpisodeManifest
-> FrameRecord
-> timestamp-based TemporalSlice
-> uniform midpoint evidence
-> human temporal gold
-> event coverage
```

The two branches have not yet been unified into cross-domain retrieval and
generation.

## 5. Evaluation contract

Every answer-oriented evaluation case has:

- stable `case_id`;
- question;
- intended input source IDs;
- `answerable` label once manually verified;
- reference answer once manually verified;
- reference evidence once manually verified;
- tags for modality/task analysis.

Day 1 created 20 draft questions but intentionally did not invent ground truth
before real assets were attached. Later cases are promoted to `verified` only
after human annotation.

Temporal evaluation additionally uses human-verified event intervals. Ambiguous
or censored boundaries must be excluded rather than manufactured.

## 6. Baseline ladder

The project keeps simple baselines frozen before adding stronger components.

### Document branch

1. direct Qwen3-VL inference;
2. BM25 text retrieval;
3. ColQwen2.5 visual retrieval;
4. RRF hybrid retrieval;
5. grounded generation + citation validation + abstention.

### Robot branch

1. sample-synchronized evidence binding;
2. 2 s uniform-midpoint temporal selection;
3. next: visual-motion-aware temporal selection;
4. later: robot-state/action-aware temporal selection.

The later selector must be compared against the same human temporal gold and a
controlled evidence budget.

## 7. Current smoke results and limitations

Current data scale is intentionally small:

- one 8-page STS3215 datasheet;
- two verified PDF retrieval queries;
- three grounded-generation smoke cases;
- one SO-ARM101 robot episode;
- three verified visual temporal events.

Day 6 produces perfect deterministic grounded-generation contract metrics on
three smoke cases. This is not a general answer-quality benchmark.

Day 7 uniform midpoint covers 2 of 3 verified temporal events and misses the
0.268 s `object_lift` event. This real miss is preserved as the motivation for
the next temporal selector.

## 8. Future evaluation metrics

Retrieval:

- Recall@5
- nDCG@5

Answer/evidence:

- answer accuracy
- citation/evidence accuracy
- abstention accuracy

Temporal localization:

- event coverage
- nearest-evidence temporal distance
- temporal IoU and/or boundary error when the task supports interval prediction

Systems:

- P50/P95 latency
- throughput
- peak GPU memory

Formal system performance will later be measured on the RTX 4090. Current RTX
4070 SUPER timings are development diagnostics only.

## 9. Non-goals for the current stage

The following are intentionally not part of the Day 7.5 calibration:

- cross-domain robot + document generation;
- failed-grasp diagnosis;
- `q_t` / action-aware selection;
- FastAPI service;
- LangGraph / MCP orchestration;
- robot control policy;
- LoRA / QLoRA.

These remain later work and must not be presented as completed features.
