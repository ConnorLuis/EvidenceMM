# EvidenceMM Task Definition

## 1. Problem

EvidenceMM answers questions over multimodal operational evidence rather than chatting with a single uploaded image.

The final system should retrieve and verify evidence from:

1. PDF manuals, diagrams, and tables;
2. still images;
3. wrist/front operation videos;
4. robot state/action streams.

A valid answer should be traceable to evidence such as page number, timestamp, key frame, camera view, or image region. If the available evidence is insufficient, the system should abstain instead of fabricating a diagnosis.

## 2. Flagship scenario

Given a failed robot-grasp episode, locate the failure interval and distinguish hypotheses such as:

- target offset / perception error;
- gripper-close timing error;
- trajectory execution deviation;
- insufficient evidence.

The final answer should combine robot video, state/action changes, and manual evidence.

## 3. Evidence contract

Evidence is represented with a generic `EvidenceRef`.

- PDF: `source_id + page_number`, optionally `bbox`.
- image: `source_id`, optionally `bbox`.
- video: `source_id + time_start_sec/time_end_sec`, optionally `frame_index` and `camera`.
- robot state/action: `source_id + time_start_sec/time_end_sec`, with diagnostic notes stored outside the raw evidence locator.

`bbox` uses normalized coordinates in [0, 1] with origin at the top-left:

```text
(x1, y1) --------
   |             |
   |             |
   -------- (x2, y2)
```

Page numbers are 1-based. Times are seconds from the beginning of the source stream.

## 4. Evaluation contract

Every evaluation case has:

- stable `case_id`;
- question;
- intended input source ids;
- `answerable` label once manually verified;
- reference answer once manually verified;
- reference evidence once manually verified;
- tags for modality/task analysis.

Day1 creates 20 *draft* questions but intentionally does not invent ground truth before real assets are attached. Later, draft cases are promoted to `verified` only after human annotation.

## 5. Baseline

Day1 baseline is direct Qwen3-VL inference with no retrieval. It answers from explicitly supplied image/video inputs only.

This baseline exists so later retrieval systems can be compared against a fixed no-retrieval reference.

## 6. Future evaluation metrics

Retrieval:
- Recall@5
- nDCG@5

Answer/evidence:
- answer accuracy
- citation/evidence accuracy
- abstention accuracy

Temporal localization:
- temporal IoU and/or boundary error

Systems:
- P50/P95 latency
- peak GPU memory

These metrics are future acceptance targets, not Day1 claims.
