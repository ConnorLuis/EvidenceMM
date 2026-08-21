# EvidenceMM Task Definition

## 1. Problem

EvidenceMM answers questions over traceable multimodal operational evidence.

The evidence space includes:

1. PDF manuals, diagrams, and tables;
2. rendered document pages;
3. synchronized robot-operation image sequences;
4. robot state/action streams.

A valid answer must be grounded in evidence locators such as PDF page number,
robot frame index, timestamp, camera view, or region. When evidence is
insufficient, the system should abstain.

## 2. Flagship scenario

Given a real failed robot-grasp episode, the final system should locate the
relevant failure interval and assess hypotheses such as:

- target offset / perception error;
- gripper-close timing error;
- trajectory execution deviation;
- insufficient evidence.

The final answer should combine robot visual evidence, state/action evidence,
and relevant manual evidence.

This physical root-cause diagnosis is still a target. It is not yet a completed
capability.

## 3. Evidence contract

Generic source locations use `EvidenceRef`.

Document evidence binds:

```text
source_id + page_number
```

Robot sequence evidence binds:

```text
episode/source_id
+ frame_index
+ camera
+ timestamp interval
```

Robot `observation`, `action`, and `tracking_error` are attached to the canonical
robot sample rather than represented as disconnected evidence sources.

Cross-domain evidence is carried in `UnifiedEvidenceBundle`.

## 4. Canonical source identity

The canonical robot source is:

```text
metadata.json
samples.csv
front/*.jpg
wrist/*.jpg
```

`samples.csv:elapsed_ns` is the canonical relative timeline.

Camera source timestamp and age remain camera-specific.

MP4 is optional future input/display adaptation, not required canonical
evidence.

## 5. Current canonical integration

Day15 canonical document retrieval:

```text
BM25 Top-5
       \
        -> candidate union -> BGE reranker
       /
BGE-M3 Top-5
```

Cross-domain integration:

```text
document ranking
+
robot signal ranking
        ↓
fixed 3 + 2 budget
        ↓
UnifiedEvidenceBundle
        ↓
Qwen3-VL
        ↓
compact citation IDs
        ↓
EvidenceRef resolution
        ↓
citation / fact validation
        ↓
pipeline diagnosis
```

The Day12 BM25-only document mode remains selectable as a frozen baseline.

## 6. Validated non-canonical modules

The following are validated but are not silently claimed as part of the
current Day15 canonical cross-domain path:

- ColQwen2.5 visual page retrieval;
- BM25 + ColQwen RRF;
- historical temporal-selector micro-baselines;
- generic experimental RRF adapter.

ColQwen remains relevant to the project, but canonical visual integration must
be evaluated explicitly rather than assumed.

## 7. Evaluation contract

Evaluation labels are frozen before metric calculation.

Gold labels must never be read by retrievers.

Current smoke evaluation covers:

- document retrieval rank;
- citation policy;
- required citation coverage;
- required-fact coverage;
- abstention;
- robot temporal evidence;
- ranking trace;
- deterministic pipeline failure diagnosis.

Current scale is not sufficient for broad quality claims.

## 8. Failure taxonomy

Current Day14 diagnosis concerns EvidenceMM system behavior:

```text
retrieval
evidence
generation
```

It includes retrieval misses, missing evidence, citation violations, incomplete
answers, and abstention errors.

It does not infer physical robot failure causes.

## 9. Next flagship evaluation

The next major dataset/evaluation step should bind real successful and failed
episodes with:

- failure time interval;
- failure class / hypothesis;
- uncertainty / unanswerable label;
- visible camera(s);
- relevant state/action span;
- supporting manual page(s);
- held-out episode split.

Target metrics may include:

- document Recall/nDCG;
- temporal event recall and boundary error/IoU;
- failure-cause macro-F1;
- citation precision/recall;
- abstention accuracy;
- end-to-end latency and GPU memory.

## 10. Non-goals

EvidenceMM does not own:

- robot control policy;
- ACT training;
- LangGraph orchestration;
- MCP planning;
- multi-agent memory;
- LoRA/QLoRA by default.

Those concerns belong to separate projects or are deferred until evidence
analysis justifies them.
