# EvidenceMM

EvidenceMM is a traceable multimodal RAG system for complex technical documents
and robot-operation sequences.

Its purpose is not to chat with a single uploaded image. The system binds
canonical source bytes, retrieves evidence, constructs cross-domain evidence
bundles, generates grounded answers, resolves citations deterministically, and
diagnoses pipeline-contract failures.

## Current status

The project has completed the core evidence-system layer and is now moving from
module validation toward the flagship robot-failure application.

Implemented and validated:

- real PDF/image source binding with SHA256 provenance;
- page-level BM25 text retrieval;
- ColQwen2.5 visual page retrieval;
- BM25 + ColQwen visual RRF document baseline;
- Qwen3-VL grounded generation with citation validation and abstention;
- synchronized SO-ARM101 front/wrist robot-sequence evidence;
- timestamp, observation, action, and tracking-error binding;
- multiple frozen temporal-selection baselines;
- `UnifiedEvidenceBundle` for document + robot evidence;
- query-driven cross-domain retrieval-to-generation smoke;
- BGE-M3 dense retrieval;
- sparse+dense candidate union;
- BGE cross-encoder reranking;
- ranking trace;
- EvidenceMM pipeline failure diagnosis.

Not completed and not claimed:

- robot failed-grasp root-cause diagnosis;
- benchmark-scale retrieval/generation quality;
- large multi-episode evaluation;
- production API/deployment;
- Agent/MCP integration.

## Canonical Day15 E2E

The current default main path is:

```text
Query
  |
  +--> Document
  |      BM25 Top-5
  |        +
  |      BGE-M3 Top-5
  |        ↓
  |      candidate union
  |        ↓
  |      BGE reranker
  |
  +--> Robot
         signal/state-action retrieval
             |
             v
       fixed 3 document + 2 robot evidence
             |
             v
      UnifiedEvidenceBundle
             |
             v
      Qwen3-VL grounded generation
             |
             v
      compact citation IDs
             |
             v
      deterministic EvidenceRef resolution
             |
             v
      citation / required-fact validation
             |
             v
      pipeline failure diagnosis
```

The canonical smoke supports `--document-mode bm25` and
`--document-mode hybrid` so the frozen Day12 BM25 baseline remains comparable
with the Day15 hybrid+rereanked document path.

### ColQwen status

ColQwen2.5 visual retrieval is a real validated component from the earlier
document branch. It is not silently claimed as part of the current canonical
cross-domain path. Visual retrieval will only be added to that path after a
controlled integration/evaluation step.

## Robot source

The canonical robot source is the original synchronized operation sequence:

```text
metadata.json
samples.csv
front/*.jpg
wrist/*.jpg
```

Evidence retains:

- `episode_id`;
- `frame_index`;
- canonical relative timestamp;
- camera-specific source timestamp / age;
- front/wrist image hashes;
- observation;
- action;
- tracking error.

MP4 is not required for canonical evidence.

## Traceability

Document evidence is traceable to PDF page number.

Robot evidence is traceable to frame, timestamp, camera, and bound episode.

Generation uses compact citation IDs and deterministic resolution to the frozen
`EvidenceRef` contract rather than asking the VLM to reproduce complex
references reliably.

## Ranking explainability

The document ranking trace records:

- BM25 rank/score;
- BGE-M3 rank/cosine similarity;
- sparse+dense candidate-union provenance;
- cross-encoder reranker raw logit;
- final rank.

The robot ranking trace records the selected query profile, signal score, frame,
timestamp, and deterministic tie behavior.

## Failure diagnosis

Current failure diagnosis is for the EvidenceMM pipeline:

- retrieval miss;
- missing document/robot/required evidence;
- hallucinated/out-of-bundle citation;
- duplicate citation;
- citation gap;
- incomplete generation;
- false abstention;
- over-answering.

This is not a physical robot root-cause classifier.

## Evaluation scale

Current results are smoke-scale and intentionally preserve negative or neutral
results.

Examples:

- the early BM25 document baseline has a verified rank-2 gold case;
- visual/state-action temporal heuristics did not automatically beat simpler
  temporal coverage baselines;
- on the tiny Day13 document smoke set, BM25 and BGE-M3 had identical aggregate
  retrieval metrics, while the cross-encoder reranker improved top-rank quality;
- Day14 fault injection validates pipeline diagnosis but is not a set of real
  robot failures.

Do not report these results as benchmark-scale model quality.

## Flagship target

The remaining flagship task is real robot-operation failure diagnosis:

```text
failed episode
+
front/wrist evidence
+
state/action trajectory
+
manual evidence
        ↓
failure interval / hypothesis
        ↓
supporting evidence + counterevidence
        ↓
confidence / abstention
```

That capability will only be claimed after real failed episodes and held-out
annotations exist.

## Project boundary

```text
chat-api   = LLM / RAG gateway
agent-api  = Agent reasoning / orchestration
EvidenceMM = multimodal evidence retrieval / grounding / diagnosis
SO-ARM101  = robot data / imitation-learning validation
```

EvidenceMM deliberately does not duplicate LangGraph, MCP, memory, or planning
from `agent-api`.
