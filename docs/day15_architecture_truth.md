# Day15 Architecture Truth

## Current canonical E2E

The current canonical EvidenceMM main path is:

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
  |      BGE cross-encoder reranker
  |
  +--> Robot
         frozen query-conditioned signal retriever
         front/wrist + observation/action/timestamp
             |
             v
       fixed 3 + 2 evidence budget
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
      citation + fact validation
             |
             v
      pipeline failure diagnosis
```

The Day15 canonical smoke also supports `--document-mode bm25` so the frozen
Day12 BM25 cross-domain baseline remains directly comparable.

## Validated but not canonical

These modules are real and remain useful, but they are not currently claimed
as part of the canonical Day15 cross-domain path:

- ColQwen2.5 visual page retrieval and Day5 BM25+visual RRF;
- Day7-10 temporal selector micro-baselines;
- the small Day15 generic RRF `EvidenceCandidate` scaffold.

In particular, visual retrieval is not described as active in the canonical
pipeline until it is deliberately integrated and reevaluated.

## Pipeline diagnosis versus robot diagnosis

Day14 diagnoses EvidenceMM system failures such as retrieval miss, missing
evidence, citation violations, incomplete answers, and abstention errors.

It does not diagnose why a physical grasp failed.

The flagship robot-failure scenario still requires real failed episodes and
held-out failure labels.

## Current evaluation scale

Current evidence remains smoke-scale:

- one bound 8-page STS3215 PDF;
- a very small verified document query set;
- one canonical SO-ARM101 operation episode;
- a small number of verified temporal events;
- single-case cross-domain grounded-generation smoke.

Perfect smoke metrics must not be presented as general model quality.

## Next priority after Day15

Stop horizontal module expansion.

The next flagship work is:

1. bind real successful and failed robot episodes;
2. freeze failure intervals / labels / uncertainty;
3. build held-out cross-domain diagnosis cases;
4. evaluate evidence retrieval, citation, abstention, and failure-cause quality;
5. only then expose a robot-diagnosis capability to an external Agent.
