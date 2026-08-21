# Day15 Canonical E2E Consolidation

## Purpose

Day15 consolidates previously validated EvidenceMM modules into one explicit
main path rather than adding another independent capability.

The canonical document path is now:

```text
BM25 Top-5
       \
        -> candidate union -> BGE reranker -> final document ranking
       /
BGE-M3 Top-5
```

The canonical cross-domain path is:

```text
hybrid+rereanked document candidates
+
frozen robot signal candidates
        ↓
fixed 3 document + 2 robot evidence budget
        ↓
UnifiedEvidenceBundle
        ↓
Qwen3-VL grounded generation
        ↓
compact citation IDs
        ↓
deterministic EvidenceRef resolution
        ↓
citation / required-fact validation
        ↓
Day14 pipeline diagnosis
```

## Baseline switch

The Day15 smoke supports:

```text
--document-mode bm25
--document-mode hybrid
```

`bm25` preserves the Day12 document baseline.

`hybrid` is the new default and reuses the frozen Day13
BM25 + BGE-M3 candidate union + BGE reranker stack.

This allows baseline/hybrid comparison without changing Day11-Day14 frozen
contracts.

## ColQwen status

ColQwen2.5 visual page retrieval remains a real validated EvidenceMM component
from Days 4-6, but it is not silently claimed as part of the Day15 canonical
cross-domain path.

That separation is deliberate. The project will only add visual retrieval to
the canonical document path after controlled evaluation shows the integration
is justified.

## Ranking trace

Hybrid mode records:

- BM25 Top-5 ranks and scores;
- BGE-M3 Top-5 ranks and cosine similarities;
- union provenance;
- cross-encoder reranker ranks and raw logits;
- final document candidates entering the 3+2 evidence budget.

Cross-domain document and robot raw scores are never directly compared.

## Failure diagnosis

Day14 pipeline diagnosis is applied after canonical generation to verify that
the smoke answer has no detected retrieval, evidence-contract, citation, fact
coverage, or abstention failure.

This is not robot-operation root-cause diagnosis.

## Non-claims

Day15 does not claim:

- benchmark-scale hybrid retrieval gains;
- ColQwen is in the canonical path;
- natural-language semantic robot-event retrieval;
- robot failed-grasp root-cause diagnosis;
- Agent or MCP integration.
