# EvidenceMM Day 13 - Sparse + Dense Hybrid Retrieval and Reranking

## Purpose

Day 13 completes the document retrieval improvement stack started by the
frozen dense baseline:

```text
BM25 ------------------\
                        -> candidate union -> cross-encoder reranker
BGE-M3 dense ----------/
```

The robot retriever is not modified.

## Candidate union

Each branch retrieves Top-5 pages independently.

Candidates are deduplicated by:

```text
(source_id, page_number)
```

The union records the original BM25 rank/score and dense rank/score for every
page.

The union's `pool_index` is deterministic lexical ordering only. It is
explicitly not a relevance score and no BM25/dense raw-score addition is used.

## Cross-encoder reranker

The reranker is:

```text
BAAI/bge-reranker-v2-m3
```

It receives each `(query, page_text)` pair and returns a raw relevance logit.
Higher logits rank first.

The raw logit is not presented as a calibrated probability.

## Evaluation

The same frozen two Day 12 document cases are used.

Metrics:

```text
BM25:     Hit@1 / Hit@3 / Hit@5 / MRR@5
Dense:    Hit@1 / Hit@3 / Hit@5 / MRR@5
Union:    candidate coverage + pool size + branch overlap
Reranker: Hit@1 / Hit@3 / Hit@5 / MRR@5
```

The gold page labels are read only after retrieval, union construction, and
reranking outputs have been produced.

## Candidate recall limitation

The current two-case smoke set already has BM25 Hit@5=1.0.

Therefore the union can demonstrate complementary candidate composition, but
it cannot establish a benchmark-scale candidate-recall improvement claim.

The union is also variable-sized and can exceed five pages, so it is not used
for a same-budget recall claim.

## Boundaries

Day 13 does not modify:

- ColQwen2.5 visual retrieval;
- robot signal retrieval;
- the Day 12 3+2 cross-domain evidence budget;
- grounded generation;
- failure diagnosis;
- Agent or MCP integration.

<!-- DAY13_OBSERVED_START -->
## Observed Day 13 result

```text
BM25:     {'hit_at_1': 0.5, 'hit_at_3': 1.0, 'hit_at_5': 1.0, 'mrr_at_5': 0.75}
Dense:    {'hit_at_1': 0.5, 'hit_at_3': 1.0, 'hit_at_5': 1.0, 'mrr_at_5': 0.75}
Union:    {'candidate_recall': 1.0, 'average_pool_size': 5.5, 'average_branch_overlap': 4.5, 'bm25_hit_at_5_reference': 1.0, 'coverage_gain_over_bm25_top5': 0.0, 'same_budget_recall_claim': False}
Reranker: {'hit_at_1': 1.0, 'hit_at_3': 1.0, 'hit_at_5': 1.0, 'mrr_at_5': 1.0}
Delta reranker - BM25:
{'hit_at_1': 0.5, 'hit_at_3': 0.0, 'hit_at_5': 0.0, 'mrr_at_5': 0.25}
```

This remains a 2-case smoke result.
The union candidate pool is variable-sized, so no same-budget recall
improvement claim is made.

Per-case first relevant ranks:

```text
d12_ret_001: BM25=1, Dense=1, UnionContainsGold=True, Reranker=1
d12_ret_002: BM25=2, Dense=2, UnionContainsGold=True, Reranker=1
```
<!-- DAY13_OBSERVED_END -->
