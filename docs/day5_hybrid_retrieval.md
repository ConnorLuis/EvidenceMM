# EvidenceMM Day 5 - Hybrid RRF Retrieval

## Goal

Day 5 fuses the two frozen retrieval baselines without changing either
underlying retriever.

```text
query
  ├── BM25 text-only ranking
  └── ColQwen2.5 vision-only ranking
                ↓
        Reciprocal Rank Fusion
                ↓
         hybrid page ranking
```

## Why RRF

BM25 and ColQwen2.5 produce scores on unrelated numerical scales. Directly
adding or averaging those raw scores would be arbitrary.

Day 5 therefore fuses ranks rather than raw scores.

For a page with rank `r` in a retrieval channel, its contribution is:

`weight / (rrf_k + r)`

The baseline uses:

- `rrf_k = 60`
- text weight = `1.0`
- vision weight = `1.0`
- top-k = `5`

No parameter tuning is performed against the two verified queries.

## Frozen component baselines

Day 3 BM25:

| Metric | Value |
| --- | ---: |
| Recall@1 | 0.5000 |
| Recall@3 | 1.0000 |
| Recall@5 | 1.0000 |
| MRR@5 | 0.7500 |
| nDCG@5 | 0.8155 |

Day 4 ColQwen2.5:

| Metric | Value |
| --- | ---: |
| Recall@1 | 1.0000 |
| Recall@3 | 1.0000 |
| Recall@5 | 1.0000 |
| MRR@5 | 1.0000 |
| nDCG@5 | 1.0000 |

The hybrid experiment reuses the exact same two human-verified PDF queries and
page-level gold evidence.

## Interpretation constraint

Because the current evaluation set contains only one 8-page datasheet and two
verified PDF queries, Day 5 remains a pipeline smoke baseline.

If RRF also achieves perfect metrics, this is not evidence that fusion is
universally better than vision-only retrieval. The useful observation is
whether the fused ranking preserves or improves the complementary evidence
from the two frozen channels.

## Explicitly out of scope

- dense text embeddings
- Qdrant
- learned fusion
- reranking
- query rewriting
- Qwen3-VL answer generation
- video retrieval
- FastAPI
- LangGraph
- LoRA

## Observed Day 5 baseline

The Day 5 hybrid baseline was evaluated on the same 8-page STS3215
datasheet and the same two human-verified PDF queries used by the frozen
Day 3 and Day 4 baselines.

Configuration:

- RRF constant: `k = 60`
- text weight: `1.0`
- vision weight: `1.0`
- component top-k: `5`
- fused top-k: `5`

No RRF parameter tuning, modality-weight tuning, query translation, query
rewriting, or manual relevance boosting was performed.

### Three-way comparison

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 text-only | 0.5000 | 1.0000 | 1.0000 | 0.7500 | 0.8155 |
| ColQwen2.5 vision-only | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| RRF hybrid | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The hybrid baseline does not improve the aggregate metrics over the already
perfect vision-only result on this tiny smoke set. Its useful result is that
it preserves both vision-only Top-1 hits while combining independent lexical
and visual rankings without requiring comparable raw score scales.

### Case-level fusion analysis

For `d2_pdf_001`, gold page 3 is rank 1 under both retrieval channels:

- text rank: 1
- vision rank: 1
- hybrid rank: 1
- RRF score: approximately 0.032787

This is a strong agreement case.

For `d2_pdf_002`, gold page 4 receives complementary support:

- BM25: page 4 rank 2
- ColQwen2.5: page 4 rank 1
- RRF: page 4 rank 1
- RRF score: approximately 0.032522

The lexical false positive, page 8, has:

- BM25 rank 1
- ColQwen2.5 rank 5
- RRF rank 2
- RRF score: approximately 0.031778

The result illustrates the intended fusion behavior: a page supported strongly
by both channels can outrank a page that is strongly supported by only one
channel.

### Deterministic tie-breaking

RRF can produce exactly equal fusion scores. For example, in
`d2_pdf_001`, pages 4 and 6 both receive approximately `0.031514`.

Equal RRF scores are resolved deterministically by:

1. lower text rank,
2. lower vision rank,
3. source ID,
4. page number.

This rule is for reproducibility only and is not interpreted as additional
relevance evidence.

### Performance note

The two-query hybrid evaluation took approximately 0.319 seconds after model
loading on the development RTX 4070 SUPER.

The first query includes GPU warm-up effects, so these measurements are
development diagnostics rather than final latency benchmarks. Formal P50/P95,
throughput, and peak-memory measurements remain reserved for the later RTX
4090 benchmark.

### Scope limitation

The evaluation still contains only one 8-page datasheet and two verified PDF
queries. Perfect hybrid metrics on this smoke set must not be reported as a
general retrieval-quality result.
