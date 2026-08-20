# EvidenceMM Day 4 - Vision-Only Retrieval

## Gate A

Day 4 first validates the visual retrieval stack before building the full
8-page index.

Pipeline:

```text
bound PDF + SourceManifest SHA256
        ↓
144-DPI page rendering
        ↓
page-image SHA256 manifest
        ↓
ColQwen2.5 model load
        ↓
one real page image embedding
        ↓
one real query embedding
        ↓
MaxSim similarity smoke test
```

Model:

- `vidore/colqwen2.5-v0.2`
- `colpali-engine` pinned to upstream commit `c23838d920a7c426ee297034211cff2f55da65dc`
- bfloat16
- maximum 768 visual tokens
- CUDA inference

Generated page images and later visual embeddings stay under
`data/processed/` and are not committed.

The local RTX 4070 SUPER is a development/smoke device. Formal performance
numbers will later be collected on the RTX 4090.

## Gate B

Only after Gate A passes:

- embed all 8 pages with batch size 1
- move each page multi-vector embedding to CPU
- evaluate the same two verified PDF cases as Day 3
- report Recall@1/3/5, MRR@5, nDCG@5
- compare against the frozen Day 3 BM25 baseline

No hybrid fusion, reranking, Qdrant, answer generation, or video work is part
of Day 4.

## Observed Day 4 baseline

The final Day 4 vision-only baseline was evaluated on the same 8-page
STS3215 datasheet and the same two human-verified PDF queries used by
the frozen Day 3 BM25 baseline.

### Environment

- model: `vidore/colqwen2.5-v0.2`
- torch: `2.11.0+cu130`
- transformers: `5.15.0`
- peft: `0.19.1`
- GPU: NVIDIA GeForce RTX 4070 SUPER, 12 GB
- render DPI: 144
- maximum visual tokens: 768

The ColPali engine is pinned to upstream commit:

`c23838d920a7c426ee297034211cff2f55da65dc`

This commit fixes the Transformers 5.x checkpoint conversion for
`ColQwen2_5`, specifically mapping `model.embed_tokens` and `model.norm`
to the `language_model.*` layout. The environment gate verifies these
mappings before model inference.

### Retrieval results

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 text-only | 0.5000 | 1.0000 | 1.0000 | 0.7500 | 0.8155 |
| ColQwen2.5 vision-only | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Per-case rankings:

- `d2_pdf_001`: gold page 3 was rank 1 under both BM25 and ColQwen2.5.
- `d2_pdf_002`: gold page 4 improved from BM25 rank 2 to ColQwen2.5 rank 1.

For `d2_pdf_002`, the lexical baseline ranked page 8 first because generic
parameter-related terms created strong lexical overlap. The vision-only
retriever instead ranked the page containing the `7-10 Feedback` table row
first.

No query translation, prompt rewriting, relevance feedback, parameter tuning,
or manual score boosting was applied after observing the Day 3 baseline.

### Development-machine performance

The 8-page visual index was built successfully with page embeddings of shape
`[747, 128]`.

Observed development-machine measurements:

- model load during index build: approximately 18.97 s
- total 8-page index encoding: approximately 2.75 s
- warm page encoding after first-page initialization: approximately 0.21 s/page
- index-build peak allocated GPU memory: approximately 7.67 GB
- two-query evaluation time after model load: approximately 0.63 s
- evaluation peak allocated GPU memory: approximately 7.29 GB

The first page/query includes CUDA and kernel warm-up effects, so it is not
used as steady-state latency.

These RTX 4070 SUPER measurements are development diagnostics only. Formal
latency, throughput, P50/P95, and peak-VRAM measurements are reserved for the
later RTX 4090 benchmark.

### Scope warning

The result is a pipeline smoke baseline over one 8-page document and two
verified queries. A perfect score on this tiny set is not treated as a
headline retrieval-quality claim.
