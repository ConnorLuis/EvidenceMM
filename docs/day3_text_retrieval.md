# EvidenceMM Day 3 - PDF Standardization and Text-Only Retrieval

## Goal

Day 3 isolates the lexical retrieval baseline before any visual retrieval,
dense retrieval, reranking, or generation is added.

Pipeline:

```text
bound PDF + SHA256 manifest
        ↓
page-level text extraction
        ↓
Unicode/whitespace normalization
        ↓
1-based PageDocument JSONL
        ↓
mixed Chinese/English tokenization
        ↓
BM25 page retrieval
        ↓
verified page-level evaluation
```

## Corpus contract

Each standardized page contains:

- `source_id`
- `page_number` (1-based)
- normalized `text`
- `text_sha256`
- `char_count`

The generated corpus is stored under `data/processed/` and remains ignored by
Git. The source manifest and evaluation gold remain versioned.

Before extraction, the PDF bytes are re-hashed and must match the bound
`SourceManifest.sha256`. This prevents silent source drift.

## Tokenization

The Day 3 lexical baseline deliberately avoids a language-specific tokenizer.

- English/alphanumeric terms are lowercased and tokenized as words.
- Chinese runs contribute both single characters and character bigrams.
- Input text is NFKC-normalized.

This is a deterministic baseline, not the final retrieval strategy.

## Evaluation

Day 3 evaluates only verified PDF cases from the existing Day 2 set.

Metrics:

- Recall@1
- Recall@3
- Recall@5
- MRR@5
- nDCG@5

The initial corpus is only one 8-page STS3215 datasheet with two verified PDF
queries. Results are therefore a pipeline-correctness smoke baseline, not a
headline retrieval-quality claim.

## Explicitly out of scope

- dense embedding
- ColQwen / ColPali
- Qdrant
- RRF
- reranking
- Qwen3-VL answer generation
- OCR fallback
- LangGraph

## Observed Day 3 baseline

The initial reproducible baseline was evaluated on the bound 8-page
STS3215 datasheet and the two human-verified PDF cases from Day 2.

Results:

| Metric | Value |
| --- | ---: |
| Recall@1 | 0.5000 |
| Recall@3 | 1.0000 |
| Recall@5 | 1.0000 |
| MRR@5 | 0.7500 |
| nDCG@5 | 0.8155 |

Per-case result:

- `d2_pdf_001`: gold page 3 retrieved at rank 1.
- `d2_pdf_002`: gold page 4 retrieved at rank 2; page 8 was ranked first.

### Baseline error analysis

For `d2_pdf_002`, the query asks for feedback-state parameters. The gold
evidence is the `7-10 Feedback` row on page 4. The lexical baseline instead
ranks page 8 first.

This is a useful baseline failure rather than a tuning target. Page 8 contains
the phrase `Control signal parameter`, which lexically overlaps with the
query term for parameters. It is also substantially shorter than page 4, so
BM25 document-length normalization can amplify these matching terms.

The page-level corpus additionally retains repeated document boilerplate such
as model name and specification headers. BM25 has no table semantics or
semantic understanding that `Feedback` is more relevant than a generic
occurrence of `parameter`.

No query rewriting, parameter tuning, manual boosting, or relevance feedback
was applied after observing this result. The failure is intentionally
preserved as the text-only baseline for later comparison with semantic,
visual, and hybrid retrieval.
