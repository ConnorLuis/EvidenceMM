# EvidenceMM Day 13 - Dense Retrieval Baseline

## Gate A.1 purpose

Day 13 starts the document retrieval improvement phase.

This gate introduces a text-semantic dense retriever and compares it with the
existing BM25 document retriever on the frozen Day 12 two-case cross-domain
queries.

It does not yet perform sparse+dense fusion or reranking.

## Existing retrieval that remains frozen

EvidenceMM already contains two older document retrieval branches:

```text
Day 3: BM25 page retrieval
Day 4: ColQwen2.5 visual page retrieval
Day 5: BM25 + ColQwen2.5 RRF
```

Day 13 does not relabel ColQwen2.5 as a text-dense retriever and does not modify
the Day 5 RRF baseline.

The new dense branch operates on extracted PDF page text.

## Dense model

The baseline model is:

```text
BAAI/bge-m3
```

The implementation uses the existing Transformers dependency and the model's
CLS token representation, followed by L2 normalization and cosine similarity.

No new Python dependency is required.

## Frozen Gate A.1 settings

```text
top_k = 5
max_length = 2048
batch_size = 4
similarity = cosine
pooling = CLS
```

The same page corpus and same two Day 12 document labels are used for BM25 and
dense retrieval.

## Evaluation

For both BM25 and dense retrieval the report records:

```text
Hit@1
Hit@3
Hit@5
MRR@5
first relevant rank per case
full top-5 ranking trace
```

Gold labels are used only after retrieval outputs have been produced.

## Why no improvement claim yet

The current verified document evaluation contains only two cases, and the
existing BM25 baseline already has Hit@3=1.0 and Hit@5=1.0 on those cases.

Therefore candidate recall cannot meaningfully demonstrate improvement on this
smoke set.

Gate A.1 only establishes the dense baseline and preserves its result whether it
is better, equal, or worse.

A broader frozen document evaluation set is required before EvidenceMM claims
candidate-recall improvement.

## Planned Day 13 continuation

After Gate A.1:

```text
A.2 sparse+dense candidate union
B.1 cross-encoder reranker
B.2 broader candidate-recall evaluation
C.1 evidence ranking trace analysis
```

Robot retrieval remains frozen during the document hybrid work.

Failure diagnosis and Agent tools are explicitly later phases.

## Non-claims

This gate does not claim:

- candidate recall improvement;
- hybrid fusion;
- reranking;
- replacement of ColQwen2.5;
- robot event retrieval improvement;
- failure diagnosis;
- Agent or MCP integration.
