# EvidenceMM Day 12 - Retrieval Ranking & Budget Contract

## Purpose

Gate A.1 froze the public retrieval boundary:

```text
query -> EvidenceRetriever -> UnifiedEvidenceBundle
```

Gate A.2 freezes the internal ranking and cross-domain budget semantics before
any BM25 or robot-search implementation is added.

## Frozen baseline budget

The Day 12 cross-domain baseline uses:

```text
total_top_k = 5
document_quota = 3
robot_quota = 2
```

The quota values are source-family caps. If a retriever has fewer valid
candidates, the composer returns fewer than five items. It does not duplicate
or fabricate filler evidence.

The baseline requires both source families to return at least one candidate
because Day 12 evaluates cross-domain retrieval.

## Why fixed quota instead of raw-score fusion

Document and robot retrievers do not share a calibrated score space.

For example:

```text
BM25 raw score:              8.7
robot state/metadata score:  0.81
```

Those values are not directly comparable.

Therefore Day 12 explicitly forbids sorting document and robot candidates by a
shared raw-score column.

Each retriever owns its within-domain ranking. The cross-domain composer keeps
those ranks and applies the fixed 3+2 quota.

## Candidate contract

Every ranked candidate records:

```text
domain
retriever_name
rank
raw_score
UnifiedEvidenceItem
```

`raw_score` is diagnostic metadata only.

The authoritative baseline ordering is `rank`, which must be:

```text
1, 2, 3, ...
```

with no gaps and in list order.

A ranked list must come from exactly one retriever and must not contain
duplicate evidence IDs.

## Deterministic composition

The final Day 12 bundle order is:

```text
document rank 1
document rank 2
document rank 3
robot rank 1
robot rank 2
```

subject to candidate availability.

This order is deterministic and does not imply that document evidence is
globally more relevant than robot evidence.

## Retrieval trace vs generation contract

Ranking metadata is intentionally kept outside `UnifiedEvidenceItem`.

The generator continues to receive only the frozen Day 11 evidence contract.

The retrieval/evaluation layer may log:

```text
retriever_name
rank
raw_score
domain
evidence_id
```

without polluting evidence provenance or the generation schema.

## Existing RRF boundary

EvidenceMM already has RRF for the Day 5 document-only text/vision hybrid
retriever.

Day 12 does not reuse that RRF to merge document and robot raw scores. These are
different source families and different retrieval semantics.

A future cross-domain fusion experiment may be added as a separate controlled
comparison after the fixed-quota baseline is measured.

## Anti-tuning rules

Before the real Day 12 retrieval evaluation:

- total budget remains 5;
- document quota remains 3;
- robot quota remains 2;
- final composition uses within-domain rank only;
- no cross-domain score normalization;
- no learned fusion;
- no RRF across document and robot candidates;
- no reranker;
- no LLM query routing or rewriting.

The values are not changed after seeing retrieval results.

## Gate A.2 non-claims

Passing this gate does not mean search is implemented.

It only fixes how future document and robot candidate lists will be represented
and composed into the already-frozen `UnifiedEvidenceBundle`.

The next gate may implement the first real document candidate generator and the
first real robot candidate generator under these rules.
