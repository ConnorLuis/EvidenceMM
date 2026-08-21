# Day15 Hybrid Retrieval

## Objective

Connect sparse and dense retrieval into the canonical evidence retrieval layer.

Pipeline:

Query
-> BM25
-> Dense Retrieval
-> Reciprocal Rank Fusion
-> EvidenceCandidate
-> UnifiedEvidenceBundle

## Design constraints

- Existing Day11-Day14 contracts remain unchanged.
- Retrieval output uses EvidenceCandidate.
- No generation logic is modified.
