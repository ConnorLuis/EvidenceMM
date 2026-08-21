# EvidenceMM Day 13 - Evidence Ranking Analysis

## Purpose

This report makes the frozen retrieval decisions inspectable.

It answers two concrete questions:

```text
Why does the target STS3215 page rank where it does?
Why are robot frames such as 155/156 selected?
```

## Document trace

For the selected document case the trace records:

```text
BM25 rank and exact term-level score contributions
BGE-M3 dense cosine similarity and rank
candidate-union provenance
cross-encoder raw reranker logit and final rank
```

BM25 term contributions reproduce the frozen BM25 formula exactly.

Dense cosine similarity and reranker logits are reported as model scores only;
the system does not claim token-level attribution for either neural model.

## Robot trace

The robot trace reuses the frozen Day 12 signal retriever.

For the canonical query:

```text
robot gripper action
```

the parser selects:

```text
joint = gripper
signal = action
```

The ranking analysis computes all episode scores, counts ties at the maximum,
and records the already-frozen rule:

```text
raw_score descending
frame_index ascending on ties
```

This explains deterministic selection under the robot quota without introducing
temporal NMS, diversity heuristics, event labels, or semantic event retrieval.

## Boundary

Ranking explanation is not failure diagnosis. It describes why evidence was
ranked and selected under the existing scoring rules; it does not infer whether
a robot operation succeeded or failed or why an outcome occurred.

<!-- DAY13_OBSERVED_START -->
## Observed ranking trace

Document case `d12_ret_001`, target page
`3`:

```text
BM25 rank: 1
Dense rank: 1
Reranker rank: 1
```

Robot query `robot gripper action`:

```text
profile: {'joints': ['gripper'], 'signals': ['action'], 'explicit_joint_terms': True, 'explicit_signal_terms': True}
top_score: 1.6666666666666679
exact_top_score_tie_count: 14
exact_top_score_tied_frames: [155, 156, 381, 382, 383, 384, 623, 624, 838, 847, 848, 863, 869, 870]
near_top_tolerance: 1e-12
near_top_score_count: 28
near_top_score_frames: [155, 156, 381, 382, 383, 384, 623, 624, 838, 847, 848, 863, 869, 870, 157, 158, 159, 252, 253, 369, 370, 371, 380, 630, 631, 835, 836, 837]
selected_frames: [155, 156]
selection_rule: raw_score descending; frame_index ascending only when raw_score values are exactly equal
```

The document trace explains exact BM25 term contributions, dense cosine
similarity, union provenance, and cross-encoder reranker rank. The robot trace
records the canonical signal profile and deterministic tie break. Neither trace
is a causal failure diagnosis.
<!-- DAY13_OBSERVED_END -->
