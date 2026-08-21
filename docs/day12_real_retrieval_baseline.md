# EvidenceMM Day 12 - Real Evidence Search Baseline

## Gate B.1 goal

Gate A froze the retrieval contract, total budget 5, document quota 3, robot
quota 2, and no cross-domain raw-score fusion.

Gate B.1 adds the first real candidate generators while keeping generation,
gold labels, failure diagnosis, and Agent orchestration out of the run.

## Document baseline

The document generator reuses the frozen page-level BM25 implementation over the
real standardized STS3215 PDF pages.

Each candidate binds the real PDF source SHA256, page number, normalized text
SHA256, extracted page text, rendered page image path and image SHA256, and the
PDF `EvidenceRef`.

The BM25 score is retrieval trace metadata only.

## Robot baseline

The robot baseline uses only the canonical real episode data:

```text
samples.csv observation
samples.csv action
samples.csv tracking_error
front/wrist FrameRecord
```

The query parser recognizes only exact canonical identifiers.

Joints:

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
gripper
```

Signals:

```text
observation
action
tracking_error
```

There is no synonym injection, translation, LLM rewrite, event-label lookup, or
Day 7 temporal-gold access.

If no canonical joint is present, all six joints are used. If no canonical
signal is present, observation and action changes are used.

For the selected joint set:

```text
observation score = RMS adjacent observation delta
action score      = RMS adjacent action delta
tracking_error    = RMS current tracking error
```

If multiple signals are present, their scores are fused by `max`.

Frames rank by score descending, then lower frame index. There is no temporal
NMS or diversity heuristic.

## Frozen smoke query

```text
STS3215 typical operating voltage; robot gripper action
```

The document phrase is an ordinary technical retrieval phrase. The robot clause
uses exact canonical identifiers, so the first baseline does not hide a synonym
dictionary.

This is an integration smoke query, not a benchmark question.

## Candidate and output budget

Each domain produces five candidates:

```text
candidate_pool_k = 5
```

The frozen composer then keeps:

```text
document rank 1..3
robot rank 1..2
```

Document and robot raw scores are never compared.

## Gate B.1 acceptance

The real smoke must show:

```text
gold_read = false
generation_called = false
failure_diagnosis_attempted = false
agent_used = false

candidate_pool_k_per_domain = 5

bundle.item_count = 5
bundle.document_item_count = 3
bundle.robot_item_count = 2

cross_domain_score_fusion = false
query_rewrite = false
```

The smoke is repeated byte-for-byte to verify determinism.

## Non-claims

Passing Gate B.1 does not mean natural-language robot semantic retrieval is
solved, robot events are recognized from text, retrieval quality has been
benchmarked, grounded generation has consumed retrieved evidence, or failure
diagnosis exists.

The next gate can freeze a small evaluation set and connect retrieved evidence
to the existing Day 11 grounded-generation layer.

## Observed Gate B.1 smoke result

The first real retrieval smoke passed the frozen protocol.

```text
pytest = 99 passed
smoke exit_code = 0
deterministic output = PASS
gold_read = false
generation_called = false

candidate pool per domain = 5

selected bundle:
  document items = 3
  robot items = 2
  total items = 5
