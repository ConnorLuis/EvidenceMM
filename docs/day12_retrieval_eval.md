# EvidenceMM Day 12 - Retrieval Evaluation

## Gate B.2.1 purpose

This gate evaluates the frozen Day 12 retrieval baseline before any retrieved
bundle is connected to the generator.

It does not modify the Gate B.1 retrievers, ranking rule, or 3+2 budget.

## Frozen evaluation fixture

Two existing document-grounded questions are reused as document labels:

```text
case 1: STS3215 typical operating voltage -> page 3
case 2: STS3215 feedback status parameters -> page 4
```

Each query also contains an explicit canonical robot clause:

```text
robot gripper action
robot shoulder_pan action
```

Robot clauses are evaluated only for exact parser/profile correctness and
evidence binding. There is no robot event gold and no claim of natural-language
event recognition.

## Separation of retrieval and evaluation labels

The evaluator loads document gold pages only after candidate retrieval has been
executed.

The gold page labels are never passed to:

```text
DocumentBM25CandidateRetriever
RobotSignalCandidateRetriever
compose_fixed_quota
```

The report therefore distinguishes:

```text
evaluation_labels_loaded = true
gold_read_by_retriever = false
robot_event_gold_used = false
```

## Metrics

Document retrieval reports:

```text
Hit@1
Hit@3
Hit@5
MRR
```

Robot integration reports:

```text
exact query-profile match
front/wrist evidence validity
state/action frame/timestamp consistency
```

Cross-domain integration reports:

```text
3 document + 2 robot selected items
UnifiedEvidenceBundle validation
```

The evaluation script returns failure only for structural/protocol violations.
A weak document retrieval metric is recorded as a baseline result and is not
post-hoc tuned.

## Non-claims

This is a two-case smoke evaluation, not a benchmark.

It does not evaluate robot event recognition, failed-grasp diagnosis,
cross-domain score calibration, reranking, Agent orchestration, or grounded
generation.

## Observed Gate B.2.1 result

The frozen two-case retrieval smoke evaluation completed deterministically.

- pytest: 105 passed
- deterministic: PASS
- document Hit@1: 0.5
- document Hit@3: 1.0
- document Hit@5: 1.0
- document MRR: 0.75
- robot query-profile accuracy: 1.0
- bundle structural valid rate: 1.0

Case d12_ret_001: document page 3 ranked 1; robot profile gripper/action; structural validation passed.

Case d12_ret_002: document page 4 ranked 2; robot profile shoulder_pan/action; structural validation passed.

The retrievers did not receive document gold labels and no robot-event gold was used. Results are preserved without post-hoc tuning. This remains a two-case smoke evaluation, not benchmark-scale retrieval performance.
