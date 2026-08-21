# EvidenceMM Day 12 - Retrieval to Grounded Generation

## Gate B.2.2 purpose

This gate closes the Day 12 end-to-end path:

```text
query
-> real document retrieval
-> real robot signal retrieval
-> frozen 3+2 composition
-> UnifiedEvidenceBundle
-> Qwen3-VL
-> UnifiedGroundedAnswer
-> EvidenceRef validation
```

No frozen retrieval algorithm is modified.

## Frozen query

The smoke question asks for two directly supported evidence families:

1. the STS3215 typical operating voltages;
2. metadata for the two robot samples selected by the existing
   `robot gripper action` retrieval baseline.

The exact canonical terms `gripper` and `action` are retained so the robot
retriever does not require synonym injection or an LLM query rewrite.

## Retrieval

The pipeline reuses:

```text
DocumentBM25CandidateRetriever
RobotSignalCandidateRetriever
compose_fixed_quota
```

with the already-frozen budget:

```text
total = 5
document = 3
robot = 2
```

No document/robot raw-score fusion is added.

## Generation

The retrieved bundle is passed directly to the frozen Day 11
`build_unified_messages()` function and then to:

```text
Qwen/Qwen3-VL-4B-Instruct
```

The generation stage therefore receives the same Day 11 evidence contract, but
the evidence items are now produced by Day 12 retrieval instead of a fixed
hard-coded page/frame fixture.

For a full 3+2 bundle the expected visual input count is:

```text
3 document page images
+ 2 robot samples x 2 cameras
= 7 images
```

## Validation

The smoke requires:

```text
expected STS3215 page 3 is present in the retrieved bundle
structured JSON output
non-abstaining answer
generic citation policy validity
required citation coverage
required fact coverage
```

Required citations are:

```text
page 3 PDF EvidenceRef
robot sample 1 front EvidenceRef
robot sample 1 wrist EvidenceRef
robot sample 2 front EvidenceRef
robot sample 2 wrist EvidenceRef
```

The two robot frame indices and timestamps are derived dynamically from the
actual retrieved bundle. They are not robot-event gold.

## Gold separation

The expected document page is an evaluation fixture used only after retrieval.

It is never passed to either retriever or to the cross-domain composer.

The robot branch still uses no human event labels.

## Non-claims

Passing this gate supports one end-to-end retrieval-to-generation smoke case.

It does not establish benchmark-scale retrieval or generation quality,
natural-language robot event recognition, temporal diversity optimization,
failure diagnosis, causal reasoning, Agent orchestration, or MCP integration.
