# EvidenceMM Day 14 - System Failure Diagnosis

## Purpose

Day 14 adds deterministic diagnosis for failures in the EvidenceMM pipeline:

```text
retrieval failure
evidence missing
generation grounding failure
```

This is system-pipeline diagnosis, not robot-operation outcome diagnosis.

The current canonical robot episode is not labeled as a failed grasp, so Day 14
does not invent a grasp-failure cause.

## Failure taxonomy

### Retrieval failure

`retrieval_miss`

The required evaluation page is absent within the evaluated retrieval cutoff.

Gold labels are used only by evaluation/diagnosis, never by the retriever.

### Evidence missing

`evidence_missing_document`

The cross-domain bundle contains no document-page evidence.

`evidence_missing_robot`

The cross-domain bundle contains no robot-sample evidence.

`evidence_missing_required_ref`

The bundle is structurally valid but lacks one or more task-required
`EvidenceRef` values.

### Generation failure

`generation_hallucinated_citation`

The model emits a citation outside the supplied evidence bundle.

This is the deterministic hallucination signal supported by the current
EvidenceMM contract.

It does not claim to detect every unsupported natural-language statement when
the citations themselves are valid.

`generation_duplicate_citation`

The generated citation list contains duplicates.

`generation_citation_gap`

The answer is non-abstaining but lacks required supporting citations.

`generation_incomplete`

The generated answer does not cover every frozen required-fact group.

`generation_false_abstention`

An answerable evaluation fixture is incorrectly refused.

`generation_overanswer`

An unanswerable evaluation fixture receives a non-abstaining answer.

## Real-bundle fault injection

The Day 14 evaluation first builds a normal real EvidenceMM bundle through the
frozen Day 12 document and robot retrievers.

It then creates controlled interface-level fault injections:

```text
remove gold page from observed retrieval ranking
remove required page from bundle
remove robot evidence from bundle
add an out-of-bundle citation
drop required citations
drop a required fact
force false abstention
```

The diagnostic system must return the frozen expected failure code for each
scenario.

The injected failures are clearly marked and are not presented as naturally
occurring model or robot failures.

## Relationship to Day 13

Day 13 answers:

```text
why did this evidence rank here?
```

Day 14 answers:

```text
which pipeline stage failed, and what contract was violated?
```

The two layers remain separate.

## Relationship to future Agent tools

The structured `FailureDiagnosisReport` can later be exposed behind an Agent
tool such as:

```text
explain_failure()
```

Day 14 itself does not add LangGraph, MCP, or Agent orchestration.

## Non-claims

Day 14 does not claim:

- failed-grasp root-cause diagnosis;
- semantic hallucination detection for all uncited prose;
- benchmark-scale failure-detection accuracy;
- Agent or MCP integration.

<!-- DAY14_OBSERVED_START -->
## Observed Day 14 result

```text
scenario_count = 8
all_expected_diagnoses_match = True
reference_document_pages = [3, 5, 8]
reference_robot_frames = [155, 156]
```

Scenario results:

```text
healthy_reference: expected=[] actual=[] match=True
retrieval_miss_injected: expected=['retrieval_miss'] actual=['retrieval_miss'] match=True
missing_required_evidence_injected: expected=['evidence_missing_required_ref'] actual=['evidence_missing_required_ref'] match=True
missing_robot_evidence_injected: expected=['evidence_missing_robot', 'evidence_missing_required_ref'] actual=['evidence_missing_required_ref', 'evidence_missing_robot'] match=True
hallucinated_citation_injected: expected=['generation_hallucinated_citation'] actual=['generation_hallucinated_citation'] match=True
citation_gap_injected: expected=['generation_citation_gap'] actual=['generation_citation_gap'] match=True
incomplete_generation_injected: expected=['generation_incomplete'] actual=['generation_incomplete'] match=True
false_abstention_injected: expected=['generation_false_abstention', 'generation_citation_gap', 'generation_incomplete'] actual=['generation_citation_gap', 'generation_false_abstention', 'generation_incomplete'] match=True
```

This is deterministic system-pipeline failure diagnosis over fault-injected variants of a real EvidenceMM evidence bundle.
It is not robot-operation outcome or failed-grasp cause diagnosis.
<!-- DAY14_OBSERVED_END -->
