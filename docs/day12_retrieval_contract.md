# EvidenceMM Day 12 - Retrieval Contract Gate A

## Goal

Day 12 begins the query-driven retrieval phase.

Day 11 already established:

```text
fixed real document evidence
+
fixed real robot evidence
    -> UnifiedEvidenceBundle
    -> grounded generation
    -> generic EvidenceRef validation
```

Day 12 changes only the evidence acquisition boundary:

```text
user query
    -> EvidenceRetriever
    -> UnifiedEvidenceBundle
    -> existing Day 11 grounded generation
```

Gate A intentionally defines only the retrieval contract and tests. It does
not implement BM25, dense retrieval, visual retrieval, robot retrieval, or
generation changes.

## Frozen boundary

The minimal production-facing protocol is:

```python
retrieve(
    query: str,
    top_k: int = 5,
) -> UnifiedEvidenceBundle
```

The retrieval implementation is therefore hidden from grounded generation.

A future implementation may use:

- BM25;
- dense embeddings;
- ColQwen;
- temporal indexing;
- metadata/state/action retrieval;
- Qdrant or another external index.

None of those implementation details are permitted to leak through this
boundary.

## Frozen Day 12 budget

For the Day 12 baseline:

```text
DEFAULT_TOP_K = 5
```

`top_k` means the maximum number of `UnifiedEvidenceItem` objects returned in
one bundle.

Gate A does not yet define modality-specific quota allocation inside those five
items. That decision belongs to the retrieval implementation gate and must be
frozen before real retrieval evaluation.

## Query normalization

The boundary applies only one query transformation:

```text
strip leading/trailing whitespace
```

It performs no query rewriting, expansion, translation, synonym injection,
LLM reformulation, or modality routing.

A blank query is rejected.

## Bundle invariants

A retriever call is valid only when:

1. the return value is a `UnifiedEvidenceBundle`;
2. `bundle.question` equals the normalized user query;
3. bundle item count is `<= top_k`;
4. all existing Day 11 `UnifiedEvidenceBundle` schema invariants still pass.

This preserves a single evidence contract between retrieval and generation.

## Layering rule

`src/evidencemm/retrieval.py` may depend on the Day 11 unified evidence schema.

It must not depend on:

```text
unified_grounding.py
grounded_generation.py
Qwen3-VL
Agent
MCP
```

The dependency direction stays:

```text
retrieval implementation
        |
        v
retrieval.py
        |
        v
UnifiedEvidenceBundle
        |
        v
unified_grounding.py
```

Grounded generation therefore remains unchanged while retrieval evolves.

## Gate A acceptance

Gate A passes when:

```text
DEFAULT_TOP_K = 5

query normalization deterministic
blank query rejected
invalid top_k rejected

EvidenceRetriever runtime protocol valid

retrieve_with_contract(...)
    -> UnifiedEvidenceBundle only

bundle.question == normalized query

len(bundle.items) <= top_k

no import from unified_grounding
no import from grounded_generation
```

## Non-claims

Passing Gate A does not mean that EvidenceMM can already search evidence.

It proves only that a stable query-to-evidence boundary now exists.

The next gate will freeze the actual document/robot retrieval representation,
candidate budget and ranking semantics before implementing the first search
baseline.
