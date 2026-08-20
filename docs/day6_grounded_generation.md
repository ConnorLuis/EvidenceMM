# EvidenceMM Day 6 - Grounded Answer Generation

## Goal

Day 6 connects the frozen retrieval layer to evidence-constrained answer
generation.

```text
question
   ↓
BM25 + ColQwen2.5
   ↓
RRF hybrid ranking
   ↓
Top-2 evidence pages
   ↓
page image + extracted text
   ↓
Qwen3-VL-4B-Instruct
   ↓
structured grounded answer
   ↓
citation validator
   ↓
answer / abstain
```

## Generator

The baseline generator is `Qwen/Qwen3-VL-4B-Instruct`.

The prompt supplies only retrieved evidence pages. Each evidence page contains:

- source ID
- 1-based page number
- rendered page image
- normalized extracted page text
- retrieval rank

The model must output exactly one JSON object with `answer`, `abstain`, and
`citations`.

If the supplied evidence is insufficient, the model must set `abstain=true`
and return an empty citation list.

## GPU lifecycle

The development GPU has 12 GB VRAM. ColQwen2.5 and Qwen3-VL-4B are therefore
not kept resident at the same time.

Day 6 runs in two explicit phases:

1. load ColQwen2.5 once and retrieve evidence for all evaluation cases;
2. delete the retriever, run garbage collection, and empty the CUDA cache;
3. load Qwen3-VL once and generate all answers.

## Visual budget

The generator receives at most two evidence-page images.

Per-image visual-token budget:

- minimum: 128
- maximum: 384

Extracted text is supplied alongside each page image, so the generator does
not depend on image OCR alone.

## Evaluation set

Day 6 contains three cases:

- two existing answerable PDF questions with verified gold pages;
- one controlled unsupported question asking for a Wi-Fi frequency that the
  STS3215 datasheet does not provide.

The controlled abstention case is checked against the standardized full PDF
text for explicit Wi-Fi-related terms before generation.

## Deterministic metrics

Day 6 does not use an LLM judge.

It reports:

- structured output rate
- answerability accuracy
- citation-policy validity rate
- citation gold-page hit rate
- mean citation precision
- required-fact coverage
- abstention accuracy
- end-to-end pass rate

A citation is invalid if it points outside the evidence pages actually supplied
to the generator.

There is no retry or output-repair loop in the Day 6 baseline. Invalid JSON,
unsupported citations, missed facts, and failure to abstain are recorded as
real baseline failures.

## Scope limitation

Three cases are sufficient to validate the grounded-generation contract, not
to make a general answer-quality claim. Larger human-verified evaluation comes
later.

## Observed Day 6 baseline

The initial grounded-generation baseline was evaluated on three cases:

- two answerable PDF cases with human-verified gold pages;
- one controlled unsupported PDF question requiring abstention.

No prompt tuning, retry, output repair, evidence-top-k tuning, or manual
evidence selection was performed after observing the baseline.

### Results

| Metric | Value |
| --- | ---: |
| Structured output rate | 1.0000 |
| Answerability accuracy | 1.0000 |
| Citation-policy valid rate | 1.0000 |
| Citation gold-page hit rate | 1.0000 |
| Mean citation precision | 1.0000 |
| Mean required-fact coverage | 1.0000 |
| Abstention accuracy | 1.0000 |
| End-to-end pass rate | 1.0000 |

These perfect values are a smoke-test result over only three cases and must
not be interpreted as a general answer-quality benchmark.

### Answerable cases

For `d6_pdf_001`, hybrid retrieval ranked verified page 3 first. The generator
answered that the typical operating voltages are 6V and 7.4V and cited only
page 3. Both required fact groups were covered.

For `d6_pdf_002`, hybrid retrieval ranked verified page 4 first and page 8
second. The generator returned all six required feedback parameters:

- Load
- Position
- Speed
- Input Voltage
- Current
- Temperature

The answer cited only verified page 4, despite page 8 also being supplied as
retrieved evidence.

### Controlled abstention

For `d6_pdf_abstain_001`, the question asks whether the STS3215 Wi-Fi
frequency is 2.4 GHz or 5 GHz.

The full standardized PDF text does not contain the controlled Wi-Fi-related
terms used by the case validator. Hybrid retrieval still supplied pages 3 and
4, and page 3 contains generic FCC/radio-frequency language. Nevertheless,
the generator correctly treated the supplied evidence as insufficient,
returned `abstain=true`, and emitted no citations.

This demonstrates the intended distinction between retrieving superficially
related evidence and possessing sufficient evidence to answer a question.

### Development GPU lifecycle

The RTX 4070 SUPER development run used two sequential GPU phases.

Retrieval phase:

- ColQwen2.5 model load: approximately 22.30 s
- three-case retrieval: approximately 0.64 s
- peak allocated GPU memory: approximately 7.29 GB

After deleting the retrieval model, garbage collection, and CUDA cache
release, allocated GPU memory dropped to approximately 9.1 MB.

Generation phase:

- Qwen3-VL-4B model load: approximately 20.20 s
- three-case generation total: approximately 5.33 s
- peak allocated GPU memory: approximately 9.04 GB

This validates the sequential model-lifecycle strategy required by the 12 GB
development GPU.

These measurements are development diagnostics only. Formal P50/P95,
throughput, and peak-memory benchmarks remain reserved for the later RTX 4090
evaluation.
