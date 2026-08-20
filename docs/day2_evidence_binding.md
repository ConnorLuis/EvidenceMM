# EvidenceMM Day 2 - Real Source Binding and Human Verification

Day 2 introduces real source identities and traceable evidence locations.

## Rules

1. `source_id` is stable and independent of the local file name.
2. Every bound asset stores SHA-256 so later experiments can prove which bytes were used.
3. PDF page numbers exposed in evidence are 1-based.
4. PDF/image regions use normalized top-left-origin `bbox`.
5. Text search may suggest a PDF page, but it does **not** make a case verified.
6. A case becomes `verified` only after a human checks the rendered page/image and records:
   - `answerable`
   - `expected_answer`
   - `expected_evidence`
   - `verified_by`
   - `verified_at`

## Day 2 target assets

- `robot_image_wrist_001`: a real local wrist-camera image.
- `sts3215_datasheet`: a real STS3215 PDF specification used as the first PDF evidence source.

The STS3215 PDF is a component specification, not a substitute for the future full SO-ARM101 operating manual. It is used on Day 2 to validate the PDF evidence contract with real bytes and real page numbers.

## Day 2 verified target

Promote four manually checked cases:

- 2 image cases
- 2 PDF cases

Keep all original Day 1 cases unchanged.
