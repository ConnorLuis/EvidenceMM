# EvidenceMM Day28 raw-audit package

This package implements Day28 only:

- 92 registered-attempt fresh raw re-audit;
- 93-directory retention accounting;
- one anonymous technical exclusion bound by tree SHA256;
- label-safe registered source manifest;
- byte-level source binding;
- Day28 validator and unit tests.

It does not modify Day24–Day27 artifacts and does not materialize Day29 Ground Truth or the Day30 split.

## Supplied files

- `configs/day28_raw_audit.yaml`
- `docs/day28_raw_audit.md`
- `scripts/analyze_day28_raw_audit.py`
- `scripts/validate_day28_raw_audit.py`
- `src/evidencemm/day28_raw_audit.py`
- `tests/test_day28_raw_audit.py`

The analyzer generates:

- `data/protocol/day28_registered_source_manifest.csv`
- `data/protocol/day28_raw_audit_analysis.json`
- `data/protocol/day28_raw_audit_analysis.csv`

## Run order

```bash
cd ~/projects/evidencemm
python scripts/analyze_day28_raw_audit.py
python scripts/validate_day28_raw_audit.py
pytest -q tests/test_day28_raw_audit.py
pytest -q
```

The validator intentionally performs a fresh byte-level re-audit and therefore re-reads the raw episode trees.

Do not stage or commit until all Day28 gates pass.
