# Day35 Reproducibility and Audit Guide

## Repository state

Day35 closes the current research phase on top of the frozen Day34 commit:

`c80525f18062ada2613518f16f47868552ede0ea`

The Day35 release audit verifies that the Day22-Day34 milestone commits remain
ancestors and that frozen Day34 metrics/error/efficiency artifacts retain their
known SHA256 values.

## Environment

The package is defined by `pyproject.toml`.

Minimal development installation:

```bash
conda activate evidencemm
cd ~/projects/evidencemm
python -m pip install -e '.[dev]'
```

The final local experiments used a CUDA-capable environment with Qwen3-VL
available in the local Hugging Face cache. Large model caches are not committed.

## Full regression

```bash
pytest -q
```

Day34 closed with 356 passing tests before Day35 release tests were added.

## Frozen audit chain

The core post-benchmark audits are:

```bash
python scripts/day31_root_cause_baseline.py audit
python scripts/day32_development_calibration.py audit
python scripts/day33_single_heldout_eval.py audit
python scripts/day34_metrics_error_efficiency.py audit
python scripts/day35_release_audit.py audit
```

Day33 audit requires the frozen final artifacts and verifies the single held-out
result. Day34 recomputes the final metrics/error report and verifies that no
held-out rerun or post-result tuning was introduced.

## Raw data and media

The Git repository intentionally ignores local raw datasets/media and model
caches. Canonical robot media are bound through manifests, hashes and local
compatibility paths established by the raw-audit stage.

For a machine with the original raw source mounted, the Day31-Day34 runtime
checks validate source hashes before evidence is accepted.

## Frozen artifact hashes

### Day33 final evaluation

- scoring predictions:
  `787dd204275c0f453d3700e95d165b607ad91946e38f866a7f0250a1bc8fde06`
- final predictions:
  `c6046cc34de98cf0ab892236028a96d78a990145dfdbdd98fe68f90a53ea3289`
- final metrics:
  `4165d639e9ff31bec09f76d133e45ba344d462cba04f0c72b1d97e1c65958369`
- freeze receipt:
  `18663b49577187c305be349aa733d4697976142036b81de8457b9dfeb2f9c711`

### Day34 post-hoc analysis

- final metrics report:
  `536d49aff909545310867294a2cdd2f2626498f6d0220a459d07d873e31845e9`
- error analysis:
  `f32f44492d26cf344e7e056a4d8edef94a3af7f85743dd0dc959e8e323cd4e55`
- per-case analysis:
  `130beb67cd9df54faa7b621338c60c5575d26df619d9f55b41dd7184c5853bf6`
- development E2E profile:
  `a97d4d3e3a8c709c9b9d11999fd221cba91446438daeb2cf7e470d35b07fa208`
- efficiency report:
  `2002015a54afc95d8aa181dcc2c14ed394b193d4d45586b5673d7a7d35c6873d`
- freeze receipt:
  `0c7fe40e62a4086b7de8d86b4c13ebc273cf82e6f69963dee2da3bff558b9c2b`

## Reproduction levels

There are three useful levels of reproduction.

### 1. Static audit

Requires only the Git repository. Verify committed hashes, split/annotation
contracts and final metric recomputation.

### 2. Evidence/runtime reproduction

Requires the canonical raw robot source paths. Rebuild selected evidence and
verify source/evidence fingerprints.

### 3. Model inference reproduction

Requires the raw sources plus the local Qwen3-VL model/processor cache and a
compatible GPU environment.

Because generative inference and GPU timings depend on software/hardware, exact
latency should not be treated as portable. Frozen prediction artifacts are the
canonical final benchmark output.

## Starting future research

Do not rewrite Day22-Day35 artifacts to improve the benchmark result. Future
changes such as constrained JSON decoding, alternative calibration, improved
clean-success discrimination or new retrieval evidence should start in a new
versioned experimental phase with a new held-out protocol.
