# Post-Day35 Release Hardening

Day35 remains the frozen experimental/project-release boundary. This hardening
commit does **not** alter Day22-Day35 predictions, metrics, calibration, benchmark
claims, README, benchmark card, limitations, or release manifest.

It closes two repository-engineering gaps identified after Day35:

1. no GitHub Actions workflow;
2. key runtime/build/dev dependency versions were not captured as exact resolved
   constraints.

## CPU-only GitHub Actions

`.github/workflows/cpu-static-audit.yml` deliberately avoids model inference and
GPU setup.

The workflow:

- uses Python 3.11.15;
- installs only `pytest` under the committed constraints;
- runs `compileall`;
- validates the frozen Day35 release;
- recomputes/audits Day34 frozen metrics and hashes;
- validates the post-Day35 hardening contract;
- runs the Day31-Day35 static schema/hash test subset.

It does **not** instantiate Qwen3-VL, import model weights, run CUDA, access raw
robot media, or consume a new held-out evaluation.

## Environment constraints

`constraints.txt` is generated from the same successful local environment used
for the frozen benchmark/release.

`data/protocol/day35_environment_snapshot.json` records:

- Python version;
- direct runtime/dev/build dependency versions;
- exact local Torch/TorchVision builds;
- Transformers version;
- CUDA runtime and GPU identity when available;
- SHA256 of `constraints.txt`.

The known frozen Day34 runtime evidence is:

- Python 3.11.15;
- Torch 2.11.0+cu130;
- Transformers 5.15.0;
- CUDA 13.0;
- NVIDIA GeForce RTX 4070 SUPER.

The exporter fails closed if the current environment does not match the known
Python/Torch/Transformers evidence before writing the lock artifacts.


### Torch version semantics

PyTorch exposes two useful version identifiers in this environment:

- package/distribution metadata: `2.11.0`;
- imported runtime build: `torch.__version__ == 2.11.0+cu130`.

`constraints.txt` pins the installable distribution version (`torch==2.11.0`),
while the JSON environment snapshot separately freezes the runtime CUDA build
identifier (`2.11.0+cu130`). This avoids treating the local CUDA build suffix as
a PyPI distribution-version requirement while still preserving the successful
runtime provenance.

## Reproduction semantics

The constraints file is a **project-scoped resolved direct-dependency lock**, not
a claim that all transitive wheels are universally portable across CUDA/OS
combinations. The JSON snapshot preserves local build identifiers such as
`+cu130`.

For GitHub Actions, the workflow installs only pytest, so the CPU audit never
tries to install the CUDA Torch build.

For full model reproduction, use a CUDA/PyTorch index compatible with the frozen
snapshot and then install the remaining project dependencies under
`constraints.txt`.

## Release integrity

`scripts/release_hardening_audit.py` verifies that:

- the Day35 release commit remains an ancestor;
- every Day35 frozen public/release file is byte-identical to the Day35 commit;
- the Day35 release manifest Git blob is unchanged;
- every declared direct runtime/dev/build dependency has an exact constraint;
- Torch and TorchVision are explicitly frozen;
- the environment snapshot is bound to the constraints SHA;
- the CI workflow remains CPU/static and read-only.

This hardening is repository infrastructure only. It is not Day36 and it does
not reopen the frozen benchmark.
