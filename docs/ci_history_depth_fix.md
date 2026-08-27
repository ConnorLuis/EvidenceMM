# CI full-history correction

The first remote `CPU Static Audit` run for post-Day35 hardening commit
`7389a3f58204b300ceec709ad782805adad1fb4d` failed at the Day35 release
validation step.

The failure was caused by the default shallow checkout from
`actions/checkout@v4` (`fetch-depth: 1`). The Day35 audit intentionally verifies
that the frozen Day22-Day34 milestone commits are ancestors of the current
`HEAD`. In a one-commit shallow checkout, those historical commit objects are
not available, so Git reports each milestone as an invalid commit.

The workflow now uses:

```yaml
- uses: actions/checkout@v4
  with:
    fetch-depth: 0
```

The hardening audit and test suite also require this setting.

This correction changes only post-Day35 CI/repository-hardening infrastructure.
No Day22-Day35 prediction, metric, annotation, split, calibration, benchmark
document, README, limitations file, or Day35 release manifest is modified.
