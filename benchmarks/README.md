# Benchmarks

The benchmark suites live outside the installed package and the pytest test suite.
Each top-level package below `benchmarks/` corresponds to one top-level pydynopt
component. The initial suite is `benchmarks/interpolate/`; future suites should be
added as sibling packages.

## Running a suite

Run the interpolation suite from the repository root:

```bash
uv run --frozen python -m benchmarks interpolate
```

The runner pins itself to exactly one logical CPU before importing NumPy or Numba.
The selected CPU is
`min(7, os.cpu_count() - 1)`, which is CPU 7 on machines with at least eight logical
CPUs. The run fails if that CPU is unavailable or affinity cannot be enforced.
Numba, OpenMP, and common numerical-library thread counts are fixed at one.

Compilation and warm-up are excluded. Each case is measured in 11 samples of at
least 100 ms each. A sample contains enough complete workload rounds to meet that
target, so it normally executes millions of interpolation operations. Results are
reported as the median nanoseconds per query or output element, together with the
relative median absolute deviation and coefficient of variation. A CV above 2% is
flagged as noisy.

## Comparing changes

Create a machine-local baseline before changing the implementation:

```bash
uv run --frozen python -m benchmarks interpolate \
    --output /tmp/pydynopt-interpolate-before.json
```

Compare the changed implementation with that baseline:

```bash
uv run --frozen python -m benchmarks interpolate \
    --baseline /tmp/pydynopt-interpolate-before.json \
    --output /tmp/pydynopt-interpolate-after.json
```

The comparison requires matching suite, CPU, CPU model, Python, NumPy, and Numba
metadata. It reports changes for each case and each aggregate group. Raw baselines
should not be committed because timings are meaningful only on the same machine and
software environment. Increment `SUITE_VERSION` in `benchmarks/_runner.py` whenever
workloads, inputs, operation counts, or measurement settings change.

Changes below roughly 2-3% should be confirmed by running the complete suite three
times. Inspect individual cases and dispersion rather than accepting an aggregate
improvement that hides a common-case regression.

## Scope

The interpolation suite covers:

- local, same-interval, and distant search patterns;
- scalar and array calls from Numba-compiled kernels;
- C-contiguous and arbitrary-strided function arrays;
- caller-provided and internally allocated output arrays;
- actual out-of-grid extrapolation;
- direct Python/NumPy array calls;
- downstream-style pipelines that locate once and evaluate several fields.

Benchmarks are deliberately not collected by pytest and should not run in ordinary
shared CI. A dedicated benchmark host can archive the JSON output and apply
project-specific regression thresholds.
