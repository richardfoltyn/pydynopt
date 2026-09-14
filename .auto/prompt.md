# Autoresearch: Optimize Numba linear interpolation

## Objective

Improve the runtime performance of pydynopt's Numba-compiled one- and
two-dimensional linear interpolation routines. Python-only entry points exist for
validation and debugging and are not optimization targets.

The representative downstream workload is
`/home/richard/repos/jmp/work/model/python/src/EBL/model/eval_funcs.py`. It calls
public pydynopt interpolation functions from nested Numba kernels, frequently
reuses the previous 1D lower-bound index, and evaluates several function arrays at
the same located coordinates. The 2D synthetic benchmark uses that repeated-field,
scalar evaluation pattern but benchmarks only pydynopt functions, never JMP's
handwritten interpolation helpers.

Do not overfit or cheat. All calls must perform their documented work and consume
the results.

## Metrics

- **Primary**: `geomean_ns` (ns/query, lower is better) - geometric mean across a
  frozen, balanced suite of 1D/2D scalar and array workloads.
- **Secondary**: `one_d_ns`, `two_d_ns`, `scalar_ns`, `array_ns`,
  `locate1d_local_ns`, `eval2d_ebl_ns`, and `max_cv_pct`.

## How to Run

`./.auto/measure.sh` emits structured `METRIC name=value` lines. It pins syntax
checks and the benchmark process to CPU affinity mask 0-7. The benchmark itself
verifies that exact mask before timing. Never run performance measurements outside
cores 0-7: cores 8-15 have different L3 caches and maximum clocks.

Compilation and warm-up are excluded from internal timings. Each case runs seven
approximately 40 ms samples and reports the median. Runtime arrays, changing query
positions, accumulated checksums, local/random search patterns, nonuniform grids,
C-contiguous and strided 2D values, and supplied output buffers prevent trivial or
benchmark-specific shortcuts.

## Files in Scope

- `src/pydynopt/interpolate/numba/search.py` - unchecked lower-bound search kernel.
- `src/pydynopt/interpolate/numba/linear.py` - Numba-compatible 1D/2D kernels.
- `src/pydynopt/interpolate/linear.py` - Numba overload dispatch only, and only if
  dispatch or inlining changes are required.
- `src/tests/interpolation/` - add correctness coverage only when an optimization
  exposes a contract gap.
- `.auto/prompt.md` and `.auto/ideas.md` - persistent experiment knowledge.

## Off Limits

- `.auto/benchmark.py`, `.auto/measure.sh`, and benchmark workloads are frozen
  after the baseline. Do not alter them to improve the score.
- JMP source files and handwritten interpolation helpers.
- Python-only validation/debugging paths in `src/pydynopt/interpolate/linear.py`.
- Public API semantics, dtype behavior, output-buffer behavior, extrapolation
  behavior, or supported array layouts.
- Dependency changes.

## Constraints

- Benchmark processes must have exact CPU affinity 0-7.
- No `fastmath`, reduced precision, hard-coded benchmark values, skipped work, or
  benchmark-specific branches.
- Preserve numerical behavior within the existing interpolation contract.
- After each passing benchmark, `.auto/checks.sh` runs interpolation tests and
  Ruff, formatting, and ty checks on the files in scope.
- Keep an experiment only when the primary metric improves. Monitor individual
  workloads so the aggregate does not conceal a severe common-case regression.
- Use `uv` for every Python, test, lint, formatting, and type-check command.

## Initial Optimization Directions

1. Improve `bsearch_impl` for neighboring interval movement while retaining its
   current same-bin fast path and logarithmic behavior for distant jumps.
2. Inspect generated LLVM/assembly and test explicit inlining for tiny scalar
   locate/evaluate helpers.
3. Reduce repeated loads, tuple/index handling, and unnecessary conversions in
   fused scalar kernels.
4. Improve bilinear arithmetic and memory access without relaxed floating-point
   semantics.
5. Reduce `.flat` and helper-call overhead in array kernels while retaining
   arbitrary supported layouts.

## What's Been Tried

- Frozen baseline established in run 3 at `geomean_ns=17.711749` (the dashboard's
  baseline includes two earlier benchmark-check failures).
- Run 4 added one-interval upward/downward fast paths to `bsearch_impl`. This was a
  clear win for local and array queries, with a small random-query cost.
- Run 5 forced all small helpers inline and reached `11.747198` ns, but Numba hit
  SSA-scope failures in public 2D tests. Do not retry this exact nested-inline
  structure.
- Runs 6-8 isolated safe inline boundaries: evaluation helpers and fused scalar
  kernels use `JIT_OPTIONS_INLINE`; `interp1d_locate_scalar` is forced inline, but
  `bsearch_impl` deliberately is not. Run 8 passes all checks at `11.975886` ns.
  This structure captures most of run 5's speed without the compiler bug.
- Repeated 2D field evaluation remains about 6.9-7.0 ns per state component and has
  changed little. Current larger costs are 2D locate/combined paths and random
  searches.
