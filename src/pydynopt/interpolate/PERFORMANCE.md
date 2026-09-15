# Optimizing Numba linear interpolation

This guide records the principles behind the optimized one- and two-dimensional
linear interpolation kernels. It is intended for agents or developers optimizing a
similar code base, not as a promise that every choice is optimal on every CPU or
Numba release.

On the frozen benchmark used for this work, the geometric mean fell from about
17.92 ns to 6.98 ns per query, a reduction of about 61%. Compilation and warm-up
were excluded. The research harness allowed execution on CPUs 0-7, so these values
are not directly comparable to results from the permanent, single-CPU benchmark
suite documented in [`benchmarks/README.md`](../../../benchmarks/README.md). The
largest gains came from search locality, selective inlining, Numba-friendly control
flow, and layout-specific 2D addressing rather than from changing interpolation
arithmetic.

## Start with a representative workload

Interpolation performance depends on how callers compose the API. The workload
that guided these changes has the following important properties:

- Scalar interpolation is called inside Numba kernels.
- A previous lower-bound index is reused as the next search hint.
- Several function arrays are evaluated at the same located coordinates.
- Local and random query movement both matter.
- 2D values may be C-contiguous or arbitrary-strided.
- Array calls may reuse caller-provided output buffers.

Use a frozen, balanced suite. Include same/adjacent and distant searches, scalar and
array calls, contiguous and strided values, supplied output buffers, nonuniform
grids, changing positions, and consumed results. Pin measurements to comparable
cores, exclude compilation, take repeated samples, and monitor individual cases in
addition to an aggregate metric.

A narrow direct-case improvement is not enough when unchanged common cases regress.
Small source changes can alter Numba's register allocation and caller code layout,
so rerun borderline results.

## Reuse located scalar components

Pass length-two index and weight tuples to `interp2d_eval` when an outer Numba
kernel already holds their components as scalars and evaluates several fields at
those coordinates:

```python
from numba import njit

from pydynopt.interpolate import interp2d_eval


@njit
def evaluate_fields(index0, index1, weight0, weight1, fp0, fp1):
    index = (index0, index1)
    weight = (weight0, weight1)
    value0 = interp2d_eval(index, weight, fp0)
    value1 = interp2d_eval(index, weight, fp1)
    return value0, value1
```

Numba represents these fixed-size tuples without NRT allocation. This form uses the
same lower-grid-point weight convention, extrapolation checks, arithmetic order,
and layout-specialized leaves as array-based `interp2d_eval` calls.

## Separate hot local search from cold distant search

The central 1D primitive is `interp1d_locate_scalar` in
`numba/linear.py`. Its structure is deliberate:

1. Load the hinted lower and upper grid points once.
2. Return immediately when the hint still brackets the query.
3. Check one adjacent interval while reusing an already loaded endpoint.
4. Call `_bsearch_range` only for a distant query.
5. Retain the selected endpoints through the weight calculation.

This avoids a general search call and repeated grid loads on the dominant local
paths. It also preserves logarithmic behavior for distant movement.

The distant binary loop is kept in `_bsearch_range` in `numba/search.py`. Keep it
out of line. Inlining it duplicates the loop in 1D arrays, both dimensions of 2D
callers, and fused interpolation kernels. That improved an isolated random search
but worsened the balanced workload and greatly increased generated code.

The fallback accepts known lower and upper bounds. Do not move same/adjacent checks
back into that helper: the interpolation caller needs its loaded endpoints to
calculate the weight without loading them again.

### Preserve unordered-value behavior

Do not tighten the downward fallback upper bound from the original hint to
`hint - 1`. That inference is valid for ordered finite queries but not for NaN,
because both ordered comparisons are false. Passing the original hint preserves
the Python search path's observable lower-bound index while the weight propagates
NaN.

## Write control flow for Numba SSA

Equivalent Python can produce very different Numba intermediate representation.
The locate kernel originally reassigned `index`, `lower`, and `upper` in several
branches and merged them before one return. Rewriting it as branch-local early
returns produced a major speedup and eliminated `NumbaIRAssumptionWarning` messages.

For small always-inlined kernels:

- Prefer branch-local names and returns over many SSA phi merges.
- Keep values live only on paths that need them.
- Avoid returning a large tuple from a cold helper. Returning only the located
  integer index was much cheaper than returning index, endpoints, and weight.
- Treat source-level temporary variables as performance decisions. Extra cached
  bounds, indices, or complements can increase register pressure even when they
  appear to remove arithmetic.

Inspect Numba LLVM or assembly when possible, but validate every source rewrite with
timings. A visible call is not automatically harmful, and fewer source operations
do not guarantee faster generated code.

## Inline selectively

Inlining policy was as important as kernel arithmetic. The useful policy is not
"inline everything."

### Inline these narrow scalar boundaries

- Scalar locate/evaluate/fused leaf kernels use `JIT_OPTIONS_INLINE` where their
  arithmetic should be exposed to a compact caller.
- Public scalar locate overloads are always inline so tuple results and two-element
  index/weight views can scalarize.
- Public scalar 1D combined interpolation has its own always-inline overload.
- The C-layout scalar 2D combined overload is isolated and always inline.
- The arbitrary-strided one-point 2D evaluator is selected by its own always-inline
  overload so caller-side index and weight views scalarize.
- `_initial_indices` uses compile-time `None` and indexable implementations and is
  always inline.

### Keep these boundaries out of line

- `_bsearch_range`, to prevent binary-loop duplication.
- Public 1D evaluation in repeated-field callers; inlining it enlarged the caller
  and slowed the pipeline.
- Public C-layout 2D evaluation, including scalar-tuple `interp2d_eval`; its compact
  call boundary schedules better when several fields are evaluated at one point.
- Array loop implementations and public array wrappers; forcing them inline caused
  severe code-size and code-layout regressions.

Use mutually exclusive overload templates when policies differ by scalar/array or
memory layout. A scalar-only inline overload and a normal array overload performed
far better than one shared always-inline template. Explicit templates were also
more robust than callable Numba inline cost models, which failed around folded
default and keyword output arguments.

## Specialize 2D addressing by layout

Four generic multidimensional corner addresses are expensive in a tiny bilinear
kernel. Compile-time layout information supports two different strategies.

### C-contiguous values

`_interp2d_eval_point_c` and `_interp2d_scalar_c` calculate one row-major base
offset:

```text
offset = index0 * number_of_columns + index1
```

The four corners are then `offset`, `offset + 1`, `offset + number_of_columns`, and
`offset + number_of_columns + 1`. This avoids four generic multidimensional address
calculations.

Load adjacent values from one row together while preserving the arithmetic tree.
Do not force the public C-layout evaluation overload inline; only its compact flat
leaf should inline into that overload.

### Arbitrary-strided values

For `interp2d_eval_point`, cache the lower and upper row views, then load adjacent
columns from each row. Construct the lower row and then the upper row before loading
all four corners. Reordering row creation or serializing row creation with corner
loads materially slowed strided arrays.

Keep the explicit shared dimension-0 complement in this evaluator. Writing
`1.0 - weight0` twice regressed the strided path even though a compiler might appear
able to eliminate it.

### Search order is caller-specific

The fused C-layout scalar kernel locates dimension 1 before dimension 0. Dimension 1
is contiguous and fast-moving in the representative workload, and this order
improved the fused caller. Do not generalize it:

- Standalone buffered 2D locate is faster with dimension 0 first.
- The 2D array kernel is faster with dimension 0 first after loop versioning and row
  caching.

The nearby line comments encode these ordering constraints.

## Version invariant array-loop modes

Passing `extrapolate` through a shared evaluator left avoidable control flow in
array loops. Both `interp1d_array_impl` and `interp2d_array_impl` now branch on the
mode outside the loop:

- The extrapolating loop performs direct weighted evaluation with no per-element
  mode check.
- The non-extrapolating loop retains boundary behavior and NaN output semantics.

This source-level loop versioning improved both dimensions even though an optimizing
compiler could theoretically unswitch the loop itself.

Within the extrapolating 2D loop, cache lower and upper row views and load corners in
row-contiguous order. Keep direct `.flat` traversal for arbitrary-shaped coordinate
and output arrays. Hoisting flat iterator objects or adding a dedicated flat-offset
array implementation did not improve the balanced suite.

Keep array implementations out of line. Large always-inline loops expanded every
allocation/output wrapper and degraded unrelated scalar code.

## Preserve arithmetic and contracts

The optimized code retains the documented lower-grid-point weight and weighted-sum
formulations. Do not introduce `fastmath`, reduced precision, or algebraic rewrites
without a separately approved numerical contract.

In particular:

- Difference-form 1D interpolation had fewer arithmetic operations but was slower,
  likely because of its longer dependency chain.
- Reversing the bilinear interpolation dimensions changed the arithmetic tree and
  regressed performance.
- Explicit corner temporaries help compact evaluation-only kernels, but can hurt
  fused kernels whose searches already create register pressure.
- Reordering output stores in buffered 2D locate was slower. Keep contiguous index
  stores before contiguous weight stores.

Continue testing dtype behavior, extrapolation, NaN propagation and located indices,
C and strided layouts, arbitrary coordinate shapes, supplied output identity, and
integer inputs. Low-level unchecked kernels may assume validated grids and hints,
but public overload behavior must remain aligned with the Python path.

## Approaches that did not generalize

These are useful warnings for similar work:

- Inlining the full binary search or array loops.
- Inlining every public evaluation overload.
- Returning complete cold locate tuples from a helper.
- Branchless binary-search updates.
- Caching final grid indices or trivial `index + 1` expressions.
- Caching `.flat` iterator objects.
- Automatically treating every 2D array as a flat C-layout kernel.
- Adding source temporaries for complements or endpoints without measuring register
  pressure.
- Reordering both 2D searches based only on one successful fused C-layout caller.
- Assuming callable inline policies handle Numba defaults and keyword arguments
  reliably.

## Transfer checklist

When optimizing another interpolation package:

1. Freeze correctness tests and a representative balanced benchmark.
2. Identify which callers reuse hints and which evaluate multiple fields per locate.
3. Add same/adjacent fast paths without removing logarithmic distant fallback.
4. Reuse grid endpoints through weight calculation.
5. Rewrite tiny inlined kernels with branch-local early returns.
6. Inspect overload signatures and split scalar, array, C-layout, and strided
   policies at typing time.
7. Keep cold loops and large array loops out of line.
8. Specialize C-layout scalar corner addressing with one flat offset.
9. Cache row bases for arbitrary-strided one-point 2D evaluation.
10. Version invariant modes outside array loops.
11. Preserve arithmetic order and public semantics.
12. Measure the aggregate and every directly affected workload; rerun marginal
    results and reject changes whose apparent gain comes from unrelated noise.
13. Recheck generated warnings, formatting, lint, typing, and the full test suite.

Treat the comments around ordering and inlining in the implementation as tested
constraints. If a future Numba release changes code generation, re-benchmark before
simplifying them rather than assuming an algebraically equivalent form is equally
fast.
