# Fix unsafe factor-level serialization in `patsy_add_levels`

## Repository and scope

Implement this issue in the sibling `pydynopt` repository, not in
`alp-beliefs`:

- Source: `src/pydynopt/stats/patsy.py`
- Focused tests: `src/tests/stats/test_patsy.py`
- Public API: `pydynopt.stats.patsy.patsy_add_levels()`

Do not modify `alp-beliefs` as part of the upstream patch.

## Problem

`patsy_add_levels()` currently inserts factor levels into formula text with:

```python
code += ', levels=[' + ','.join(str(v) for v in values) + '])'
```

This does not serialize values as literals. With pandas 3.0's inferred `str`
dtype, a factor such as `['female', 'male']` produces:

```text
C(group, levels=[female,male])
```

Patsy interprets these labels as variable names. Labels containing spaces,
commas, quotes, parentheses, or text resembling a formula expression can also
produce invalid or semantically different formulas. For example, the current
implementation mishandles labels such as:

- `base level`
- `O'Reilly`
- `x, y`
- `C(other)`
- `looks_like_a_variable`

The current `sorted(df[name].unique())` path also does not state the desired
missing-value policy and can fail when `pd.NA` is compared with ordinary values.

This is an upstream formula-construction defect rather than an
`alp-beliefs`-specific behavior, so it should be fixed once in `pydynopt`.

## Required behavior

1. Insert every factor level as a valid, quoted Python literal rather than with
   `str(value)`. Serializing the complete normalized level list with `repr()` is
   acceptable if every element has first been converted to a literal-safe Python
   scalar.
2. Convert NumPy scalar values to the corresponding Python scalar before
   serialization so output does not contain expressions such as `np.int64(1)`.
3. Compute levels from observed, nonmissing values only:
   - exclude `None`, `np.nan`, and `pd.NA`;
   - exclude unused pandas categorical levels;
   - do not create a Patsy missing-value level.
4. Preserve the established deterministic sorted order of observed levels. If
   some supported values cannot be ordered together, raise a clear `ValueError`
   rather than producing nondeterministic formula text.
5. Preserve all existing API behavior unrelated to level serialization:
   - formulas that already specify `levels=` remain unchanged;
   - intercept handling remains unchanged;
   - interactions and left-hand-side factors remain supported;
   - the returned factor-name list remains unique and order preserving;
   - explicit `Treatment(...)` selections remain intact.
6. The returned formula must be accepted by Patsy and produce columns in the
   inserted level order. An explicit treatment containing punctuation or quotes
   must select the intended reference level.
7. If a scalar value cannot be represented safely in Patsy formula text, raise a
   contextual `ValueError` naming the factor and value. Do not silently convert
   values to strings when that changes their identity or dtype.

## Required regression tests

Extend `src/tests/stats/test_patsy.py` with focused tests that call both
`patsy_add_levels()` and `patsy.dmatrix()`.

Cover at least:

1. A pandas 3.0 `str` Series with labels `female` and `male`. Verify that the
   generated formula contains quoted levels and builds a matrix without looking
   for variables named `female` or `male`.
2. String labels containing spaces, punctuation, a comma, embedded single and
   double quotes, parentheses, and text that looks like a variable or formula.
3. A nondefault `Treatment(...)` whose reference label contains an apostrophe or
   other punctuation. Verify the actual omitted Patsy column, not only the
   formula string.
4. A categorical Series with unused declared categories. Verify that only
   observed categories are inserted.
5. Ordinary and categorical numeric data, including NumPy scalar values. Verify
   that the generated text uses literal numeric values rather than `np.*`
   expressions.
6. Missing values in pandas `str`, object, and categorical Series. Verify they
   are excluded and do not cause sorting failures.
7. Existing explicit `levels=`, no-intercept formulas, interactions, and LHS
   factors to guard the current API contract.
8. Deterministic level order and the returned factor-name order.

A representative integration assertion should resemble:

```python
formula, factors = patsy_add_levels(
    "C(group, Treatment(\"O'Reilly\"))",
    pd.DataFrame(
        {
            "group": pd.Series(
                ["base level", "O'Reilly", "x, y", "C(other)"],
                dtype="str",
            )
        }
    ),
)
matrix = patsy.dmatrix(formula, data, return_type="dataframe")
```

Assert that `factors == ['group']`, the matrix builds successfully, and the
`O'Reilly` level is the treatment category.

## Verification

Run in `pydynopt` using its existing environment and project instructions:

```bash
pytest -q src/tests/stats/test_patsy.py
ruff format src/pydynopt/stats/patsy.py src/tests/stats/test_patsy.py
ruff check src/pydynopt/stats/patsy.py src/tests/stats/test_patsy.py
```

Also run the repository's configured type checker on the changed scope. Do not
install or update dependencies merely for this issue.

## Downstream context

`alp-beliefs/python/src/alp_beliefs/model/estimator.py::_init_derived_data()` calls
`patsy_add_levels()` for all five estimator formulas. Block 5 of
`MODEL_REVIEW_IMPLEMENTATION_SESSIONS.md` requires pandas `str` factors and labels
with spaces, punctuation, quotes, and variable-like text to generate valid Patsy
design matrices. Once this upstream issue is merged, `alp-beliefs` can add the
corresponding integration regression without carrying a duplicate local Patsy
formula implementation.
