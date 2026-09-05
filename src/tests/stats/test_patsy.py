"""Unit tests for patsy utilities."""

import ast

import numpy as np
import pandas as pd
import pytest

from pydynopt.stats.patsy import (
    patsy_add_levels,
    patsy_formula_to_categorical_treatments,
    patsy_formula_to_categorical_varnames,
    patsy_formula_to_varnames,
    patsy_strip_categorical,
    patsy_strip_formula,
)


def test_patsy_formula_to_varnames():
    """Test extracting unique variable names from patsy formulas."""
    # Simple formula
    assert patsy_formula_to_varnames('y ~ x1 + x2') == ['y', 'x1', 'x2']

    # Categorical and interaction/nesting/functions
    assert patsy_formula_to_varnames('y ~ C(x1) + I(x2**2) + x3:x4') == [
        'y',
        'x1',
        'x2',
        'x3',
        'x4',
    ]

    # With intercept/numbers/operators
    assert patsy_formula_to_varnames('y ~ 1 + x1 + np.log(x2)') == ['y', 'x1', 'x2']


def test_patsy_formula_to_categorical_varnames():
    """Test extracting unique categorical variable names from patsy formulas."""
    assert patsy_formula_to_categorical_varnames('y ~ x1 + C(x2)') == ['x2']
    assert patsy_formula_to_categorical_varnames('y ~ C(x1, Treatment(1)) + C(x2)') == [
        'x1',
        'x2',
    ]


def test_patsy_formula_to_categorical_treatments():
    """Test extracting categorical variable names and their treatments."""
    import patsy

    assert patsy_formula_to_categorical_treatments(
        'y ~ C(x1, Treatment(1)) + C(x2)'
    ) == {'x1': '1'}

    formula = "C(x1, Treatment('a'))"
    assert patsy_formula_to_categorical_treatments(formula) == {'x1': "'a'"}

    matrix = patsy.dmatrix(  # type: ignore
        formula,
        pd.DataFrame({'x1': ['a', 'b']}),
        return_type='dataframe',
    )
    assert list(matrix.columns) == ['Intercept', "C(x1, Treatment('a'))[T.b]"]


def test_patsy_categorical_treatments_add_levels_round_trip():
    """Parse a quoted treatment from a formula with generated levels."""
    import patsy

    labels = ['base level', "O'Reilly", 'x, y', 'C(other)']
    data = pd.DataFrame({'group': pd.Series(labels, dtype='str')})
    formula, _ = patsy_add_levels(
        'C(group, Treatment("O\'Reilly"))',
        data,
    )

    treatments = patsy_formula_to_categorical_treatments(formula)
    assert treatments == {'group': '"O\'Reilly"'}
    assert ast.literal_eval(treatments['group']) == "O'Reilly"

    matrix = patsy.dmatrix(formula, data, return_type='dataframe')  # type: ignore
    columns = list(matrix.columns)
    assert len(columns) == len(labels)
    for label in labels:
        is_encoded = any(column.endswith(f'[T.{label}]') for column in columns)
        assert is_encoded == (label != "O'Reilly")


def test_patsy_categorical_treatments_ignore_special_level_literals():
    """Ignore punctuation and expression-like text in explicit level literals."""
    import patsy

    labels = [
        'base level',
        'C(other)',
        'x, y',
        "O'Reilly",
        'say "hello"',
        'both \'single\' and "double"',
        'looks_like_a_variable',
        'Treatment(fake_variable)',
    ]
    data = pd.DataFrame({'group': pd.Series(labels, dtype='str')})
    formula, _ = patsy_add_levels(
        'C(group, Treatment("base level"))',
        data,
    )

    treatments = patsy_formula_to_categorical_treatments(formula)
    assert treatments == {'group': '"base level"'}

    matrix = patsy.dmatrix(formula, data, return_type='dataframe')  # type: ignore
    columns = list(matrix.columns)
    assert len(columns) == len(labels)
    for label in labels:
        is_encoded = any(column.endswith(f'[T.{label}]') for column in columns)
        assert is_encoded == (label != 'base level')


@pytest.mark.parametrize(
    'treatment',
    ['ends)', 'comma, label', "single'quote", 'double"quote'],
)
def test_patsy_categorical_treatments_with_special_treatment(treatment):
    """Parse treatment labels containing parentheses, commas, and quotes."""
    import patsy

    labels = ['other', treatment, 'third']
    data = pd.DataFrame({'group': pd.Series(labels, dtype='str')})
    formula, _ = patsy_add_levels(
        f'C(group, Treatment({treatment!r}))',
        data,
    )

    treatments = patsy_formula_to_categorical_treatments(formula)
    assert ast.literal_eval(treatments['group']) == treatment

    matrix = patsy.dmatrix(formula, data, return_type='dataframe')  # type: ignore
    columns = list(matrix.columns)
    assert len(columns) == len(labels)
    for label in labels:
        is_encoded = any(column.endswith(f'[T.{label}]') for column in columns)
        assert is_encoded == (label != treatment)


def test_patsy_categorical_treatments_numeric_levels():
    """Parse a numeric treatment from a formula with generated numeric levels."""
    import patsy

    labels = [1, 2, 3]
    data = pd.DataFrame({'group': labels})
    formula, _ = patsy_add_levels('C(group, Treatment(2))', data)

    assert 'levels=[1, 2, 3]' in formula
    assert patsy_formula_to_categorical_treatments(formula) == {'group': '2'}

    matrix = patsy.dmatrix(formula, data, return_type='dataframe')  # type: ignore
    columns = list(matrix.columns)
    assert len(columns) == len(labels)
    for label in labels:
        is_encoded = any(column.endswith(f'[T.{label}]') for column in columns)
        assert is_encoded == (label != 2)


def test_patsy_categorical_treatments_multiple_factors():
    """Parse only Treatment contrasts from a formula with multiple factors."""
    import patsy

    data = pd.DataFrame(
        {
            'group': ['a', 'b', 'c'],
            'region': ['north', 'south', 'north'],
            'kind': ['first', 'second', 'first'],
        }
    )
    formula, _ = patsy_add_levels(
        'C(group, Treatment("b")) + C(region) + C(kind, Treatment("first"))',
        data,
    )

    assert patsy_formula_to_categorical_treatments(formula) == {
        'group': '"b"',
        'kind': '"first"',
    }

    matrix = patsy.dmatrix(formula, data, return_type='dataframe')  # type: ignore
    columns = list(matrix.columns)
    assert len(columns) == 5
    assert not any(
        'C(group,' in column and column.endswith('[T.b]') for column in columns
    )
    assert not any(
        'C(kind,' in column and column.endswith('[T.first]') for column in columns
    )
    assert any(
        'C(region,' in column and column.endswith('[T.south]') for column in columns
    )


def test_patsy_categorical_treatments_without_argument_raises():
    """Raise a contextual error for Treatment() without an argument."""
    with pytest.raises(
        ValueError,
        match=r"Malformed Treatment\(\) call.*'group'.*expected at least one argument",
    ):
        patsy_formula_to_categorical_treatments('C(group, Treatment())')


def test_patsy_add_levels():
    """Test updating formula with added factor levels."""
    df = pd.DataFrame({'x1': ['a', 'b', 'a'], 'x2': [1, 2, 3]})
    formula = 'y ~ C(x1) + x2'
    formula_upd, factors = patsy_add_levels(formula, df)
    assert formula_upd is not None
    assert "levels=['a', 'b']" in formula_upd
    assert factors == ['x1']


def test_patsy_add_levels_quoted():
    """Test patsy_add_levels with quoted string levels and design matrix generation."""
    import patsy

    # 1. A pandas 3.0 `str` Series with labels `female` and `male`.
    df = pd.DataFrame({'group': pd.Series(['female', 'male', 'female'], dtype='str')})
    formula = 'C(group)'
    formula_upd, factors = patsy_add_levels(formula, df)
    assert factors == ['group']
    assert "levels=['female', 'male']" in formula_upd

    # Verify we can build the matrix without looking for variables named female/male
    matrix = patsy.dmatrix(formula_upd, df, return_type='dataframe')  # type: ignore
    assert list(matrix.columns) == [
        'Intercept',
        "C(group, levels=['female', 'male'])[T.male]",
    ]


def test_patsy_add_levels_escapes():
    """Test patsy_add_levels with special characters in string labels."""
    import patsy

    # 2. String labels containing spaces, punctuation, a comma, embedded quotes, parentheses.
    labels = ['base level', "O'Reilly", 'x, y', 'C(other)', 'looks_like_a_variable']
    df = pd.DataFrame({'group': pd.Series(labels, dtype='str')})

    # 3. A nondefault `Treatment(...)` whose reference label contains an apostrophe.
    formula = 'C(group, Treatment("O\'Reilly"))'
    formula_upd, factors = patsy_add_levels(formula, df)

    assert factors == ['group']
    matrix = patsy.dmatrix(formula_upd, df, return_type='dataframe')  # type: ignore

    expected_cols = [
        'Intercept',
        "C(group, Treatment(\"O'Reilly\"), levels=['C(other)', \"O'Reilly\", 'base level', 'looks_like_a_variable', 'x, y'])[T.C(other)]",
        "C(group, Treatment(\"O'Reilly\"), levels=['C(other)', \"O'Reilly\", 'base level', 'looks_like_a_variable', 'x, y'])[T.base level]",
        "C(group, Treatment(\"O'Reilly\"), levels=['C(other)', \"O'Reilly\", 'base level', 'looks_like_a_variable', 'x, y'])[T.looks_like_a_variable]",
        "C(group, Treatment(\"O'Reilly\"), levels=['C(other)', \"O'Reilly\", 'base level', 'looks_like_a_variable', 'x, y'])[T.x, y]",
    ]
    for col in expected_cols:
        assert col in matrix.columns

    for col in matrix.columns:
        if col != 'Intercept':
            assert "T.O'Reilly" not in col


def test_patsy_add_levels_categorical():
    """Test patsy_add_levels with categorical Series and unused categories."""
    # 4. A categorical Series with unused declared categories.
    df = pd.DataFrame(
        {'group': pd.Series(pd.Categorical(['a', 'c'], categories=['a', 'b', 'c']))}
    )
    formula = 'C(group)'
    formula_upd, _ = patsy_add_levels(formula, df)
    # 'b' should be excluded because it's unused/unobserved
    assert "levels=['a', 'c']" in formula_upd


def test_patsy_add_levels_numeric():
    """Test patsy_add_levels with numeric and numpy scalar data."""
    import patsy

    # 5. Ordinary and categorical numeric data, including NumPy scalar values.
    df = pd.DataFrame({'group': pd.Series([np.int64(2), np.int64(1), np.int64(2)])})
    formula = 'C(group)'
    formula_upd, _ = patsy_add_levels(formula, df)
    assert 'levels=[1, 2]' in formula_upd

    matrix = patsy.dmatrix(formula_upd, df, return_type='dataframe')  # type: ignore
    assert list(matrix.columns) == ['Intercept', 'C(group, levels=[1, 2])[T.2]']


def test_patsy_add_levels_missing():
    """Test patsy_add_levels with missing values in different Series types."""
    # 6. Missing values in pandas `str`, object, and categorical Series.
    # string series
    df_str = pd.DataFrame({'group': pd.Series(['b', None, 'a'], dtype='str')})
    formula_upd, _ = patsy_add_levels('C(group)', df_str)
    assert "levels=['a', 'b']" in formula_upd

    # object series with pd.NA
    df_obj = pd.DataFrame({'group': pd.Series(['b', pd.NA, 'a'], dtype='object')})
    formula_upd, _ = patsy_add_levels('C(group)', df_obj)
    assert "levels=['a', 'b']" in formula_upd

    # categorical series with NaN
    df_cat = pd.DataFrame(
        {
            'group': pd.Series(
                pd.Categorical(['b', np.nan, 'a'], categories=['a', 'b', 'c'])
            )
        }
    )
    formula_upd, _ = patsy_add_levels('C(group)', df_cat)
    assert "levels=['a', 'b']" in formula_upd


def test_patsy_add_levels_api_contracts():
    """Test patsy_add_levels preserves existing API contract details."""
    # 7. Existing explicit `levels=`, no-intercept formulas, interactions, and LHS factors.
    df = pd.DataFrame(
        {
            'y': [1, 2, 3],
            'x1': ['a', 'b', 'a'],
            'x2': ['c', 'd', 'c'],
        }
    )

    # explicit levels remains unchanged (except for normalization/intercept)
    formula1 = 'y ~ C(x1, levels=[a,b])'
    formula_upd1, _ = patsy_add_levels(formula1, df)
    assert formula_upd1 == 'y ~ 1 + C(x1, levels=[a, b])'

    # no-intercept formula gets -1
    formula2 = 'y ~ C(x1) - 1'
    formula_upd2, _ = patsy_add_levels(formula2, df)
    assert formula_upd2 == "y ~ C(x1, levels=['a', 'b']) -1"

    # interactions C(x1):C(x2)
    formula3 = 'y ~ C(x1):C(x2)'
    formula_upd3, factors = patsy_add_levels(formula3, df)
    assert formula_upd3 == "y ~ 1 + C(x1, levels=['a', 'b']):C(x2, levels=['c', 'd'])"
    assert set(factors) == {'x1', 'x2'}

    # LHS factors
    formula4 = 'C(x1) ~ C(x2)'
    formula_upd4, factors = patsy_add_levels(formula4, df)
    assert formula_upd4 == "C(x1, levels=['a', 'b']) ~ 1 + C(x2, levels=['c', 'd'])"
    assert set(factors) == {'x1', 'x2'}


def test_patsy_add_levels_unsortable_raises():
    """Test patsy_add_levels raises ValueError when levels cannot be sorted."""
    # If some supported values cannot be ordered together, raise a clear `ValueError`
    df = pd.DataFrame({'group': pd.Series([1, 'a'], dtype='object')})
    with pytest.raises(ValueError) as exc_info:
        patsy_add_levels('C(group)', df)
    assert 'cannot be sorted' in str(exc_info.value)


def test_patsy_add_levels_unsafe_literal_raises():
    """Test patsy_add_levels raises ValueError when values cannot be safely serialized."""

    class UnsafeCustomObject:
        pass

    df = pd.DataFrame({'group': pd.Series([UnsafeCustomObject()], dtype='object')})
    with pytest.raises(ValueError) as exc_info:
        patsy_add_levels('C(group)', df)
    assert 'cannot be represented safely' in str(exc_info.value)


def test_patsy_strip_categorical():
    """Test stripping categorical metadata from variable definitions."""
    # String input
    assert patsy_strip_categorical('C(x1, Treatment(1))') == 'C(x1)'
    # List/Iterable input
    assert patsy_strip_categorical(['C(x1, Treatment(1))', 'x2']) == ['C(x1)', 'x2']


def test_patsy_strip_formula():
    """Test stripping redundant white space from formula strings."""
    assert patsy_strip_formula('  y  ~  x1+x2  ') == 'y ~ x1 + x2'
    assert patsy_strip_formula('') == ''
