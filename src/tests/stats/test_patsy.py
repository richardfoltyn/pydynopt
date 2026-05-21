"""
Unit tests for patsy utilities
"""

import pandas as pd

from pydynopt.stats.patsy import (
    patsy_add_levels,
    patsy_formula_to_categorical_treatments,
    patsy_formula_to_categorical_varnames,
    patsy_formula_to_varnames,
    patsy_strip_categorical,
    patsy_strip_formula,
)


def test_patsy_formula_to_varnames():
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
    assert patsy_formula_to_categorical_varnames('y ~ x1 + C(x2)') == ['x2']
    assert patsy_formula_to_categorical_varnames('y ~ C(x1, Treatment(1)) + C(x2)') == [
        'x1',
        'x2',
    ]


def test_patsy_formula_to_categorical_treatments():
    assert patsy_formula_to_categorical_treatments(
        'y ~ C(x1, Treatment(1)) + C(x2)'
    ) == {'x1': '1'}
    assert patsy_formula_to_categorical_treatments("y ~ C(x1, Treatment('a'))") == {
        'x1': "'a'"
    }


def test_patsy_add_levels():
    df = pd.DataFrame({'x1': ['a', 'b', 'a'], 'x2': [1, 2, 3]})
    formula = 'y ~ C(x1) + x2'
    formula_upd, factors = patsy_add_levels(formula, df)
    assert formula_upd is not None
    assert 'levels=[a,b]' in formula_upd
    assert factors == ['x1']


def test_patsy_strip_categorical():
    # String input
    assert patsy_strip_categorical('C(x1, Treatment(1))') == 'C(x1)'
    # List/Iterable input
    assert patsy_strip_categorical(['C(x1, Treatment(1))', 'x2']) == ['C(x1)', 'x2']


def test_patsy_strip_formula():
    assert patsy_strip_formula('  y  ~  x1+x2  ') == 'y ~ x1 + x2'
    assert patsy_strip_formula('') == ''


if __name__ == '__main__':
    print('Running test_patsy_formula_to_varnames...')
    test_patsy_formula_to_varnames()
    print('Running test_patsy_formula_to_categorical_varnames...')
    test_patsy_formula_to_categorical_varnames()
    print('Running test_patsy_formula_to_categorical_treatments...')
    test_patsy_formula_to_categorical_treatments()
    print('Running test_patsy_add_levels...')
    test_patsy_add_levels()
    print('Running test_patsy_strip_categorical...')
    test_patsy_strip_categorical()
    print('Running test_patsy_strip_formula...')
    test_patsy_strip_formula()
    print('All tests passed successfully!')
