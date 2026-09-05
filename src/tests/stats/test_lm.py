"""Unit tests for linear-model utilities."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pydynopt.stats.lm import areg, demean_within


@pytest.fixture
def df_areg() -> pd.DataFrame:
    """Create deterministic, unbalanced panel data for ``areg`` tests."""
    df_data = pd.DataFrame(
        {
            'group': ['a'] * 4 + ['b'] * 3 + ['c'] * 5 + ['d'] * 4,
            'x1': [
                0.0,
                1.0,
                3.0,
                4.0,
                -1.0,
                2.0,
                5.0,
                0.0,
                2.0,
                3.0,
                6.0,
                7.0,
                -2.0,
                1.0,
                4.0,
                8.0,
            ],
            'x2': [
                2.0,
                -1.0,
                0.0,
                3.0,
                1.0,
                4.0,
                -2.0,
                -3.0,
                2.0,
                5.0,
                1.0,
                4.0,
                0.0,
                -2.0,
                3.0,
                6.0,
            ],
            'weight': [
                1.0,
                2.0,
                1.0,
                3.0,
                2.0,
                4.0,
                1.0,
                3.0,
                1.0,
                2.0,
                4.0,
                1.0,
                2.0,
                3.0,
                1.0,
                5.0,
            ],
        }
    )
    effects = df_data['group'].map({'a': -2.0, 'b': 1.5, 'c': 4.0, 'd': 8.0})
    noise = np.array(
        [
            0.2,
            -0.1,
            0.3,
            -0.2,
            -0.4,
            0.1,
            0.5,
            -0.3,
            0.4,
            -0.2,
            0.1,
            0.3,
            0.2,
            -0.5,
            0.4,
            -0.1,
        ]
    )
    df_data['y'] = 1.25 + 0.8 * df_data['x1'] - 0.35 * df_data['x2']
    df_data['y'] += effects + noise
    return df_data


def test_demean_within_series_no_weights():
    # Setup data: 2 groups, 3 observations in group A, 2 in group B
    index = pd.Index(['A', 'A', 'A', 'B', 'B'], name='group')
    x = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0], index=index, name='val')

    # Demean without weights
    x_dem, x_mean = demean_within(x, 'group')

    # Expected means: group A mean is 2.0, group B mean is 15.0
    # Expected demeaned:
    # A: [1.0 - 2.0, 2.0 - 2.0, 3.0 - 2.0] = [-1.0, 0.0, 1.0]
    # B: [10.0 - 15.0, 20.0 - 15.0] = [-5.0, 5.0]
    expected_dem = pd.Series([-1.0, 0.0, 1.0, -5.0, 5.0], index=index, name='val')

    # Expected x_mean_first (group means, one per group)
    expected_mean = pd.Series(
        [2.0, 15.0], index=pd.Index(['A', 'B'], name='group'), name='val'
    )

    pd.testing.assert_series_equal(x_dem, expected_dem)
    pd.testing.assert_series_equal(x_mean, expected_mean)


def test_demean_within_series_weighted():
    # Setup data
    index = pd.Index(['A', 'A', 'B', 'B'], name='group')
    x = pd.Series([1.0, 2.0, 10.0, 20.0], index=index, name='val')

    # Case 1: rescale_weights=False (weights already sum to 1 within group)
    # Group A weights: 0.25, 0.75 (sum to 1) -> mean: 1.0*0.25 + 2.0*0.75 = 1.75
    # Group B weights: 0.4, 0.6 (sum to 1) -> mean: 10.0*0.4 + 20.0*0.6 = 16.0
    weights = pd.Series([0.25, 0.75, 0.4, 0.6], index=index, name='weight')

    x_dem, x_mean = demean_within(x, 'group', weights=weights, rescale_weights=False)

    expected_dem = pd.Series(
        [1.0 - 1.75, 2.0 - 1.75, 10.0 - 16.0, 20.0 - 16.0], index=index, name='val'
    )
    expected_mean = pd.Series(
        [1.75, 16.0], index=pd.Index(['A', 'B'], name='group'), name='val'
    )

    pd.testing.assert_series_equal(x_dem, expected_dem, check_names=False)
    pd.testing.assert_series_equal(x_mean, expected_mean, check_names=False)

    # Case 2: rescale_weights=True (weights do not sum to 1 within group)
    # Group A weights: 1.0, 3.0 -> sum=4.0 -> normalized: 0.25, 0.75
    # Group B weights: 2.0, 3.0 -> sum=5.0 -> normalized: 0.4, 0.6
    weights_raw = pd.Series([1.0, 3.0, 2.0, 3.0], index=index, name='weight')

    x_dem, x_mean = demean_within(x, 'group', weights=weights_raw, rescale_weights=True)

    pd.testing.assert_series_equal(x_dem, expected_dem, check_names=False)
    pd.testing.assert_series_equal(x_mean, expected_mean, check_names=False)


def test_demean_within_dataframe():
    # Setup data with DataFrame
    index = pd.Index(['A', 'A', 'B', 'B'], name='group')
    df = pd.DataFrame(
        {'val1': [1.0, 3.0, 10.0, 30.0], 'val2': [2.0, 4.0, 20.0, 40.0]}, index=index
    )

    # Without weights
    df_dem, df_mean = demean_within(df, 'group')

    # Expected means:
    # val1: A -> 2.0, B -> 20.0
    # val2: A -> 3.0, B -> 30.0
    # Expected demeaned:
    # val1: A -> [-1.0, 1.0], B -> [-10.0, 10.0]
    # val2: A -> [-1.0, 1.0], B -> [-10.0, 10.0]
    expected_dem = pd.DataFrame(
        {'val1': [-1.0, 1.0, -10.0, 10.0], 'val2': [-1.0, 1.0, -10.0, 10.0]},
        index=index,
    )

    expected_mean = pd.DataFrame(
        {'val1': [2.0, 20.0], 'val2': [3.0, 30.0]},
        index=pd.Index(['A', 'B'], name='group'),
    )

    pd.testing.assert_frame_equal(df_dem, expected_dem)
    pd.testing.assert_frame_equal(df_mean, expected_mean)


def test_demean_within_multiple_groups():
    # Setup data with multi-group variables
    index = pd.MultiIndex.from_tuples(
        [('A', 1), ('A', 1), ('A', 2), ('B', 1), ('B', 1)], names=['group1', 'group2']
    )

    x = pd.Series([1.0, 3.0, 10.0, 100.0, 200.0], index=index, name='val')

    x_dem, x_mean = demean_within(x, ['group1', 'group2'])

    # Expected means:
    # (A, 1) -> mean of [1.0, 3.0] = 2.0
    # (A, 2) -> mean of [10.0] = 10.0
    # (B, 1) -> mean of [100.0, 200.0] = 150.0
    # Expected demeaned:
    # (A, 1) -> [-1.0, 1.0]
    # (A, 2) -> [0.0]
    # (B, 1) -> [-50.0, 50.0]
    expected_dem = pd.Series([-1.0, 1.0, 0.0, -50.0, 50.0], index=index, name='val')

    expected_mean = pd.Series(
        [2.0, 10.0, 150.0],
        index=pd.MultiIndex.from_tuples(
            [('A', 1), ('A', 2), ('B', 1)], names=['group1', 'group2']
        ),
        name='val',
    )

    pd.testing.assert_series_equal(x_dem, expected_dem)
    pd.testing.assert_series_equal(x_mean, expected_mean)


@pytest.mark.parametrize('weighted', [False, True], ids=['ols', 'wls'])
def test_areg_matches_explicit_fixed_effects(
    df_areg: pd.DataFrame,
    weighted: bool,
) -> None:
    """Match OLS/WLS with an explicit indicator for every fixed effect."""
    formula = 'y ~ x1 + x2'
    reference_formula = f'{formula} + C(group)'

    if weighted:
        result, _ = areg('group', formula=formula, data=df_areg, weights='weight')
        weights = df_areg['weight'] / df_areg['weight'].sum()
        reference = smf.wls(reference_formula, data=df_areg, weights=weights).fit()
    else:
        result, _ = areg('group', formula=formula, data=df_areg)
        reference = smf.ols(reference_formula, data=df_areg).fit()

    regressors = ['x1', 'x2']
    np.testing.assert_allclose(result.params[regressors], reference.params[regressors])
    np.testing.assert_allclose(result.bse[regressors], reference.bse[regressors])
    np.testing.assert_allclose(result.resid, reference.resid)
    assert result.nobs == reference.nobs
    assert result.df_resid == reference.df_resid
    assert result.rsquared == pytest.approx(reference.rsquared)


def test_areg_is_invariant_to_group_outcome_shifts(df_areg: pd.DataFrame) -> None:
    """Absorb arbitrary group-level shifts without changing within estimates."""
    result, metrics = areg(
        'group', formula='y ~ x1 + x2', data=df_areg, weights='weight'
    )
    df_shifted = df_areg.copy()
    shifts = df_shifted['group'].map({'a': 10.0, 'b': -3.0, 'c': 25.0, 'd': 1.0})
    df_shifted['y'] += shifts

    shifted_result, shifted_metrics = areg(
        'group', formula='y ~ x1 + x2', data=df_shifted, weights='weight'
    )

    regressors = ['x1', 'x2']
    np.testing.assert_allclose(
        result.params[regressors], shifted_result.params[regressors]
    )
    np.testing.assert_allclose(result.resid, shifted_result.resid)
    assert shifted_metrics.within == pytest.approx(metrics.within)


def test_areg_formula_and_matrix_interfaces_agree(df_areg: pd.DataFrame) -> None:
    """Produce identical estimates through formula and matrix interfaces."""
    formula_result, formula_metrics = areg(
        'group', formula='y ~ x1 + x2', data=df_areg, weights='weight'
    )

    df_indexed = df_areg.set_index('group')
    endog = df_indexed['y']
    df_exog = df_indexed[['x1', 'x2']].copy()
    df_exog.insert(0, 'Intercept', 1.0)
    matrix_result, matrix_metrics = areg(
        'group',
        endog=endog,
        exog=df_exog,
        weights=df_indexed['weight'],
    )

    np.testing.assert_allclose(formula_result.params, matrix_result.params)
    np.testing.assert_allclose(formula_result.bse, matrix_result.bse)
    np.testing.assert_allclose(formula_result.resid, matrix_result.resid)
    assert matrix_result.df_resid == formula_result.df_resid
    assert matrix_result.rsquared == pytest.approx(formula_result.rsquared)
    for name in ('within', 'between', 'overall'):
        assert getattr(matrix_metrics, name) == pytest.approx(
            getattr(formula_metrics, name)
        )


def test_areg_r_squared_metrics(df_areg: pd.DataFrame) -> None:
    """Match independently calculated fixed-effect R-squared measures."""
    result, metrics = areg(
        'group', formula='y ~ x1 + x2', data=df_areg, weights='weight'
    )
    weights = df_areg['weight']

    weighted_y = df_areg['y'] * weights
    weight_sum_obs = weights.groupby(df_areg['group'], observed=True).transform('sum')
    y_mean_obs = (
        weighted_y.groupby(df_areg['group'], observed=True).transform('sum')
        / weight_sum_obs
    )
    ssr = np.sum(weights.to_numpy() * np.square(np.asarray(result.resid)))
    tss_within = np.sum(weights * np.square(df_areg['y'] - y_mean_obs))
    expected_within = 1.0 - ssr / tss_within

    df_weighted = df_areg.assign(
        weighted_y=weights * df_areg['y'],
        weighted_x1=weights * df_areg['x1'],
        weighted_x2=weights * df_areg['x2'],
    )
    grouped = df_weighted.groupby('group', observed=True)
    weight_sum = grouped['weight'].sum()
    y_mean = grouped['weighted_y'].sum() / weight_sum
    x1_mean = grouped['weighted_x1'].sum() / weight_sum
    x2_mean = grouped['weighted_x2'].sum() / weight_sum

    y_mean_hat = (
        result.params['Intercept']
        + result.params['x1'] * x1_mean
        + result.params['x2'] * x2_mean
    )
    expected_between = np.corrcoef(y_mean, y_mean_hat)[0, 1] ** 2

    y_hat = (
        result.params['Intercept']
        + result.params['x1'] * df_areg['x1']
        + result.params['x2'] * df_areg['x2']
    )
    expected_overall = np.corrcoef(df_areg['y'], y_hat)[0, 1] ** 2

    reference = smf.wls('y ~ x1 + x2 + C(group)', data=df_areg, weights=weights).fit()

    assert metrics.within == pytest.approx(expected_within)
    assert metrics.between == pytest.approx(expected_between)
    assert metrics.overall == pytest.approx(expected_overall)
    assert result.rsquared == pytest.approx(reference.rsquared)


def test_areg_ignores_zero_weight_observations(df_areg: pd.DataFrame) -> None:
    """Treat zero-weight observations exactly like omitted observations."""
    df_zero = df_areg.copy()
    df_zero.loc[1, 'weight'] = 0.0
    df_zero.loc[df_zero['group'].eq('d'), 'weight'] = 0.0
    keep = df_zero['weight'] > 0.0
    df_dropped = df_zero.loc[keep].copy()

    result, metrics = areg(
        'group', formula='y ~ x1 + x2', data=df_zero, weights='weight'
    )
    expected, expected_metrics = areg(
        'group', formula='y ~ x1 + x2', data=df_dropped, weights='weight'
    )

    np.testing.assert_allclose(result.params, expected.params)
    np.testing.assert_allclose(result.bse, expected.bse)
    np.testing.assert_allclose(result.resid, expected.resid)
    assert result.nobs == expected.nobs
    assert result.df_resid == expected.df_resid
    assert result.rsquared == pytest.approx(expected.rsquared)
    for name in ('within', 'between', 'overall'):
        assert getattr(metrics, name) == pytest.approx(getattr(expected_metrics, name))


def test_areg_aligns_weights_after_patsy_drops_rows(df_areg: pd.DataFrame) -> None:
    """Align weights and fixed effects with rows retained by Patsy."""
    df_missing = df_areg.copy()
    df_missing.loc[2, 'x1'] = np.nan
    df_missing.loc[df_missing['group'].eq('d'), 'x2'] = np.nan
    keep = df_missing[['y', 'x1', 'x2']].notna().all(axis=1)
    df_dropped = df_missing.loc[keep].copy()

    result, metrics = areg(
        'group', formula='y ~ x1 + x2', data=df_missing, weights='weight'
    )
    expected, expected_metrics = areg(
        'group', formula='y ~ x1 + x2', data=df_dropped, weights='weight'
    )

    np.testing.assert_allclose(result.params, expected.params)
    np.testing.assert_allclose(result.bse, expected.bse)
    np.testing.assert_allclose(result.resid, expected.resid)
    assert result.nobs == expected.nobs
    assert result.df_resid == expected.df_resid
    assert result.rsquared == pytest.approx(expected.rsquared)
    for name in ('within', 'between', 'overall'):
        assert getattr(metrics, name) == pytest.approx(getattr(expected_metrics, name))
