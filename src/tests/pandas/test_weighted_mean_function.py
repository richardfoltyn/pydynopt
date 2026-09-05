"""Unit tests for weighted_mean."""

import numpy as np
import pandas as pd
import pytest

from pydynopt.pandas import weighted_mean


def _sample_data() -> pd.DataFrame:
    """Return a deterministic sample DataFrame with NaNs."""
    return pd.DataFrame(
        {
            'x': [1.0, 2.0, np.nan, 4.0],
            'y': [2.0, np.nan, 6.0, 8.0],
            'weight': [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_weighted_mean_values_with_weight_column() -> None:
    """Verify weighted means match manual reference values."""
    df_data = _sample_data()

    result = weighted_mean(df_data, varlist=['x', 'y'], weights='weight')

    assert isinstance(result, pd.Series)
    assert np.isclose(result.loc['x'], 3.0)
    assert np.isclose(result.loc['y'], 6.5)


def test_unweighted_mean_values() -> None:
    """Verify unweighted means ignore NaNs and match pandas mean."""
    df_data = _sample_data()

    result = weighted_mean(df_data, varlist=['x', 'y'], weights=None)
    expected = df_data[['x', 'y']].mean()

    assert isinstance(result, pd.Series)
    assert np.allclose(result.to_numpy(), expected.to_numpy())


def test_weights_argument_forms_are_equivalent() -> None:
    """Verify string, Series, and ndarray weights produce identical output."""
    df_data = _sample_data()

    res_str = weighted_mean(df_data, varlist=['x', 'y'], weights='weight')
    res_series = weighted_mean(
        df_data.drop(columns=['weight']),
        varlist=['x', 'y'],
        weights=df_data['weight'],
    )
    res_array = weighted_mean(
        df_data.drop(columns=['weight']),
        varlist=['x', 'y'],
        weights=df_data['weight'].to_numpy(),
    )

    assert np.allclose(res_str.to_numpy(), res_series.to_numpy())
    assert np.allclose(res_str.to_numpy(), res_array.to_numpy())


def test_scalar_return_for_dataframe_when_requested() -> None:
    """Verify DataFrame input can return scalar for single variable."""
    df_data = _sample_data()

    result = weighted_mean(
        df_data,
        varlist='x',
        weights='weight',
        index_varlist=False,
        multi_index=False,
    )

    assert isinstance(result, float)
    assert np.isclose(result, 3.0)


def test_scalar_return_for_series_when_requested() -> None:
    """Verify Series input returns scalar with explicit scalar flags."""
    series = pd.Series([1.0, 2.0, np.nan, 4.0], name='x')
    weights = np.array([1.0, 2.0, 3.0, 4.0])

    result = weighted_mean(
        series,
        weights=weights,
        index_varlist=False,
        multi_index=False,
    )

    assert isinstance(result, float)
    assert np.isclose(result, 3.0)


def test_multi_index_output() -> None:
    """Verify multi-index output structure for weighted means."""
    df_data = _sample_data()

    result = weighted_mean(
        df_data, varlist=['x', 'y'], weights='weight', multi_index=True
    )

    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ['Variable', 'Moment']
    assert list(result.index.get_level_values('Moment')) == ['Mean', 'Mean']


def test_nonfinite_weights_are_excluded() -> None:
    """Verify rows with non-finite weights do not contribute to means."""
    df_data = pd.DataFrame(
        {
            'x': [1.0, 2.0, 3.0, 4.0],
            'weight': [1.0, np.nan, np.inf, 2.0],
        }
    )

    result = weighted_mean(
        df_data,
        varlist='x',
        weights='weight',
        index_varlist=False,
        multi_index=False,
    )

    assert np.isclose(result, 3.0)


def test_invalid_weight_column_raises() -> None:
    """Verify missing weight column raises ValueError."""
    df_data = _sample_data()

    with pytest.raises(ValueError, match='Unsupported weight argument'):
        weighted_mean(df_data, varlist=['x', 'y'], weights='missing_weight')


def test_weight_array_length_mismatch_raises() -> None:
    """Verify ndarray length mismatch raises ValueError."""
    df_data = _sample_data().drop(columns=['weight'])

    with pytest.raises(ValueError, match='Length of weights does not match'):
        weighted_mean(df_data, varlist=['x', 'y'], weights=np.array([1.0, 2.0]))


def test_legacy_weight_var_matches_weights_argument() -> None:
    """Verify legacy weight_var parameter matches weights string behavior."""
    df_data = _sample_data()

    res_new = weighted_mean(df_data, varlist=['x', 'y'], weights='weight')
    res_legacy = weighted_mean(df_data, varlist=['x', 'y'], weight_var='weight')

    assert np.allclose(res_new.to_numpy(), res_legacy.to_numpy())
