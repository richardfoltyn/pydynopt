"""Unit tests for percentile."""

import numpy as np
import pandas as pd

from pydynopt.pandas import percentile


def _sample_data() -> pd.DataFrame:
    """Return a deterministic sample DataFrame for percentiles."""
    return pd.DataFrame(
        {
            'x': [1.0, 2.0, 3.0, 4.0],
            'y': [10.0, 20.0, 30.0, 40.0],
            'weight': [1.0, 1.0, 1.0, 1.0],
        }
    )


def test_percentile_scalar_return() -> None:
    """Verify that percentile returns a float when prank and varlist are scalar."""
    df_data = _sample_data()

    result = percentile(df_data, prank=50.0, varlist='x', weight_var='weight')

    assert isinstance(result, float)
    assert np.isclose(result, 2.0)


def test_percentile_multi_index_return() -> None:
    """Verify that percentile returns a DataFrame when multi_index is True."""
    df_data = _sample_data()

    result = percentile(
        df_data,
        prank=50.0,
        varlist='x',
        weight_var='weight',
        multi_index=True,
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 1)
    assert np.isclose(result.iloc[0, 0], 2.0)


def test_percentile_vector_prank_return() -> None:
    """Verify that percentile returns a DataFrame when prank is a sequence."""
    df_data = _sample_data()

    result = percentile(df_data, prank=[25.0, 75.0], varlist='x', weight_var='weight')

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1)
    assert np.isclose(result.iloc[0, 0], 1.0)
    assert np.isclose(result.iloc[1, 0], 3.0)


def test_percentile_vector_varlist_return() -> None:
    """Verify that percentile returns a DataFrame when varlist is a sequence."""
    df_data = _sample_data()

    result = percentile(df_data, prank=50.0, varlist=['x', 'y'], weight_var='weight')

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 2)
    assert np.isclose(result.loc[50, 'x'], 2.0)
    assert np.isclose(result.loc[50, 'y'], 20.0)


def test_percentile_default_varlist_return() -> None:
    """Verify that percentile returns a DataFrame when varlist is None."""
    df_data = _sample_data()

    result = percentile(df_data, prank=50.0, weight_var='weight')

    assert isinstance(result, pd.DataFrame)
    assert 'x' in result.columns
    assert 'y' in result.columns
    assert np.isclose(result.loc[50, 'x'], 2.0)
    assert np.isclose(result.loc[50, 'y'], 20.0)
