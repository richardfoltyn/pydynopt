"""Unit tests for interpolate_bin_weights."""

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
import pytest

from pydynopt.pandas import interpolate_bin_weights


def test_interpolate_bin_weights_basic_values() -> None:
    """Verify basic bin interpolation against a hand-computed reference."""
    values = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0], name='cdf')

    result = interpolate_bin_weights(edges=[0.0, 0.5, 1.0], values=values)

    expected_idx = pd.MultiIndex.from_tuples(
        [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (1, 3)],
        names=['ibin', 'cdf'],
    )
    expected = pd.Series([1.0, 1.0, 0.0, 0.0, 1.0, 1.0], index=expected_idx)
    expected.name = 'weight'

    assert_series_equal(result, expected)


def test_interpolate_bin_weights_custom_names() -> None:
    """Verify custom output names for bin level, value level, and series name."""
    result = interpolate_bin_weights(
        edges=[0.0, 0.5, 1.0],
        values=[0.0, 0.25, 0.5, 0.75, 1.0],
        name_bins='bin',
        name_values='grid',
        generate='w',
    )

    assert result.name == 'w'
    assert list(result.index.names) == ['bin', 'grid']


def test_interpolate_bin_weights_grouped_edges() -> None:
    """Verify grouped edge grids produce independent bin maps per group."""
    values = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0], name='cdf')

    idx = pd.MultiIndex.from_tuples(
        [('A', 0), ('A', 1), ('A', 2), ('B', 0), ('B', 1), ('B', 2)],
        names=['grp', 'iedge'],
    )
    df_edges = pd.DataFrame({'edge': [0.0, 0.5, 1.0, 0.0, 0.3, 1.0]}, index=idx)

    result = interpolate_bin_weights(edges=df_edges, values=values, name_bins='bin')

    assert list(result.index.names) == ['grp', 'bin', 'cdf']

    group_sums = result.groupby(level=['grp', 'cdf']).sum()
    assert np.allclose(group_sums.to_numpy(), 1.0)

    bins_a = sorted(set(result.xs('A', level='grp').index.get_level_values('bin')))
    bins_b = sorted(set(result.xs('B', level='grp').index.get_level_values('bin')))
    assert bins_a == [0, 1]
    assert bins_b == [0, 1]


def test_interpolate_bin_weights_values_need_two_points() -> None:
    """Verify values input with fewer than two points raises ValueError."""
    with pytest.raises(ValueError, match='at least 2 points'):
        interpolate_bin_weights(edges=[0.0, 1.0], values=[0.0])


def test_interpolate_bin_weights_values_must_be_nondecreasing() -> None:
    """Verify non-monotone values input raises ValueError."""
    with pytest.raises(ValueError, match='nondecreasing'):
        interpolate_bin_weights(edges=[0.0, 0.5, 1.0], values=[0.0, 0.8, 0.6, 1.0])


def test_interpolate_bin_weights_values_dataframe_single_column_only() -> None:
    """Verify multi-column values DataFrame input raises ValueError."""
    df_values = pd.DataFrame({'v1': [0.0, 0.5, 1.0], 'v2': [0.0, 0.5, 1.0]})

    with pytest.raises(ValueError, match='multiple columns'):
        interpolate_bin_weights(edges=[0.0, 1.0], values=df_values)


def test_interpolate_bin_weights_edges_dataframe_single_column_only() -> None:
    """Verify multi-column edges DataFrame input raises ValueError."""
    df_edges = pd.DataFrame({'e1': [0.0, 0.5, 1.0], 'e2': [0.0, 0.5, 1.0]})

    with pytest.raises(ValueError, match='multiple columns'):
        interpolate_bin_weights(edges=df_edges, values=[0.0, 0.5, 1.0])


def test_interpolate_bin_weights_values_series_multiindex_raises() -> None:
    """Verify values Series with MultiIndex raises ValueError."""
    idx = pd.MultiIndex.from_product([['g1', 'g2'], [0, 1]], names=['grp', 'i'])
    values = pd.Series([0.0, 0.5, 0.5, 1.0], index=idx)

    with pytest.raises(ValueError, match='multiple index levels'):
        interpolate_bin_weights(edges=[0.0, 1.0], values=values)


def test_interpolate_bin_weights_flat_regions_supported() -> None:
    """Verify interpolation remains bounded for values with flat segments."""
    values = [0.0, 0.0, 0.5, 0.5, 1.0]
    edges = [0.0, 0.25, 0.75, 1.0]

    result = interpolate_bin_weights(edges=edges, values=values)

    assert np.all(np.isfinite(result.to_numpy()))
    assert np.all((result.to_numpy() >= 0.0) & (result.to_numpy() <= 1.0))


def test_interpolate_bin_weights_randomized_full_cover_property() -> None:
    """Verify full-support bins partition all represented value-grid mass."""
    rng = np.random.default_rng(123)

    for _ in range(40):
        # Build nondecreasing values with occasional flat regions.
        values = np.cumsum(rng.choice([0.0, 0.01, 0.02, 0.05], size=40))
        if values[-1] == 0.0:
            values[-1] = 1.0
        values = values / values[-1]

        # Bins cover the full support from min(values) to max(values).
        edges = np.linspace(float(values[0]), float(values[-1]), 7)

        result = interpolate_bin_weights(
            edges=edges,
            values=values,
            name_values='grid',
            validate=True,
        )

        # With full-support bins, each represented grid location should be fully
        # partitioned across bins.
        sums = result.groupby(level='grid').sum().to_numpy()
        assert np.allclose(sums, 1.0, atol=1.0e-10)

        weights = result.to_numpy()
        assert np.all(weights >= -1.0e-12)
        assert np.all(weights <= 1.0 + 1.0e-12)
