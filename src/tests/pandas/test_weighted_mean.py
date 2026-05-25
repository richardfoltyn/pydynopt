"""Unit tests for df_weighted_mean."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from pydynopt.pandas import df_weighted_mean
from pydynopt.utils import anything_to_list

# Dimension sizes to be used for tests
DIMS = (0, 1, 2, 11)


@pytest.fixture(params=[1234])
def rng(request):
    """Return a reproducible NumPy random generator."""
    seed = request.param
    rng = np.random.default_rng(seed)
    return rng


@pytest.fixture(params=[1, 2, 3])
def nlevels(request) -> int:
    """Return the number of index levels used for generated data."""
    return request.param


@pytest.fixture
def shapes(nlevels) -> np.ndarray:
    """Return all nlevels-dimensional shape combinations from DIMS."""
    dims = np.array(DIMS)
    ii = np.meshgrid(*((dims,) * nlevels))
    ii = np.hstack([i.reshape((-1, 1)) for i in ii])

    return ii


def create_data(
    shape: tuple[int],
    weights: bool,
    rng,
    na_count: int = 0,
    columns: str | Sequence[str] = 'data',
) -> pd.DataFrame:
    """
    Create a DataFrame with MultiIndex levels determined by given `shape`.

    Parameters
    ----------
    shape : tuple of int
        Shape of the MultiIndex.
    weights : bool
        If True, add a column 'weight'
    rng
        RNG instance to generate data and weights.
    na_count : int
        Number of NaNs to insert within each inner-most index level.
    columns : str or Sequence of str
        Data columns to generate.

    Returns
    -------
    pd.DataFrame
    """
    nlevels = len(shape)
    levels = [f'level{i}' for i in range(nlevels)]

    if nlevels == 1:
        idx = pd.Index(np.arange(shape[0]), name='level0')
    else:
        idx = pd.MultiIndex.from_product([np.arange(i) for i in shape], names=levels)

    columns = anything_to_list(columns)
    df = pd.DataFrame(index=idx, columns=columns, dtype=float)
    df.iloc[:, :] = rng.normal(size=df.shape)

    if na_count > 0 and len(df) > 0:
        if len(levels) == 1:
            size = min(na_count, len(df))
            ii = rng.choice(len(df), size=size, replace=False)
            df.iloc[ii, 0] = np.nan
        else:
            # Inject NaNs by inner-most level
            levels = levels[:-1]

            def _fcn(x):
                size = min(len(x), na_count)
                ii = rng.choice(len(x), size=size, replace=False)
                x = x.copy(deep=True)
                x.iloc[ii] = np.nan
                return x

            df = df.groupby(levels).transform(_fcn)

    if weights:
        df['weight'] = rng.uniform(size=len(df))

    return df


def get_groups(data: pd.DataFrame) -> list[list[str]]:
    """Return all non-trivial index-level combinations for groupby."""
    levels = list(data.index.names)
    nlevels = len(levels)

    # List of unique combinations of nlevels - 1 elements
    ii = np.arange(2)
    ii = np.meshgrid(*((ii,) * nlevels))
    ii = np.hstack([i.reshape((-1, 1)) for i in ii])

    s = np.sum(ii, axis=1)
    keep = (s > 0) & (s < nlevels)
    ii = ii[keep]

    groups = [[levels[j] for j, flag in enumerate(row) if flag == 1] for row in ii]

    return groups


@pytest.fixture(params=[True, False])
def use_weights(request) -> bool:
    """Return whether weighted aggregation should be used in a test case."""
    return request.param


@pytest.fixture(params=[0, 1, 2])
def na_count(request):
    """Return the number of NaNs to inject per innermost group."""
    return request.param


@pytest.fixture(params=[1, 2, 3])
def na_min_count(request):
    """Return the minimum observations required before computing a mean."""
    return request.param


def test_means(shapes, use_weights: bool, na_count: int, na_min_count: int, rng):
    """Verify grouped means against a manual weighted/unweighted reference."""
    for shape in shapes:
        shape = tuple(shape)
        data = create_data(shape, weights=use_weights, rng=rng, na_count=na_count)

        groups = get_groups(data)

        # Add case of no groups
        groups.insert(0, None)

        for group in groups:
            res = df_weighted_mean(
                data,
                groups=group,
                weights='weight' if use_weights else None,
                na_min_count=na_min_count,
            )

            d = data.copy(deep=True)
            if not use_weights:
                d['weight'] = 1.0
            d['data'] *= d['weight']
            d['weight'] = np.where(d['data'].notna(), d['weight'], 0.0)

            if group:
                d['data'] /= d['weight'].groupby(group).sum()
                desired = (
                    d['data']
                    .groupby(group)
                    .sum(min_count=max(1, na_min_count))
                    .to_frame()
                )
            else:
                d['data'] /= d['weight'].sum()
                desired = pd.DataFrame(
                    d['data'].sum(min_count=max(1, na_min_count)),
                    index=[0],
                    columns=['data'],
                )

            assert np.all(res.notna() == desired.notna())

            # Pandas has some issues with multidimensional boolean masks, convert to
            # numpy to compare non-NaN values
            res = res.to_numpy()
            desired = desired.to_numpy()

            mask = ~np.isnan(res)
            assert np.all(np.abs(res[mask] - desired[mask]) < 1.0e-10)


def test_arg_weights(rng):
    """Verify equivalent results for default, string, and Series weights."""
    shape = (2, 3, 4)

    data = create_data(shape, weights=True, rng=rng)
    weights = data['weight'].copy(deep=True)

    groups = get_groups(data)

    for group in groups:
        # Use default value
        res1 = df_weighted_mean(data, groups=group)

        # Pass weights as string
        res2 = df_weighted_mean(data, groups=group, weights='weight')

        # Pass weights as Series argument
        d2 = data.drop(columns=['weight'])
        res3 = df_weighted_mean(d2, groups=group, weights=weights)

        assert np.all(res1.notna() == res2.notna())
        assert np.all(res1.notna() == res3.notna())

        res1 = res1.to_numpy()
        res2 = res2.to_numpy()
        res3 = res3.to_numpy()

        mask = ~np.isnan(res1)

        assert np.all(np.abs(res1[mask] - res2[mask]) < 1.0e-10)
        assert np.all(np.abs(res1[mask] - res3[mask]) < 1.0e-10)


def test_arg_weights_ndarray(rng):
    """Verify ndarray weights match results from a weight column."""
    shape = (2, 3, 4)

    data = create_data(shape, weights=True, rng=rng)
    weights = data['weight'].to_numpy(copy=True)

    groups = get_groups(data)
    groups.insert(0, None)

    for group in groups:
        res1 = df_weighted_mean(data, groups=group, weights='weight')

        d2 = data.drop(columns=['weight'])
        res2 = df_weighted_mean(d2, groups=group, weights=weights)

        assert np.all(res1.notna() == res2.notna())

        res1 = res1.to_numpy()
        res2 = res2.to_numpy()

        mask = ~np.isnan(res1)
        assert np.all(np.abs(res1[mask] - res2[mask]) < 1.0e-10)


def test_multiindex(rng):
    """Verify MultiIndex output has expected level names and moments."""
    shape = (5, 4, 11)

    data = create_data(shape, weights=True, rng=rng)

    groups = get_groups(data)

    for group in groups:
        res = df_weighted_mean(data, groups=group, weights='weight', multi_index=True)

        levels = list(res.columns.names)
        assert levels == ['Variable', 'Moment']

        mom = list(res.columns.get_level_values('Moment'))
        assert mom == ['Mean']


def test_series(rng):
    """Verify Series input matches DataFrame input for identical data."""
    shape = (1, 3, 1)

    data = create_data(shape, weights=True, rng=rng)
    weights = data['weight'].copy()
    series = data['data'].copy()

    groups = get_groups(data)

    for group in groups:
        # DataFrame
        res1 = df_weighted_mean(data, groups=group, weights='weight')

        # Series
        res2 = df_weighted_mean(series, groups=group, weights=weights)

        assert np.all(np.abs(res1 - res2.to_frame()) < 1.0e-10)


def test_varlist_scalar_returns_series(rng):
    """Verify scalar varlist returns a named Series output."""
    shape = (3, 5)

    data = create_data(shape, weights=True, rng=rng, columns=['v1', 'v2'])
    res = df_weighted_mean(
        data,
        groups=['level0'],
        varlist='v1',
        weights='weight',
    )

    assert isinstance(res, pd.Series)
    assert res.name == 'v1'


def test_na_min_count_validation(rng):
    """Verify invalid na_min_count raises ValueError."""
    data = create_data((2, 3), weights=True, rng=rng)

    with pytest.raises(ValueError, match='na_min_count'):
        df_weighted_mean(data, weights='weight', na_min_count=0)


def test_nonfinite_weights_are_ignored():
    """Verify observations with non-finite weights are excluded."""
    data = pd.DataFrame(
        {
            'data': [1.0, 2.0, 3.0, 4.0],
            'weight': [1.0, np.nan, np.inf, 2.0],
        }
    )

    res = df_weighted_mean(data, weights='weight', varlist=['data'])
    desired = (1.0 * 1.0 + 4.0 * 2.0) / (1.0 + 2.0)

    assert np.isclose(res.loc[0, 'data'], desired)


def test_varlist(rng):
    """Verify implicit and explicit varlist selection rules."""
    shape = (4, 10)

    data = create_data(shape, weights=True, rng=rng, columns=['v1', 'v2'])
    weights = data['weight'].copy()

    groups = get_groups(data)

    for group in groups:
        # Use default, all columns other than weight
        res1 = df_weighted_mean(data, groups=group, weights='weight')

        # Use default, all columns since no weight is in DataFrame
        d2 = data.drop(columns='weight')
        res2 = df_weighted_mean(d2, groups=group, weights=weights)

        assert np.all(np.abs(res1 - res2) < 1.0e-10)

        # Use subset of columns (v1)
        res1 = df_weighted_mean(data, groups=group, weights='weight', varlist=['v1'])

        # Use default, all columns since no weight is in DataFrame
        res2 = df_weighted_mean(data[['v1']], groups=group, weights=weights)

        assert np.all(np.abs(res1 - res2) < 1.0e-10)

        # Use subset of columns (v2)
        res1 = df_weighted_mean(data, groups=group, weights='weight', varlist=['v2'])

        # Use default, all columns since no weight is in DataFrame
        res2 = df_weighted_mean(data[['v2']], groups=group, weights=weights)

        assert np.all(np.abs(res1 - res2) < 1.0e-10)


def test_nobs(shapes, use_weights: bool, na_count: int, na_min_count: int, rng):
    """Verify the optional observation-count output column."""
    columns = 'data'
    nobs_column = 'Nobs'

    for shape in shapes:
        shape = tuple(shape)
        data = create_data(
            shape, weights=use_weights, rng=rng, na_count=na_count, columns=columns
        )

        groups = get_groups(data)

        # Add case of no groups
        groups.insert(0, None)

        for group in groups:
            res = df_weighted_mean(
                data,
                groups=group,
                weights='weight' if use_weights else None,
                na_min_count=na_min_count,
                nobs_column=nobs_column,
            )

            d = data.copy(deep=True)
            if not use_weights:
                d['weight'] = 1.0
            d['data'] *= d['weight']
            d['weight'] = np.where(d['data'].notna(), d['weight'], 0.0)

            if group:
                desired_nobs = pd.Series(
                    (d['weight'] > 0).groupby(group).sum(), name=(columns, nobs_column)
                )
            else:
                desired_nobs = pd.Series(
                    (d['weight'] > 0).sum(), name=(columns, nobs_column)
                )

            assert np.all(res[(columns, nobs_column)] == desired_nobs)
            # assert np.all(res[(columns, 'weight')] == desired_wgt)


def test_sum_weights(shapes, na_count: int, na_min_count: int, rng):
    """Verify the optional sum-of-weights output column."""
    columns = 'data'

    for shape in shapes:
        shape = tuple(shape)
        data = create_data(
            shape, weights=True, rng=rng, na_count=na_count, columns=columns
        )

        groups = get_groups(data)

        # Add case of no groups
        groups.insert(0, None)

        for group in groups:
            res = df_weighted_mean(
                data,
                groups=group,
                weights='weight',
                na_min_count=na_min_count,
                add_weights_column=True,
            )

            d = data.copy(deep=True)
            d['data'] *= d['weight']
            d['weight'] = np.where(d['data'].notna(), d['weight'], 0.0)

            if group:
                desired_wgt = pd.Series(
                    d['weight'].groupby(group).sum(), name=(columns, 'weight')
                )
            else:
                desired_wgt = pd.Series(d['weight'].sum(), name=(columns, 'weight'))

            assert np.all(res[(columns, 'weight')] == desired_wgt)


def test_index_groups(shapes, rng):
    """Verify identical results for index-based and column-based groups."""
    for shape in shapes:
        shape = tuple(shape)
        data = create_data(shape, weights=True, rng=rng)

        groups = get_groups(data)

        # Add case of no groups
        groups.insert(0, None)

        for group in groups:
            res1 = df_weighted_mean(
                data,
                groups=group,
                weights='weight',
            )

            # Compute with grouping variables not in index
            res2 = df_weighted_mean(
                data.reset_index(drop=False),
                groups=group,
                weights='weight',
                varlist=['data'],
            )

            assert np.all(res1.notna() == res2.notna())

            # Pandas has some issues with multidimensional boolean masks, convert to
            # numpy to compare non-NaN values
            res1 = res1.to_numpy()
            res2 = res2.to_numpy()

            mask = ~np.isnan(res1)
            assert np.all(np.abs(res1[mask] - res2[mask]) < 1.0e-10)


def test_index_groups_series_weights(shapes, rng):
    """Verify Series-weight results for index-based and column-based groups."""
    for shape in shapes:
        shape = tuple(shape)
        data = create_data(shape, weights=True, rng=rng)
        weights = data['weight'].copy(deep=True)

        groups = get_groups(data)
        groups.insert(0, None)

        for group in groups:
            res1 = df_weighted_mean(
                data.drop(columns=['weight']),
                groups=group,
                weights=weights,
                varlist=['data'],
            )

            # Compute with grouping variables not in index.
            res2 = df_weighted_mean(
                data.reset_index(drop=False).drop(columns=['weight']),
                groups=group,
                weights=weights,
                varlist=['data'],
            )

            assert np.all(res1.notna() == res2.notna())

            res1 = res1.to_numpy()
            res2 = res2.to_numpy()

            mask = ~np.isnan(res1)
            assert np.all(np.abs(res1[mask] - res2[mask]) < 1.0e-10)
