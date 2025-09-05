"""
This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Iterable
from typing import Optional, Union, Sequence

import numpy as np
import pandas as pd

import pydynopt.stats
from pydynopt.numba import jit, has_numba
from pydynopt.utils import anything_to_list

__all__ = [
    "weighted_mean",
    "df_weighted_mean",
    "percentile",
    "weighted_pmf",
    "interpolate_bin_weights",
]


@jit(nopython=True, parallel=False)
def _weighed_mean_impl(data: np.ndarray, weights: np.ndarray) -> float:
    """
    Helper routine for fast weighted means of data with NAs.

    Parameters
    ----------
    data
    weights

    Returns
    -------
    float
    """

    N = len(data)

    sw = 0.0
    m = 0.0

    for i in range(N):
        xi = data[i]
        if np.isfinite(xi):
            w = weights[i]
            sw += w
            m += w * xi

    if sw > 0.0:
        m /= sw

    return m


def weighted_mean(
    data: pd.DataFrame,
    varlist: Optional[Union[str, Iterable[str]]] = None,
    weights: Optional[Union[str, pd.Series, np.ndarray]] = 'weight',
    weight_var: Optional[str] = None,
    index_varlist: bool = True,
    multi_index: bool = False,
    index_names: Optional[tuple[str]] = ('Variable', 'Moment'),
    dtype: Optional[Union[np.dtype, type]] = float,
) -> Union[float, pd.Series]:
    """
    Compute weighted mean of variable given by varname, ignoring any NaNs.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
    varlist : str or list of str, optional
        List of variables for which to compute weighted mean.
    weights : str or tuple or array_like, optional
    weight_var : str, optional
        Name of DataFrame column containing the weights.
    index_varlist : bool, optional
        If true, create index from variable list (slow).
    multi_index : bool, optional
        If true, insert an additional index level with value 'Mean'
        for each variable (very slow!).
    index_names : str or array_like, optional
        Names for (multi)-index levels of resulting Series.
    dtype : np.dtype or type, optional
        If present, determines dtype of output array.

    Returns
    -------
    pd.Series or float
        Series containing weighted means with variable names used as index.
    """

    isscalar = isinstance(varlist, str)

    if isinstance(data, pd.Series):
        data = data.to_frame('v0')
        varlist = ['v0']
        isscalar = True

    # Legacy: if weight_var is passed, interpret this is the column name
    # in df.
    if weight_var is not None:
        weights = weight_var

    if weights is not None and not isinstance(weights, (pd.Series, np.ndarray)):
        try:
            weight_var = weights
            weights = data[weight_var]
        except:
            raise ValueError('Unsupported weight argument')

    has_weights = weights is not None
    if has_weights:
        weights = np.array(weights, copy=False)

    varlist = anything_to_list(varlist)
    if varlist is None:
        varlist = [name for name in data.columns if name != weight_var]

    if not has_weights:
        means = data[varlist].sum().to_numpy()

    elif has_numba:
        means = np.full(len(varlist), fill_value=np.nan)
        for i, varname in enumerate(varlist):
            dnp = data[varname].to_numpy(copy=False)
            means[i] = _weighed_mean_impl(dnp, weights)
    else:
        # Find appropriate dtype for weighted values
        if dtype is None:
            dtypes = tuple(data[varname].dtype for varname in varlist)
            if has_weights:
                dtypes += (weights.dtype,)
            dtype = np.result_type(*dtypes)

        n = data.shape[0]
        mask = np.empty(n, dtype=np.bool_)
        var_weighted = np.empty(n, dtype=dtype)
        means = np.full(len(varlist), fill_value=np.nan)

        for i, varname in enumerate(varlist):
            dnp = data[varname].to_numpy(copy=False)
            np.isfinite(dnp, out=mask)

            if has_weights:
                sum_wgt = np.sum(weights, where=mask)
                np.multiply(data[varname].to_numpy(), weights, out=var_weighted)
                sum_var = np.sum(var_weighted, where=mask)
            else:
                sum_wgt = np.sum(mask)
                sum_var = np.sum(data[varname].to_numpy(), where=mask)

            means[i] = sum_var / sum_wgt

        # Force deallocated of temporary arrays.
        del mask, var_weighted

    if index_varlist or multi_index:
        index_names = anything_to_list(index_names)
        if multi_index:
            idx = pd.MultiIndex.from_product((varlist, ['Mean']), names=index_names)
        else:
            idx = pd.Index(varlist, name=index_names[0])
        result = pd.Series(means, index=idx)
    elif isscalar:
        result = means[0]
    else:
        result = pd.Series(means)

    return result


def df_weighted_mean(
    data: Union[pd.Series, pd.DataFrame],
    groups: Optional[Union[str, Iterable[str]]] = None,
    varlist: Optional[Union[str, Iterable[str]]] = None,
    *,
    weights: Optional[Union[pd.Series, np.ndarray, str]] = 'weight',
    na_min_count: int = 1,
    multi_index: bool = False,
    index_names: Optional[Union[str, Iterable[str]]] = ('Variable', 'Moment'),
    add_weights_column: bool = False,
    nobs_column: Optional[str] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Compute (within-group) weighted mean of variables.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
    groups : str or Iterable of str, optional
        List of variables defining groups.
    varlist : str or list of str, optional
        List of variables for which to compute weighted mean.
    weights : str or tuple or array_like, optional
    na_min_count : int, optional
        Groups with number of observations below this value are assigned NA.
    multi_index : bool, optional
        If true, insert an additional index level with value 'Mean'
        for each variable.
    index_names : str or array_like, optional
        Names for (multi)-index levels of resulting Series.
    add_weights_column : bool
        If true, add the sum of weights for each bin in the output DataFrame
        (implies `multi_index = True`)
    nobs_column: str, optional
        If not None, add the number of observations for each bin in the output DataFrame
        (implies `multi_index = True`)

    Returns
    -------
    pd.Series or pd.DataFrame
        Series containing weighted means with variable names used as index.
    """

    isscalar = isinstance(varlist, str) or isinstance(data, pd.Series)
    groups = anything_to_list(groups, force=True)

    # Force MultiIndex output if several stats are computed for each variable
    multi_index = multi_index or add_weights_column or nobs_column

    if na_min_count < 1:
        raise ValueError('Argument \'na_min_count\' must be positive')

    if isinstance(data, pd.Series):
        name = data.name or 'v0'
        data = data.to_frame(name)
        varlist = [name]

    weight_varname = None
    if isinstance(weights, str):
        weight_varname = weights
        weights = data[weights]

    # Extract default varlist
    varlist = anything_to_list(varlist)
    if varlist is None:
        varlist = [
            name
            for name in data.columns
            if name != weight_varname and name not in groups
        ]

    # Check that grouping variables are in index, otherwise put them there
    missing = any(group not in data.index.names for group in groups)
    if missing:
        data = data.reset_index(drop=list(data.index.names) == [None]).set_index(groups)

    if weights is None:
        df_means, df_nobs, df_weights = _df_weighted_mean_no_wgt(
            data, groups, varlist, nobs_column, add_weights_column, na_min_count
        )
    else:
        df_means, df_nobs, df_weights = _df_weighted_mean_wgt(
            data,
            weights,
            groups,
            varlist,
            nobs_column,
            add_weights_column,
            na_min_count,
        )

    if isscalar and not multi_index:
        result = df_means.iloc[:, 0].copy()
    elif multi_index:
        index_names = anything_to_list(index_names)
        if nobs_column or add_weights_column:
            stats = ['Mean']
            components = [df_means]
            if nobs_column:
                stats += [nobs_column]
                components += [df_nobs]
            if add_weights_column:
                stats += [weight_varname]
                components += [df_weights]

            result = pd.concat(components, axis=1, keys=stats, names=index_names[::-1])
            # Flip index order so that variables are on top, sort second level and
            # make sure that variable order is the same as in the input DF
            result = result.reorder_levels(index_names, axis=1)
            result = result.sort_index(axis=1, level=-1)[varlist].copy()
        else:
            result = df_means
            result.columns = pd.MultiIndex.from_product(
                (varlist, ['Mean']), names=index_names
            )
    else:
        result = df_means

    return result


def _df_weighted_mean_no_wgt(
    data: pd.DataFrame,
    groups: list[str],
    varlist: list[str],
    nobs_column: Optional[str],
    add_weights_column: bool,
    na_min_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Implementation of weighted mean if no weights are present.
    """

    df_nobs = None
    df_sum_weights = None

    if groups:
        df_means = (
            data[varlist].groupby(groups).sum(min_count=na_min_count).astype(float)
        )
        wsum = data[varlist].groupby(groups).count()
        df_means /= wsum
        if nobs_column:
            df_nobs = wsum
        if add_weights_column:
            # No weights provided, use N obs. as weights
            df_sum_weights = wsum
    else:
        df_means = data[varlist].sum(min_count=na_min_count).astype(float)
        df_means /= data[varlist].count()
        df_means = df_means.to_frame().T
        if nobs_column:
            df_nobs = data[varlist].count().to_frame().T
        if add_weights_column:
            # No weights provided, use N obs. as weights
            df_sum_weights = data[varlist].count().to_frame().T

    return df_means, df_nobs, df_sum_weights


def _df_weighted_mean_wgt(
    data: pd.DataFrame,
    weights: pd.Series | np.ndarray,
    groups: list[str],
    varlist: list[str],
    nobs_column: Optional[str],
    add_weights_column: bool,
    na_min_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Implementation of weighted mean if weights are present.
    """

    df_nobs = None
    df_sum_weights = None

    # Working DataFrame to store temporary data
    df_work = pd.DataFrame(index=data.index, columns=['xw', 'w'], dtype=float)

    w_np = weights.to_numpy(copy=False)

    if groups:
        df_means = None

        for i, varname in enumerate(varlist):
            var_np = data[varname].to_numpy(copy=False)
            df_work['xw'] = var_np * w_np
            # Set weights for NaN obs or NaN weights to zero
            df_work['w'] = np.where(df_work['xw'].notna(), w_np, 0.0)

            wsum = df_work['w'].groupby(groups).sum()

            # Store number of obs. before altering weights to prevent division by 0
            if nobs_column:
                nobs = (df_work['w'] > 0.0).groupby(groups).sum()
                if df_nobs is None:
                    df_nobs = nobs.to_frame(varname)
                else:
                    df_nobs[varname] = nobs

            mean = (
                df_work['xw'].groupby(groups).sum(min_count=na_min_count).astype(float)
            )
            # Replace means where condition is false (i.e., weights sum to 0) to NaN
            mask = wsum == 0.0
            mean.mask(mask, other=np.nan, inplace=True)
            mean /= wsum

            if df_means is None:
                df_means = mean.to_frame(varname)
            else:
                df_means[varname] = mean

            if add_weights_column:
                if df_sum_weights is None:
                    df_sum_weights = wsum.to_frame(varname)
                else:
                    df_sum_weights[varname] = wsum
    else:
        df_means = pd.DataFrame(columns=varlist, index=[0], dtype=float)
        df_nobs = pd.DataFrame(0, columns=varlist, index=[0], dtype=int)
        df_sum_weights = pd.DataFrame(columns=varlist, index=[0], dtype=float)

        for i, varname in enumerate(varlist):
            var_np = data[varname].to_numpy(copy=False)
            df_work.loc[:, 'xw'] = var_np * w_np
            # Set weights for NaN obs or NaN weights to zero
            df_work.loc[:, 'w'] = np.where(df_work['xw'].notna(), w_np, 0.0)

            wsum = df_work['w'].sum()
            if wsum > 0.0:
                sxw = df_work['xw'].sum(min_count=na_min_count)
                mean = sxw / wsum
            else:
                # Note: pandas mean() return NaN if all elements are NaN, but sum()
                # returns 0.0. We want to mimic the behavior of mean()
                mean = np.nan
            df_means.loc[0, varname] = mean

            if nobs_column:
                df_nobs.loc[0, varname] = (df_work['w'] > 0).sum()
            if add_weights_column:
                df_sum_weights.loc[0, varname] = wsum

    return df_means, df_nobs, df_sum_weights


def percentile(
    df: pd.DataFrame,
    prank: Union[float, Iterable[float]],
    varlist: Union[str, Iterable[str]] = None,
    weight_var: str = 'weight',
    interpolation: str = 'linear',
    multi_index: bool = False,
    index_names: Union[str, Iterable[str]] = ('Variable', 'Moment'),
) -> pd.DataFrame:
    """
    Compute (weighted) percentiles for a given list of variables.

    Parameters
    ----------
    df : pd.DataFrame
    prank : float or array_like
    varlist : str or list of str
        List of variables (columns) for which to compute percentiles
        (default: all columns except for weight variable).
    weight_var : str, optional
        Name of weight variable
    interpolation : str
        Interpolation method passed to pydynopt.stats.percentile()
    multi_index : bool, optional
        If true, insert an additional index level with value 'Mean'
        for each variable.
    index_names : str or array_like, optional
        Names for (multi)-index levels of resulting Series.

    Returns
    -------
    pd.DataFrame or float
    """

    isscalar = isinstance(varlist, str) and np.isscalar(prank)

    prank = anything_to_list(prank)
    # Use integer values if there is no loss in precision
    prank = [int(rnk) if int(rnk) == rnk else rnk for rnk in prank]

    varlist = anything_to_list(varlist)
    if varlist is None:
        varlist = [name for name in df.columns.names if name != weight_var]

    # Find appropriate dtype for weighted values
    dtypes = tuple(df[varname].dtype for varname in varlist)
    dtype = np.result_type(*dtypes)

    n = df.shape[0]
    mask = np.empty(n, dtype=np.bool_)
    var_contiguous = np.empty(n, dtype=dtype)
    pmf = np.empty(n, dtype=float)

    pctl = np.full((len(varlist), len(prank)), fill_value=np.nan)

    for i, varname in enumerate(varlist):
        np.isfinite(df[varname].to_numpy(), out=mask)
        ni = mask.sum()
        # If required, create contiguous copy of non-contiguous data
        if ni < n:
            var_contiguous[:ni] = df.loc[mask, varname]
            x = var_contiguous[:ni]
            pmf[:ni] = df.loc[mask, weight_var]
        else:
            x = df[varname].to_numpy()
            pmf[:] = df[weight_var]

        # normalize PMF
        mass = np.sum(pmf[:ni])
        if mass == 0.0:
            continue

        pmf[:ni] /= mass

        pctl[i] = pydynopt.stats.percentile(
            x[:ni],
            pmf[:ni],
            prank,
            assume_sorted=False,
            assume_unique=False,
            interpolation=interpolation,
        )

    # Force deallocated of temporary arrays.
    del mask, pmf, var_contiguous

    if isscalar and not multi_index:
        result = pctl[0, 0]
    else:
        index_names = anything_to_list(index_names)
        columns = pd.Index(varlist, name=index_names[0])
        idx = pd.Index(prank, name=index_names[1])

        result = pd.DataFrame(pctl.T, index=idx, columns=columns)

    return result


def weighted_pmf(
    df: pd.DataFrame,
    *,
    varlist_outer: Optional[Union[str, Iterable]] = None,
    varlist_inner: Union[str, Iterable],
    weights: Optional[str | pd.Series | np.ndarray] = 'weight',
    varname_weight: str = 'weight',
    skipna: bool = True,
    generate: Optional[str] = 'pmf',
) -> pd.DataFrame:
    """
    Compute weight weighted PMF over "inner" cells defined by `varlist_inner`
    within "outer" cells defined by `varlist_outer`.

    Parameters
    ----------
    df : pd.DataFrame
    varlist_outer : str or array_like
    varlist_inner: str or array_like
    weights : str or pd.Series or np.ndarray
        Weights to be used, passed either as conformable numerical or as a column
        name in `df`.
    varname_weight : str, optional
        Name of variable containing weights. Deprecated in favor of `weights`.
    skipna : bool
        If true, drop obs with missing values in any of the variables in
        `varlist_inner`, `varlist_outer` or with missing weights.
    generate : str, optional
        Name of output variable.

    Returns
    -------
    pd.DataFrame
    """

    if weights is not None:
        if isinstance(weights, str):
            weights = df[weights]
    elif varname_weight is not None:
        # Deprecated legacy way to specify weights
        weights = df[weights]
    else:
        # Degenerate uniform weights weights
        weights = np.ones(len(df))

    # Set to internal weight variable name
    varname_weight = "_weight"
    varname_generate = generate if generate else "_pmf"

    varlist_outer = anything_to_list(varlist_outer, force=True)
    varlist_inner = anything_to_list(varlist_inner)

    varlist_all = varlist_outer + varlist_inner

    df = df[varlist_all].copy()
    df[varname_weight] = weights

    if skipna:
        # Keep only obs with nonmissing values AND nonmissing weights. Missing obs with
        # nonmissing weights will otherwise create results that don't sum to 1.
        keep = df.notna().all(axis=1)
        # Do not create needless copies
        if keep.sum() < len(df):
            df = df[keep].copy()

    df_inner = df.groupby(varlist_all)[[varname_weight]].sum()

    if varlist_outer:
        df_outer = df_inner.groupby(varlist_outer)[varname_weight].sum()
        df_outer = df_outer.to_frame(name='weight_sum')
    else:
        weight_sum = df_inner[varname_weight].sum()
        df_outer = pd.DataFrame(
            weight_sum, index=df_inner.index, columns=['weight_sum']
        )

    df_inner = df_inner.join(df_outer, how='left')
    df_inner[varname_generate] = df_inner[varname_weight] / df_inner['weight_sum']

    if generate:
        # Return as DataFrame with requested column name
        pmf = df_inner[[varname_generate]].copy()
    else:
        # Return as Series
        pmf = df_inner[varname_generate].copy()

    return pmf


def interpolate_bin_weights(
    edges: pd.DataFrame | Sequence[float],
    values: pd.DataFrame | Sequence[float],
    name_bins: str = 'ibin',
    name_values: Optional[str] = None,
    generate: str = 'weight'
) -> pd.Series:
    """
    Create weights that map (increasing!) values of a CDF defined on a grid of `values`
    into bins with defined by `edges`.

    Weights are 0 if the grid point is outside a bracket, 1 if it is fully
    contained, and in (0,1) if it is partially contained.

    Parameters
    ----------
    edges : pd.DataFrame or Sequence of float
        Edges defining individual bins. DataFrame with MultiIndex can be passed
        if edges differ by some index level.
    values : pd.DataFrame or Sequence of float
        Grid of increasing CDF values or edges to be mapped into brackets.
    name_bins : str
        Index name assigned to level representing bins.
    name_values : str, optional
        Index name assigned to level representing `values`.
    generate : str, optional
        Name of resulting Series

    Returns
    -------
    pd.Series
    """

    if isinstance(values, pd.Series):
        name_values_default = values.name
        if values.index.nlevels > 1:
            raise ValueError('Series \'values\' contans multiple index levels')
    elif isinstance(values, pd.DataFrame):
        if values.shape[1] > 1:
            raise ValueError('DataFrame \'values\' contains multiple columns')
        if values.index.nlevels > 1:
            raise ValueError('DataFrame \'values\' contans multiple index levels')
        if values.index.names != [None]:
            name_values_default = values.index.names[0]
        else:
            name_values_default = values.columns[0]
        values = values.iloc[:, 0]
    else:
        name_values_default = None
        values = np.atleast_1d(values)

    name_values = name_values or name_values_default

    if not isinstance(values, pd.Series):
        values = pd.Series(values)
        values.index.name = name_values

    # --- Prepare edges ---

    name_edges = '_edges'
    if isinstance(edges, pd.Series):
        df_edges = edges.to_frame(name_edges)
    elif isinstance(edges, pd.DataFrame):
        if edges.shape[1] > 1:
            raise ValueError('edges DataFrame contains multiple columns')
        edges = edges.copy()
        df_edges = edges.rename(columns={edges.columns[0]: name_edges})
    else:
        df_edges = pd.DataFrame(np.atleast_1d(edges).flatten(), columns=[name_edges])

    # --- Create bin lower and upper bounds ---

    # Edges differ by index cell?
    by = []
    if df_edges.index.nlevels > 1:
        by = list(df_edges.index.names[:-1])

    # bin lower bound
    df_lb = df_edges.rename(columns={name_edges: 'lb'})
    # bin upper bound
    if by:
        df_ub = df_edges.groupby(by).shift(-1)
    else:
        df_ub = df_edges.shift(-1)
    df_ub = df_ub.rename(columns={name_edges: 'ub'})

    df_bins = pd.concat((df_lb, df_ub), axis=1)
    df_bins = df_bins.dropna()

    # Create linear bin index
    if by:
        ibin = df_bins.groupby(by)['lb'].transform(lambda x: np.arange(len(x)))
    else:
        ibin = np.arange(len(df_bins))

    df_bins[name_bins] = ibin
    df_bins = df_bins.reset_index(-1, drop=True).set_index(name_bins, append=bool(by))

    # --- Create weights for each cell and each value ---

    def _create_weights(x):
        # Bin lower and upper bounds
        lb, ub = float(x['lb'].iloc[0]), float(x['ub'].iloc[0])

        # Do not use interp_locate() as bsearch cannot deal with flat CDFs. Instead,
        # manually compute number of bins below given percentile.

        # 1. Find the first grid point with at least partial overlap (overlap could be
        # only the edge):
        # Number of values strictly below lb
        ifirst = max(0, np.sum(values < lb) - 1)
        # Correct for edge case
        ifirst += int(values[ifirst + 1] <= lb and values[ifirst] < lb)
        # Interval spanned by first grid point (= mass contained in that interval)
        dx = values[ifirst + 1] - values[ifirst]
        if dx > 0:
            # Weight on the first point
            wgt_first = 1.0 - (lb - values[ifirst]) / dx
        else:
            # Flat region, cannot interpolate.
            wgt_first = 1.0

        # Identify last grid point with at least partial overlap (could be the same as
        # the first point if the bin is larger than the bin).
        ilast = max(0, np.sum(values <= ub) - 1)
        ilast = min(ilast, len(values) - 2)
        # Interval spanned by last point (= mass contained in that interval)
        dx = values[ilast + 1] - values[ilast]
        if dx > 0:
            # Weight on last point. Take into account bin could be
            # fully contained in last interval (which in turn might be the same as
            # the first interval).
            wgt_last = (min(values[ilast+1], ub) - max(values[ilast], lb)) / dx
        else:
            # Flat region, cannot interpolate.
            wgt_last = 1.0

        # Do not allow for extrapolation.
        #  - If wgt_first > 1, the lower bound lies below
        #    any value, so all those grid points should receive weight = 1.
        #  - If wgt_last < 0, the upper bound lies above any values, so the weight
        #    needs to be weight = 0 since 1-weight is assigned to the right-most point.
        # The other two cases are not possible.
        wgt_first = min(1.0, wgt_first)
        wgt_last = max(0.0, wgt_last)

        # Directly copy the input Series since this ensures the correct index
        weights = values.iloc[ifirst:ilast+1].copy(deep=True)
        weights.iloc[:] = 1.0
        weights.iloc[0] = wgt_first
        weights.iloc[-1] = wgt_last
        weights.name = generate
        # Convert to DataFrame to ensure correct vertical stacking of results even if
        # there is only one bin.
        weights = weights.to_frame()

        return weights

    df_weights = df_bins.groupby(by + [name_bins]).apply(_create_weights)

    # Convert back to Series
    df_weights = df_weights[generate]

    # Check that we are not double-counting bins. Weights can be less than 1 if they
    # are outside any bin interval.
    if by:
        tmp = df_weights.groupby(by + [name_values]).sum(axis=0)
        assert np.all((tmp - 1.0) < 1.0e-10)
    else:
        assert np.all((df_weights.groupby(name_values).sum() - 1.0) < 1.0e-10)

    return df_weights
