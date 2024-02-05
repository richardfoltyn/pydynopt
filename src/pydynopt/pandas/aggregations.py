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
from pydynopt.interpolate import interp1d_locate
from pydynopt.utils import anything_to_list


from pydynopt.numba import jit, has_numba


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
    na_min_count: Optional[int] = 1,
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

    isscalar = isinstance(varlist, str)
    groups = anything_to_list(groups, force=True)

    # Force MultiIndex output if several stats are computed for each variable
    multi_index = multi_index or add_weights_column or nobs_column

    data = data.copy()

    if isinstance(data, pd.Series):
        data = data.to_frame('v0')
        varlist = ['v0']
        isscalar = True

    if isinstance(weights, str):
        weight_varname = weights
    elif weights is not None:
        weight_varname = '__weight'
        data[weight_varname] = weights
    else:
        weight_varname = None

    varlist = anything_to_list(varlist)
    if varlist is None:
        varlist = [
            name
            for name in data.columns
            if name != weight_varname and name not in groups
        ]

    if not weight_varname:
        if groups:
            df_means = data[varlist].groupby(groups).sum()

            if nobs_column:
                df_nobs = data[varlist].groupby(groups).count()
            if add_weights_column:
                # No weights provided, use N obs. as weights
                df_sum_weights = data[varlist].groupby(groups).count()
        else:
            df_means = data[varlist].sum()

            if nobs_column:
                df_nobs = data[varlist].count()
            if add_weights_column:
                # No weights provided, use N obs. as weights
                df_sum_weights = data[varlist].count()
    else:
        means = []
        sum_weights = []
        nobs = []

        wgt_notna = f'__{weight_varname}_not_na'

        for i, varname in enumerate(varlist):
            varname_wgt = f'__{varname}_wgt'
            data[varname_wgt] = data[varname] * data[weight_varname]
            data[wgt_notna] = data[weight_varname] * data[varname_wgt].notna()
            if groups:
                data[varname_wgt] /= data.groupby(groups)[wgt_notna].transform('sum')
                mean_var = data.groupby(groups)[varname_wgt].sum(min_count=na_min_count)

                if nobs_column:
                    nobs_var = data.groupby(groups)[wgt_notna].agg(
                        lambda x: np.sum(x > 0.0)
                    )
                if add_weights_column:
                    sum_weights_var = data.groupby(groups)[wgt_notna].sum()
                    sum_weights_var[mean_var.isna()] = np.nan
            else:
                data[varname_wgt] /= data[wgt_notna].sum()
                mean_var = pd.Series(data[varname_wgt].sum(min_count=na_min_count))

                if nobs_column:
                    nobs_var = pd.Series(data[varname_wgt].count())
                if add_weights_column and np.isfinite(mean_var):
                    sum_weights_var = pd.Series(data[wgt_notna].sum())
                else:
                    sum_weights_var = np.nan

            means.append(mean_var.to_frame(varname))

            if nobs_column:
                nobs.append(nobs_var.to_frame(varname))
            if add_weights_column:
                sum_weights.append(sum_weights_var.to_frame(varname))

        df_means = pd.concat(means, axis=1)

        if nobs_column:
            df_nobs = pd.concat(nobs, axis=1)
            df_nobs.columns = df_means.columns
        if add_weights_column:
            df_sum_weights = pd.concat(sum_weights, axis=1)
            df_sum_weights.columns = df_means.columns

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
                components += [df_sum_weights]

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

    grp = df.groupby(varlist_all)
    df_inner = grp.agg({varname_weight: np.sum})

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
) -> pd.DataFrame:
    """
    Create weights that map bins defined on a grid of `values`
    into bins with defined by `edges`.

    Weights are 0 if the grid point is outside a bin, 1 if it is fully
    contained, and in (0,1) if it is partially contained.

    Parameters
    ----------
    edges : pd.DataFrame or Sequence of float
        Edges defining individual bins. DataFrame with MultiIndex can be passed
        if edges differ by some index level.
    values : pd.DataFrame or Sequence of float
        Grid of values to be binned
    name_bins : str
        Index name assigned to level representing bins.
    name_values : str, optional
        Index name assigned to level representing `values`.

    Returns
    -------
    pd.DataFrame
    """

    if isinstance(values, pd.Series):
        name_values_default = values.name
        values = values.to_numpy()
    elif isinstance(values, pd.DataFrame):
        if values.shape[1] > 1:
            raise ValueError('values DataFrame contains multiple columns')
        name_values_default = values.columns[0]
        values = values.to_numpy().flatten()
    else:
        name_values_default = None
        values = np.atleast_1d(values)

    if not name_values:
        name_values = name_values_default

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
    df_bins = df_bins.reset_index(-1, drop=True).set_index(name_bins, append=True)

    # --- Create weights for each cell and each value ---

    index_values = pd.Index(np.arange(len(values)), name=name_values)

    def _create_weights(x):
        lb, ub = float(x['lb'].iloc[0]), float(x['ub'].iloc[0])
        weights = np.zeros_like(values)

        ilb, wgt_lb = interp1d_locate(lb, values)
        iub, wgt_ub = interp1d_locate(ub, values)

        weights[ilb : iub + 1] = 1.0
        weights[ilb] = wgt_lb
        weights[iub] = 1.0 - wgt_ub

        s = pd.Series(weights, index=index_values, name='weight')
        return s

    df_weights = df_bins.groupby(by + [name_bins]).apply(_create_weights)

    # Check that we are not double-counting bins. Weights can be less than 1 if CAH
    # bins are not included in any bin (e.g. if the distribution for some given age
    # tops out below the max CAH).
    tmp = df_weights.groupby(by).sum()
    assert np.all(np.abs((tmp - 1.0) < 1.0e-10))

    df_weights = df_weights.stack()

    return df_weights
