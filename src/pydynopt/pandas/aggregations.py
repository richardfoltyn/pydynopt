"""
Pandas aggregations utilities.

This module provides helper functions for fast weighted mean, percentile,
weighted PMF, and CDF bin weights interpolation.

This work is licensed under CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Sequence
from typing import Any, Literal, overload

import numpy as np
import pandas as pd

import pydynopt.stats
from pydynopt.typing import SequenceNotStr
from pydynopt.utils import anything_to_list

__all__ = [
    'df_weighted_mean',
    'interpolate_bin_weights',
    'percentile',
    'weighted_mean',
    'weighted_pmf',
]


@overload
def weighted_mean(
    data: pd.Series,
    varlist: str | SequenceNotStr[str] | None = None,
    weights: str | pd.Series | np.ndarray | None = 'weight',
    weight_var: str | None = None,
    *,
    index_varlist: Literal[False],
    multi_index: Literal[False],
    index_names: tuple[str, ...] | None = ('Variable', 'Moment'),
    dtype: np.dtype | type | None = float,
) -> float: ...


@overload
def weighted_mean(
    data: pd.DataFrame,
    varlist: str,
    weights: str | pd.Series | np.ndarray | None = 'weight',
    weight_var: str | None = None,
    *,
    index_varlist: Literal[False],
    multi_index: Literal[False],
    index_names: tuple[str, ...] | None = ('Variable', 'Moment'),
    dtype: np.dtype | type | None = float,
) -> float: ...


@overload
def weighted_mean(
    data: pd.DataFrame,
    varlist: SequenceNotStr[str] | None = None,
    weights: str | pd.Series | np.ndarray | None = 'weight',
    weight_var: str | None = None,
    *,
    index_varlist: Literal[False],
    multi_index: Literal[False],
    index_names: tuple[str, ...] | None = ('Variable', 'Moment'),
    dtype: np.dtype | type | None = float,
) -> pd.Series: ...


@overload
def weighted_mean(
    data: pd.DataFrame | pd.Series,
    varlist: str | SequenceNotStr[str] | None = None,
    weights: str | pd.Series | np.ndarray | None = 'weight',
    weight_var: str | None = None,
    index_varlist: Literal[True] = True,
    multi_index: bool = False,
    index_names: tuple[str, ...] | None = ('Variable', 'Moment'),
    dtype: np.dtype | type | None = float,
) -> pd.Series: ...


@overload
def weighted_mean(
    data: pd.DataFrame | pd.Series,
    varlist: str | SequenceNotStr[str] | None = None,
    weights: str | pd.Series | np.ndarray | None = 'weight',
    weight_var: str | None = None,
    index_varlist: bool = True,
    multi_index: Literal[True] = True,
    index_names: tuple[str, ...] | None = ('Variable', 'Moment'),
    dtype: np.dtype | type | None = float,
) -> pd.Series: ...


def weighted_mean(
    data: pd.DataFrame | pd.Series,
    varlist: str | SequenceNotStr[str] | None = None,
    weights: str | pd.Series | np.ndarray | None = 'weight',
    weight_var: str | None = None,
    index_varlist: bool = True,
    multi_index: bool = False,
    index_names: tuple[str, ...] | None = ('Variable', 'Moment'),
    dtype: np.dtype | type | None = float,
) -> float | pd.Series:
    """
    Compute weighted mean of variables, ignoring any NaNs.

    Parameters
    ----------
    data
        DataFrame or Series containing the data variables.
    varlist
        List of variables for which to compute weighted mean.
    weights
        Name of column, Series, or array containing the weights.
    weight_var
        Name of DataFrame column containing the weights. Deprecated in favor of
        `weights`.
    index_varlist
        If true, create index from variable list (slow).
    multi_index
        If true, insert an additional index level with value 'Mean'
        for each variable (very slow!).
    index_names
        Names for (multi)-index levels of resulting Series.
    dtype
        If present, determines dtype of output array.

    Returns
    -------
    Series containing weighted means with variable names used as index.
    """
    isscalar = isinstance(varlist, str)

    if isinstance(data, pd.Series):
        data = data.to_frame('v0')
        varlist = ['v0']
        isscalar = True

    # Legacy: if weight_var is passed, interpret this as the column name in data.
    if weight_var is not None:
        weights = weight_var

    weights_np: np.ndarray | None = None
    if isinstance(weights, str):
        weight_var = weights
        if weight_var not in data:
            raise ValueError(f'Unsupported weight argument: {weight_var}')
        weights_np = data[weight_var].to_numpy(dtype=float, copy=False)
    elif isinstance(weights, pd.Series):
        weights_np = weights.reindex(data.index).to_numpy(dtype=float, copy=False)
    elif isinstance(weights, np.ndarray):
        if len(weights) != len(data):
            raise ValueError('Length of weights does not match input data')
        weights_np = np.asarray(weights, dtype=float)
    elif weights is not None:
        raise ValueError('Unsupported weight argument')

    varlist = anything_to_list(varlist)
    if varlist is None:
        varlist = [name for name in data.columns if name != weight_var]

    if weights_np is None:
        means = data[varlist].mean(axis=0, skipna=True).to_numpy()
    else:
        nobs = data.shape[0]
        means = np.full(len(varlist), np.nan)
        mask = np.empty(nobs, dtype=np.bool_)
        finite_weights = np.isfinite(weights_np)
        var_weighted = np.empty(nobs, dtype=float)

        for i, varname in enumerate(varlist):
            dnp = data[varname].to_numpy(dtype=float, copy=False)
            np.isfinite(dnp, out=mask)
            np.logical_and(mask, finite_weights, out=mask)

            sum_weights = np.sum(weights_np, where=mask)
            if sum_weights > 0.0:
                np.multiply(dnp, weights_np, out=var_weighted)
                sum_weighted = np.sum(var_weighted, where=mask)
                means[i] = sum_weighted / sum_weights

    if dtype is not None:
        means = means.astype(dtype, copy=False)

    if index_varlist or multi_index:
        idx_names = anything_to_list(index_names, force=True)
        if multi_index:
            idx = pd.MultiIndex.from_product((varlist, ['Mean']), names=idx_names)
        else:
            name = idx_names[0] if idx_names else None
            idx = pd.Index(varlist, name=name)
        result = pd.Series(means, index=idx)
    elif isscalar:
        result = means[0]
    else:
        result = pd.Series(means)

    return result


@overload
def df_weighted_mean(
    data: pd.Series,
    groups: str | SequenceNotStr[str] | None = None,
    varlist: str | SequenceNotStr[str] | None = None,
    *,
    weights: pd.Series | np.ndarray | str | None = 'weight',
    na_min_count: int = 1,
    multi_index: Literal[False] = False,
    index_names: str | SequenceNotStr[str] | None = ('Variable', 'Moment'),
    add_weights_column: Literal[False] = False,
    nobs_column: None = None,
) -> pd.Series: ...


@overload
def df_weighted_mean(
    data: pd.Series | pd.DataFrame,
    groups: str | SequenceNotStr[str] | None = None,
    varlist: str | SequenceNotStr[str] | None = None,
    *,
    weights: pd.Series | np.ndarray | str | None = 'weight',
    na_min_count: int = 1,
    multi_index: Literal[True],
    index_names: str | SequenceNotStr[str] | None = ('Variable', 'Moment'),
    add_weights_column: bool = False,
    nobs_column: str | None = None,
) -> pd.DataFrame: ...


@overload
def df_weighted_mean(
    data: pd.Series | pd.DataFrame,
    groups: str | SequenceNotStr[str] | None = None,
    varlist: str | SequenceNotStr[str] | None = None,
    *,
    weights: pd.Series | np.ndarray | str | None = 'weight',
    na_min_count: int = 1,
    multi_index: bool = False,
    index_names: str | SequenceNotStr[str] | None = ('Variable', 'Moment'),
    add_weights_column: Literal[True],
    nobs_column: str | None = None,
) -> pd.DataFrame: ...


@overload
def df_weighted_mean(
    data: pd.Series | pd.DataFrame,
    groups: str | SequenceNotStr[str] | None = None,
    varlist: str | SequenceNotStr[str] | None = None,
    *,
    weights: pd.Series | np.ndarray | str | None = 'weight',
    na_min_count: int = 1,
    multi_index: bool = False,
    index_names: str | SequenceNotStr[str] | None = ('Variable', 'Moment'),
    add_weights_column: bool = False,
    nobs_column: str,
) -> pd.DataFrame: ...


@overload
def df_weighted_mean(
    data: pd.DataFrame,
    groups: str | SequenceNotStr[str] | None,
    varlist: str,
    *,
    weights: pd.Series | np.ndarray | str | None = 'weight',
    na_min_count: int = 1,
    multi_index: Literal[False] = False,
    index_names: str | SequenceNotStr[str] | None = ('Variable', 'Moment'),
    add_weights_column: Literal[False] = False,
    nobs_column: None = None,
) -> pd.Series: ...


@overload
def df_weighted_mean(
    data: pd.DataFrame,
    *,
    varlist: str,
    weights: pd.Series | np.ndarray | str | None = 'weight',
    na_min_count: int = 1,
    multi_index: Literal[False] = False,
    index_names: str | SequenceNotStr[str] | None = ('Variable', 'Moment'),
    add_weights_column: Literal[False] = False,
    nobs_column: None = None,
) -> pd.Series: ...


@overload
def df_weighted_mean(
    data: pd.DataFrame,
    groups: str | SequenceNotStr[str] | None = None,
    varlist: SequenceNotStr[str] | None = None,
    *,
    weights: pd.Series | np.ndarray | str | None = 'weight',
    na_min_count: int = 1,
    multi_index: Literal[False] = False,
    index_names: str | SequenceNotStr[str] | None = ('Variable', 'Moment'),
    add_weights_column: Literal[False] = False,
    nobs_column: None = None,
) -> pd.DataFrame: ...


def df_weighted_mean(
    data: pd.Series | pd.DataFrame,
    groups: str | SequenceNotStr[str] | None = None,
    varlist: str | SequenceNotStr[str] | None = None,
    *,
    weights: pd.Series | np.ndarray | str | None = 'weight',
    na_min_count: int = 1,
    multi_index: bool = False,
    index_names: str | SequenceNotStr[str] | None = ('Variable', 'Moment'),
    add_weights_column: bool = False,
    nobs_column: str | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Compute (within-group) weighted mean of variables.

    Parameters
    ----------
    data
        DataFrame or Series containing the data variables.
    groups
        List of variables defining groups.
    varlist
        List of variables for which to compute weighted mean.
    weights
        Name of column, Series, or array containing the weights.
    na_min_count
        Groups with number of observations below this value are assigned NA.
    multi_index
        If true, insert an additional index level with value 'Mean'
        for each variable.
    index_names
        Names for MultiIndex column levels or Series index levels.
    add_weights_column
        If true, add the sum of weights for each bin in the output DataFrame
        (implies `multi_index = True`)
    nobs_column
        If not None, add the number of observations for each bin in the output DataFrame
        (implies `multi_index = True`)

    Returns
    -------
    Series or DataFrame containing weighted means.
    """
    isscalar = isinstance(varlist, str) or isinstance(data, pd.Series)
    groups_list: list[str] = anything_to_list(groups, force=True)

    # Force MultiIndex output if several stats are computed for each variable
    multi_index = multi_index or add_weights_column or bool(nobs_column)

    if na_min_count < 1:
        raise ValueError("Argument 'na_min_count' must be positive")

    if isinstance(data, pd.Series):
        name = str(data.name) if data.name is not None else 'v0'
        data = data.to_frame(name)
        varlist = [name]

    weight_varname = None
    if isinstance(weights, str):
        weight_varname = weights

    # Extract default varlist
    varlist = anything_to_list(varlist)
    if varlist is None:
        varlist = [
            name
            for name in data.columns
            if name != weight_varname and name not in groups_list
        ]

    # Check that grouping variables are in index, otherwise put them there
    missing = [group for group in groups_list if group not in data.index.names]
    if missing:
        data = data.set_index(missing, append=True)

    if isinstance(weights, str):
        weights = data[weight_varname]

    if weights is None:
        df_means, df_nobs, df_weights = _df_weighted_mean_no_wgt(
            data, groups_list, varlist, nobs_column, add_weights_column, na_min_count
        )
    else:
        df_means, df_nobs, df_weights = _df_weighted_mean_wgt(
            data,
            weights,
            groups_list,
            varlist,
            nobs_column,
            add_weights_column,
            na_min_count,
        )

    if isscalar and not multi_index:
        result = df_means.iloc[:, 0].copy()
    elif multi_index:
        idx_names = anything_to_list(index_names, force=True)
        if nobs_column or add_weights_column:
            stats = ['Mean']
            components = [df_means]
            if nobs_column is not None:
                assert df_nobs is not None
                stats.append(nobs_column)
                components.append(df_nobs)
            if add_weights_column:
                assert df_weights is not None
                wname = weight_varname if weight_varname is not None else 'weight'
                stats.append(wname)
                components.append(df_weights)

            result = pd.concat(components, axis=1, keys=stats, names=idx_names[::-1])
            # Flip index order so that variables are on top, sort second level and
            # make sure that variable order is the same as in the input DF
            result = result.reorder_levels(idx_names, axis=1)
            result = result.sort_index(axis=1, level=-1)[varlist].copy()
        else:
            result = df_means
            result.columns = pd.MultiIndex.from_product(
                (varlist, ['Mean']), names=idx_names
            )
    else:
        result = df_means

    return result


def _df_weighted_mean_no_wgt(
    data: pd.DataFrame,
    groups: list[str],
    varlist: list[str],
    nobs_column: str | None,
    add_weights_column: bool,
    na_min_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Implement weighted mean when no weights are present."""
    df_nobs = None
    df_sum_weights = None

    if groups:
        grouped = data[varlist].groupby(groups)
        wsum = grouped.count()
        df_means = grouped.mean().astype(float)
        if na_min_count > 1:
            df_means = df_means.mask(wsum < na_min_count)
        if nobs_column:
            df_nobs = wsum
        if add_weights_column:
            # No weights provided, use N obs. as weights
            df_sum_weights = wsum
    else:
        counts = data[varlist].count()
        means = data[varlist].mean(axis=0, skipna=True).astype(float)
        if na_min_count > 1:
            means = means.mask(counts < na_min_count)
        df_means = means.to_frame().T
        if nobs_column:
            df_nobs = counts.to_frame().T
        if add_weights_column:
            # No weights provided, use N obs. as weights
            df_sum_weights = counts.to_frame().T

    return df_means, df_nobs, df_sum_weights


def _df_weighted_mean_wgt(
    data: pd.DataFrame,
    weights: pd.Series | np.ndarray,
    groups: list[str],
    varlist: list[str],
    nobs_column: str | None,
    add_weights_column: bool,
    na_min_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Implement weighted mean when weights are present."""
    df_nobs = None
    df_sum_weights = None

    # Internal helper: caller is expected to provide conforming/aligned weights.
    weights_np = (
        weights.to_numpy(dtype=float, copy=False)
        if isinstance(weights, pd.Series)
        else np.asarray(weights, dtype=float)
    )
    finite_weights = np.isfinite(weights_np)
    nobs = len(weights_np)

    mask = np.empty(nobs, dtype=np.bool_)
    xw = np.empty(nobs, dtype=float)
    w_eff = np.empty(nobs, dtype=float)

    if groups:
        means_cols: dict[str, pd.Series] = {}
        nobs_cols: dict[str, pd.Series] = {}
        sumw_cols: dict[str, pd.Series] = {}

        for varname in varlist:
            var_np = data[varname].to_numpy(dtype=float, copy=False)

            np.isfinite(var_np, out=mask)
            np.logical_and(mask, finite_weights, out=mask)

            np.multiply(var_np, weights_np, out=xw)
            xw[~mask] = np.nan

            np.copyto(w_eff, weights_np)
            w_eff[~mask] = 0.0

            sxw = (
                pd.Series(xw, index=data.index, copy=False)
                .groupby(groups)
                .sum(min_count=na_min_count)
            )
            sw = pd.Series(w_eff, index=data.index, copy=False).groupby(groups).sum()
            means_cols[varname] = sxw.div(sw.where(sw > 0.0)).astype(float)

            if nobs_column:
                nobs_cols[varname] = (
                    pd.Series(w_eff > 0.0, index=data.index)
                    .groupby(groups)
                    .sum()
                    .astype(int)
                )

            if add_weights_column:
                sumw_cols[varname] = sw

        df_means = pd.DataFrame(means_cols)
        if nobs_column:
            df_nobs = pd.DataFrame(nobs_cols)
        if add_weights_column:
            df_sum_weights = pd.DataFrame(sumw_cols)
    else:
        means_arr = np.full(len(varlist), np.nan)
        nobs_arr = np.zeros(len(varlist), dtype=int)
        sumw_arr = np.zeros(len(varlist), dtype=float)

        for i, varname in enumerate(varlist):
            var_np = data[varname].to_numpy(dtype=float, copy=False)

            np.isfinite(var_np, out=mask)
            np.logical_and(mask, finite_weights, out=mask)

            np.copyto(w_eff, weights_np)
            w_eff[~mask] = 0.0
            sw = float(np.sum(w_eff))

            if sw > 0.0 and int(mask.sum()) >= na_min_count:
                np.multiply(var_np, weights_np, out=xw)
                sxw = float(np.sum(xw, where=mask))
                means_arr[i] = sxw / sw

            if nobs_column:
                nobs_arr[i] = int(np.sum(w_eff > 0.0))
            if add_weights_column:
                sumw_arr[i] = sw

        columns = pd.Index(varlist)
        df_means = pd.DataFrame([means_arr], columns=columns)
        if nobs_column:
            df_nobs = pd.DataFrame([nobs_arr], columns=columns)
        if add_weights_column:
            df_sum_weights = pd.DataFrame([sumw_arr], columns=columns)

    return df_means, df_nobs, df_sum_weights


@overload
def percentile(
    df: pd.DataFrame,
    prank: float,
    varlist: str,
    weight_var: str = 'weight',
    interpolation: str = 'linear',
    multi_index: Literal[False] = False,
    index_names: str | SequenceNotStr[str] | None = ('Variable', 'Moment'),
) -> float: ...


@overload
def percentile(
    df: pd.DataFrame,
    prank: float | Sequence[float] | np.ndarray,
    varlist: str | SequenceNotStr[str] | None = None,
    weight_var: str = 'weight',
    interpolation: str = 'linear',
    multi_index: bool = False,
    index_names: str | SequenceNotStr[str] | None = ('Variable', 'Moment'),
) -> pd.DataFrame: ...


def percentile(
    df: pd.DataFrame,
    prank: float | Sequence[float] | np.ndarray,
    varlist: str | SequenceNotStr[str] | None = None,
    weight_var: str = 'weight',
    interpolation: str = 'linear',
    multi_index: bool = False,
    index_names: str | SequenceNotStr[str] | None = ('Variable', 'Moment'),
) -> float | pd.DataFrame:
    """
    Compute (weighted) percentiles for a given list of variables.

    Parameters
    ----------
    df
        DataFrame containing the variables to compute percentiles for.
    prank
        Percentile ranks to compute, between 0 and 100.
    varlist
        List of variables (columns) for which to compute percentiles
        (default: all columns except for weight variable).
    weight_var
        Name of weight variable.
    interpolation
        Interpolation method passed to pydynopt.stats.percentile().
    multi_index
        If true, insert an additional index level with value 'Mean'
        for each variable.
    index_names
        Names for (multi)-index levels of resulting Series.
    """
    is_scalar = isinstance(varlist, str) and np.isscalar(prank)

    prank_list = anything_to_list(prank, force=True)
    # Use integer values if there is no loss in precision
    prank_list = [int(rnk) if int(rnk) == rnk else rnk for rnk in prank_list]

    varlist_list = anything_to_list(varlist, force=True)
    if not varlist_list:
        varlist_list = [name for name in df.columns if name != weight_var]

    weights_np = df[weight_var].to_numpy(dtype=float, copy=False)

    pctl = np.full((len(varlist_list), len(prank_list)), fill_value=np.nan)

    for i, varname in enumerate(varlist_list):
        x = df[varname].to_numpy(dtype=float, copy=False)
        valid = np.isfinite(x) & np.isfinite(weights_np)
        if not np.any(valid):
            continue

        x = x[valid]
        pmf = weights_np[valid].copy()

        # normalize PMF
        mass = np.sum(pmf)
        if mass <= 0.0:
            continue

        pmf /= mass

        pctl[i] = pydynopt.stats.percentile(
            x,
            pmf,
            prank_list,
            assume_sorted=False,
            assume_unique=False,
            interpolation=interpolation,
        )

    if is_scalar and not multi_index:
        result = float(pctl[0, 0])
    else:
        idx_names = anything_to_list(index_names, force=True)
        columns = pd.Index(
            varlist_list, name=idx_names[0] if len(idx_names) > 0 else None
        )
        idx = pd.Index(prank_list, name=idx_names[1] if len(idx_names) > 1 else None)

        result = pd.DataFrame(pctl.T, index=idx, columns=columns)

    return result


@overload
def weighted_pmf(
    df: pd.DataFrame,
    *,
    varlist_outer: str | SequenceNotStr[Any] | None = None,
    varlist_inner: str | SequenceNotStr[Any],
    weights: str | pd.Series | np.ndarray | None = 'weight',
    varname_weight: str = 'weight',
    skipna: bool = True,
    generate: str = 'pmf',
) -> pd.DataFrame: ...


@overload
def weighted_pmf(
    df: pd.DataFrame,
    *,
    varlist_outer: str | SequenceNotStr[Any] | None = None,
    varlist_inner: str | SequenceNotStr[Any],
    weights: str | pd.Series | np.ndarray | None = 'weight',
    varname_weight: str = 'weight',
    skipna: bool = True,
    generate: None,
) -> pd.Series: ...


def weighted_pmf(
    df: pd.DataFrame,
    *,
    varlist_outer: str | SequenceNotStr[Any] | None = None,
    varlist_inner: str | SequenceNotStr[Any],
    weights: str | pd.Series | np.ndarray | None = 'weight',
    varname_weight: str = 'weight',
    skipna: bool = True,
    generate: str | None = 'pmf',
) -> pd.DataFrame | pd.Series:
    """
    Compute weight weighted PMF over "inner" cells.

    This computes the PMF over "inner" cells defined by `varlist_inner`
    within "outer" cells defined by `varlist_outer`.

    Parameters
    ----------
    df
        DataFrame containing the variables.
    varlist_outer
        List of variables defining the outer group.
    varlist_inner
        List of variables defining the inner cells for which the PMF is computed.
    weights
        Weights to be used, passed either as a numerical array/Series or as a column
        name in `df`.
    varname_weight
        Name of variable containing weights. Deprecated in favor of `weights`.
    skipna
        If true, drop obs with missing values in any of the variables in
        `varlist_inner`, `varlist_outer` or with missing weights.
    generate
        Name of output variable.
    """
    if weights is not None:
        weights_val = df[weights] if isinstance(weights, str) else weights
    elif varname_weight is not None:
        # Deprecated legacy way to specify weights
        weights_val = df[varname_weight]
    else:
        # Degenerate uniform weights weights
        weights_val = np.ones(len(df))

    # Set to internal weight variable name
    varname_weight = '_weight'
    varname_generate = generate if generate else '_pmf'

    varlist_outer_list: list[Any] = anything_to_list(varlist_outer, force=True)
    varlist_inner_list: list[Any] = anything_to_list(varlist_inner, force=True)

    varlist_all = varlist_outer_list + varlist_inner_list

    df = df[varlist_all].copy()
    df[varname_weight] = weights_val

    if skipna:
        # Keep only obs with nonmissing values AND nonmissing weights. Missing obs with
        # nonmissing weights will otherwise create results that don't sum to 1.
        keep = df.notna().all(axis=1)
        # Do not create needless copies
        if keep.sum() < len(df):
            df = df[keep].copy()

    df_inner = df.groupby(varlist_all)[[varname_weight]].sum()

    if varlist_outer_list:
        df_outer = df_inner.groupby(varlist_outer_list)[varname_weight].sum()
        df_outer = df_outer.to_frame(name='weight_sum')
    else:
        weight_sum = df_inner[varname_weight].sum()
        df_outer = pd.DataFrame(
            weight_sum,
            index=df_inner.index,
            columns=pd.Index(['weight_sum']),
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
    edges: pd.DataFrame | pd.Series | Sequence[float] | np.ndarray,
    values: pd.DataFrame | pd.Series | Sequence[float] | np.ndarray,
    name_bins: str = 'ibin',
    name_values: str | None = None,
    generate: str = 'weight',
    validate: bool = True,
) -> pd.Series:
    """
    Create weights that map grid values into bins.

    This maps (increasing!) values of a CDF defined on a grid of `values`
    into bins defined by `edges`.

    Weights are 0 if the grid point is outside a bracket, 1 if it is fully
    contained, and in (0,1) if it is partially contained.

    Parameters
    ----------
    edges
        Edges defining individual bins. DataFrame with MultiIndex can be passed
        if edges differ by some index level.
    values
        Grid of increasing CDF values or edges to be mapped into brackets.
    name_bins
        Index name assigned to level representing bins.
    name_values
        Index name assigned to level representing `values`.
    generate
        Name of resulting Series
    validate
        If true, validate that the resulting bin weights sum to 1.0.
    """
    values_series: pd.Series
    if isinstance(values, pd.Series):
        name_values_default = values.name
        if values.index.nlevels > 1:
            raise ValueError("Series 'values' contans multiple index levels")
        values_series = values
    elif isinstance(values, pd.DataFrame):
        if values.shape[1] > 1:
            raise ValueError("DataFrame 'values' contains multiple columns")
        if values.index.nlevels > 1:
            raise ValueError("DataFrame 'values' contans multiple index levels")
        if values.index.names != [None]:
            name_values_default = values.index.names[0]
        else:
            name_values_default = values.columns[0]
        values_series = values.iloc[:, 0]
    else:
        name_values_default = None
        values_series = pd.Series(np.atleast_1d(values), dtype=float)

    if len(values_series) < 2:
        raise ValueError("Argument 'values' must contain at least 2 points")

    values_np = values_series.to_numpy(dtype=float, copy=False)
    if np.any(np.diff(values_np) < 0.0):
        raise ValueError("Argument 'values' must be nondecreasing")

    name_values_str: str | None = (
        str(name_values_default) if name_values_default is not None else None
    )
    name_values = name_values or name_values_str

    if name_values is not None:
        values_series.index.name = name_values

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
        df_edges = pd.DataFrame(
            np.atleast_1d(edges).flatten(),
            columns=pd.Index([name_edges]),
        )

    # --- Create bin lower and upper bounds ---

    # Edges differ by index cell?
    by = []
    if df_edges.index.nlevels > 1:
        by = list(df_edges.index.names[:-1])

    # bin lower bound
    df_lb = df_edges.rename(columns={name_edges: 'lb'})
    # bin upper bound
    df_ub = df_edges.groupby(by).shift(-1) if by else df_edges.shift(-1)
    df_ub = df_ub.rename(columns={name_edges: 'ub'})

    df_bins = pd.concat((df_lb, df_ub), axis=1)
    df_bins = df_bins.dropna()

    # Create linear bin index
    ibin = df_bins.groupby(by)['lb'].cumcount() if by else np.arange(len(df_bins))

    df_bins[name_bins] = ibin
    df_bins = df_bins.reset_index(-1, drop=True).set_index(name_bins, append=bool(by))

    # --- Create weights for each cell and each value ---

    def _create_weights(x):
        # Bin lower and upper bounds
        lb, ub = float(x['lb'].iloc[0]), float(x['ub'].iloc[0])

        # Do not use interp_locate() as bsearch cannot deal with flat CDFs. Instead,
        # manually compute number of bins below given percentile.

        # 1. Find the first grid point with at least partial overlap (overlap could be
        # only the edge).
        ifirst = int(np.searchsorted(values_np, lb, side='left')) - 1
        ifirst = max(0, min(ifirst, len(values_np) - 2))
        # Interval spanned by first grid point (= mass contained in that interval)
        dx = values_np[ifirst + 1] - values_np[ifirst]
        wgt_first = 1.0 - (lb - values_np[ifirst]) / dx if dx > 0 else 1.0

        # Identify last grid point with at least partial overlap (could be the same as
        # the first point if the bin is larger than the bin).
        ilast = int(np.searchsorted(values_np, ub, side='right')) - 1
        ilast = max(0, min(ilast, len(values_np) - 2))
        # Interval spanned by last point (= mass contained in that interval)
        dx = values_np[ilast + 1] - values_np[ilast]
        if dx > 0:
            # Weight on last point. Take into account bin could be
            # fully contained in last interval (which in turn might be the same as
            # the first interval).
            wgt_last = (min(values_np[ilast + 1], ub) - max(values_np[ilast], lb)) / dx
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

        # Copy input Series index
        weights: pd.Series = pd.Series(
            1.0, index=values_series.index[ifirst : ilast + 1], name=generate
        )
        weights.iloc[0] = float(wgt_first)
        weights.iloc[-1] = float(wgt_last)
        # Convert to DataFrame to ensure correct vertical stacking of results even if
        # there is only one bin.
        df_weights = weights.to_frame()

        return df_weights

    df_weights = df_bins.groupby([*by, name_bins]).apply(_create_weights)

    # Convert back to Series
    df_weights = df_weights[generate]

    if validate:
        # Check that we are not double-counting bins. Weights can be less than 1 if they
        # are outside any bin interval.
        if by:
            levels = [*range(len(by)), df_weights.index.nlevels - 1]
        else:
            levels = df_weights.index.nlevels - 1
        tmp = df_weights.groupby(level=levels).sum()
        assert np.all(tmp.to_numpy() <= 1.0 + 1.0e-10)

    return df_weights
