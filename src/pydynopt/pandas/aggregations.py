
import numpy as np
import pandas as pd

import pydynopt.stats
from pydynopt.utils import anything_to_list


def weighted_mean(data, varlist=None, weights='weight', weight_var=None,
                  multi_index=False, index_names=('Variables', 'Moments')):
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
    multi_index : bool, optional
        If true, insert an additional index level with value 'Mean'
        for each variable.
    index_names : str or array_like, optional
        Names for (multi)-index levels of resulting Series.

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

    has_weights = (weights is not None)
    if has_weights:
        weights = np.array(weights, copy=False)

    varlist = anything_to_list(varlist)
    if varlist is None:
        varlist = [name for name in data.columns if name != weight_var]

    # Find appropriate dtype for weighted values

    dtypes = tuple(data[varname].dtype for varname in varlist)
    if has_weights:
        dtypes += (weights.dtype, )
    dtype = np.result_type(*dtypes)

    n = data.shape[0]
    mask = np.empty(n, dtype=np.bool_)
    var_weighted = np.empty(n, dtype=dtype)

    means = np.full(len(varlist), fill_value=np.nan)

    for i, varname in enumerate(varlist):
        np.isfinite(data[varname].to_numpy(), out=mask)

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

    if isscalar and not multi_index:
        result = means[0]
    else:
        index_names = anything_to_list(index_names)
        if multi_index:
            idx = pd.MultiIndex.from_product((varlist, ['Mean']),
                                             names=index_names)
        else:
            idx = pd.Index(varlist, name=index_names[0])
        result = pd.Series(means, index=idx)

    return result


def percentile(df, prank, varlist=None, weight_var='weight', multi_index=False,
               index_names=('Variables', 'Moments')):
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
    pmf = np.empty(n, dtype=df[weight_var].dtype)

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

        pctl[i] = pydynopt.stats.percentile(x[:ni], pmf[:ni], prank,
                                            assume_sorted=False,
                                            assume_unique=False,
                                            interpolation='linear')

    # Force deallocated of temporary arrays.
    del mask, pmf, var_contiguous

    if isscalar and not multi_index:
        result = pctl[0, 0]
    else:
        index_names = anything_to_list(index_names)
        idx = pd.Index(varlist, name=index_names[0])

        columns = pd.Index(prank, name=index_names[1])
        result = pd.DataFrame(pctl, index=idx, columns=columns)
        # Move column index into rows, create Series
        result = result.stack()

    return result


def weighted_pmf(df, varlist_outer, varlist_inner, varname_weight='weight',
                 generate='pmf'):
    """
    Compute weight weighted PMF over "inner" cells defined by `varlist_inner`
    within "outer" cells defined by `varlist_outer`.

    Parameters
    ----------
    df : pd.DataFrame
    varlist_outer : str or array_like
    varlist_inner: str or array_like
    varname_weight : str, optional
        Name of variable containing weights.
    generate : str, optional
        Name of output variable.

    Returns
    -------
    pd.DataFrame
    """

    varlist_outer = anything_to_list(varlist_outer, force=True)
    varlist_inner = anything_to_list(varlist_inner)

    varlist_all = varlist_outer + varlist_inner

    grp = df.groupby(varlist_all)
    df_inner = grp.agg({varname_weight: np.sum})
    if varlist_outer:
        df_outer = df_inner.groupby(varlist_outer)[varname_weight].sum()
        df_outer = df_outer.to_frame(name='weight_sum')
    else:
        weight_sum = df_inner[varname_weight].sum()
        df_outer = pd.DataFrame(weight_sum, index=df_inner.index, columns=['weight_sum'])

    df_inner = df_inner.join(df_outer, how='left')
    df_inner[generate] = df_inner[varname_weight]/df_inner['weight_sum']

    df_pmf = df_inner[[generate]].copy()

    return df_pmf
