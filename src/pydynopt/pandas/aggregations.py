
import numpy as np
import pandas as pd

import pydynopt.stats
from pydynopt.utils import anything_to_list


def weighted_mean(df, varlist=None, weight_var='weight', multi_index=False,
                  index_names=('Variables', 'Moments')):
    """
    Compute weighted mean of variable given by varname, ignoring any NaNs.

    Parameters
    ----------
    df : pd.DataFrame
    varlist : str or list of str, optional
        List of variables for which to compute weighted mean.
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

    varlist = anything_to_list(varlist)
    if varlist is None:
        varlist = [name for name in df.columns if name != weight_var]

    # Find appropriate dtype for weighted values
    dtypes = tuple(df[varname].dtype for varname in varlist + [weight_var])
    dtype = np.result_type(*dtypes)

    n = df.shape[0]
    mask = np.empty(n, dtype=np.bool_)
    var_weighted = np.empty(n, dtype=dtype)

    means = np.full(len(varlist), fill_value=np.nan)

    for i, varname in enumerate(varlist):
        np.isfinite(df[varname].to_numpy(), out=mask)
        sum_wgt = np.sum(df[weight_var].to_numpy(), where=mask)
        np.multiply(df[varname].to_numpy(), df[weight_var].to_numpy(), out=var_weighted)

        sum_var = np.sum(var_weighted, where=mask)

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
