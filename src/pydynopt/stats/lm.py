"""
This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from typing import Optional

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults
import patsy
import numpy as np

__all__ = ['areg']


def demean_within(
    x: pd.Series | pd.DataFrame,
    groups: str | list[str],
    weights: Optional[pd.Series] = None,
    rescale_weights: bool = False,
) -> pd.Series | pd.DataFrame:
    """
    Demean given variables within groups, optionally used weighted group means.

    Parameters
    ----------
    x : pd.Series or pd.DataFrame
    groups : str or list of str
        Variable name(s) defining groups
    weights : pd.Series, optional
        Weights used to compute within-group means
    rescale_weights : bool, optional
        If true, assume that weights are NOT normalized within group
        such that they sum to 1. In this case, normalized is performed
        before computed weighted means.

    Returns
    -------
    pd.Series or pd.DataFrame
    """

    # Rescale weights if normalization is requested
    if weights is not None and rescale_weights:
        wsum = weights.groupby(groups).transform('sum')
        weights = weights / wsum

    # Apply (the same) weights to all variables in DataFrame or Series
    if weights is not None:
        weights = weights.to_numpy(copy=False)
        if isinstance(x, pd.DataFrame):
            # Additional dimension required to make element-wise * work
            weights = weights[:, None]

        xw = x * weights
    else:
        xw = x

    x_mean = xw.groupby(groups).transform('sum')
    x_demean = x - x_mean

    return x_demean


def areg(
    absorb: str,
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    endog: Optional[pd.DataFrame | pd.Series] = None,
    exog: Optional[pd.DataFrame] = None,
    weights: Optional[str | np.ndarray | pd.Series | pd.DataFrame] = None,
) -> RegressionResults:
    """
    Run FE regression similar to Stata's areg, obsorbing FE.

    Parameters
    ----------
    absorb : str
        Column or index name which contains group identifiers defining the level
        at which FE are created.
    formula : str, optional
        Patsy formula used to determine LHS and RHS factors.
    data : pd.DataFrame
        Data from which to generate endogenous and exogenous variables using
        `formula`. Assume to contain weights if `weights` is a str
    endog: array_like, optional
        Endogenous variable. Takes precedence over `formula` & `data`
        if both are given.
    exog: array_like, optional
        Exogenuos variables. Takes precedence over `formula` & `data`
        if both are given.
    weights : str or array_like, optional
        Column name of weights stored in `data` or array containing sample
        weights.

    Returns
    -------
    RegressionResults
    """

    def _fix_index(d):
        if d is not None and absorb not in d.index.names:
            idx = pd.Index(d[absorb], name=absorb)
            d = d.copy(deep=False)
            d.index = idx
        return d

    data = _fix_index(data)
    endog = _fix_index(endog)
    exog = _fix_index(exog)

    if endog is not None and exog is not None:
        y = endog.copy()
        X = exog.copy()
    elif formula is not None and data is not None:
        y, X = patsy.dmatrices(formula, data, return_type='dataframe')
    else:
        raise ValueError('Either data or endog + exog arguments required')

    has_const = any((v == 1.0).all() for name, v in X.items())

    has_weights = weights is not None
    weights_indiv = None
    if weights is not None:
        if data is not None and weights in data.columns:
            weights_name = weights
            weights = data[weights]
        else:
            weights_name = '_weights'

        weights = np.array(weights, dtype=float, copy=True)
        sw = pd.Series(weights, name=weights_name, index=X.index, copy=True)

        keep = sw > 0.0

        if not keep.all():
            y = y.loc[keep].copy()
            X = X.loc[keep].copy()
            sw = sw.loc[keep].copy()
            if data is not None:
                data = data.loc[keep].copy()

        # Weights normalized within FE unit
        sw_sum = sw.groupby(absorb).transform('sum')
        weights_indiv = sw / sw_sum

        # Normalize so that weights sum to 1.0
        weights /= weights.sum()

    if data is not None:
        groups = data.index.get_level_values(absorb)
    else:
        groups = exog.index.get_level_values(absorb)

    ybar = float(np.average(y, axis=0, weights=weights))
    Xbar = np.average(X, axis=0, weights=weights)

    if has_weights:
        # demean outcome within FE cells
        y = demean_within(y, absorb, weights_indiv, rescale_weights=False)
        # add back total mean
        y += ybar

        # demean regressors within FE cells
        X = demean_within(X, absorb, weights_indiv, rescale_weights=False)
        # add back total mean
        X += Xbar

        reg = sm.WLS(y, X, weights=weights, hasconst=has_const)
    else:
        y = y - y.groupby(absorb).transform('mean') + ybar
        X = X - X.groupby(absorb).transform('mean') + Xbar

        reg = sm.OLS(y, X, hasconst=has_const)

    # Account for df loss from FE transform
    reg.df_resid -= groups.nunique() - 1

    result = reg.fit()

    return result
