"""
Module containing functions for linear models.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from dataclasses import dataclass
from typing import Any, overload

import numpy as np
import pandas as pd
import patsy as _patsy
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults

__all__ = ['FeRsquared', 'areg', 'demean_within']


patsy: Any = _patsy


@dataclass
class FeRsquared:
    """Stata-like FE R-squared metrics."""

    within: float
    between: float
    overall: float


@overload
def demean_within(
    x: pd.Series,
    groups: str | list[str],
    weights: pd.Series | None = None,
    rescale_weights: bool = False,
) -> tuple[pd.Series, pd.Series]: ...


@overload
def demean_within(
    x: pd.DataFrame,
    groups: str | list[str],
    weights: pd.Series | None = None,
    rescale_weights: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]: ...


def demean_within(
    x: pd.Series | pd.DataFrame,
    groups: str | list[str],
    weights: pd.Series | None = None,
    rescale_weights: bool = False,
) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
    """
    Demean given variables within groups, optionally used weighted group means.

    Parameters
    ----------
    x
        Series or DataFrame containing the variables to demean.
    groups
        Variable name(s) defining groups.
    weights
        Weights used to compute within-group means.
    rescale_weights
        If true, assume that weights are NOT normalized within group
        such that they sum to 1. In this case, normalized is performed
        before computed weighted means.

    Returns
    -------
    demeaned
        Demeaned values.
    means
        Group means, one observation per group.
    """
    if isinstance(groups, str):
        if groups in x.index.names:
            group_keys = x.index.get_level_values(groups)
        else:
            group_keys = x[groups]
        groupby_keys = group_keys
        reindex_keys = group_keys
    else:
        arrays = []
        for grp in groups:
            if grp in x.index.names:
                arrays.append(x.index.get_level_values(grp))
            else:
                arrays.append(x[grp])
        groupby_keys = arrays
        reindex_keys = pd.MultiIndex.from_arrays(arrays)

    if weights is not None:
        if rescale_weights:
            w_sum = weights.groupby(groupby_keys, observed=True).sum()
            xw_sum = x.mul(weights, axis=0).groupby(groupby_keys, observed=True).sum()
            means = xw_sum.div(w_sum, axis=0)
        else:
            means = x.mul(weights, axis=0).groupby(groupby_keys, observed=True).sum()
    else:
        means = x.groupby(groupby_keys, observed=True).mean()

    # Reindex means to align with the original index structure
    x_mean = means.reindex(reindex_keys)
    x_mean.index = x.index

    # Compute: x_demean = x - x_mean
    x_demean = x - x_mean

    return x_demean, means




@overload
def _areg_fix_index(d: pd.DataFrame, absorb: str) -> pd.DataFrame: ...


@overload
def _areg_fix_index(d: pd.Series, absorb: str) -> pd.Series: ...


@overload
def _areg_fix_index(d: None, absorb: str) -> None: ...


def _areg_fix_index(
    d: pd.DataFrame | pd.Series | None,
    absorb: str,
) -> pd.DataFrame | pd.Series | None:
    """Fix index to align with absorb column if not already present."""
    if d is not None and absorb not in d.index.names:
        idx = pd.Index(d[absorb], name=absorb)
        d = d.copy(deep=False)
        d.index = idx
    return d


def areg(
    absorb: str,
    formula: str | None = None,
    data: pd.DataFrame | None = None,
    endog: pd.DataFrame | pd.Series | None = None,
    exog: pd.DataFrame | None = None,
    weights: str | np.ndarray | pd.Series | pd.DataFrame | None = None,
) -> tuple[RegressionResults, FeRsquared]:
    """
    Run fixed-effects regression similar to Stata's areg, absorbing the FE.

    The `rsquared` attribute in the returned `RegressionResults` contains the R^2
    as computed by Stata's -areg-, i.e., it is based on predicted values that
    include the (absorbed) fixed effect. The other FE R^2 metrics are returned
    separately in a dataclass.

    Parameters
    ----------
    absorb
        Column or index name which contains group identifiers defining the level
        at which FE are created.
    formula
        Patsy formula used to determine LHS and RHS factors.
    data
        Data from which to generate endogenous and exogenous variables using
        `formula`. Assume to contain weights if `weights` is a str.
    endog
        Endogenous variable. Takes precedence over `formula` & `data`
        if both are given.
    exog
        Exogenous variables. Takes precedence over `formula` & `data`
        if both are given.
    weights
        Column name of weights stored in `data` or array containing sample
        weights.

    Returns
    -------
    result
        Regression results containing estimated parameters and diagnostics.
    metrics
        Dataclass containing Stata-like FE R-squared metrics.
    """
    data = _areg_fix_index(data, absorb)
    endog = _areg_fix_index(endog, absorb)
    exog = _areg_fix_index(exog, absorb)

    y_raw: pd.DataFrame | pd.Series
    X: pd.DataFrame

    if endog is not None and exog is not None:
        y_raw = endog.copy()
        X = exog.copy()
    elif formula is not None and data is not None:
        y_raw, X = patsy.dmatrices(formula, data, return_type='dataframe')
    else:
        raise ValueError('Either data or endog + exog arguments required')

    has_weights = weights is not None
    weights_indiv: pd.Series | None = None
    weights_arr: np.ndarray | None = None

    if has_weights:
        if data is not None and isinstance(weights, str) and weights in data.columns:
            weights_name = weights
            weights_series = data[weights]
        else:
            weights_name = '_weights'
            weights_series = weights

        weights_arr = np.array(weights_series, dtype=float, copy=True)
        sw = pd.Series(weights_arr, name=weights_name, index=X.index, copy=True)

        keep = sw > 0.0

        if not keep.all():
            if isinstance(y_raw, pd.DataFrame):
                y_raw = y_raw.loc[keep].copy()
            else:
                y_raw = y_raw.loc[keep].copy()
            X = X.loc[keep].copy()
            sw = sw.loc[keep].copy()
            if data is not None:
                data = data.loc[keep].copy()

        # Weights normalized within FE unit
        sw_sum = sw.groupby(absorb).transform('sum')
        weights_indiv = sw / sw_sum

        # Normalize so that weights sum to 1.0
        weights_arr /= weights_arr.sum()

    # Convert to Series
    y = y_raw.iloc[:, 0].copy() if isinstance(y_raw, pd.DataFrame) else y_raw.copy()

    # Detect constant columns (std is NaN for single obs)
    has_const = any(not (v.std() > 0) for name, v in X.items())

    if data is not None:
        groups = data.index.get_level_values(absorb)
    elif exog is not None:
        groups = exog.index.get_level_values(absorb)
    else:
        raise ValueError('Either data or exog must be provided')

    ybar = float(np.average(y, weights=weights_arr))
    Xbar = np.average(X, axis=0, weights=weights_arr)

    # demean outcome within FE cells
    y_dem, y_mean = demean_within(y, absorb, weights_indiv, rescale_weights=False)
    # add back total mean
    y_dem += ybar

    # demean regressors within FE cells
    X_dem, X_mean = demean_within(X, absorb, weights_indiv, rescale_weights=False)
    # add back total mean
    X_dem += Xbar

    if has_weights:
        reg = sm.WLS(y_dem, X_dem, weights=weights_arr, hasconst=has_const)
    else:
        reg = sm.OLS(y_dem, X_dem, hasconst=has_const)

    # Account for df loss from FE transform
    reg.df_resid -= groups.nunique() - 1

    result = reg.fit()

    rsquared_within = result.rsquared

    # --- Between regression R^2 ---

    y_mean_hat = result.predict(X_mean)

    # NOTE: Stata does not seem to use weights for between R^2, so neither do we.
    VCV = np.cov(y_mean, y_mean_hat, aweights=None)
    corr_bw = VCV[0, 1] / np.sqrt(VCV[0, 0] * VCV[1, 1])
    rsquared_bw = corr_bw**2.0

    # --- Overall R^2 ---

    y_hat = result.predict(X)
    # NOTE: Stata does not seem to use weights for overall R^2, so neither do we.
    VCV = np.cov(y_hat, y, aweights=None)
    corr_overall = VCV[0, 1] / np.sqrt(VCV[0, 0] * VCV[1, 1])
    rsquared_overall = corr_overall**2.0

    # --- R^2 as computed by Stata's -areg- ---

    resid = y - y_hat
    # Predicted fixed effects
    if has_weights:
        assert weights_indiv is not None
        resid *= weights_indiv
        fe = resid.groupby(absorb).transform('sum')
    else:
        fe = resid.groupby(absorb).transform('mean')

    y_hat_total = y_hat + fe
    VCV = np.cov(y_hat_total, y, aweights=weights_arr)
    corr_total = VCV[0, 1] / np.sqrt(VCV[0, 0] * VCV[1, 1])
    rsquared_total = corr_total**2.0

    # Replace original R^2 with the one that is the same as Stata's -areg-
    result.rsquared = rsquared_total

    metrics = FeRsquared(
        within=rsquared_within,
        between=rsquared_bw,
        overall=rsquared_overall,
    )

    return result, metrics
