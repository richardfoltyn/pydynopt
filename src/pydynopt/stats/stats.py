"""
Module providing core statistical functions

Author: Richard Foltyn
"""

import numpy as np


def gini(states, pmf):
    """
    Compute Gini coefficient from a normalized histogram (or a discrete RV
    with finite support)

    Formula taken directory from Wikipedia.

    Parameters
    ----------
    states : array_like
    pmf : array_like

    Returns
    -------
    gini : float
        Gini coefficient for given PMF or histogram
    """

    states = np.atleast_1d(states)
    pmf = np.atleast_1d(pmf)

    S = np.cumsum(pmf*states)
    S = np.insert(S, 0, 0.0)
    midS = (S[:-1] + S[1:])
    gini = 1.0 - np.sum(pmf*midS)/S[-1]

    return gini


def percentile(x, pmf, prank):
    """
    Compute percentiles of a given distribution characterized by its (finite)
    state space and PMF.

    If `x` and `pmf` are of equal length, they are assumed to represent a
    discrete random variable and no (linear) interpolation is attempted.

    If `x` has one element more than `pmf`, it is assumed that `x` contains
    the bin edges that define intervals with the corresponding mass given by
    `pmf`. Percentile ranks are linearly interpolated, but *only* within the bin
    that brackets a percentile.

    This function differs from Numpy's percentile() in that in attempts to
    correctly deal with points masses in PMFs, yet still allow for linear
    interpolation within a bin.

    Parameters
    ----------
    x : array_like
    pmf : array_like
    prank : array_like or float
        Percentile ranks (valid range: [0,100])

    Returns
    -------
    pctl : np.ndarray or float
        Percentiles corresponding to given percentile ranks.
    """

    x = np.atleast_1d(x).flatten()
    pmf = np.atleast_1d(pmf).flatten()
    if np.any(prank < 0.0) or np.any(prank > 1.0):
        msg = 'Invalid percentile rank argument'
        raise ValueError(msg)

    if len(x) == len(pmf):
        # Assume that RV is discrete and that x contains the discrete support
        # with corresponding probabilities stored in pmf

        cdf = np.hstack((0.0, np.cumsum(pmf)))
        # trim (constant) right tail as that confuses digitize()
        ii = np.where(np.abs(cdf - 1.0) > 1.0e-14)[0]
        cdf = cdf[ii]
        ii = np.digitize(prank, cdf) - 1
        ii = np.fmin(ii, len(pmf) - 1)
        pctl = x[ii]

    elif len(x) == (len(pmf) + 1):
        cdf = np.hstack((0.0, np.cumsum(pmf)))
        # trim (constant) right tail as that confuses digitize()
        ii = np.where(np.abs(cdf - 1.0) > 1.0e-14)[0]
        cdf = cdf[ii]
        x = x[ii]
        cdf[-1] = 1.0
        ii = np.digitize(prank, cdf) - 1
        # include only CDF values that bracket percentiles of interest.
        ii = np.union1d(ii, np.fmin(ii + 1, len(cdf) - 1))
        # linearly interpolate *within* brackets
        pctl = np.interp(prank, cdf[ii], x[ii])
    else:
        msg = 'Non-conformable arrays'
        raise ValueError(msg)

    return pctl


def quantile(x, pmf, qrank):
    """
    Convenience wrapper around percentile() function that accepts quantile
    rank argument in the interval [0.0, 1.0] instead of percentile ranks
    in [0.0, 100.0]

    Parameters
    ----------
    x : array_like
    pmf : array_like
    qrank : array_like or float
        Quantiles ranks (valid range: [0.0, 1.0])

    Returns
    -------
    qntl : float or np.ndarray
        Quantiles corresponding to given quantile ranks
    """

    qntl = percentile(x, pmf, qrank*100.0)
    return qntl
