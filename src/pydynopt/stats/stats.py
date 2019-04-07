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


def quantile(x, pmf, qrank):
    """
    Compute quantiles of a given distribution characterized by its (finite)
    state space and PMF.

    If `x` and `pmf` are of equal length, they are assumed to represent a
    discrete random variable and no (linear) interpolation is attempted.

    If `x` has one element more than `pmf`, it is assumed that `x` contains
    the bin edges that define intervals with the corresponding mass given by
    `pmf`. Quantile ranks are linearly interpolated, but *only* within the bin
    that brackets a percentile.

    This function differs from Numpy's quantile() in that in attempts to
    correctly deal with points masses in PMFs, yet still allow for linear
    interpolation within a bin.

    Parameters
    ----------
    x : array_like
    pmf : array_like
    qrank : array_like or float
        Quantile ranks (valid range: [0,1])

    Returns
    -------
    pctl : np.ndarray or float
        Quantile corresponding to given percentile ranks.
    """

    x = np.atleast_1d(x).flatten()
    pmf = np.atleast_1d(pmf).flatten()
    if np.any(qrank < 0.0) or np.any(qrank > 1.0):
        msg = 'Invalid percentile rank argument'
        raise ValueError(msg)

    if len(x) == len(pmf):
        # Assume that RV is discrete and that x contains the discrete support
        # with corresponding probabilities stored in pmf

        cdf = np.hstack((0.0, np.cumsum(pmf)))
        cdf /= cdf[-1]
        # trim (constant) right tail as that confuses digitize()
        ii = np.where(np.abs(cdf - 1.0) > 1.0e-14)[0]
        cdf = cdf[ii]
        ii = np.digitize(qrank, cdf) - 1
        ii = np.fmin(ii, len(pmf) - 1)
        pctl = x[ii]

    elif len(x) == (len(pmf) + 1):
        cdf = np.hstack((0.0, np.cumsum(pmf)))
        cdf /= cdf[-1]
        # trim (constant) right tail as that confuses digitize()
        iub = np.amin(np.where(cdf == 1.0)[0]) + 1
        cdf = cdf[:iub]
        x = x[:iub]
        cdf[-1] = 1.0
        ii = np.digitize(qrank, cdf) - 1
        # include only CDF values that bracket percentiles of interest.
        ii = np.union1d(ii, np.fmin(ii + 1, len(cdf) - 1))
        # linearly interpolate *within* brackets
        pctl = np.interp(qrank, cdf[ii], x[ii])
    else:
        msg = 'Non-conformable arrays'
        raise ValueError(msg)

    return pctl


def percentile(x, pmf, prank):
    """
    Convenience wrapper around quantile() function that accepts percentile
    rank argument in the interval [0,100] instead of quantile ranks
    in [0.0,1.0]

    Parameters
    ----------
    x : array_like
    pmf : array_like
    prank : array_like or float
        Percentile ranks (valid range: [0,100])

    Returns
    -------
    pctl : float or np.ndarray
        Percentiles corresponding to given percentile ranks
    """

    isscalar = np.isscalar(prank)
    qrank = np.array(prank) / 100.0
    pctl = quantile(x, pmf, qrank)

    if isscalar:
        pctl = np.asscalar(pctl)

    return pctl


def quantile_rank(x, pmf, qntl):
    """
    (Approximate) inverse function of quantile().
    Returns the quantile ranks corresponding to a given array of quantiles.

    Note: this basically is a CDF function except that it returns NaN
    for those quantiles that are outside of the distribution's support.

    Additionally, this function should correctly handle point masses.

    Parameters
    ----------
    x : array_like
        (flattened) state space
    pmf : array_like
        PMF corresponding to (flattened) state space
    qntl : float or array_like
        List of quantiles

    Returns
    -------
    rank : array_like
        Quantile ranks corresponding to given quantiles
    """

    is_scalar = np.isscalar(qntl)
    shp_in = np.array(qntl).shape

    x = np.atleast_1d(x).flatten()
    pmf = np.atleast_1d(pmf).flatten()

    qntl = np.atleast_1d(qntl)

    if len(x) == len(pmf):
        # (x, pmf) describe a discrete random variable and qntl should
        # match the values in x exactly for this to make any sense.
        raise NotImplementedError()
    elif len(x) == (len(pmf) + 1):
        # assume that x contains bin edges and pmf contains the mass
        # within these edges.
        cdf = np.hstack((0.0, np.cumsum(pmf)))
        cdf /= cdf[-1]

        # remove all points where CDF is exactly 0.0, except for the last
        ifrom = np.amax(np.where(cdf == 0.0)[0])
        # remove all trailing points where CDF is exactly 1.0, except for the
        # first one
        ito = min(np.amin(np.where(cdf == 1.0)[0]) + 1, len(cdf))
        cdf = cdf[ifrom:ito]
        x = x[ifrom:ito]

        ii = np.digitize(qntl, x, right=True)
        # include only CDF values that bracket percentiles of interest.
        jj = np.fmin(np.fmax(0, ii), len(x) - 2)
        jj = np.union1d(jj, jj+1)

        rank = np.interp(qntl, x[jj], cdf[jj], left=np.nan, right=np.nan)
    else:
        rank = None

    if rank is not None:
        if is_scalar:
            rank = np.asscalar(rank)
        else:
            rank = rank.reshape(shp_in)

    return rank


def percentile_rank(x, pmf, pctl):
    """
    Convenience wrapper around quantile_rank() that returns percentiles
    instead of quantiles.

    Parameters
    ----------
    x : array_like
        (flattened) state space
    pmf : array_like
        PMF corresponding to (flattened) state space
    qntl : float or array_like
        List of percentiles

    Returns
    -------
    rank : array_like
        Percentile ranks corresponding to given percentiles
    """
    rank = quantile_rank(x, pmf, pctl)
    rank *= 100.0

    return rank
