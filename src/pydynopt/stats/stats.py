"""
Module providing core statistical functions

Author: Richard Foltyn
"""

import numpy as np

from pydynopt.numba import jit, register_jitable, overload
from pydynopt.numba import to_array


def gini(states, pmf, assume_sorted=False):
    """
    Compute Gini coefficient from a normalized histogram (or a discrete RV
    with finite support)

    Formula taken directly from Wikipedia.

    Parameters
    ----------
    states : array_like
        State space of discrete random variable, or histogam bin midpoints.
        Higher-dimensional arrays will be flattened.
    pmf : array_like
        Probability corresponding to each element in `states`.
        Higher-dimensional arrays will be flattened.
    assume_sorted : bool
        If true, assume that `states` array is sorted (ignored for higher-
        dimensional arrays which are always sorted)

    Returns
    -------
    gini : float
        Gini coefficient for given PMF or histogram
    """

    states = np.atleast_1d(states)
    pmf = np.atleast_1d(pmf)

    needs_sort = states.ndim > 1 or not assume_sorted
    states = states.reshape((-1, ))
    pmf = pmf.reshape((-1, ))

    if needs_sort:
        iorder = np.argsort(states)
        states = states[iorder]
        pmf = pmf[iorder]

    S = np.cumsum(pmf*states)
    S = np.insert(S, 0, 0.0)
    midS = (S[:-1] + S[1:])
    gini = 1.0 - np.sum(pmf*midS)/S[-1]

    return gini


@jit(nopython=True, nogil=True)
def create_unique_pmf(x, pmf):
    """
    Collapses discrete distribution with potentially duplicate values in the
    state space to a state space with unique values and appropriately summed
    up probabilities.

    Parameters
    ----------
    x : np.ndarray
    pmf : np.ndarray

    Returns
    -------
    xuniq : np.ndarray
        Sorted, unique state space
    pmf_uniq: np.ndarray
        Probabilities corresponding to unique states
    """

    xuniq = np.unique(x)
    if len(xuniq) == len(x) and np.all(xuniq == x):
        return x, pmf

    pmf_uniq = np.zeros_like(xuniq)

    j = 0
    nx = len(x)

    for i, xi in enumerate(xuniq):
        while j < nx and (xi == x[j]):
            pmf_uniq[i] += pmf[j]
            j += 1

    pmf_uniq /= np.sum(pmf_uniq)

    return xuniq, pmf_uniq


def quantile_array(x, pmf, qrank, assume_sorted=False, assume_unique=False):
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
    qrank : array_like
        Quantile ranks (valid range: [0,1])
    assume_sorted : bool
        If true, assume that state space `x` is sorted.
    assume_unique : bool
        If true, assume that elements in state space `x` are unique.

    Returns
    -------
    q : np.ndarray
        Quantile corresponding to given quantile ranks.
    """

    x1d = np.atleast_1d(x).flatten()
    pmf1d = np.atleast_1d(pmf).flatten()
    qrank1d = np.atleast_1d(qrank).flatten()

    if np.any(qrank1d < 0.0) or np.any(qrank1d > 1.0):
        raise ValueError('Invalid percentile rank argument')

    if len(x1d) == len(pmf1d):
        # Assume that RV is discrete and that x contains the discrete support
        # with corresponding probabilities stored in pmf

        if not assume_sorted:
            iorder = np.argsort(x1d)
            x1d = x1d[iorder]
            pmf1d = pmf1d[iorder]

        if not assume_unique:
            x1d, pmf1d = create_unique_pmf(x1d, pmf1d)

        cdf = np.empty((pmf1d.size + 1, ), dtype=pmf1d.dtype)
        cdf[0] = 0.0
        cdf[1:] = np.cumsum(pmf1d)
        cdf /= cdf[-1]

        # trim (constant) right tail as that confuses digitize()
        # Note that there is at least one such element as we set the last
        # element to 1.0
        ii = np.where(cdf > (1.0 - 1.0e-14))[0]
        imax = ii[0]
        cdf = cdf[:imax+1]
        cdf[-1] = 1.0

        # trim (constant) left tail
        ii = np.where(cdf > 0.0)[0]
        imin = max(0, ii[0] - 1)
        cdf = cdf[imin:]
        ii = np.digitize(qrank1d, cdf)
        ii += (imin - 1)
        ii = np.fmin(ii, pmf1d.size - 1)
        q = x1d[ii]

    elif len(x1d) == (len(pmf1d) + 1):

        cdf = np.empty((pmf1d.size + 1,), dtype=pmf1d.dtype)
        cdf[0] = 0.0
        cdf[1:] = np.cumsum(pmf1d)
        cdf /= cdf[-1]

        # trim (constant) right tail as that confuses digitize()
        iub = np.amin(np.where(cdf == 1.0)[0]) + 1
        cdf = cdf[:iub]
        x1d = x1d[:iub]
        cdf[-1] = 1.0
        ii = np.digitize(qrank1d, cdf)
        ii -= 1
        # include only CDF values that bracket percentiles of interest.
        # This is a work-around as Numba does not support np.union1d()
        iip1 = np.fmin(ii + 1, cdf.size - 1)
        ii = np.hstack((ii, iip1))
        ii = np.unique(ii)
        # linearly interpolate *within* brackets
        q = np.interp(qrank1d, cdf[ii], x1d[ii])
    else:
        raise ValueError('Non-conformable arrays')

    return q


def quantile_scalar(x, pmf, qrank, assume_sorted=False, assume_unique=False):
    """
    Implementation of quantile() function for scalar-valued `qrank` arguments.

    Parameters
    ----------
    x : array_like
    pmf : array_like
    qrank : float
    assume_sorted : bool
    assume_unique : bool

    Returns
    -------
    q : float
    """
    qrank1d = np.array(qrank)
    q1d = quantile(x, pmf, qrank1d, assume_sorted, assume_unique)

    q = q1d[0]

    return q


def quantile(x, pmf, qrank, assume_sorted=False, assume_unique=False):
    """
    Compute quantiles of a given distribution characterized by its (finite)
    state space and PMF.

    For implementation details see the documentation for quantile_array().

    Parameters
    ----------
    x : array_like
    pmf : array_like
    qrank : array_like or float
        Quantile ranks (valid range: [0,1])
    assume_sorted : bool
        If true, assume that state space `x` is sorted.
    assume_unique : bool
        If true, assume that elements in state space `x` are unique.

    Returns
    -------
    q : np.ndarray or float
        Quantile corresponding to given quantile ranks.
    """

    qrank1d = np.array(qrank)
    q = quantile_array(x, pmf, qrank1d, assume_sorted, assume_unique)

    if np.isscalar(qrank):
        q = q.item()

    return q


@overload(quantile, jit_options={'nogil': True, 'parallel': False})
def quantile_generic(x, pmf, qrank, assume_sorted=False, assume_unique=False):
    from numba.types import Number

    f = None
    if isinstance(qrank, Number):
        f = quantile_scalar
    else:
        f = quantile_array

    return f


def percentile_array(x, pmf, prank, assume_sorted=False, assume_unique=False):
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
    assume_sorted : bool
        If true, assume that state space `x` is sorted.
    assume_unique : bool
        If true, assume that elements in state space `x` are unique.

    Returns
    -------
    pctl : np.ndarray
        Percentiles corresponding to given percentile ranks
    """

    qrank = to_array(prank)
    qrank /= 100.0
    pctl = quantile(x, pmf, qrank, assume_sorted, assume_unique)

    return pctl


def percentile_scalar(x, pmf, prank, assume_sorted=False, assume_unique=False):

    qrank = np.array(prank)
    qrank /= 100.0
    pctl1d = quantile(x, pmf, qrank, assume_sorted, assume_unique)

    pctl = pctl1d[0]

    return pctl


def percentile(x, pmf, prank, assume_sorted=False, assume_unique=False):
    """
    Compute percentiles of a given distribution characterized by its (finite)
    state space and PMF.

    See documentation for quantile() for implementation details.

    Parameters
    ----------
    x : array_like
    pmf : array_like
    prank : array_like or float
        Percentile ranks (valid range: [0,100])
    assume_sorted : bool
        If true, assume that state space `x` is sorted.
    assume_unique : bool
        If true, assume that elements in state space `x` are unique.

    Returns
    -------
    pctl : np.ndarray or float
        Percentiles corresponding to given percentile ranks.
    """

    qrank = np.array(prank, dtype=np.float64)
    qrank /= 100.0
    pctl = quantile_array(x, pmf, qrank, assume_sorted, assume_unique)

    if np.isscalar(prank):
        pctl = pctl.item()

    return pctl


@overload(percentile, jit_options={'nogil': True, 'parallel': False})
def percentile_generic(x, pmf, prank, assume_sorted=False, assume_unique=False):

    from numba.types import Number

    f = None
    if isinstance(prank, Number):
        f = percentile_scalar
    else:
        f = percentile_array

    return f


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
