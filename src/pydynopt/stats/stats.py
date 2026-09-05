"""Provide core statistical functions.

- Compute inequality measures, quantiles, and percentile ranks.
- Discretize continuous random variables.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Callable, Sequence
from typing import Any, Protocol, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pydynopt.numba import JIT_OPTIONS, jit, overload as numba_overload

type NumericScalar = int | float | complex | np.number[Any]
type NumericResult = NumericScalar | NDArray[Any]


class _ContinuousDistribution(Protocol):
    """Describe the continuous distribution methods used by this module."""

    def ppf(self, q: ArrayLike, **kwargs: Any) -> ArrayLike:
        """Evaluate the percentile-point function."""
        ...

    def expect(
        self,
        func: Callable[[float], float],
        *,
        lb: float,
        ub: float,
        conditional: bool,
        **kwargs: Any,
    ) -> float | np.number[Any]:
        """Evaluate an expectation over an interval."""
        ...


@jit(**JIT_OPTIONS)
def gini(
    states: ArrayLike,
    pmf: ArrayLike,
    assume_sorted: bool = False,
) -> float:
    """Compute the Gini coefficient of a normalized finite distribution.

    Parameters
    ----------
    states
        Discrete state space or histogram bin midpoints. Higher-dimensional
        arrays are flattened.
    pmf
        Probabilities corresponding to ``states``. Higher-dimensional arrays
        are flattened.
    assume_sorted
        If true, assume ``states`` is sorted. Higher-dimensional arrays are
        always sorted.

    Returns
    -------
    Gini coefficient for the given distribution.
    """
    states_arr = np.atleast_1d(states)
    pmf_arr = np.atleast_1d(pmf)

    needs_sort = states_arr.ndim > 1 or not assume_sorted
    states1d = states_arr.reshape((-1,))
    pmf1d = pmf_arr.reshape((-1,))

    if needs_sort:
        iorder = np.argsort(states1d)
        states1d = states1d[iorder]
        pmf1d = pmf1d[iorder]

    S = np.cumsum(pmf1d * states1d)
    # Numba does not support hstack() with scalar args
    zero = np.zeros(1, dtype=S.dtype)
    S = np.hstack((zero, S))
    midS = S[:-1] + S[1:]
    gini = 1.0 - np.dot(pmf1d, midS) / S[-1]

    return float(gini)


@jit(**JIT_OPTIONS)
def create_unique_pmf(
    x: NDArray[Any],
    pmf: NDArray[Any],
    assume_sorted: bool = False,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Collapse a discrete distribution to a unique state space.

    Parameters
    ----------
    x
        State space, potentially containing duplicate values.
    pmf
        Probabilities corresponding to ``x``.
    assume_sorted
        If true, assume ``x`` is sorted in ascending order.

    Returns
    -------
    xuniq
        Sorted, unique state space.
    pmf_uniq
        Normalized probabilities corresponding to the unique states.
    """
    if not assume_sorted:
        iorder = np.argsort(x)
        x = x[iorder]
        pmf = pmf[iorder]

    xuniq = np.unique(x)
    pmf_uniq = np.zeros(xuniq.size, dtype=pmf.dtype)

    j = 0
    nx = len(x)

    for i, xi in enumerate(xuniq):
        while j < nx and (xi == x[j]):
            pmf_uniq[i] += pmf[j]
            j += 1

    mass = np.sum(pmf_uniq)
    if mass != 0.0 and np.isfinite(mass):
        pmf_uniq /= mass

    return xuniq, pmf_uniq


@jit(**JIT_OPTIONS)
def _ppf_nearest(
    rank: NDArray[Any],
    cdf: NDArray[Any],
    x: NDArray[Any],
    qntl: NDArray[Any],
) -> None:
    """Store nearest quantiles while handling flat CDF regions.

    For each rank, select ``x[j]`` such that
    ``cdf[j - 1] < rank[i] <= cdf[j]``. Boundary violations select the first
    or last applicable support value.

    Parameters
    ----------
    rank
        Quantile ranks.
    cdf
        Non-decreasing cumulative probabilities.
    x
        Sorted, unique distribution support.
    qntl
        Output array in which to store the quantiles.
    """
    # Skip over any potential initially flat region without mass
    imin = 0
    for imin in range(cdf.size - 1):
        if cdf[imin + 1] > 0.0:
            break

    for i in range(rank.size):
        ri = rank[i]
        if ri <= cdf[0]:
            # In this case we cannot find a bracketing interval, so return the smallest
            # value on the support
            qntl[i] = x[0]
        else:
            j = imin
            for j in range(imin, cdf.size - 1):
                if cdf[j] < ri <= cdf[j + 1]:
                    break
            # Desired quantile is in the half-open interval (cdf[j], cdf[j+1]], so
            # it falls into the bin j+1
            qntl[i] = x[j + 1]


@jit(**JIT_OPTIONS)
def _ppf_interp(
    rank: NDArray[Any],
    cdf: NDArray[Any],
    x: NDArray[Any],
    qntl: NDArray[Any],
) -> None:
    """Store interpolated quantiles while handling flat CDF regions.

    Each rank is first assigned to an interval satisfying
    ``cdf[j - 1] < rank[i] <= cdf[j]`` and then interpolated within that
    interval.

    Parameters
    ----------
    rank
        Quantile ranks.
    cdf
        Non-decreasing cumulative probabilities.
    x
        Sorted, unique distribution support.
    qntl
        Output array in which to store the quantiles.
    """
    # Skip over any potential initially flat region without mass
    imin = 0
    for imin in range(cdf.size - 1):
        if cdf[imin + 1] > 0.0:
            break

    for i in range(rank.size):
        ri = rank[i]

        if ri <= cdf[0]:
            # We cannot interpolate to the left, so return min. value on the support
            qntl[i] = x[0]
        else:
            ilb = imin
            for ilb in range(imin, cdf.size - 1):
                if cdf[ilb] < ri <= cdf[ilb + 1]:
                    break

            cdf_lb = cdf[ilb]
            cdf_ub = cdf[ilb + 1]

            wgt_lb = (cdf_ub - ri) / (cdf_ub - cdf_lb)

            q = wgt_lb * x[ilb] + (1.0 - wgt_lb) * x[ilb + 1]
            qntl[i] = q


def quantile_array(
    x: ArrayLike,
    pmf: ArrayLike,
    qrank: ArrayLike,
    assume_sorted: bool = False,
    assume_unique: bool = False,
    interpolation: str = 'nearest',
) -> NDArray[Any]:
    """Compute quantiles of a finite distribution for an array of ranks.

    Equal-length ``x`` and ``pmf`` inputs describe a discrete random variable.
    When ``x`` has one more element than ``pmf``, ``x`` contains bin edges and
    quantiles are linearly interpolated only within the bracketing bin.

    Unlike :func:`numpy.quantile`, this function handles point masses while
    permitting linear interpolation within bins.

    Parameters
    ----------
    x
        Discrete support or bin edges.
    pmf
        Probability masses corresponding to ``x`` or its bins.
    qrank
        Quantile ranks in the interval [0, 1].
    assume_sorted
        If true, assume ``x`` is sorted.
    assume_unique
        If true, assume the elements of ``x`` are unique.
    interpolation
        Interpolation method. Its interpretation depends on the size of ``x``.

    Returns
    -------
    Quantiles corresponding to the given ranks.
    """
    x1d = np.atleast_1d(x).flatten()
    pmf1d = np.atleast_1d(pmf).flatten()
    qrank1d = np.atleast_1d(qrank).flatten()

    interpolation = interpolation.lower()

    if qrank1d.size == 0:
        q = np.empty(0, dtype=x1d.dtype)
        return q

    if np.any(qrank1d < 0.0) or np.any(qrank1d > 1.0):
        raise ValueError('Invalid percentile rank argument')

    q = np.full_like(qrank1d, dtype=x1d.dtype, fill_value=np.nan)

    if len(x1d) == len(pmf1d):
        # Assume that RV is discrete and that x contains the discrete support
        # with corresponding probabilities stored in pmf

        if not assume_sorted:
            iorder = np.argsort(x1d)
            x1d = x1d[iorder]
            pmf1d = pmf1d[iorder]

        if not assume_unique:
            x1d, pmf1d = create_unique_pmf(x1d, pmf1d, assume_sorted=True)

        cdf = np.cumsum(pmf1d)
        mass = cdf[-1]
        if mass == 0.0 or not np.isfinite(mass):
            return q
        cdf /= mass

    elif len(x1d) == (len(pmf1d) + 1):
        # Assume that this is a continuous RV and the values in x are BIN
        # EDGES while PMF contains the mass in the bin between any two edges.
        # This should be combined with (linear) interpolation as returning
        # the nearest edge does not make any sense.

        cdf = np.empty((pmf1d.size + 1,), dtype=pmf1d.dtype)
        cdf[0] = 0.0
        cdf[1:] = np.cumsum(pmf1d)
        mass = cdf[-1]
        if mass == 0.0 or not np.isfinite(mass):
            return q
        cdf /= mass

        # Force linear interpolation
        interpolation = 'linear'
    else:
        raise ValueError('Non-conformable arrays')

    if interpolation == 'nearest':
        _ppf_nearest(qrank1d, cdf, x1d, q)
    elif interpolation == 'linear':
        _ppf_interp(qrank1d, cdf, x1d, q)
    else:
        raise ValueError('Unsupported interpolation method')

    return q


def quantile_scalar(
    x: ArrayLike,
    pmf: ArrayLike,
    qrank: NumericScalar,
    assume_sorted: bool = False,
    assume_unique: bool = False,
    interpolation: str = 'nearest',
) -> np.number[Any]:
    """Compute a quantile for a scalar-valued rank.

    Parameters
    ----------
    x
        Discrete support or bin edges.
    pmf
        Probability masses corresponding to ``x`` or its bins.
    qrank
        Quantile rank in the interval [0, 1].
    assume_sorted
        If true, assume ``x`` is sorted.
    assume_unique
        If true, assume the elements of ``x`` are unique.
    interpolation
        Interpolation method. Its interpretation depends on the size of ``x``.

    Returns
    -------
    Quantile corresponding to the given rank.
    """
    qrank1d = np.asarray(qrank, dtype=np.asarray(x).dtype)
    q1d = quantile(
        x,
        pmf,
        qrank1d,
        assume_sorted,
        assume_unique,
        interpolation,
    )

    q = q1d[0]

    return q


@overload
def quantile(
    x: ArrayLike,
    pmf: ArrayLike,
    qrank: NumericScalar,
    assume_sorted: bool = False,
    assume_unique: bool = False,
    interpolation: str = 'nearest',
) -> NumericScalar: ...


@overload
def quantile(
    x: ArrayLike,
    pmf: ArrayLike,
    qrank: NDArray[Any],
    assume_sorted: bool = False,
    assume_unique: bool = False,
    interpolation: str = 'nearest',
) -> NDArray[Any]: ...


@overload
def quantile(
    x: ArrayLike,
    pmf: ArrayLike,
    qrank: ArrayLike,
    assume_sorted: bool = False,
    assume_unique: bool = False,
    interpolation: str = 'nearest',
) -> NumericResult: ...


def quantile(
    x: ArrayLike,
    pmf: ArrayLike,
    qrank: ArrayLike,
    assume_sorted: bool = False,
    assume_unique: bool = False,
    interpolation: str = 'nearest',
) -> NumericResult:
    """Compute quantiles of a finite distribution.

    See :func:`quantile_array` for implementation details.

    Parameters
    ----------
    x
        Discrete support or bin edges.
    pmf
        Probability masses corresponding to ``x`` or its bins.
    qrank
        Quantile ranks in the interval [0, 1].
    assume_sorted
        If true, assume ``x`` is sorted.
    assume_unique
        If true, assume the elements of ``x`` are unique.
    interpolation
        Interpolation method. Its interpretation depends on the size of ``x``.

    Returns
    -------
    Quantiles corresponding to the given ranks.
    """
    qrank1d = np.asarray(qrank)
    q = quantile_array(x, pmf, qrank1d, assume_sorted, assume_unique, interpolation)

    if np.isscalar(qrank):
        q = q.item()

    return q


@numba_overload(quantile, jit_options=JIT_OPTIONS)
def quantile_generic(
    x: Any,
    pmf: Any,
    qrank: Any,
    assume_sorted: Any = False,
    assume_unique: Any = False,
    interpolation: Any = 'nearest',
) -> Callable[..., Any]:
    """Select a quantile implementation for Numba compilation."""
    from numba import types

    if isinstance(qrank, types.Number):
        return quantile_scalar

    return quantile_array


def percentile_array(
    x: ArrayLike,
    pmf: ArrayLike,
    prank: ArrayLike,
    assume_sorted: bool = False,
    assume_unique: bool = False,
    interpolation: str = 'nearest',
) -> NDArray[Any]:
    """Compute percentiles for an array of ranks in [0, 100].

    Parameters
    ----------
    x
        Discrete support or bin edges.
    pmf
        Probability masses corresponding to ``x`` or its bins.
    prank
        Percentile ranks in the interval [0, 100].
    assume_sorted
        If true, assume ``x`` is sorted.
    assume_unique
        If true, assume the elements of ``x`` are unique.
    interpolation
        Interpolation method. Its interpretation depends on the size of ``x``.

    Returns
    -------
    Percentiles corresponding to the given ranks.
    """
    qrank = np.asarray(prank) / 100.0
    pctl = quantile(
        x,
        pmf,
        qrank,
        assume_sorted,
        assume_unique,
        interpolation,
    )

    return pctl


def percentile_scalar(
    x: ArrayLike,
    pmf: ArrayLike,
    prank: NumericScalar,
    assume_sorted: bool = False,
    assume_unique: bool = False,
    interpolation: str = 'nearest',
) -> np.number[Any]:
    """Compute a percentile for a scalar-valued rank.

    Parameters
    ----------
    x
        Discrete support or bin edges.
    pmf
        Probability masses corresponding to ``x`` or its bins.
    prank
        Percentile rank in the interval [0, 100].
    assume_sorted
        If true, assume ``x`` is sorted.
    assume_unique
        If true, assume the elements of ``x`` are unique.
    interpolation
        Interpolation method. Its interpretation depends on the size of ``x``.

    Returns
    -------
    Percentile corresponding to the given rank.
    """
    qrank = np.atleast_1d(np.asarray(prank, dtype=np.asarray(x).dtype)) / 100.0
    pctl1d = quantile(
        x,
        pmf,
        qrank,
        assume_sorted,
        assume_unique,
        interpolation,
    )

    pctl = pctl1d[0]

    return pctl


@overload
def percentile(
    x: ArrayLike,
    pmf: ArrayLike,
    prank: NumericScalar,
    assume_sorted: bool = False,
    assume_unique: bool = False,
    interpolation: str = 'nearest',
) -> NumericScalar: ...


@overload
def percentile(
    x: ArrayLike,
    pmf: ArrayLike,
    prank: NDArray[Any],
    assume_sorted: bool = False,
    assume_unique: bool = False,
    interpolation: str = 'nearest',
) -> NDArray[Any]: ...


@overload
def percentile(
    x: ArrayLike,
    pmf: ArrayLike,
    prank: ArrayLike,
    assume_sorted: bool = False,
    assume_unique: bool = False,
    interpolation: str = 'nearest',
) -> NumericResult: ...


def percentile(
    x: ArrayLike,
    pmf: ArrayLike,
    prank: ArrayLike,
    assume_sorted: bool = False,
    assume_unique: bool = False,
    interpolation: str = 'nearest',
) -> NumericResult:
    """Compute percentiles of a finite distribution.

    See :func:`quantile` for implementation details.

    Parameters
    ----------
    x
        Discrete support or bin edges.
    pmf
        Probability masses corresponding to ``x`` or its bins.
    prank
        Percentile ranks in the interval [0, 100].
    assume_sorted
        If true, assume ``x`` is sorted.
    assume_unique
        If true, assume the elements of ``x`` are unique.
    interpolation
        Interpolation method. Its interpretation depends on the size of ``x``.

    Returns
    -------
    Percentiles corresponding to the given ranks.
    """
    qrank = np.asarray(prank) / 100.0
    pctl = quantile_array(x, pmf, qrank, assume_sorted, assume_unique, interpolation)

    if np.isscalar(prank):
        pctl = pctl.item()

    return pctl


@numba_overload(percentile, jit_options=JIT_OPTIONS)
def percentile_generic(
    x: Any,
    pmf: Any,
    prank: Any,
    assume_sorted: Any = False,
    assume_unique: Any = False,
    interpolation: Any = 'nearest',
) -> Callable[..., Any]:
    """Select a percentile implementation for Numba compilation."""
    from numba import types

    if isinstance(prank, types.Number):
        return percentile_scalar

    return percentile_array


def quantile_rank(
    x: ArrayLike,
    pmf: ArrayLike,
    qntl: ArrayLike,
    interpolation: str = 'linear',
) -> float | NDArray[Any] | None:
    """Compute the approximate inverse of :func:`quantile`.

    This function acts like a CDF but returns NaN for quantiles outside the
    distribution support and handles point masses.

    Parameters
    ----------
    x
        Flattened state space or bin edges.
    pmf
        Probability masses corresponding to ``x`` or its bins.
    qntl
        Quantiles for which to compute ranks.
    interpolation
        Interpolation method for quantiles between support points.

    Returns
    -------
    Quantile ranks, or ``None`` if the input lengths are incompatible.
    """
    is_scalar = np.isscalar(qntl)
    shp_in = np.asarray(qntl).shape
    interpolation = interpolation.lower()

    x1d = np.atleast_1d(x).flatten()
    pmf1d = np.atleast_1d(pmf).flatten()
    qntl1d = np.atleast_1d(qntl)

    rank: NDArray[Any] | None
    if len(x1d) == len(pmf1d):
        cdf = np.cumsum(pmf1d)
        cdf /= cdf[-1]

        # Remove initial zero-mass points, retaining the last one.
        ii = np.where(cdf == 0.0)[0]
        ifrom = np.amax(ii) if len(ii) > 0 else 0
        # Remove trailing unit-mass points, retaining the first one.
        ii = np.where(cdf == 1.0)[0]
        ito = min(np.amin(ii) + 1, len(cdf)) if len(ii) > 0 else len(cdf)
        cdf = cdf[ifrom:ito]
        x1d = x1d[ifrom:ito]

        if interpolation == 'linear':
            rank = np.interp(qntl1d, x1d, cdf, left=np.nan, right=np.nan)
        else:
            raise NotImplementedError('Interpolation method not implemented')

    elif len(x1d) == len(pmf1d) + 1:
        # x contains bin edges and pmf contains the mass within these edges.
        cdf = np.hstack((0.0, np.cumsum(pmf1d)))
        cdf /= cdf[-1]

        # Remove initial zero-mass points, retaining the last one.
        ii = np.where(cdf == 0.0)[0]
        ifrom = np.amax(ii) if len(ii) > 0 else 0
        # Remove trailing unit-mass points, retaining the first one.
        ii = np.where(cdf == 1.0)[0]
        ito = min(np.amin(ii) + 1, len(cdf)) if len(ii) > 0 else len(cdf)
        cdf = cdf[ifrom:ito]
        x1d = x1d[ifrom:ito]

        ii = np.digitize(qntl1d, x1d, right=True)
        # Include only CDF values bracketing the quantiles of interest.
        jj = np.fmin(np.fmax(0, ii), len(x1d) - 2)
        jj = np.union1d(jj, jj + 1)

        rank = np.interp(qntl1d, x1d[jj], cdf[jj], left=np.nan, right=np.nan)
    else:
        rank = None

    if rank is None:
        return None
    if is_scalar:
        return float(rank.item())

    return rank.reshape(shp_in)


def percentile_rank(
    x: ArrayLike,
    pmf: ArrayLike,
    pctl: ArrayLike,
    interpolation: str = 'linear',
) -> float | NDArray[Any] | None:
    """Compute percentile ranks for values from a finite distribution.

    Parameters
    ----------
    x
        Flattened state space or bin edges.
    pmf
        Probability masses corresponding to ``x`` or its bins.
    pctl
        Percentiles for which to compute ranks.
    interpolation
        Interpolation method for percentiles between support points.

    Returns
    -------
    Percentile ranks, or ``None`` if the input lengths are incompatible.
    """
    rank = quantile_rank(x, pmf, pctl, interpolation)
    if rank is None:
        return None

    rank *= 100.0
    return rank


def discretize_rv(
    *,
    n: int | None = None,
    q: Sequence[float] | NDArray[Any] | None = None,
    dist: _ContinuousDistribution | None = None,
    return_edges: bool = False,
    **kwargs: Any,
) -> (
    tuple[NDArray[Any], NDArray[Any]] | tuple[NDArray[Any], NDArray[Any], NDArray[Any]]
):
    """Discretize a continuous random variable into finite bins.

    Each discrete realization is the conditional expectation within its bin.

    Parameters
    ----------
    n
        Number of bins to create.
    q
        Quantile ranks in [0, 1] defining the bin edges. Takes precedence over
        ``n`` when supplied.
    dist
        Continuous distribution implementing the SciPy ``ppf`` and ``expect``
        interfaces. Defaults to the standard normal distribution.
    return_edges
        If true, also return the bin edges.
    **kwargs
        Keyword arguments passed to the distribution's ``ppf`` and ``expect``
        methods.

    Returns
    -------
    grid
        Conditional expectations within the bins.
    pmf
        Normalized probability masses of the bins.
    edges
        Bin edges, returned when ``return_edges`` is true.
    """
    if dist is None:
        from scipy.stats import norm

        dist = norm

    if q is not None:
        q_arr = np.atleast_1d(q)
        n_bins = len(q_arr) - 1
    else:
        n_bins = 1 if n is None else n
        # Create equidistant bins in terms of quantile ranks.
        q_arr = np.linspace(0.0, 1.0, n_bins + 1)

    edges = np.asarray(dist.ppf(q_arr, **kwargs))
    grid = np.empty(n_bins)
    pmf = q_arr[1:] - q_arr[:-1]
    pmf /= np.sum(pmf)

    for i in range(n_bins):
        lb, ub = edges[i], edges[i + 1]

        # Compute the conditional expectation.
        xcond = dist.expect(
            lambda x: x,
            lb=lb,
            ub=ub,
            conditional=True,
            **kwargs,
        )
        grid[i] = float(xcond)

    if return_edges:
        return grid, pmf, edges

    return grid, pmf
