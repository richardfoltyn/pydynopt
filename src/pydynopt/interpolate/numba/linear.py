"""
Numba implementations of linear interpolation routines.

NOTE: Do not add @jit decorators as that does not work with @overload.

Author: Richard Foltyn
"""

import numpy as np

from pydynopt.numba import jit
from .search import bsearch, bsearch_impl


def interp1d_locate_scalar(x, xp, ilb=0, index_out=None, weight_out=None):
    """
    Numba implementation for computing the interpolation bracketing interval
    and weight for a scalar value.

    Parameters
    ----------
    x : float
        Sample point at which to interpolate.
    xp : np.ndarray
        Grid points representing domain over which to interpolate.
    ilb : int
        Initial guess for index of lower bound of bracketing interval.
        NOTE: No error-checking is performed on its value!
    index_out : None
        Ignored in scalar version
    weight_out : None
        Ignored in scalar version

    Returns
    -------
    ilb : int
        Index of lower bound of bracketing interval
    weight : float
        Weight on lower bound of bracketing interval
    """

    ilb = bsearch_impl(x, xp, ilb)

    weight = (xp[ilb+1] - x)/(xp[ilb+1] - xp[ilb])

    return ilb, weight


def interp1d_locate_array(x, xp, ilb=0, index_out=None, weight_out=None):
    """
    Numba implementation for computing the interpolation bracketing intervals
    and weights for array-valued arguments.

    Parameters
    ----------
    x : np.ndarray
        Sample points at which to interpolate.
    xp : np.ndarray
        Grid points representing domain over which to interpolate.
    ilb : int
        Initial guess for index of lower bound of bracketing interval of
        the first element.
        NOTE: No error-checking is performed on its value!
    index_out : np.ndarray or None
        Optional pre-allocated output array for indices of lower bounds
        of bracketing interval.
    weight_out : np.ndarray or None
        Optional pre-allocated output array for lower bound weights.

    Returns
    -------
    index_out : np.ndarray
    weight_out : np.ndarray
    """

    if index_out is None:
        index_out = np.empty_like(x, dtype=np.int64)
        weight_out = np.empty_like(x, dtype=np.float64)

    index_out_flat = index_out.reshape((-1, ))
    weight_out_flat = weight_out.reshape((-1, ))

    for i, xi in enumerate(x.flat):
        ilb = bsearch_impl(xi, xp, ilb)
        index_out_flat[i] = ilb

        # Interpolation weight on lower bound
        wgt_lb = (xp[ilb + 1] - xi)/(xp[ilb + 1] - xp[ilb])
        weight_out_flat[i] = wgt_lb

    return index_out, weight_out


def interp1d_eval_scalar(index, weight, fp, extrapolate=True, left=np.nan,
                         right=np.nan, out=None):
    """
    Numba implementation to evaluate an interpolant at a single scalar value.

    Parameters
    ----------
    index : int
        Index of lower bound of bracketing interval.
    weight : float
        Weight on lower bound of bracketing interval.
    fp : np.ndarray
        Function values defined on original grid points.
    extrapolate : bool
        If true, extrapolate values outside of domain.
    left : float
        Value to return if sample point is below the domain lower bound.
    right : float
        Value to return if sample point is above the domain upper bound.
    out : None
        Ignored for scalar sample points.

    Returns
    -------
    fx : float
        Interpolant evaluated at sample point.
    """

    fx = weight * fp[index] + (1.0 - weight) * fp[index+1]

    if not extrapolate:
        if weight > 1.0:
            fx = left
        elif weight < 0.0:
            fx = right

    return fx


def interp1d_eval_array(index, weight, fp, extrapolate=True, left=np.nan,
                        right=np.nan, out=None):
    """
    Numba implementation to evaluate an intepolant and multiple sample points.

    Parameters
    ----------
    index : np.ndarray
        Indices of lower bounds of bracketing intervals.
    weight : np.ndarray
        Weights on lower bounds of bracketing intervals.
    fp : np.ndarray
        Function values defined on original grid points.
    extrapolate : bool
        If true, extrapolate values outside of domain.
    left : float
        Value to return if sample point is below the domain lower bound.
    right : float
        Value to return if sample point is above the domain upper bound.
    out : np.ndarray or None
        Optional pre-allocated output array.

    Returns
    -------
    out : np.ndarray
        Interpolant evaluated at sample points.
    """

    if out is None:
        out = np.empty_like(weight, dtype=np.float64)

    index_flat = index.reshape((-1, ))
    weight_flat = weight.reshape((-1, ))
    out_flat = out.reshape((-1, ))

    for i in range(out_flat.shape[0]):
        wgt_lb = weight_flat[i]
        ilb = index_flat[i]

        fx = wgt_lb * fp[ilb] + (1.0 - weight) * fp[ilb+1]

        if not extrapolate:
            if wgt_lb > 1.0:
                out[i] = left
            elif wgt_lb < 0.0:
                out[i] = right

        out_flat[i] = fx

    return out


interp1d_locate_scalar_jit = jit(interp1d_locate_scalar, nopython=True)
interp1d_eval_scalar_jit = jit(interp1d_eval_scalar, nopython=True)


def interp1d_scalar(x, xp, fp, extrapolate=True, left=np.nan, right=np.nan,
                    out=None):
    """
    Combined routine to both locate and evaluate linear interpolant
    at a single sample point.

    Parameters
    ----------
    x
    xp
    fp
    extrapolate
    left
    right
    out

    Returns
    -------

    """

    ilb, wgt = interp1d_locate_scalar_jit(x, xp)
    fx = interp1d_eval_scalar_jit(ilb, wgt, fp, extrapolate, left, right, out)

    return fx


def interp1d_array(x, xp, fp, extrapolate=True, left=np.nan, right=np.nan,
                   out=None):
    """
    Combined routine to both locate and evaluate linear interpolant
    at a collection of sample points.

    Parameters
    ----------
    x
    xp
    fp
    extrapolate
    left
    right
    out

    Returns
    -------

    """

    if out is None:
        out = np.empty_like(x, dtype=np.float64)

    out_flat = out.reshape((-1, ))

    ilb = 0
    for i, xi in enumerate(x.flat):
        ilb, wgt = interp1d_locate_scalar_jit(xi, xp, ilb)
        fx = interp1d_eval_scalar_jit(ilb, wgt, fp, extrapolate, left, right, out)

        out_flat[i] = fx

    return out


def interp_bilinear_scalar(x1, x2, xp1, xp2, fp, extrapolate=True, out=None):
    """
    Perform bilinear interpolation at given tuple of scalar sample points.

    Parameters
    ----------
    x1 : float
        Sample point in first dimension
    x2 : float
        Sample point in second dimension
    xp1 : np.ndarray
        Grid in first dimension
    xp2 : np.ndarray
        Grid in second dimension
    fp : np.ndarray
        Function evaluated at Cartesian product of `xp1` and  `xp2`
    extrapolate : bool
        If true, extrapolate values at points outside of given domain. Otherwise
        non-interior points will be set to NaN.
    out : None
        Ignored by scalar implementation, present for compatibility with
        array-valued function.

    Returns
    -------
    out : float
        Interpolated function value at given sample point
    """

    if not extrapolate:
        interior = xp1[0] <= x1 <= xp1[-1] and xp2[0] <= x2 <= xp2[-1]
        if not interior:
            fx = np.nan
            return fx

    ilb1 = bsearch(x1, xp1)
    ilb2 = bsearch(x2, xp2)

    # Interpolate in dimension 1
    w1 = (x1 - xp1[ilb1]) / (xp1[ilb1+1] - xp1[ilb1])
    w2 = (x2 - xp2[ilb2]) / (xp2[ilb2+1] - xp2[ilb2])

    fx1_lb = (1.0-w1) * fp[ilb1,ilb2] + w1 * fp[ilb1+1,ilb2]
    fx1_ub = (1.0-w1) * fp[ilb1,ilb2+1] + w1 * fp[ilb1+1,ilb2+1]

    # Interpolate in dimension 2
    fx = (1.0 - w2) * fx1_lb + w2 * fx1_ub

    return fx


def interp_bilinear_array(x1, x2, xp1, xp2, fp, extrapolate=True, out=None):
    """
    Perform bilinear interpolation at array-valued collection of sample points.

    Parameters
    ----------
    x1 : np.ndarray
        Sample points in first dimension
    x2 : np.ndarray
        Sample points in second dimension
    xp1 : np.ndarray
        Grid in first dimension
    xp2 : np.ndarray
        Grid in second dimension
    fp : np.ndarray
        Function evaluated at Cartesian product of `xp1` and  `xp2`
    extrapolate : bool
        If true, extrapolate values at points outside of given domain. Otherwise
        non-interior points will be set to NaN.
    out : np.ndarray or None
        Optional output array

    Returns
    -------
    out : np.ndarray
        Interpolated function values at given sample points
    """

    nx = x1.shape[0]

    if out is None:
        out = np.empty_like(x1)

    for i in range(nx):
        if not extrapolate:
            interior = xp1[0] <= x1[i] <= xp1[-1] and xp2[0] <= x2[i] <= xp2[-1]
            if not interior:
                out[i] = np.nan
                continue

        ilb1 = bsearch(x1[i], xp1)
        ilb2 = bsearch(x2[i], xp2)

        # Interpolate in dimension 1
        w1 = (x1[i] - xp1[ilb1])/(xp1[ilb1 + 1] - xp1[ilb1])
        w2 = (x2[i] - xp2[ilb2])/(xp2[ilb2 + 1] - xp2[ilb2])

        fx1_lb = (1.0 - w1)*fp[ilb1, ilb2] + w1*fp[ilb1 + 1, ilb2]
        fx1_ub = (1.0 - w1)*fp[ilb1, ilb2 + 1] + w1*fp[ilb1 + 1, ilb2 + 1]

        # Interpolate in dimension 2
        out[i] = (1.0 - w2)*fx1_lb + w2*fx1_ub

    return out


# @overload(interp_bilinear)
def interp_bilinear_generic(x1, x2, xp1, xp2, fp, extrapolate=True, out=None):
    """
    Generic interface around interp_bilinear() to be used to return the correct
    function to numbafy depending on argument types.

    Parameters
    ----------
    x1
    x2
    xp1
    xp2
    fp
    extrapolate
    out

    Returns
    -------
    fcn : callable or None
    """

    fcn = None

    if all(isinstance(x, numba.types.scalars.Number) for x in (x1, x2)):
        fcn = interp_bilinear_scalar
    elif all(isinstance(x, numba.types.npytypes.Array) for x in (x1, x2)):
        fcn = interp_bilinear_array

    return fcn
