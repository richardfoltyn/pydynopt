"""
Numba implementations of linear interpolation routines.

NOTE: Do not add @jit decorators as that does not work with @overload.

Author: Richard Foltyn
"""

import numpy as np

from pydynopt.numba import jit
from .search import bsearch_impl


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
    if weight_out is None:
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
        out = np.empty_like(weight, dtype=fp.dtype)

    index_flat = index.reshape((-1, ))
    weight_flat = weight.reshape((-1, ))
    out_flat = out.reshape((-1, ))

    for i in range(out_flat.size):
        ilb, wgt = index_flat[i], weight_flat[i]
        fx = interp1d_eval_scalar_jit(ilb, wgt, fp, extrapolate, left, right)

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
        out = np.empty_like(x, dtype=x.dtype)

    out_flat = out.reshape((-1, ))

    ilb = 0
    for i, xi in enumerate(x.flat):
        ilb, wgt = interp1d_locate_scalar_jit(xi, xp, ilb)
        fx = interp1d_eval_scalar_jit(ilb, wgt, fp, extrapolate, left, right, out)

        out_flat[i] = fx

    return out


def interp2d_locate_scalar(x0, x1, xp0, xp1, ilb=None, index_out=None,
                           weight_out=None):

    if index_out is None:
        index_out = np.empty(2, dtype=np.int64)
    if weight_out is None:
        weight_out = np.empty(2, dtype=np.float64)

    ilb0 = 0
    ilb1 = 0

    if ilb is not None:
        ilb0, ilb1 = ilb[0], ilb[1]

    ilb0, wgt0 = interp1d_locate_scalar_jit(x0, xp0, ilb0)
    ilb1, wgt1 = interp1d_locate_scalar_jit(x1, xp1, ilb1)

    index_out[0] = ilb0
    index_out[1] = ilb1

    weight_out[0] = wgt0
    weight_out[1] = wgt1

    return index_out, weight_out


def interp2d_locate_array(x0, x1, xp0, xp1, ilb=None, index_out=None,
                          weight_out=None):

    if index_out is None:
        # Numba does not support constructing tuples to be passed as
        # shape arguments to Numpy's array construction routines, so
        # use some tricks
        index_out = np.stack((np.empty_like(x0, dtype=np.int64),
                              np.empty_like(x0, dtype=np.int64)), axis=-1)
    if weight_out is None:
        weight_out = np.stack((np.empty_like(x0), np.empty_like(x0)), axis=-1)

    index_out_flat = index_out.reshape((-1, 2))
    weight_out_flat = weight_out.reshape((-1, 2))

    ilb0 = 0
    ilb1 = 0

    if ilb is not None:
        ilb0, ilb1 = ilb[0], ilb[1]

    for i, (x0i, x1i) in enumerate(zip(x0.flat, x1.flat)):
        ilb0, wgt0 = interp1d_locate_scalar_jit(x0i, xp0, ilb0)
        ilb1, wgt1 = interp1d_locate_scalar_jit(x1i, xp1, ilb1)

        index_out_flat[i, 0] = ilb0
        index_out_flat[i, 1] = ilb1

        weight_out_flat[i, 0] = wgt0
        weight_out_flat[i, 1] = wgt1

    return index_out, weight_out


def interp2d_eval_scalar(index, weight, fp, extrapolate=True, out=None):

    # interpolate in dimension 0

    wgt0, wgt1 = weight[0], weight[1]
    ilb0, ilb1 = index[0], index[1]

    if not extrapolate:
        if not np.all(weight >= 0.0) or not np.all(weight <= 1.0):
            fx = np.nan
            return fx

    fx0_lb = wgt0 * fp[ilb0, ilb1] + (1.0-wgt0) * fp[ilb0+1, ilb1]
    fx0_ub = wgt0 * fp[ilb0, ilb1+1] + (1.0-wgt0) * fp[ilb0+1, ilb1+1]

    # Interpolate in dimension 1
    fx = wgt1 * fx0_lb + (1.0 - wgt1) * fx0_ub

    return fx


interp2d_locate_scalar_jit = jit(interp2d_locate_scalar, nopython=True)
interp2d_eval_scalar_jit = jit(interp2d_eval_scalar, nopython=True)


def interp2d_eval_array(index, weight, fp, extrapolate=True, out=None):

    if out is None:
        # Numba does not support constructing tuples to be passed as
        # shape arguments to Numpy's array construction routines, so
        # use some tricks
        out = np.empty_like(weight[..., 0], dtype=fp.dtype)

    index_flat = index.reshape((-1, 2))
    weight_flat = weight.reshape((-1, 2))
    out_flat = out.reshape((-1, ))

    for i in range(out.size):
        fx = interp2d_eval_scalar_jit(index_flat[i], weight_flat[i], fp,
                                      extrapolate=extrapolate)
        out_flat[i] = fx

    return out


def interp2d_scalar(x0, x1, xp0, xp1, fp, ilb=None, extrapolate=True, out=None):

    index = np.empty(2, dtype=np.int64)
    weight = np.empty(2, dtype=x0.dtype)

    interp2d_locate_scalar_jit(x0, x1, xp0, xp1, ilb, index, weight)
    fx = interp2d_eval_scalar_jit(index, weight, fp, extrapolate)

    return fx


def interp2d_array(x0, x1, xp0, xp1, fp, extrapolate=True, out=None):

    if out is None:
        out = np.empty_like(x0, dtype=x0.dtype)

    out_flat = out.reshape((-1, 1))

    ilb = np.zeros(2, dtype=np.int64)
    wgt = np.zeros(2, dtype=x0.dtype)

    for i, (x0i, x1i) in enumerate(zip(x0, x1)):
        interp2d_locate_scalar_jit(x0i, x1i, xp0, xp1, ilb, ilb, wgt)
        fx = interp2d_eval_scalar_jit(ilb, wgt, fp, extrapolate)
        out_flat[i] = fx

    return out

