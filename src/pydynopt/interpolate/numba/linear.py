"""
Numba implementations of linear interpolation routines.

NOTE: Do not add @jit decorators as that does not work with @overload.

Author: Richard Foltyn
"""

import numpy as np

from pydynopt.numba import jit, register_jitable
from .search import bsearch_impl


@register_jitable(parallel=False)
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


@register_jitable(parallel=False)
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

    lind_out = np.empty_like(x, dtype=np.int64) if index_out is None else index_out
    lwgt_out = np.empty_like(x, dtype=np.float64) if weight_out is None else weight_out

    lind_out_flat = lind_out.reshape((-1, ))
    lwgt_out_flat = lwgt_out.reshape((-1, ))

    for i, xi in enumerate(x.flat):
        ilb = bsearch_impl(xi, xp, ilb)
        lind_out_flat[i] = ilb

        # Interpolation weight on lower bound
        wgt_lb = (xp[ilb + 1] - xi)/(xp[ilb + 1] - xp[ilb])
        lwgt_out_flat[i] = wgt_lb

    return lind_out, lwgt_out


@register_jitable(parallel=False)
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


@register_jitable(parallel=False)
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

    lout = np.empty_like(weight, dtype=fp.dtype) if out is None else out

    index_flat = index.reshape((-1, ))
    weight_flat = weight.reshape((-1, ))
    out_flat = lout.reshape((-1, ))

    for i in range(out_flat.size):
        ilb, wgt = index_flat[i], weight_flat[i]

        fx = wgt*fp[ilb] + (1.0 - wgt)*fp[ilb + 1]

        if not extrapolate:
            if wgt > 1.0:
                fx = left
            elif wgt < 0.0:
                fx = right

        out_flat[i] = fx

    return lout


def interp1d_scalar(x, xp, fp, ilb=0, extrapolate=True, left=np.nan,
                    right=np.nan, out=None):
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

    ilb, wgt = interp1d_locate_scalar(x, xp, ilb)
    fx = interp1d_eval_scalar(ilb, wgt, fp, extrapolate, left, right, out)

    return fx


def interp1d_array(x, xp, fp, ilb=0, extrapolate=True, left=np.nan, right=np.nan,
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

    lout = np.empty_like(x, dtype=x.dtype) if out is None else out
    lout_flat = lout.reshape((-1, 1))

    for i, xi in enumerate(x.flat):
        ilb, wgt = interp1d_locate_scalar(xi, xp, ilb)
        fx = interp1d_eval_scalar(ilb, wgt, fp, extrapolate, left, right)

        lout_flat[i] = fx

    return lout


def interp2d_locate_scalar(x0, x1, xp0, xp1, ilb=None, index_out=None,
                           weight_out=None):

    if index_out is None:
        lind_out = np.empty(2, dtype=np.int64)
    else:
        lind_out = index_out

    if weight_out is None:
        lwgt_out = np.empty(2, dtype=np.float64)
    else:
        lwgt_out = weight_out

    if ilb is not None:
        lilb = np.zeros(2, dtype=np.int64)
    else:
        lilb = ilb

    interp2d_locate_scalar_impl(x0, x1, xp0, xp1, lilb, lind_out, lwgt_out)

    return lind_out, lwgt_out


@register_jitable(nogil=True)
def interp2d_locate_scalar_impl(x0, x1, xp0, xp1, ilb, index_out, weight_out):

    ilb0, ilb1 = ilb[0], ilb[1]

    ilb0, wgt0 = interp1d_locate_scalar(x0, xp0, ilb0)
    ilb1, wgt1 = interp1d_locate_scalar(x1, xp1, ilb1)

    index_out[0], index_out[1] = ilb0, ilb1
    weight_out[0], weight_out[1] = wgt0, wgt1

    return index_out, weight_out


def interp2d_locate_array(x0, x1, xp0, xp1, ilb=None, index_out=None,
                          weight_out=None):

    shp = tuple(x0.shape) + (2, )

    lind_out = np.empty(shp, dtype=np.int64) if index_out is None else index_out
    lwgt_out = np.empty(shp, dtype=x0.dtype) if weight_out is None else weight_out

    lind_out_flat = lind_out.reshape((-1, 2))
    lwgt_out_flat = lwgt_out.reshape((-1, 2))

    ilb0 = 0
    ilb1 = 0

    if ilb is not None:
        ilb0, ilb1 = ilb[0], ilb[1]

    for i, (x0i, x1i) in enumerate(zip(x0.flat, x1.flat)):
        ilb0 = bsearch_impl(x0i, xp0, ilb0)
        ilb1 = bsearch_impl(x1i, xp1, ilb1)

        wgt0 = (xp0[ilb0 + 1] - x0i)/(xp0[ilb0 + 1] - xp0[ilb0])
        wgt1 = (xp1[ilb1 + 1] - x1i)/(xp1[ilb1 + 1] - xp1[ilb1])

        lind_out_flat[i, 0] = ilb0
        lind_out_flat[i, 1] = ilb1

        lwgt_out_flat[i, 0] = wgt0
        lwgt_out_flat[i, 1] = wgt1

    return lind_out, lwgt_out


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

    lout = np.empty_like(weight[..., 0], dtype=fp.dtype) if out is None else out

    index_flat = index.reshape((-1, 2))
    weight_flat = weight.reshape((-1, 2))
    lout_flat = lout.reshape((-1, ))

    for i in range(lout.size):
        wgt0, wgt1 = weight_flat[i, 0], weight_flat[i, 1]
        ilb0, ilb1 = index_flat[i, 0], index_flat[i, 1]

        if not extrapolate:
            if wgt0 < 0.0 or wgt0 > 1.0 or wgt1 < 0.0 or wgt1 > 1.0:
                fx = np.nan
                lout_flat[i] = fx
                continue

        # Interpolate in dimension 0
        fx0_lb = wgt0*fp[ilb0, ilb1] + (1.0 - wgt0)*fp[ilb0 + 1, ilb1]
        fx0_ub = wgt0*fp[ilb0, ilb1 + 1] + (1.0 - wgt0)*fp[ilb0 + 1, ilb1 + 1]

        # Interpolate in dimension 1
        fx = wgt1*fx0_lb + (1.0 - wgt1)*fx0_ub

        lout_flat[i] = fx

    return lout


def interp2d_scalar(x0, x1, xp0, xp1, fp, ilb=None, extrapolate=True, out=None):

    index = np.empty(2, dtype=np.int64)
    weight = np.empty(2, dtype=xp0.dtype)

    interp2d_locate_scalar_jit(x0, x1, xp0, xp1, ilb, index, weight)
    fx = interp2d_eval_scalar_jit(index, weight, fp, extrapolate)

    return fx


def interp2d_array(x0, x1, xp0, xp1, fp, ilb=None, extrapolate=True, out=None):

    lout = np.empty_like(x0) if out is None else out

    lout_flat = lout.reshape((-1, 1))

    lilb = np.zeros(2, dtype=np.int64)
    wgt = np.zeros(2, dtype=x0.dtype)

    if ilb is not None:
        lilb[:] = ilb

    for i, (x0i, x1i) in enumerate(zip(x0, x1)):
        interp2d_locate_scalar_jit(x0i, x1i, xp0, xp1, lilb, lilb, wgt)
        fx = interp2d_eval_scalar_jit(lilb, wgt, fp, extrapolate)
        lout_flat[i] = fx

    return lout

