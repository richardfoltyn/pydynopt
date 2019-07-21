"""
Module that implements routines for linear interpolation.

Author: Richard Foltyn
"""

import numpy as np
from scipy.interpolate import interpn

from pydynopt.numba import jit
from .numba.linear import interp1d_eval_array, interp1d_locate_array
from .numba.linear import interp1d_array

# Add @jit wrappers around Numba implementations of interpolation routines
interp1d_locate_jit = jit(interp1d_locate_array, nopython=True)
interp1d_eval_jit = jit(interp1d_eval_array, nopython=True)
interp1d_jit = jit(interp1d_array, nopython=True)


def interp1d_locate(x, xp, ilb=0, index_out=None, weight_out=None):
    """

    Parameters
    ----------
    x
    xp
    index_out
    weight_out

    Returns
    -------

    """

    xx = np.atleast_1d(x)

    if xp.shape[0] < 2:
        msg = 'Invalid input array xp'
        raise ValueError(msg)

    if index_out is None:
        index_out = np.empty_like(xx, dtype=np.int64)
    if weight_out is None:
        weight_out = np.empty_like(xx, dtype=np.float64)

    ilb = max(0, min(xp.shape[0] - 2, ilb))

    # Use Numba-fied implementation to do the actual work
    interp1d_locate_jit(xx, xp, ilb, index_out, weight_out)

    if np.isscalar(x):
        index_out = index_out.item()
        weight_out = weight_out.item()

    return index_out, weight_out


def interp1d_eval(index, weight, fp, extrapolate=True,
                  left=np.nan, right=np.nan, out=None):
    """

    Parameters
    ----------
    index
    weight
    fp
    extrapolate
    out

    Returns
    -------

    """

    ilb = np.atleast_1d(index)
    wgt_lb = np.atleast_1d(weight)

    if out is None:
        out = np.empty_like(wgt_lb, dtype=np.float64)

    # Use numba-fied function to perform actual evaluation
    interp1d_eval_jit(ilb, wgt_lb, fp, extrapolate, left, right, out)

    if np.isscalar(index):
        out = out.item()

    return out


def interp1d(x, xp, fp, extrapolate=True, left=np.nan, right=np.nan,
             out=None):
    """

    Parameters
    ----------
    x
    xp
    fp
    extrapolate
    out

    Returns
    -------

    """
    xx = np.atleast_1d(x)

    if xp.shape[0] < 2:
        msg = 'Invalid input array xp'
        raise ValueError(msg)

    if out is None:
        out = np.empty_like(xx, dtype=np.float64)

    interp1d_jit(x, xp, fp, extrapolate, left, right, out)

    if np.isscalar(x):
        out = out.item()

    return out


def interp_bilinear(x1, x2, xp1, xp2, fp, extrapolate=True, out=None):
    """
    Perform bilinear interpolation at given sample points.

    This is just a wrapper around scipy's implementation in interpn() and is
    only provided such that the Numba-fied version can overload this function,
    and so that non-Numba-fied code works as well.

    Should not be used in code which will never be run via Numba.

    Parameters
    ----------
    x1 : float or np.ndarray
        Sample points in first dimension
    x2 : float or np.ndarray
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
    out : float or np.ndarray
        Interpolated function values at given sample points
    """

    isscalar = np.isscalar(x1) and np.isscalar(x2)
    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)

    xx = np.vstack((x1, x2)).T

    if out is None:
        out = np.empty_like(x1)

    fill_value = None if extrapolate else np.nan
    out[:] = interpn((xp1, xp2), fp, xx, bounds_error=False,
                     fill_value=fill_value)

    if isscalar:
        out = out.item()

    return out
