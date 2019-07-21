"""
Module that implements routines for linear interpolation.

Author: Richard Foltyn
"""

import numpy as np

from pydynopt.numba import jit
from .numba.linear import interp1d_eval_array, interp1d_locate_array
from .numba.linear import interp1d_array

from .numba.linear import interp2d_locate_array, interp2d_eval_array
from .numba.linear import interp2d_array

# Add @jit wrappers around Numba implementations of interpolation routines
interp1d_locate_jit = jit(interp1d_locate_array, nopython=True)
interp1d_eval_jit = jit(interp1d_eval_array, nopython=True)
interp1d_jit = jit(interp1d_array, nopython=True)

interp2d_locate_jit = jit(interp2d_locate_array, nopython=True)
interp2d_eval_jit = jit(interp2d_eval_array, nopython=True)
interp2d_jit = jit(interp2d_array, nopython=True)


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


def interp2d_locate(x0, x1, xp0, xp1, ilb=None, index_out=None,
                    weight_out=None):

    xx0 = np.atleast_1d(x0)
    xx1 = np.atleast_1d(x1)

    xx0, xx1 = np.broadcast_arrays(xx0, xx1)

    if np.any(xx0.shape !=  xx1.shape):
        msg = 'Non-conformable sample data arrays x0, x1'
        raise ValueError(msg)

    shp = None

    if index_out is None or weight_out is None:
        shp = list(x0.shape) + [2]

    if index_out is None:
        index_out = np.empty(shp, dtype=np.int64)

    if weight_out is None:
        weight_out = np.empty(shp, dtype=x0.dtype)

    interp2d_locate_jit(xx0, xx1, xp0, xp1, ilb, index_out, weight_out)

    if np.isscalar(x0) and np.isscalar(x1):
        index_out = index_out.flatten()
        weight_out = weight_out.flatten()

    return index_out, weight_out


def interp2d_eval(index, weight, fp, extrapolate=True, out=None):

    if out is None:
        shp = index.shape[:-1]
        out = np.empty(shp, dtype=fp.dtype)

    interp2d_eval_jit(index, weight, fp, extrapolate, out)

    if index.ndim == 1:
        out = out.item()

    return out


def interp2d(x0, x1, xp0, xp1, fp, extrapolate=True, out=None):
    """
    Perform bilinear interpolation at given sample points.

    This is just a wrapper around scipy's implementation in interpn() and is
    only provided such that the Numba-fied version can overload this function,
    and so that non-Numba-fied code works as well.

    Should not be used in code which will never be run via Numba.

    Parameters
    ----------
    x0 : float or np.ndarray
        Sample points in first dimension
    x1 : float or np.ndarray
        Sample points in second dimension
    xp0 : np.ndarray
        Grid in first dimension
    xp1 : np.ndarray
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

    xx0 = np.atleast_1d(x0)
    xx1 = np.atleast_1d(x1)

    xx0, xx1 = np.broadcast_arrays(xx0, xx1)

    if xp0.shape[0] != fp.shape[0] or xp1.shape[0] != fp.shape[1]:
        msg = 'Non-conformable input arrays'
        raise ValueError(msg)

    if any(n < 2 for n in fp.shape):
        msg = 'At least two grid points needed in each dimension!'
        raise ValueError(msg)

    if np.any(xx0.shape != xx1.shape):
        msg = 'Non-conformable sample data arrays x0, x1'
        raise ValueError(msg)

    if out is None:
        out = np.empty_like(xx0)

    # Let Numba version perform the actual work
    interp2d_jit(xx0, xx1, xp0, xp1, fp, extrapolate, out)

    if np.isscalar(x0) and np.isscalar(x1):
        out = out.item()

    return out
