"""
Module that implements routines for linear interpolation.

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from pydynopt.numba import jit, JIT_OPTIONS, overload
from .numba.linear import (
    interp1d_array,
    interp1d_eval_array,
    interp1d_eval_scalar,
    interp1d_locate_array,
    interp1d_locate_scalar,
    interp1d_scalar,
    interp2d_array,
    interp2d_eval_array,
    interp2d_eval_scalar,
    interp2d_locate_array,
    interp2d_locate_scalar,
    interp2d_scalar,
    interp1d_locate_array_alloc,
    interp1d_eval_array_alloc,
)

__all__ = [
    'interp1d_locate',
    'interp1d_eval',
    'interp1d',
    'interp2d_locate',
    'interp2d_eval',
    'interp2d',
]

# Add @jit wrappers around Numba implementations of interpolation routines
interp1d_locate_jit = jit(interp1d_locate_array, **JIT_OPTIONS)
interp1d_eval_jit = jit(interp1d_eval_array, **JIT_OPTIONS)
interp1d_jit = jit(interp1d_array, **JIT_OPTIONS)

interp2d_locate_jit = jit(interp2d_locate_array, **JIT_OPTIONS)
interp2d_eval_jit = jit(interp2d_eval_array, **JIT_OPTIONS)
interp2d_jit = jit(interp2d_array, **JIT_OPTIONS)


def interp1d_locate(
    x: ArrayLike,
    xp: np.ndarray,
    ilb: int = 0,
    index_out: Optional[np.ndarray] = None,
    weight_out: Optional[np.ndarray] = None,
):
    """
    Python wrapper around Numba implementation of 1d interpolation search.

    Note: This function cannot be made directly available as a Numba function
    via @register_jitable() since the dimensions of inputs and outputs vary.

    Parameters
    ----------
    x
    xp
    ilb
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


@overload(interp1d_locate, jit_options=JIT_OPTIONS)
def _ov_interp1d_locate(x, xp, ilb=0, index_out=None, weight_out=None):
    """
    Overload for inter1d_locate with a scalar argument.
    """
    from numba import types

    f = None

    if isinstance(x, types.Number):
        f = interp1d_locate_scalar
    elif isinstance(x, types.Array):
        f = interp1d_locate_array_alloc

    return f


def interp1d_eval(
    index: np.ndarray | np.number,
    weight: ArrayLike | np.number,
    fp: ArrayLike,
    extrapolate: bool = True,
    left: np.floating = np.nan,
    right: np.floating = np.nan,
    out: Optional[np.ndarray] = None,
):
    """
    Python wrapper around Numba implementation of 1d interpolation.

    Note: This function cannot be made directly available as a Numba function
    via @register_jitable() since the dimensions of inputs and outputs vary.

    Parameters
    ----------
    index
    weight
    fp
    extrapolate
    left : float
    right : float
    out

    Returns
    -------

    """

    ilb = np.atleast_1d(index)
    wgt_lb = np.atleast_1d(weight)

    if ilb.ndim != wgt_lb.ndim or np.any(ilb.shape != wgt_lb.shape):
        msg = 'Arguments index and weight have non-conformable shapes'
        raise ValueError(msg)

    if out is None:
        out = np.empty_like(wgt_lb, dtype=np.float64)

    # Use numba-fied function to perform actual evaluation
    interp1d_eval_jit(ilb, wgt_lb, fp, extrapolate, left, right, out)

    if np.isscalar(index):
        out = out.item()

    return out


@overload(interp1d_eval, jit_options=JIT_OPTIONS)
def _ov_interp1d_eval_array(
    index, weight, fp, extrapolate=True, left=np.nan, right=np.nan, out=None
):
    from numba import types

    f = None
    if isinstance(index, types.Number):
        f = interp1d_eval_scalar
    elif isinstance(index, types.Array):
        f = interp1d_eval_array_alloc

    return f


def interp1d(
    x: ArrayLike | np.number,
    xp: ArrayLike,
    fp: ArrayLike,
    ilb: int = 0,
    extrapolate: bool = True,
    left: np.floating = np.nan,
    right: np.floating = np.nan,
    out: Optional[np.ndarray[np.floating]] = None,
    axis: int = -1,
):
    """
    Python wrapper around Numba implementation of 1d interpolation.

    Note: This function cannot be made directly available as a Numba function
    via @register_jitable() since the dimensions of inputs and outputs vary.

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
    x1d = np.ascontiguousarray(np.atleast_1d(x))

    if xp.shape[0] < 2:
        msg = 'Invalid input array xp'
        raise ValueError(msg)

    if np.atleast_2d(fp).shape[axis] != xp.shape[0]:
        msg = 'Non-conformable arrays xp, fp'
        raise ValueError(msg)

    # Recover "true" axis along which to interpolate
    if axis < 0:
        axis += fp.ndim

    out_shp = list(fp.shape[:axis]) + list(fp.shape[axis + 1 :]) + list(x1d.shape)
    out_shp = tuple(out_shp)

    # Move interpolation axis to the very end, reshape into two dimensions
    # with the interpolation axis last.
    fp_work = fp
    if fp.ndim > 1:
        fp_work = np.moveaxis(fp, axis, -1)
    fp_work = fp_work.reshape((-1, xp.shape[0]))

    # Allocate output array if required
    if out is None:
        out = np.empty(out_shp, dtype=x1d.dtype)
    else:
        shp_ok = np.all(np.array(out_shp) == out.shape)
        if not shp_ok:
            msg = 'Non-conformable output array shape, expected {}'
            raise ValueError(msg.format(out_shp))

    # Reshape output array such that sample points are along last axis.
    # Manually compute size of first dimension to correctly handle 0-sized
    # input arrays.
    d0 = out.size // x1d.size if x1d.size > 0 else 1
    shp = tuple(np.hstack((d0, x1d.shape)))
    out_work = out.reshape(shp)

    # Find interpolation indices and weights: this has to be done only once
    # and applied to all remaining axis of fp
    index = np.empty_like(x1d, dtype=np.int64)
    weight = np.empty_like(x1d, dtype=out.dtype)

    interp1d_locate_jit(x1d, xp, ilb, index_out=index, weight_out=weight)

    fp1d = np.empty_like(fp_work[0])
    out1d = np.empty_like(out_work[0])
    for i in range(fp_work.shape[0]):
        # Copy into contiguous array
        fp1d[:] = fp_work[i]
        interp1d_eval_jit(index, weight, fp1d, extrapolate, left, right, out1d)
        out_work[i] = out1d

    if fp.ndim > 1 and axis != (fp.ndim - 1):
        # Move interpolating axis back to where it was
        out = np.moveaxis(out, -1, axis)
        out = np.ascontiguousarray(out)

    if np.isscalar(x):
        out = out.item()

    return out


@overload(interp1d, jit_options=JIT_OPTIONS)
def _interp1d_generic(
    x, xp, fp, ilb=0, extrapolate=True, left=np.nan, right=np.nan, out=None
):
    from numba import types

    f = None

    if isinstance(x, types.Number):
        f = interp1d_scalar
    elif isinstance(x, types.Array):
        f = interp1d_array

    return f


@overload(interp1d, jit_options=JIT_OPTIONS)
def _interp1d_impl_generic(
    x, xp, fp, out, ilb=0, extrapolate=True, left=np.nan, right=np.nan
):
    from numba import types

    f = None

    from .numba.linear import interp1d_array_impl

    if isinstance(x, types.Number):
        pass
    elif isinstance(x, types.Array):
        f = interp1d_array_impl

    return f


def interp2d_locate(x0, x1, xp0, xp1, ilb=None, index_out=None, weight_out=None):

    xx0 = np.atleast_1d(x0)
    xx1 = np.atleast_1d(x1)

    xx0, xx1 = np.broadcast_arrays(xx0, xx1)

    if np.any(xx0.shape != xx1.shape):
        msg = 'Non-conformable sample data arrays x0, x1'
        raise ValueError(msg)

    shp = None

    if index_out is None or weight_out is None:
        shp = list(x0.shape) + [2]

    if index_out is None:
        index_out = np.empty(shp, dtype=np.int64)

    if weight_out is None:
        weight_out = np.empty(shp, dtype=x0.dtype)

    index_out = np.atleast_2d(index_out)
    weight_out = np.atleast_2d(weight_out)

    interp2d_locate_jit(xx0, xx1, xp0, xp1, ilb, index_out, weight_out)

    if np.isscalar(x0) and np.isscalar(x1):
        index_out = index_out.reshape((-1,))
        weight_out = weight_out.reshape((-1,))

    return index_out, weight_out


def interp2d_eval(index, weight, fp, extrapolate=True, out=None):

    if out is None:
        shp = index.shape[:-1]
        out = np.empty(shp, dtype=fp.dtype)

    interp2d_eval_jit(index, weight, fp, extrapolate, out)

    if index.ndim == 1:
        out = out.item()

    return out


def interp2d(x0, x1, xp0, xp1, fp, ilb=None, extrapolate=True, out=None):
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
    ilb : np.narray or None
        Optional initial guess for search routine used to locate interpolating
        bracket.
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
    interp2d_jit(xx0, xx1, xp0, xp1, fp, ilb, extrapolate, out)

    if np.isscalar(x0) and np.isscalar(x1):
        out = out.item()

    return out


@overload(interp2d, jit_options=JIT_OPTIONS)
def _interp2d_generic(x0, x1, xp0, xp1, fp, ilb=None, extrapolate=True, out=None):
    from numba import types

    f = None

    if isinstance(x0, types.Number):
        f = interp2d_scalar
    elif isinstance(x0, types.Array):
        f = interp2d_array

    return f


@overload(interp2d_locate, jit_options=JIT_OPTIONS)
def _interp2d_locate_generic(
    x0, x1, xp0, xp1, ilb=None, index_out=None, weight_out=None
):
    from numba import types

    f = None

    if isinstance(x0, types.Number):
        if ilb is None or index_out is None or weight_out is None:
            f = interp2d_locate_scalar
    elif isinstance(x0, types.Array):
        f = interp2d_locate_array

    return f


@overload(interp2d_locate, jit_options=JIT_OPTIONS)
def _interp2d_locate_impl_generic(x0, x1, xp0, xp1, ilb, index_out, weight_out):
    from numba import types

    from .numba.linear import interp2d_locate_scalar_impl

    f = None

    if isinstance(x0, types.Number):
        f = interp2d_locate_scalar_impl

    return f


@overload(interp2d_eval, jit_options=JIT_OPTIONS)
def _interp2d_eval_generic(index, weight, fp, extrapolate=True, out=None):

    from numba import types

    f = None

    # For whatever reason, index might be inferred as optional type, so
    # first recover underlying type if necessary.
    if isinstance(index, types.Optional):
        index = index.type

    if isinstance(index, types.Array) and index.ndim == 1:
        f = interp2d_eval_scalar
    elif isinstance(index, types.Array):
        f = interp2d_eval_array

    return f
