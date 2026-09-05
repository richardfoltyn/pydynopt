"""
Routines for 1D and 2D linear interpolation.

- 1D linear interpolation, bracket location, and interpolant evaluation
- 2D bilinear interpolation, bracket location, and interpolant evaluation
- Numba overload dispatchers for scalar and array inputs

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from pydynopt.numba import JIT_OPTIONS, jit, overload

from .numba.linear import (
    interp1d_array,
    interp1d_array_impl,
    interp1d_eval_array,
    interp1d_eval_array_alloc,
    interp1d_eval_scalar,
    interp1d_locate_array,
    interp1d_locate_array_alloc,
    interp1d_locate_scalar,
    interp1d_scalar,
    interp2d_array,
    interp2d_eval_array,
    interp2d_eval_scalar,
    interp2d_locate_array,
    interp2d_locate_scalar,
    interp2d_locate_scalar_impl,
    interp2d_scalar,
)

__all__ = [
    'interp1d',
    'interp1d_eval',
    'interp1d_locate',
    'interp2d',
    'interp2d_eval',
    'interp2d_locate',
]

# Add @jit wrappers around Numba implementations of interpolation routines
interp1d_locate_jit = jit(interp1d_locate_array, **JIT_OPTIONS)
interp1d_eval_jit = jit(interp1d_eval_array, **JIT_OPTIONS)
interp1d_jit = jit(interp1d_array, **JIT_OPTIONS)

interp2d_locate_jit = jit(interp2d_locate_array, **JIT_OPTIONS)
interp2d_eval_jit = jit(interp2d_eval_array, **JIT_OPTIONS)
interp2d_jit = jit(interp2d_array, **JIT_OPTIONS)


def interp1d_locate(
    x: Sequence[float] | np.ndarray | float,
    xp: np.ndarray,
    ilb: int = 0,
    index_out: np.ndarray | None = None,
    weight_out: np.ndarray | None = None,
) -> tuple[int, float] | tuple[np.ndarray, np.ndarray]:
    """
    Locate bracketing interval indices and lower bound weights for 1D interpolation.

    Parameters
    ----------
    x
        Sample points at which to interpolate.
    xp
        Grid points representing the domain over which to interpolate.
    ilb
        Initial guess for index of the bracketing interval lower bound.
    index_out
        Optional pre-allocated output array for lower bound indices.
    weight_out
        Optional pre-allocated output array for lower bound weights.

    Returns
    -------
    index_out
        Lower bound indices of bracketing intervals.
    weight_out
        Weights on lower bounds of bracketing intervals.
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
        return int(index_out.item()), float(weight_out.item())

    return index_out, weight_out


@overload(interp1d_locate, jit_options=JIT_OPTIONS)
def _ov_interp1d_locate(
    x: Any,
    xp: Any,
    ilb: Any = 0,
    index_out: Any = None,
    weight_out: Any = None,
) -> Callable[..., Any] | None:
    from numba import types

    f = None

    if isinstance(x, types.Number):
        f = interp1d_locate_scalar
    elif isinstance(x, types.Array):
        f = interp1d_locate_array_alloc

    return f


def interp1d_eval(
    index: np.ndarray | int,
    weight: Sequence[float] | np.ndarray | float,
    fp: np.ndarray,
    extrapolate: bool = True,
    left: float = np.nan,
    right: float = np.nan,
    out: np.ndarray | None = None,
) -> float | np.ndarray:
    """
    Evaluate a 1D linear interpolant using pre-computed indices and weights.

    Parameters
    ----------
    index
        Lower bound indices of bracketing intervals.
    weight
        Weights on lower bounds of bracketing intervals.
    fp
        Function values defined on original grid points.
    extrapolate
        If True, extrapolate values outside the domain. If False, return
        ``left`` or ``right`` for out-of-bounds points.
    left
        Value to return if sample point is below the domain lower bound.
    right
        Value to return if sample point is above the domain upper bound.
    out
        Optional pre-allocated output array.

    Returns
    -------
    Interpolated values at the sample points.
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
        return float(out.item())

    return out


@overload(interp1d_eval, jit_options=JIT_OPTIONS)
def _ov_interp1d_eval_array(
    index: Any,
    weight: Any,
    fp: Any,
    extrapolate: Any = True,
    left: Any = np.nan,
    right: Any = np.nan,
    out: Any = None,
) -> Callable[..., Any] | None:
    from numba import types

    f = None
    if isinstance(index, types.Number):
        f = interp1d_eval_scalar
    elif isinstance(index, types.Array):
        f = interp1d_eval_array_alloc

    return f


def interp1d(
    x: Sequence[float] | np.ndarray | float,
    xp: np.ndarray,
    fp: np.ndarray,
    ilb: int = 0,
    extrapolate: bool = True,
    left: float = np.nan,
    right: float = np.nan,
    out: np.ndarray | None = None,
    axis: int = -1,
) -> float | np.ndarray:
    """
    Perform 1D linear interpolation.

    Parameters
    ----------
    x
        Sample points at which to interpolate.
    xp
        Grid points representing the domain over which to interpolate.
    fp
        Function values defined on original grid points.
    ilb
        Initial guess for index of the bracketing interval lower bound.
    extrapolate
        If True, extrapolate values outside the domain. If False, return
        ``left`` or ``right`` for out-of-bounds points.
    left
        Value to return if sample point is below the domain lower bound.
    right
        Value to return if sample point is above the domain upper bound.
    out
        Optional pre-allocated output array.
    axis
        Axis along which to interpolate.

    Returns
    -------
    Interpolated values at the sample points.
    """
    x1d = np.ascontiguousarray(np.atleast_1d(x))

    if xp.shape[0] < 2:
        msg = 'Invalid input array xp'
        raise ValueError(msg)

    if np.atleast_2d(fp).shape[axis] != xp.shape[0]:
        msg = 'Non-conformable arrays xp, fp'
        raise ValueError(msg)

    # Recover "true" axis along which to interpolate
    actual_axis = axis + fp.ndim if axis < 0 else axis

    out_shp = (
        list(fp.shape[:actual_axis])
        + list(fp.shape[actual_axis + 1 :])
        + list(x1d.shape)
    )
    out_shp_tuple = tuple(out_shp)

    # Move interpolation axis to the very end, reshape into two dimensions
    # with the interpolation axis last.
    fp_work = fp
    if fp.ndim > 1:
        fp_work = np.moveaxis(fp, actual_axis, -1)
    fp_work = fp_work.reshape((-1, xp.shape[0]))

    # Allocate output array if required
    if out is None:
        out = np.empty(out_shp_tuple, dtype=x1d.dtype)
    else:
        shp_ok = np.all(np.array(out_shp_tuple) == out.shape)
        if not shp_ok:
            msg = f'Non-conformable output array shape, expected {out_shp_tuple}'
            raise ValueError(msg)

    # Reshape output array such that sample points are along last axis.
    # Manually compute size of first dimension to correctly handle 0-sized
    # input arrays.
    d0 = out.size // x1d.size if x1d.size > 0 else 1
    shp = (d0, *x1d.shape)
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

    if fp.ndim > 1 and actual_axis != (fp.ndim - 1):
        # Move interpolating axis back to where it was
        out = np.moveaxis(out, -1, actual_axis)
        out = np.ascontiguousarray(out)

    if np.isscalar(x):
        return float(out.item())

    return out


@overload(interp1d, jit_options=JIT_OPTIONS)
def _interp1d_generic(
    x: Any,
    xp: Any,
    fp: Any,
    ilb: Any = 0,
    extrapolate: Any = True,
    left: Any = np.nan,
    right: Any = np.nan,
    out: Any = None,
) -> Callable[..., Any] | None:
    from numba import types

    f = None

    if isinstance(x, types.Number):
        f = interp1d_scalar
    elif isinstance(x, types.Array):
        f = interp1d_array

    return f


@overload(interp1d, jit_options=JIT_OPTIONS)
def _interp1d_impl_generic(
    x: Any,
    xp: Any,
    fp: Any,
    out: Any,
    ilb: Any = 0,
    extrapolate: Any = True,
    left: Any = np.nan,
    right: Any = np.nan,
) -> Callable[..., Any] | None:
    from numba import types

    f = None

    if isinstance(x, types.Number):
        pass
    elif isinstance(x, types.Array):
        f = interp1d_array_impl

    return f


def interp2d_locate(
    x0: Sequence[float] | np.ndarray | float,
    x1: Sequence[float] | np.ndarray | float,
    xp0: np.ndarray,
    xp1: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None = None,
    index_out: np.ndarray | None = None,
    weight_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Locate bracketing interval indices and weights for 2D bilinear interpolation.

    Parameters
    ----------
    x0
        Sample points in first dimension.
    x1
        Sample points in second dimension.
    xp0
        Grid in first dimension.
    xp1
        Grid in second dimension.
    ilb
        Optional initial guess for lower bound indices in each dimension.
    index_out
        Optional pre-allocated output array for lower bound indices.
    weight_out
        Optional pre-allocated output array for lower bound weights.

    Returns
    -------
    index_out
        Lower bound indices in each dimension.
    weight_out
        Weights on lower bounds in each dimension.
    """
    xx0 = np.atleast_1d(x0)
    xx1 = np.atleast_1d(x1)

    if xx0.shape != xx1.shape:
        msg = 'Non-conformable sample data arrays x0, x1'
        raise ValueError(msg)

    xx0, xx1 = np.broadcast_arrays(xx0, xx1)

    shp = (*xx0.shape, 2)

    if index_out is None:
        index_out = np.empty(shp, dtype=np.int64)

    if weight_out is None:
        weight_out = np.empty(shp, dtype=xx0.dtype)

    index_out_2d = np.atleast_2d(index_out)
    weight_out_2d = np.atleast_2d(weight_out)

    interp2d_locate_jit(xx0, xx1, xp0, xp1, ilb, index_out_2d, weight_out_2d)

    if np.isscalar(x0) and np.isscalar(x1):
        return index_out.reshape((-1,)), weight_out.reshape((-1,))

    return index_out, weight_out


def interp2d_eval(
    index: np.ndarray,
    weight: np.ndarray,
    fp: np.ndarray,
    extrapolate: bool = True,
    out: np.ndarray | None = None,
) -> float | np.ndarray:
    """
    Evaluate a 2D bilinear interpolant using pre-computed indices and weights.

    Parameters
    ----------
    index
        Lower bound indices in each dimension.
    weight
        Weights on lower bounds in each dimension.
    fp
        Function values evaluated on the Cartesian product of ``xp0`` and ``xp1``.
    extrapolate
        If True, extrapolate values outside domain. Otherwise non-interior
        points will be set to NaN.
    out
        Optional pre-allocated output array.

    Returns
    -------
    Interpolated function values at given sample points.
    """
    if out is None:
        shp = index.shape[:-1]
        out = np.empty(shp, dtype=fp.dtype)

    interp2d_eval_jit(index, weight, fp, extrapolate, out)

    if index.ndim == 1:
        return float(out.item())

    return out


def interp2d(
    x0: Sequence[float] | np.ndarray | float,
    x1: Sequence[float] | np.ndarray | float,
    xp0: np.ndarray,
    xp1: np.ndarray,
    fp: np.ndarray,
    ilb: Sequence[int] | np.ndarray | None = None,
    extrapolate: bool = True,
    out: np.ndarray | None = None,
) -> float | np.ndarray:
    """
    Perform bilinear interpolation at given sample points.

    Parameters
    ----------
    x0
        Sample points in first dimension.
    x1
        Sample points in second dimension.
    xp0
        Grid in first dimension.
    xp1
        Grid in second dimension.
    fp
        Function evaluated at Cartesian product of ``xp0`` and ``xp1``.
    ilb
        Optional initial guess for search routine used to locate interpolating
        bracket.
    extrapolate
        If True, extrapolate values at points outside of given domain. Otherwise
        non-interior points will be set to NaN.
    out
        Optional pre-allocated output array.

    Returns
    -------
    Interpolated function values at given sample points.
    """
    xx0 = np.atleast_1d(x0)
    xx1 = np.atleast_1d(x1)

    if xx0.shape != xx1.shape:
        msg = 'Non-conformable sample data arrays x0, x1'
        raise ValueError(msg)

    xx0, xx1 = np.broadcast_arrays(xx0, xx1)

    if xp0.shape[0] != fp.shape[0] or xp1.shape[0] != fp.shape[1]:
        msg = 'Non-conformable input arrays'
        raise ValueError(msg)

    if any(n < 2 for n in fp.shape):
        msg = 'At least two grid points needed in each dimension!'
        raise ValueError(msg)

    if out is None:
        out = np.empty_like(xx0)

    # Let Numba version perform the actual work
    interp2d_jit(xx0, xx1, xp0, xp1, fp, ilb, extrapolate, out)

    if np.isscalar(x0) and np.isscalar(x1):
        return float(out.item())

    return out


@overload(interp2d, jit_options=JIT_OPTIONS)
def _interp2d_generic(
    x0: Any,
    x1: Any,
    xp0: Any,
    xp1: Any,
    fp: Any,
    ilb: Any = None,
    extrapolate: Any = True,
    out: Any = None,
) -> Callable[..., Any] | None:
    from numba import types

    f = None

    if isinstance(x0, types.Number):
        f = interp2d_scalar
    elif isinstance(x0, types.Array):
        f = interp2d_array

    return f


@overload(interp2d_locate, jit_options=JIT_OPTIONS)
def _interp2d_locate_generic(
    x0: Any,
    x1: Any,
    xp0: Any,
    xp1: Any,
    ilb: Any = None,
    index_out: Any = None,
    weight_out: Any = None,
) -> Callable[..., Any] | None:
    from numba import types

    f = None

    if isinstance(x0, types.Number):
        if ilb is None or index_out is None or weight_out is None:
            f = interp2d_locate_scalar
    elif isinstance(x0, types.Array):
        f = interp2d_locate_array

    return f


@overload(interp2d_locate, jit_options=JIT_OPTIONS)
def _interp2d_locate_impl_generic(
    x0: Any,
    x1: Any,
    xp0: Any,
    xp1: Any,
    ilb: Any,
    index_out: Any,
    weight_out: Any,
) -> Callable[..., Any] | None:
    from numba import types

    f = None

    if isinstance(x0, types.Number):
        f = interp2d_locate_scalar_impl

    return f


@overload(interp2d_eval, jit_options=JIT_OPTIONS)
def _interp2d_eval_generic(
    index: Any,
    weight: Any,
    fp: Any,
    extrapolate: Any = True,
    out: Any = None,
) -> Callable[..., Any] | None:
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
