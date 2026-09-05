"""
Routines for converting between flat indices and multidimensional coordinates.

- Flat index to coordinates conversion (ind2sub)
- Coordinates to flat index conversion (sub2ind)
- JIT-compiled dispatchers and implementations for scalar and array inputs

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from pydynopt.arrays.numba.indexing import (
    ind2sub_array,
    ind2sub_array_impl,
    ind2sub_axis_array,
    ind2sub_axis_array_impl,
    ind2sub_axis_scalar,
    ind2sub_axis_scalar_impl,
    ind2sub_scalar,
    ind2sub_scalar_impl,
    sub2ind_array,
    sub2ind_array_impl,
    sub2ind_scalar,
)
from pydynopt.numba import JIT_OPTIONS, JIT_OPTIONS_INLINE, jit, overload

__all__ = [
    'ind2sub',
    'sub2ind',
]

ind2sub_scalar_jit = jit(ind2sub_scalar, **JIT_OPTIONS)
ind2sub_array_jit = jit(ind2sub_array, **JIT_OPTIONS)
ind2sub_scalar_impl_jit = jit(ind2sub_scalar_impl, **JIT_OPTIONS)
ind2sub_array_impl_jit = jit(ind2sub_array_impl, **JIT_OPTIONS)
ind2sub_axis_scalar_jit = jit(ind2sub_axis_scalar, **JIT_OPTIONS)
ind2sub_axis_array_jit = jit(ind2sub_axis_array, **JIT_OPTIONS)
ind2sub_axis_scalar_impl_jit = jit(ind2sub_axis_scalar_impl, **JIT_OPTIONS)
ind2sub_axis_array_impl_jit = jit(ind2sub_axis_array_impl, **JIT_OPTIONS)

sub2ind_scalar_jit = jit(sub2ind_scalar, **JIT_OPTIONS)
sub2ind_array_jit = jit(sub2ind_array, **JIT_OPTIONS)


def ind2sub(
    indices: int | Sequence[int] | np.ndarray,
    shape: Sequence[int],
    axis: int | None = None,
    out: np.ndarray | None = None,
) -> int | np.ndarray:
    """
    Convert flat indices into coordinate arrays.

    Equivalent to NumPy's ``unravel_index()``, but with fewer features and thus
    potentially faster.

    Parameters
    ----------
    indices
        An integer or integer array whose elements are flat indices into an array
        of dimensions ``shape``.
    shape
        Shape of the array to use for unraveling indices.
    axis
        If specified, restricts the return value to contain only the coordinates
        along ``axis``.
    out
        Optional output array.

    Returns
    -------
    Array of coordinates or scalar coordinate along the requested axis.
    """
    if np.isscalar(indices):
        if axis is None:
            if out is None:
                # Pre-allocate array here so we don't have array dtype
                # conflicts in the JIT-able code.
                out = np.empty(len(shape), dtype=np.asarray(indices).dtype)

            out = ind2sub_scalar_impl_jit(indices, shape, axis, out)
        elif out is not None:
            # JIT-able routine writes index into first element of out!
            out = ind2sub_axis_scalar_jit(indices, shape, axis, out)
        else:
            # Implementation routine ignores out argument and only returns
            # a scalar.
            out = ind2sub_axis_scalar_impl_jit(indices, shape, axis)
    else:
        indices_arr = np.asarray(indices)
        if out is None:
            shp = (
                (len(shape), len(indices_arr)) if axis is None else (len(indices_arr),)
            )
            out = np.empty(shp, dtype=indices_arr.dtype)

        if axis is None:
            out = ind2sub_array_impl_jit(indices_arr, shape, axis, out)
        else:
            out = ind2sub_axis_array_impl_jit(indices_arr, shape, axis, out)

    return out


@overload(ind2sub, jit_options=JIT_OPTIONS_INLINE)
def ind2sub_impl_generic(
    indices: Any, shape: Any, axis: Any, out: Any
) -> Callable[..., Any] | None:
    """
    Return JIT-able function when all arguments are present.

    The routine requires that the ``out`` argument is not explicitly passed
    as None, since the JIT-able functions returned here for the most part
    assume that the ``out`` array is allocated (except for the scalar case
    when axis=None).
    """
    from numba import types

    f = None
    if isinstance(indices, types.Integer):
        if axis is not None and out is not None:
            f = ind2sub_axis_scalar
        elif axis is not None:
            f = ind2sub_scalar_impl
    elif isinstance(indices, types.Array):
        if axis is not None and out is not None:
            f = ind2sub_axis_array_impl
        elif out is not None:
            f = ind2sub_array_impl

    return f


@overload(ind2sub, jit_options=JIT_OPTIONS_INLINE)
def ind2sub_generic(
    indices: Any, shape: Any, axis: Any = None, out: Any = None
) -> Callable[..., Any] | None:
    """
    Return JIT-able function appropriate for the given arguments.

    If both axis and out are provided, this function will not be called in
    the first place! We therefore only need to handle the case when one of
    them is missing.
    """
    from numba import types

    f = None
    if isinstance(indices, types.Integer):
        f = ind2sub_axis_scalar_impl if axis is not None else ind2sub_scalar
    elif isinstance(indices, types.Array):
        f = ind2sub_axis_array if axis is not None else ind2sub_array

    return f


def sub2ind(
    coords: Sequence[int] | np.ndarray,
    shape: Sequence[int],
    out: np.ndarray | None = None,
) -> int | np.ndarray:
    """
    Convert an array of coordinates into flat indices.

    Parameters
    ----------
    coords
        Integer array or sequence of coordinates. Coordinates for each dimension
        are arranged along the first axis.
    shape
        Shape of array into which indices from ``coords`` apply.
    out
        Optional output array of flat indices.

    Returns
    -------
    Array or scalar of indices into the flattened array.
    """
    coords_arr = np.atleast_1d(coords)

    if coords_arr.ndim == 1:
        res = sub2ind_scalar_jit(coords_arr, shape, out)
    elif coords_arr.ndim == 2:
        res = sub2ind_array_jit(coords_arr, shape, out)
    else:
        msg = f'Invalid coordinates dimension: {coords_arr.ndim}'
        raise ValueError(msg)

    return res


@overload(sub2ind, jit_options=JIT_OPTIONS_INLINE)
def sub2ind_generic(
    coords: Any, shape: Any, out: Any = None
) -> Callable[..., Any] | None:
    """
    Dispatcher for sub2ind when out argument is not provided.
    """
    from numba import types

    from .numba.indexing import sub2ind_array, sub2ind_scalar

    f = None
    if out is None:
        if isinstance(coords, types.Array):
            if coords.ndim == 1:
                f = sub2ind_scalar
            elif coords.ndim >= 2:
                f = sub2ind_array
        else:
            # Assume one-dimensional sequence or tuple
            f = sub2ind_scalar

    return f


@overload(sub2ind, jit_options=JIT_OPTIONS_INLINE)
def sub2ind_impl_generic(
    coords: Any, shape: Any, out: Any
) -> Callable[..., Any] | None:
    """
    Dispatcher for sub2ind when out argument is provided.
    """
    from numba import types

    f = None
    if isinstance(coords, types.Array) and out is not None and coords.ndim >= 2:
        f = sub2ind_array_impl

    return f
