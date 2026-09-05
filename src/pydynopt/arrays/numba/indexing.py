"""
Low-level indexing routines for Numba code.

- Conversion of flat indices to multidimensional sub-indices (ind2sub)
- Conversion of multidimensional coordinates to flat indices (sub2ind)
- Specialized implementations for scalar and array arguments, with optional axis selection

This work is licensed under CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/

Author: Richard Foltyn
"""

from collections.abc import Sequence

import numpy as np

from pydynopt.numba import JIT_OPTIONS_INLINE, register_jitable

__all__ = [
    'ind2sub_array',
    'ind2sub_array_impl',
    'ind2sub_axis_array',
    'ind2sub_axis_array_impl',
    'ind2sub_axis_scalar',
    'ind2sub_axis_scalar_impl',
    'ind2sub_scalar',
    'ind2sub_scalar_impl',
    'sub2ind_array',
    'sub2ind_array_impl',
    'sub2ind_scalar',
]


def ind2sub_array(
    indices: np.ndarray,
    shape: Sequence[int],
    axis: int | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert an array of flat indices into multidimensional coordinates.

    Parameters
    ----------
    indices
        Integer array whose elements are flat indices into an array of dimensions
        ``shape``.
    shape
        Shape of the array to use for unraveling indices.
    axis
        Ignored, present to ensure compatible function signatures.
    out
        Optional output array.

    Returns
    -------
    Array of coordinates of shape ``(len(shape), len(indices))``.
    """
    unravel_ndim = len(shape)
    n = len(indices)

    coords = (
        out if out is not None else np.empty((unravel_ndim, n), dtype=indices.dtype)
    )

    coords = ind2sub_array_impl(indices, shape, 0, coords)

    return coords


@register_jitable(**JIT_OPTIONS_INLINE)
def ind2sub_array_impl(
    indices: np.ndarray,
    shape: Sequence[int],
    axis: int | None,
    out: np.ndarray,
) -> np.ndarray:
    """
    Unravel an array of flat indices into coordinates in place.

    Parameters
    ----------
    indices
        Integer array whose elements are flat indices into an array of dimensions
        ``shape``.
    shape
        Shape of the array to use for unraveling indices.
    axis
        Ignored, present to ensure compatible function signatures.
    out
        Pre-allocated output array to store the coordinates.

    Returns
    -------
    Array of coordinates of shape ``(len(shape), len(indices))``.
    """
    unravel_ndim = len(shape)
    unravel_size = 1
    for i in range(unravel_ndim):
        unravel_size *= shape[i]

    n = len(indices)

    for i in range(n):
        val = indices[i]

        if val < 0 or val >= unravel_size:
            msg = 'Invalid flat index'
            raise ValueError(msg)

        for j in range(unravel_ndim - 1, -1, -1):
            k = shape[j]
            tmp = val // k
            out[j, i] = val % k
            val = tmp

    return out


def ind2sub_axis_array(
    indices: np.ndarray,
    shape: Sequence[int],
    axis: int | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert an array of flat indices into coordinates along a specific axis.

    Parameters
    ----------
    indices
        Integer array whose elements are flat indices into an array of dimensions
        ``shape``.
    shape
        Shape of the array to use for unraveling indices.
    axis
        Axis along which coordinate array should be returned. If None,
        coordinates for the leading axis (0) are returned.
    out
        Optional output array.

    Returns
    -------
    Coordinate array along the requested axis.
    """
    n = len(indices)
    laxis = 0 if axis is None else axis

    coords = out if out is not None else np.empty((n,), dtype=indices.dtype)

    coords = ind2sub_axis_array_impl(indices, shape, laxis, coords)

    return coords


@register_jitable(**JIT_OPTIONS_INLINE)
def ind2sub_axis_array_impl(
    indices: np.ndarray,
    shape: Sequence[int],
    axis: int,
    out: np.ndarray,
) -> np.ndarray:
    """
    Unravel an array of flat indices along a specific axis in place.

    Parameters
    ----------
    indices
        Integer array whose elements are flat indices into an array of dimensions
        ``shape``.
    shape
        Shape of the array to use for unraveling indices.
    axis
        Axis along which coordinates are extracted.
    out
        Pre-allocated output array to store the coordinates.

    Returns
    -------
    Coordinate array along the requested axis.
    """
    unravel_ndim = len(shape)
    unravel_size = 1
    for i in range(unravel_ndim):
        unravel_size *= shape[i]

    n = len(indices)

    for i in range(n):
        val = indices[i]

        if val < 0 or val >= unravel_size:
            msg = 'Invalid flat index'
            raise ValueError(msg)

        for j in range(unravel_ndim - 1, -1, -1):
            k = shape[j]
            tmp = val // k
            if j == axis:
                out[i] = val % k
                break
            val = tmp

    return out


def ind2sub_scalar(
    indices: int,
    shape: Sequence[int],
    axis: int | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert a scalar flat index into coordinates across all dimensions.

    Parameters
    ----------
    indices
        Index into the flattened version of an array of dimensions ``shape``.
    shape
        Shape of the array to use for unraveling indices.
    axis
        Ignored, present to ensure compatible function signatures.
    out
        Optional output array.

    Returns
    -------
    Coordinate array containing coordinates along all dimensions.
    """
    unravel_ndim = len(shape)

    coords = (
        out
        if out is not None
        else np.empty((unravel_ndim,), dtype=np.asarray(indices).dtype)
    )

    coords = ind2sub_scalar_impl(indices, shape, 0, coords)

    return coords


@register_jitable(**JIT_OPTIONS_INLINE)
def ind2sub_scalar_impl(
    indices: int,
    shape: Sequence[int],
    axis: int | None,
    out: np.ndarray,
) -> np.ndarray:
    """
    Unravel a scalar flat index into coordinates in place.

    Parameters
    ----------
    indices
        Index into the flattened version of an array of dimensions ``shape``.
    shape
        Shape of the array to use for unraveling indices.
    axis
        Ignored, present to ensure compatible function signatures.
    out
        Pre-allocated output array to store coordinates.

    Returns
    -------
    Coordinate array containing coordinates along all dimensions.
    """
    unravel_ndim = len(shape)
    val = indices

    for j in range(unravel_ndim - 1, -1, -1):
        k = shape[j]
        tmp = val // k
        out[j] = val % k
        val = tmp

    if val >= shape[0]:
        msg = 'Invalid flat index'
        raise ValueError(msg)

    return out


@register_jitable(**JIT_OPTIONS_INLINE)
def ind2sub_axis_scalar(
    indices: int,
    shape: Sequence[int],
    axis: int | None,
    out: np.ndarray,
) -> int:
    """
    Convert a flat index into a coordinate for the given axis, writing into output.

    Parameters
    ----------
    indices
        Index into the flattened version of an array of dimensions ``shape``.
    shape
        Shape of the array to use for unraveling indices.
    axis
        Axis along which coordinate should be returned. If None,
        coordinates for the leading axis (0) are returned.
    out
        Array to store coordinate along requested axis as its first element.

    Returns
    -------
    Coordinate along the requested axis.
    """
    lout = ind2sub_axis_scalar_impl(indices, shape, axis)
    out[0] = lout
    return lout


@register_jitable(**JIT_OPTIONS_INLINE)
def ind2sub_axis_scalar_impl(
    indices: int,
    shape: Sequence[int],
    axis: int | None = None,
    out: np.ndarray | None = None,
) -> int:
    """
    Convert a flat index into a coordinate for the given axis.

    Parameters
    ----------
    indices
        Index into the flattened version of an array of dimensions ``shape``.
    shape
        Shape of the array to use for unraveling indices.
    axis
        Axis along which coordinate should be returned. If None,
        coordinates for the leading axis (0) are returned.
    out
        Ignored, present to ensure compatible function signatures.

    Returns
    -------
    Coordinate along the requested axis.
    """
    laxis = 0 if axis is None else axis

    unravel_ndim = len(shape)
    unravel_size = 1
    for i in range(unravel_ndim):
        unravel_size *= shape[i]

    val = indices

    if val < 0 or val >= unravel_size:
        msg = 'Invalid flat index'
        raise ValueError(msg)

    lout = 0

    for j in range(unravel_ndim - 1, -1, -1):
        k = shape[j]
        tmp = val // k
        if j == laxis:
            lout = val % k
            break
        val = tmp

    return lout


def sub2ind_array(
    coords: np.ndarray,
    shape: Sequence[int],
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert a 2D array of coordinates into an array of flat indices.

    Parameters
    ----------
    coords
        Two-dimensional integer array of coordinates. Each row contains
        the coordinates for one dimension.
    shape
        Shape of array into which indices from ``coords`` apply.
    out
        Optional output array of flat indices.

    Returns
    -------
    Array of indices into the flattened array.
    """
    if out is not None:
        sub2ind_array_impl(coords, shape, out)
        return out

    shp = coords.shape[1:]
    lout = np.empty(shp, dtype=coords.dtype)
    sub2ind_array_impl(coords, shape, lout)
    return lout


@register_jitable(**JIT_OPTIONS_INLINE)
def sub2ind_array_impl(
    coords: np.ndarray,
    shape: Sequence[int],
    out: np.ndarray,
) -> None:
    """
    Compute flat indices from a 2D coordinate array in place.

    Parameters
    ----------
    coords
        Two-dimensional integer array of coordinates. Each row contains
        the coordinates for one dimension.
    shape
        Shape of array into which indices from ``coords`` apply.
    out
        Array to store flat indices in place.
    """
    ndim = len(shape)
    stride = np.empty(ndim, dtype=np.int64)
    stride[-1] = 1

    for j in range(1, ndim):
        stride[ndim - j - 1] = shape[ndim - j] * stride[ndim - j]

    out[...] = 0
    out_flat = out.reshape((-1,))
    coords_flat = coords.reshape((-1, ndim))

    n = coords_flat.shape[0]

    for i in range(n):
        for j in range(ndim):
            stride_j = stride[j]

            k = coords_flat[i, j]
            if k < 0 or k >= shape[j]:
                msg = 'Invalid coordinates'
                raise ValueError(msg)
            out_flat[i] += k * stride_j


@register_jitable(**JIT_OPTIONS_INLINE)
def sub2ind_scalar(
    coords: Sequence[int] | np.ndarray,
    shape: Sequence[int],
    out: object = None,
) -> int:
    """
    Convert a sequence of coordinates into an index into a flat array.

    Parameters
    ----------
    coords
        One-dimensional array or sequence of coordinates into a multidimensional array.
    shape
        Shape of array into which indices from ``coords`` apply.
    out
        Ignored, present for API compatibility.

    Returns
    -------
    Index into the flattened array.
    """
    ndim = len(shape)
    if len(coords) != ndim:
        msg = 'Incompatible coordinate array size'
        raise ValueError(msg)

    lidx = 0
    stride_ = 1
    for j in range(ndim - 1, -1, -1):
        lidx += int(coords[j]) * stride_
        stride_ *= shape[j]

    return lidx
